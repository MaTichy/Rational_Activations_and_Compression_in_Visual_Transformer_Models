"""Simple Vision Transformer (SimpleViT) with learnable activation functions.

Based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) with
simplifications from "Better plain ViT baselines for ImageNet-1k" (Beyer et al., 2022):
  - No CLS token, uses mean pooling over patch tokens
  - Sinusoidal 2D positional embeddings (not learned)
  - Pre-norm architecture (LayerNorm before attention/FFN)

Extended with support for learnable rational activation functions (Pade approximants)
that replace fixed activations like GELU, allowing the network to learn optimal
activation shapes during training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from einops import rearrange
from einops.layers.torch import Rearrange
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from src.activations import RationalActivation
from src.schedulers import WarmupStepLR, WarmupCosineLR


def pair(t: int | tuple[int, int]) -> tuple[int, int]:
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(patches: torch.Tensor, temperature: int = 10000) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    Creates position embeddings using sine/cosine functions at different frequencies
    for both spatial dimensions (height and width), concatenated together.
    """
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    assert (dim % 4) == 0, "Feature dimension must be multiple of 4 for sincos embedding"
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def _resolve_activation(identifier: str) -> nn.Module:
    """Create an activation module from a string identifier."""
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    if identifier in activations:
        return activations[identifier]()
    if identifier == "rational":
        return RationalActivation()
    raise ValueError(f"Unknown activation: {identifier}. Choose from {list(activations.keys()) + ['rational']}")


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        # Use PyTorch's optimized scaled dot-product attention (Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head),
                    FeedForward(dim, mlp_dim, activation),
                ])
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(pl.LightningModule):
    """Simple Vision Transformer with learnable activation support.

    Args:
        image_size: Input image resolution (int or tuple).
        patch_size: Patch resolution (int or tuple).
        num_classes: Number of output classes.
        dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_dim: Hidden dimension of feed-forward network.
        channels: Number of input channels (default: 3).
        dim_head: Dimension per attention head (default: 64).
        activation: Activation identifier string ('gelu', 'relu', 'silu', 'rational').
        lr: Learning rate for model parameters.
        activation_lr: Separate learning rate for rational activation parameters.
        scheduler: LR scheduler type ('step', 'cosine', 'warmup_cosine').
        scheduler_step_size: Step size for StepLR scheduler.
        scheduler_gamma: Gamma for StepLR scheduler.
    """

    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        activation="gelu",
        lr=3e-4,
        activation_lr=1e-3,
        scheduler="cosine",
        scheduler_step_size=30,
        scheduler_gamma=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by patch size."

        patch_dim = channels * patch_height * patch_width
        self.activation_id = activation
        self.lr = lr
        self.activation_lr = activation_lr
        self.scheduler_type = scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        act = _resolve_activation(activation)
        self.uses_rational = activation == "rational"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, act)
        self.linear_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    @classmethod
    def from_config(cls, config: "ViTConfig") -> "SimpleViT":
        """Create a SimpleViT from a ViTConfig dataclass."""
        from src.config import ViTConfig
        from dataclasses import asdict
        return cls(**asdict(config))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, "b ... d -> b (...) d") + pe
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.linear_head(x)

    def configure_optimizers(self):
        if self.uses_rational:
            # Use parameter groups with different learning rates in a single optimizer
            activation_param_ids = set()
            activation_params = []
            for module in self.modules():
                if isinstance(module, RationalActivation):
                    for p in module.parameters():
                        activation_params.append(p)
                        activation_param_ids.add(id(p))
            model_params = [p for p in self.parameters() if id(p) not in activation_param_ids]

            optimizer = torch.optim.AdamW([
                {"params": model_params, "lr": self.lr, "weight_decay": 0.01},
                {"params": activation_params, "lr": self.activation_lr, "weight_decay": 0.0},
            ])
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        scheduler = self._make_scheduler(optimizer)
        return [optimizer], [scheduler]

    def _make_scheduler(self, optimizer):
        if self.scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs if self.trainer else 100)
        if self.scheduler_type == "warmup_cosine":
            return WarmupCosineLR(optimizer, warmup_epochs=5, cycle_length=50)
        return StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits.argmax(dim=1), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits.argmax(dim=1), y)
        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)

