"""Model configuration presets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for SimpleViT model and training.

    Args:
        image_size: Input image resolution.
        patch_size: Patch resolution.
        num_classes: Number of output classes.
        dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        mlp_dim: Hidden dimension of feed-forward network.
        channels: Number of input channels.
        dim_head: Dimension per attention head.
        activation: Activation identifier ('gelu', 'relu', 'silu', 'rational').
        lr: Learning rate for model parameters.
        activation_lr: Learning rate for rational activation parameters.
        scheduler: LR scheduler type ('step', 'cosine', 'warmup_cosine').
        scheduler_step_size: Step size for StepLR scheduler.
        scheduler_gamma: Gamma for StepLR scheduler.
    """

    image_size: int = 32
    patch_size: int = 4
    num_classes: int = 10
    dim: int = 256
    depth: int = 6
    heads: int = 4
    mlp_dim: int = 512
    channels: int = 3
    dim_head: int = 64
    activation: str = "gelu"
    lr: float = 3e-4
    activation_lr: float = 1e-3
    scheduler: str = "cosine"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1


CIFAR10_CONFIG = ViTConfig(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=6,
    heads=4,
    mlp_dim=512,
)

IMAGENETTE_CONFIG = ViTConfig(
    image_size=160,
    patch_size=16,
    num_classes=10,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    dim_head=64,
)
