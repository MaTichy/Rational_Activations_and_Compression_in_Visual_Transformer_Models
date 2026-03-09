"""Verification script - runs a tiny SimpleViT forward/backward pass to confirm everything works.

This script creates a minimal model and runs it on synthetic data to verify
the installation and code are functional. Works on CPU, MPS (Apple Silicon), and CUDA.
"""

import torch
import lightning as pl
from src.model import SimpleViT
from src.activations import RationalActivation


def verify_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return "cuda", "gpu"
    if torch.backends.mps.is_available():
        return "mps", "mps"
    return "cpu", "cpu"


def verify_rational_activation():
    """Test that the rational activation forward/backward works."""
    print("Testing RationalActivation...")
    act = RationalActivation(num_numerator=5, num_denominator=4)
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = act(x)
    y.sum().backward()
    assert x.grad is not None, "Gradients not computed"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Coefficients - numerator: {act.coeff_numerator.data.tolist()}")
    print(f"  Coefficients - denominator: {act.coeff_denominator.data.tolist()}")
    print("  PASSED\n")


def verify_model(activation="gelu"):
    """Test a tiny SimpleViT forward/backward pass."""
    device_name, accelerator = verify_device()
    print(f"Testing SimpleViT with activation='{activation}' on {device_name}...")

    model = SimpleViT(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
        channels=3,
        dim_head=32,
        activation=activation,
        lr=1e-3,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")

    device = torch.device(device_name)
    model = model.to(device)

    # Synthetic batch
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)

    # Forward
    logits = model(x)
    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    print(f"  Forward pass: input {x.shape} -> output {logits.shape}")

    # Backward
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    print(f"  Backward pass: loss = {loss.item():.4f}")
    print("  PASSED\n")


def verify_training():
    """Run a 2-step training loop with Lightning."""
    device_name, accelerator = verify_device()
    print(f"Testing Lightning training loop on {accelerator}...")

    model = SimpleViT(
        image_size=32, patch_size=8, num_classes=10,
        dim=64, depth=2, heads=2, mlp_dim=128,
        channels=3, dim_head=32, activation="gelu",
    )

    # Synthetic dataset
    dataset = torch.utils.data.TensorDataset(
        torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    trainer = pl.Trainer(
        max_epochs=1, accelerator=accelerator, devices=1,
        fast_dev_run=2, enable_progress_bar=False, logger=False,
    )
    trainer.fit(model, loader, loader)
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("SimpleViT Verification Suite")
    print("=" * 60)
    print()

    verify_rational_activation()
    verify_model("gelu")
    verify_model("rational")
    verify_training()

    print("=" * 60)
    print("All checks passed!")
    print("=" * 60)
