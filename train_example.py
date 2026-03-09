"""Example training script for SimpleViT with different activation functions.

Usage:
    # Train with GELU (baseline)
    python train_example.py --activation gelu --epochs 20

    # Train with learnable rational activations
    python train_example.py --activation rational --epochs 20 --activation-lr 1e-3

    # Quick smoke test
    python train_example.py --activation gelu --epochs 1 --fast-dev-run
"""

import argparse
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.model import SimpleViT
from src.data import get_cifar10


def main():
    parser = argparse.ArgumentParser(description="Train SimpleViT on CIFAR-10")
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu", "silu", "rational"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--activation-lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "warmup_cosine"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Small ViT config for CIFAR-10 (32x32 images, patch size 4)
    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=4,
        mlp_dim=512,
        channels=3,
        dim_head=64,
        activation=args.activation,
        lr=args.lr,
        activation_lr=args.activation_lr,
        scheduler=args.scheduler,
    )

    train_loader, val_loader, test_loader, _ = get_cifar10(
        batch_size=args.batch_size, num_workers=4,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-{epoch}-{val_loss:.3f}"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    accelerator = "auto"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        precision="32-true",
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
