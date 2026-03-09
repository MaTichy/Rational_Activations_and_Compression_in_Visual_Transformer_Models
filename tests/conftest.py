"""Shared test fixtures."""

import pytest
import torch
from src.model import SimpleViT


TINY_MODEL_KWARGS = dict(
    image_size=32,
    patch_size=8,
    num_classes=10,
    dim=64,
    depth=2,
    heads=2,
    mlp_dim=128,
    channels=3,
    dim_head=32,
)


@pytest.fixture
def dummy_batch():
    """A synthetic (images, labels) batch for CIFAR-10-like data."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture
def tiny_model():
    """A minimal SimpleViT with GELU activation."""
    return SimpleViT(**TINY_MODEL_KWARGS, activation="gelu")


@pytest.fixture
def tiny_rational_model():
    """A minimal SimpleViT with rational activation."""
    return SimpleViT(**TINY_MODEL_KWARGS, activation="rational")
