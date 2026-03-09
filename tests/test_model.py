"""Tests for SimpleViT model."""

import pytest
import torch
from src.model import SimpleViT
from tests.conftest import TINY_MODEL_KWARGS


class TestSimpleViT:
    def test_forward_shape(self, tiny_model, dummy_batch):
        images, _ = dummy_batch
        logits = tiny_model(images)
        assert logits.shape == (4, 10)

    def test_rational_forward(self, tiny_rational_model, dummy_batch):
        images, _ = dummy_batch
        logits = tiny_rational_model(images)
        assert logits.shape == (4, 10)

    @pytest.mark.parametrize("activation", ["gelu", "relu", "silu", "rational"])
    def test_all_activations(self, activation, dummy_batch):
        model = SimpleViT(**TINY_MODEL_KWARGS, activation=activation)
        images, _ = dummy_batch
        logits = model(images)
        assert logits.shape == (4, 10)

    def test_backward_pass(self, tiny_model, dummy_batch):
        images, labels = dummy_batch
        logits = tiny_model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        for p in tiny_model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_from_config(self):
        from src.config import CIFAR10_CONFIG
        model = SimpleViT.from_config(CIFAR10_CONFIG)
        x = torch.randn(2, 3, 32, 32)
        assert model(x).shape == (2, 10)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            SimpleViT(**TINY_MODEL_KWARGS, activation="nonexistent")

    def test_patch_size_assertion(self):
        with pytest.raises(AssertionError):
            SimpleViT(
                image_size=32, patch_size=7, num_classes=10,
                dim=64, depth=2, heads=2, mlp_dim=128,
            )

    def test_uses_rational_flag(self, tiny_model, tiny_rational_model):
        assert not tiny_model.uses_rational
        assert tiny_rational_model.uses_rational
