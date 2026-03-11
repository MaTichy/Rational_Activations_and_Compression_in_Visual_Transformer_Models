"""Dynamic INT8 quantisation utilities for SimpleViT models.

Provides post-training dynamic quantisation targeting nn.Linear layers,
model size measurement, and CPU-based evaluation of quantised models.
"""

from __future__ import annotations

import tempfile
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization
from torchmetrics import Accuracy


def _remove_pruning_reparametrizations(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing forward hooks and reparametrisation buffers."""
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
    return model


def quantise_dynamic(model: nn.Module, remove_pruning: bool = True) -> nn.Module:
    """Apply dynamic INT8 quantisation to all nn.Linear layers.

    Args:
        model: The model to quantise (will be moved to CPU).
        remove_pruning: If True, call prune.remove() on pruned layers first.

    Returns:
        Quantised model on CPU.
    """
    model = model.cpu()
    if remove_pruning:
        _remove_pruning_reparametrizations(model)
    # Use qnnpack backend on ARM/Apple Silicon (fbgemm unavailable)
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    quantised = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantised


def get_model_size_mb(model: nn.Module) -> float:
    """Measure model size in MB by saving to a temp file.

    This captures the actual storage footprint including quantised weights.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save(model.state_dict(), f)
        tmp_path = f.name
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.remove(tmp_path)
    return round(size_mb, 2)


@torch.no_grad()
def evaluate_quantised(model: nn.Module, test_loader, num_classes: int = 10) -> float:
    """Evaluate a (quantised) model on CPU.

    Args:
        model: Model to evaluate (should already be on CPU).
        test_loader: DataLoader for test data.
        num_classes: Number of classes for accuracy metric.

    Returns:
        Test accuracy as a float (0-1).
    """
    model.eval()
    acc = Accuracy(task="multiclass", num_classes=num_classes)
    for batch in test_loader:
        x, y = batch
        x, y = x.cpu(), y.cpu()
        logits = model(x)
        acc(logits.argmax(dim=1), y)
    return acc.compute().item()
