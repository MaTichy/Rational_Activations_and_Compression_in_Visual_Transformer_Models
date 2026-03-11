"""Structural compression for pruned SimpleViT models.

After iterative structured pruning, zero rows/columns remain in the weight
matrices. This module physically removes them, creating a genuinely smaller
dense model with real inference speed benefits — not just theoretical sparsity.

For each transformer layer's FeedForward block:
  - net.1 (Linear dim→hidden): remove dead output neurons
  - net.3 (Linear hidden→dim): remove matching dead input neurons

A neuron is only considered dead if it is zeroed in BOTH net.1 (row) and
net.3 (column). This is critical for non-ReLU activations where f(0) != 0
(e.g. rational activations with R(0) = a_0). A neuron dead in net.1 but
alive in net.3 still contributes through f(0) * net3_column.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def compress_pruned_model(model: nn.Module) -> tuple[nn.Module, dict]:
    """Structurally compress a pruned model by removing zero neurons.

    Creates a new model with smaller FeedForward layers where pruned neurons
    have been physically removed rather than just masked to zero.

    A neuron is only removed if it is dead in BOTH net.1 (all-zero row) AND
    net.3 (all-zero column), ensuring correctness for activations where f(0)!=0.

    Args:
        model: A pruned SimpleViT model (with or without pruning hooks).

    Returns:
        Tuple of (compressed_model, compression_info) where compression_info
        contains per-layer statistics about the compression.
    """
    model = copy.deepcopy(model)

    # Make pruning permanent (remove hooks and masks)
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")

    compression_info = {"layers": []}

    for layer_idx, layer_pair in enumerate(model.transformer.layers):
        ff = layer_pair[1]  # (Attention, FeedForward)
        net1 = ff.net[1]    # Linear(dim, hidden_dim)
        net3 = ff.net[3]    # Linear(hidden_dim, dim)

        original_dim = net1.out_features

        # A neuron is alive if it has non-zero weights in EITHER net.1 or net.3.
        # This handles activations where f(0) != 0 (e.g. rational: R(0) = a_0).
        alive_in_net1 = net1.weight.data.abs().sum(dim=1) > 0
        alive_in_net3 = net3.weight.data.abs().sum(dim=0) > 0
        surviving = alive_in_net1 | alive_in_net3
        pruned_dim = surviving.sum().item()

        if pruned_dim == original_dim:
            compression_info["layers"].append({
                "layer": layer_idx,
                "original_dim": original_dim,
                "compressed_dim": original_dim,
                "removed": 0,
            })
            continue

        # Build compressed net.1: keep only surviving rows
        new_net1 = nn.Linear(net1.in_features, pruned_dim, bias=net1.bias is not None)
        new_net1.weight.data = net1.weight.data[surviving]
        if net1.bias is not None:
            new_net1.bias.data = net1.bias.data[surviving]

        # Build compressed net.3: keep only surviving columns
        new_net3 = nn.Linear(pruned_dim, net3.out_features, bias=net3.bias is not None)
        new_net3.weight.data = net3.weight.data[:, surviving]
        if net3.bias is not None:
            new_net3.bias.data = net3.bias.data.clone()

        # Replace in the sequential
        ff.net[1] = new_net1
        ff.net[3] = new_net3

        compression_info["layers"].append({
            "layer": layer_idx,
            "original_dim": original_dim,
            "compressed_dim": pruned_dim,
            "removed": original_dim - pruned_dim,
        })

    total_original = sum(l["original_dim"] for l in compression_info["layers"])
    total_compressed = sum(l["compressed_dim"] for l in compression_info["layers"])
    compression_info["total_neurons_original"] = total_original
    compression_info["total_neurons_compressed"] = total_compressed
    compression_info["neuron_reduction_pct"] = round(
        1 - total_compressed / total_original, 4
    ) if total_original > 0 else 0.0

    return model, compression_info
