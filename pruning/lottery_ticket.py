"""Lottery Ticket Hypothesis - Iterative structured pruning for SimpleViT.

Implements the iterative magnitude pruning procedure from Frankle & Carlin (2019):
  1. Train a network to completion
  2. Prune the smallest-magnitude weights (structured, by rows/columns)
  3. Reset surviving weights to their initial random values
  4. Repeat

Supports separate pruning ratios for feed-forward and attention layers,
with structured pruning along appropriate dimensions to maintain dense
sub-networks after compression.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import lightning as pl


def calculate_per_iteration_prune_ratio(total_prune_pct: float, iterations: int) -> float:
    """Calculate per-iteration pruning ratio to achieve a total pruning percentage.

    Uses the formula: p_iter = 1 - (1 - p_total)^(1/iterations)

    Args:
        total_prune_pct: Target total pruning percentage (0 to 1).
        iterations: Number of pruning iterations.

    Returns:
        Per-iteration pruning ratio.
    """
    if not 0 < total_prune_pct < 1:
        raise ValueError("total_prune_pct must be between 0 and 1")
    return round(1 - (1 - total_prune_pct) ** (1 / iterations), 4)


def _compute_sparsity(model: nn.Module) -> dict[str, float]:
    """Compute sparsity percentages for FF and attention layers."""
    ff_total = ff_zeros = attn_total = attn_zeros = 0
    for name, module in model.named_modules():
        if "transformer.layers" not in name or not isinstance(module, nn.Linear):
            continue
        w = module.weight
        if hasattr(module, "weight_mask"):
            w = w * module.weight_mask.to(w.device)
        numel = w.numel()
        zeros = (w == 0).sum().item()
        if ".net.1" in name or ".net.3" in name:
            ff_total += numel
            ff_zeros += zeros
        elif ".to_qkv" in name or ".to_out" in name:
            attn_total += numel
            attn_zeros += zeros
    return {
        "sparsity_ff": round(ff_zeros / ff_total, 4) if ff_total else 0.0,
        "sparsity_attn": round(attn_zeros / attn_total, 4) if attn_total else 0.0,
    }


def run_iterative_pruning(
    trained_model: pl.LightningModule,
    initial_model: nn.Module,
    train_loader,
    val_loader,
    test_loader=None,
    pruning_iterations: int = 3,
    total_prune_pct_ff: float = 0.75,
    total_prune_pct_attn: float = 0.50,
    norm: float = float("inf"),
    epochs_per_round: int = 30,
    devices=1,
    accelerator="auto",
    logger=None,
) -> tuple[pl.LightningModule, list[dict]]:
    """Run iterative structured pruning with weight rewinding.

    Args:
        trained_model: The trained model to prune.
        initial_model: Copy of the model at initialization (for weight rewinding).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Optional test data loader for final evaluation.
        pruning_iterations: Number of prune-retrain cycles.
        total_prune_pct_ff: Total pruning target for feed-forward layers.
        total_prune_pct_attn: Total pruning target for attention layers.
        norm: Norm type for structured pruning (e.g., 1, 2, float('inf')).
        epochs_per_round: Training epochs per pruning round.
        devices: Number of devices for Lightning Trainer.
        accelerator: Accelerator type ('gpu', 'mps', 'cpu', 'auto').
        logger: Optional Lightning logger (e.g., CSVLogger) for training logs.

    Returns:
        Tuple of (pruned_model, round_stats) where round_stats is a list of
        dicts with keys: round, sparsity_ff, sparsity_attn, test_acc_after_round.
    """
    ratio_ff = calculate_per_iteration_prune_ratio(total_prune_pct_ff, pruning_iterations)
    ratio_attn = calculate_per_iteration_prune_ratio(total_prune_pct_attn, pruning_iterations)

    print(f"Feed-forward: {total_prune_pct_ff*100:.1f}% total, {ratio_ff*100:.2f}% per iteration")
    print(f"Attention:    {total_prune_pct_attn*100:.1f}% total, {ratio_attn*100:.2f}% per iteration")

    # Keep initial model on CPU (only used for weight rewinding during pruning)
    initial_model = initial_model.cpu()
    round_stats = []

    for i in range(pruning_iterations):
        print(f"\n--- Pruning iteration {i + 1}/{pruning_iterations} ---")

        # Move model to CPU for pruning to avoid MPS/CPU device conflicts
        # in PyTorch's pruning internals (PruningContainer._combine_masks).
        # Lightning's trainer.fit() will move it back to the accelerator.
        trained_model = trained_model.cpu()

        # PyTorch pruning's forward_pre_hook caches `module.weight` as a plain
        # attribute (not a parameter/buffer) from the last forward pass on MPS.
        # model.cpu() doesn't move plain attributes, so we must recompute them.
        for _, m in trained_model.named_modules():
            if isinstance(m, nn.Linear) and hasattr(m, "weight_orig"):
                m.weight = m.weight_orig * m.weight_mask

        for name, module in trained_model.named_modules():
            if "transformer.layers" not in name or not isinstance(module, nn.Linear):
                continue

            original = dict(initial_model.named_modules())[name]

            if ".net.1" in name:
                # FF first layer: prune rows (output neurons)
                _prune_and_rewind(module, original, ratio_ff, norm, dim=0, prune_bias=True)
            elif ".net.3" in name:
                # FF second layer: prune columns (matching pruned output neurons)
                _prune_and_rewind(module, original, ratio_ff, norm, dim=1)
            elif ".to_qkv" in name:
                _prune_and_rewind(module, original, ratio_attn, norm, dim=0)
            elif ".to_out" in name:
                _prune_and_rewind(module, original, ratio_attn, norm, dim=1)

        trainer = pl.Trainer(
            max_epochs=epochs_per_round, devices=devices,
            accelerator=accelerator, logger=logger if logger else True,
        )
        trainer.fit(trained_model, train_loader, val_loader)

        # Collect per-round stats
        sparsity = _compute_sparsity(trained_model)
        round_acc = None
        if test_loader is not None:
            test_trainer = pl.Trainer(devices=devices, accelerator=accelerator, logger=False)
            results = test_trainer.test(trained_model, test_loader, verbose=False)
            if results:
                round_acc = results[0].get("test_acc")

        round_stats.append({
            "round": i + 1,
            "sparsity_ff": sparsity["sparsity_ff"],
            "sparsity_attn": sparsity["sparsity_attn"],
            "test_acc_after_round": round_acc,
        })

    return trained_model, round_stats


def _prune_and_rewind(module, original, ratio, norm, dim, prune_bias=False) -> None:
    """Apply structured pruning and rewind surviving weights to initial values."""
    prune.ln_structured(module, name="weight", n=norm, amount=ratio, dim=dim)

    mask = module.weight_mask
    module.weight.data = torch.where(mask != 0, original.weight.data, module.weight.data)

    if prune_bias and module.bias is not None:
        bias_mask = mask.sum(dim=1) != 0
        module.bias.data = torch.where(bias_mask, original.bias.data, torch.zeros_like(module.bias.data))
