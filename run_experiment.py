"""Full experiment pipeline: Train → Prune → Quantise → Save JSON.

Compares ReLU vs Rational activations on CIFAR-10 using SimpleViT.
Results are saved incrementally to experiments/results.json.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger

from src import SimpleViT, CIFAR10_CONFIG, get_cifar10
from src.quantisation import quantise_dynamic, get_model_size_mb, evaluate_quantised
from pruning.lottery_ticket import run_iterative_pruning
from pruning.compress import compress_pruned_model

# ── Configuration ──────────────────────────────────────────────────────
SEED = 42
TRAINING_EPOCHS = 20
PRUNING_ITERATIONS = 4
EPOCHS_PER_PRUNE_ROUND = 5
TOTAL_PRUNE_FF = 0.75
TOTAL_PRUNE_ATTN = 0.50
BATCH_SIZE = 64
ACTIVATIONS = ["relu", "rational"]
EXPERIMENT_DIR = Path("experiments")
RESULTS_PATH = EXPERIMENT_DIR / "results.json"


def _save_results(results: dict) -> None:
    """Write results dict to JSON, creating directories as needed."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → Saved results to {RESULTS_PATH}")


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _read_csv_metrics(logger: CSVLogger) -> list[dict]:
    """Read epoch-level metrics from CSVLogger output."""
    import csv
    metrics_file = Path(logger.log_dir) / "metrics.csv"
    if not metrics_file.exists():
        return []
    rows = []
    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v for k, v in row.items() if v != ""})
    # CSVLogger writes step-level rows; aggregate to epoch-level
    epoch_data = {}
    for row in rows:
        epoch = row.get("epoch")
        if epoch is None:
            continue
        epoch = int(float(epoch))
        if epoch not in epoch_data:
            epoch_data[epoch] = {"epoch": epoch}
        for key in ["train_loss_epoch", "train_acc_epoch", "val_loss", "val_acc"]:
            if key in row:
                clean_key = key.replace("_epoch", "")
                epoch_data[epoch][clean_key] = float(row[key])
    # Filter out epochs with no metrics (only epoch key)
    return [d for d in epoch_data.values() if len(d) > 1]


def _detect_accelerator() -> str:
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_pipeline_for_activation(
    activation: str,
    train_loader,
    val_loader,
    test_loader,
    results: dict,
    accelerator: str,
) -> None:
    """Run the full train → prune → quantise pipeline for one activation type."""
    print(f"\n{'='*60}")
    print(f"  ACTIVATION: {activation.upper()}")
    print(f"{'='*60}")

    pl.seed_everything(SEED)
    config = replace(CIFAR10_CONFIG, activation=activation)
    model = SimpleViT.from_config(config)
    initial_weights = copy.deepcopy(model)

    # ── Phase 1: Training ──────────────────────────────────────────
    print("\n[Phase 1] Training...")
    train_logger = CSVLogger(str(EXPERIMENT_DIR / "logs"), name=f"train_{activation}")
    trainer = pl.Trainer(
        max_epochs=TRAINING_EPOCHS,
        accelerator=accelerator,
        devices=1,
        logger=train_logger,
        enable_checkpointing=False,
    )
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    training_time = round(time.time() - t0, 1)

    test_results = trainer.test(model, test_loader, verbose=False)
    test_acc = test_results[0]["test_acc"] if test_results else None

    epoch_metrics = _read_csv_metrics(train_logger)
    model_size = get_model_size_mb(model)
    param_count = _count_parameters(model)

    results["models"][activation] = {
        "training": {
            "epochs": epoch_metrics,
            "test_acc": test_acc,
            "model_size_mb": model_size,
            "param_count": param_count,
            "training_time_seconds": training_time,
        }
    }
    _save_results(results)

    # Keep a copy for quantising the unpruned model
    unpruned_model = copy.deepcopy(model)

    # ── Phase 2: Pruning (LTH) ────────────────────────────────────
    print("\n[Phase 2] Pruning (Lottery Ticket Hypothesis)...")
    prune_logger = CSVLogger(str(EXPERIMENT_DIR / "logs"), name=f"prune_{activation}")
    pruned_model, round_stats = run_iterative_pruning(
        trained_model=model,
        initial_model=initial_weights,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pruning_iterations=PRUNING_ITERATIONS,
        total_prune_pct_ff=TOTAL_PRUNE_FF,
        total_prune_pct_attn=TOTAL_PRUNE_ATTN,
        epochs_per_round=EPOCHS_PER_PRUNE_ROUND,
        accelerator=accelerator,
        logger=prune_logger,
    )

    final_pruned_acc = round_stats[-1]["test_acc_after_round"] if round_stats else None
    pruned_size = get_model_size_mb(pruned_model)
    total_sparsity = round_stats[-1]["sparsity_ff"] if round_stats else 0.0

    results["models"][activation]["pruned"] = {
        "pruning_rounds": round_stats,
        "final_test_acc": final_pruned_acc,
        "model_size_mb": pruned_size,
        "total_sparsity_ff": total_sparsity,
        "total_sparsity_attn": round_stats[-1]["sparsity_attn"] if round_stats else 0.0,
    }
    _save_results(results)

    # ── Phase 3: Structural compression ──────────────────────────
    print("\n[Phase 3] Compressing pruned model (removing zero neurons)...")
    compressed_model, compression_info = compress_pruned_model(pruned_model)
    compressed_size = get_model_size_mb(compressed_model)
    compressed_params = _count_parameters(compressed_model)

    # Evaluate compressed model
    compressed_model_device = compressed_model.to(
        "mps" if accelerator == "mps" else "cuda" if accelerator == "gpu" else "cpu"
    )
    comp_trainer = pl.Trainer(devices=1, accelerator=accelerator, logger=False)
    comp_results = comp_trainer.test(compressed_model_device, test_loader, verbose=False)
    compressed_acc = comp_results[0]["test_acc"] if comp_results else None

    results["models"][activation]["compressed"] = {
        "test_acc": compressed_acc,
        "model_size_mb": compressed_size,
        "param_count": compressed_params,
        "compression_info": compression_info,
    }
    _save_results(results)

    # ── Phase 4: Quantise unpruned model ──────────────────────────
    print("\n[Phase 4] Quantising unpruned model...")
    q_unpruned = quantise_dynamic(copy.deepcopy(unpruned_model), remove_pruning=False)
    q_unpruned_acc = evaluate_quantised(q_unpruned, test_loader)
    q_unpruned_size = get_model_size_mb(q_unpruned)

    results["models"][activation]["quantised"] = {
        "test_acc": q_unpruned_acc,
        "model_size_mb": q_unpruned_size,
    }
    _save_results(results)

    # ── Phase 5: Quantise compressed model ────────────────────────
    print("\n[Phase 5] Quantising compressed model...")
    q_compressed = quantise_dynamic(copy.deepcopy(compressed_model), remove_pruning=False)
    q_compressed_acc = evaluate_quantised(q_compressed, test_loader)
    q_compressed_size = get_model_size_mb(q_compressed)

    results["models"][activation]["compressed_quantised"] = {
        "test_acc": q_compressed_acc,
        "model_size_mb": q_compressed_size,
    }
    _save_results(results)

    print(f"\n  ✓ {activation.upper()} complete")
    print(f"    Train acc: {test_acc:.4f}  |  Pruned: {final_pruned_acc:.4f}")
    print(f"    Compressed: {compressed_acc:.4f}  |  Compressed+Q: {q_compressed_acc:.4f}")
    print(f"    Sizes: {model_size}MB → {compressed_size}MB (C) → {q_compressed_size}MB (C+Q)")


def main():
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader, _ = get_cifar10(batch_size=BATCH_SIZE)
    accelerator = _detect_accelerator()
    print(f"Accelerator: {accelerator}")

    results = {
        "metadata": {
            "seed": SEED,
            "dataset": "CIFAR-10",
            "training_epochs": TRAINING_EPOCHS,
            "pruning_iterations": PRUNING_ITERATIONS,
            "epochs_per_prune_round": EPOCHS_PER_PRUNE_ROUND,
            "total_prune_pct_ff": TOTAL_PRUNE_FF,
            "total_prune_pct_attn": TOTAL_PRUNE_ATTN,
            "batch_size": BATCH_SIZE,
            "model_config": {
                "dim": CIFAR10_CONFIG.dim,
                "depth": CIFAR10_CONFIG.depth,
                "heads": CIFAR10_CONFIG.heads,
                "mlp_dim": CIFAR10_CONFIG.mlp_dim,
                "patch_size": CIFAR10_CONFIG.patch_size,
            },
        },
        "models": {},
    }

    for activation in ACTIVATIONS:
        run_pipeline_for_activation(
            activation, train_loader, val_loader, test_loader, results, accelerator
        )

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
