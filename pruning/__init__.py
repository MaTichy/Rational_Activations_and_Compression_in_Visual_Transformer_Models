"""Iterative structured pruning via the Lottery Ticket Hypothesis."""

from pruning.lottery_ticket import run_iterative_pruning, calculate_per_iteration_prune_ratio
from pruning.compress import compress_pruned_model

__all__ = ["run_iterative_pruning", "calculate_per_iteration_prune_ratio", "compress_pruned_model"]
