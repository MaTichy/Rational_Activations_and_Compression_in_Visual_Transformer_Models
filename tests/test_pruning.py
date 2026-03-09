"""Tests for lottery ticket pruning."""

import pytest
from pruning.lottery_ticket import calculate_per_iteration_prune_ratio


class TestPruningUtils:
    def test_prune_ratio_basic(self):
        ratio = calculate_per_iteration_prune_ratio(0.75, 3)
        assert 0 < ratio < 1
        # After 3 rounds of pruning at this ratio, ~75% should be pruned
        remaining = (1 - ratio) ** 3
        assert abs(remaining - 0.25) < 0.01

    def test_prune_ratio_single_iteration(self):
        ratio = calculate_per_iteration_prune_ratio(0.5, 1)
        assert abs(ratio - 0.5) < 0.01

    def test_prune_ratio_invalid(self):
        with pytest.raises(ValueError):
            calculate_per_iteration_prune_ratio(0.0, 3)
        with pytest.raises(ValueError):
            calculate_per_iteration_prune_ratio(1.0, 3)

    def test_prune_ratio_high_iterations(self):
        ratio = calculate_per_iteration_prune_ratio(0.90, 10)
        remaining = (1 - ratio) ** 10
        assert abs(remaining - 0.10) < 0.01
