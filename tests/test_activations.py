"""Tests for learnable rational activation functions."""

import torch
import pytest
from src.activations import RationalActivation


class TestRationalActivation:
    def test_output_shape(self):
        act = RationalActivation()
        x = torch.randn(2, 8, 16)
        assert act(x).shape == x.shape

    def test_gradient_flow(self):
        act = RationalActivation()
        x = torch.randn(4, 16, requires_grad=True)
        y = act(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_coefficient_gradients(self):
        act = RationalActivation()
        x = torch.randn(4, 16)
        y = act(x)
        y.sum().backward()
        assert act.coeff_numerator.grad is not None
        assert act.coeff_denominator.grad is not None

    def test_horner_correctness(self):
        """Verify Horner's method matches naive polynomial evaluation."""
        act = RationalActivation(num_numerator=4, num_denominator=3)
        x = torch.tensor([0.0, 1.0, -1.0, 0.5])

        # Manual naive evaluation for numerator
        cn = act.coeff_numerator.data
        naive_num = sum(cn[i] * x**i for i in range(len(cn)))

        cd = act.coeff_denominator.data
        naive_den = sum(cd[i] * x ** (i + 1) for i in range(len(cd)))

        expected = naive_num / (torch.abs(naive_den) + 1)
        actual = act(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_numerically_stable(self):
        """Large inputs should not produce NaN or Inf."""
        act = RationalActivation()
        x = torch.tensor([100.0, -100.0, 1000.0])
        y = act(x)
        assert torch.isfinite(y).all()

    @pytest.mark.parametrize("init", ["uniform", "normal", "relu"])
    def test_init_strategies(self, init):
        act = RationalActivation(init=init)
        assert act.coeff_numerator.shape == (5,)
        assert act.coeff_denominator.shape == (4,)

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError, match="Unknown init"):
            RationalActivation(init="invalid")

    def test_extra_repr(self):
        act = RationalActivation(num_numerator=5, num_denominator=4)
        assert "P_degree=4" in act.extra_repr()
        assert "Q_degree=4" in act.extra_repr()
