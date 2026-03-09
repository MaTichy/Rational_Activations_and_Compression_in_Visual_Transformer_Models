"""Learnable rational activation functions based on Pade approximants.

A rational activation function R(x) = P(x) / Q(x) where P and Q are polynomials
with learnable coefficients. This allows the network to learn its own activation
function during training, adapting to the data distribution.

Reference: Molina et al., "Pade Activation Units: End-to-end Learning of Flexible
Activation Functions in Deep Networks" (ICLR 2020)
"""

import torch
import torch.nn as nn


class RationalActivation(nn.Module):
    """Learnable rational activation function using Pade approximants.

    Computes R(x) = P(x) / (|Q(x)| + 1) where:
        P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_m*x^m
        Q(x) = b_1*x + b_2*x^2 + ... + b_n*x^n

    The absolute value and +1 in the denominator ensure numerical stability.

    Args:
        num_numerator: Number of numerator coefficients (degree m).
        num_denominator: Number of denominator coefficients (degree n-1, excludes constant).
        init: Initialization strategy - 'uniform' or 'normal'.
    """

    def __init__(self, num_numerator: int = 5, num_denominator: int = 4, init: str = "uniform"):
        super().__init__()
        self.num_numerator = num_numerator
        self.num_denominator = num_denominator

        if init == "uniform":
            self.coeff_numerator = nn.Parameter(torch.rand(num_numerator))
            self.coeff_denominator = nn.Parameter(torch.rand(num_denominator))
        elif init == "normal":
            self.coeff_numerator = nn.Parameter(torch.randn(num_numerator))
            self.coeff_denominator = nn.Parameter(torch.randn(num_denominator))
        else:
            raise ValueError(f"Unknown init strategy: {init}. Use 'uniform' or 'normal'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Horner's method for numerical stability: P(x) = a_0 + x*(a_1 + x*(a_2 + ...))
        num = self.coeff_numerator[-1]
        for i in range(self.num_numerator - 2, -1, -1):
            num = num * x + self.coeff_numerator[i]

        den = self.coeff_denominator[-1]
        for i in range(self.num_denominator - 2, -1, -1):
            den = den * x + self.coeff_denominator[i]
        den = den * x  # multiply by x since denominator starts at x^1

        return num / (torch.abs(den) + 1)

    def extra_repr(self) -> str:
        return f"P_degree={self.num_numerator - 1}, Q_degree={self.num_denominator}"


class HulkBoostRationalActivation(nn.Module):
    """CUDA-accelerated rational activation using Horner's method.

    This module wraps a custom CUDA kernel that evaluates the rational function
    using Horner's method for numerical stability and shared memory for performance.
    Falls back to the pure-PyTorch RationalActivation when CUDA is not available.

    Requires the `hulk_boost_rationals` CUDA extension to be compiled and installed.
    """

    def __init__(self, num_numerator: int = 5, num_denominator: int = 4):
        super().__init__()
        self.coeff_numerator = nn.Parameter(torch.rand(num_numerator))
        self.coeff_denominator = nn.Parameter(torch.rand(num_denominator))

        try:
            import hulk_boost_rationals
            self._cuda_ext = hulk_boost_rationals
        except ImportError:
            self._cuda_ext = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cuda_ext is not None and x.is_cuda:
            return _HulkBoostFunction.apply(
                x.float(), self.coeff_numerator.float(), self.coeff_denominator.float()
            )
        # Fallback: Horner's method (numerically stable)
        cn, cd = self.coeff_numerator, self.coeff_denominator
        num = cn[-1]
        for i in range(len(cn) - 2, -1, -1):
            num = num * x + cn[i]
        den = cd[-1]
        for i in range(len(cd) - 2, -1, -1):
            den = den * x + cd[i]
        return num / (torch.abs(den * x) + 1)


class _HulkBoostFunction(torch.autograd.Function):
    """Custom autograd function for CUDA rational activation."""

    @staticmethod
    def forward(ctx, x, coeff_numerator, coeff_denominator):
        import hulk_boost_rationals
        ctx.save_for_backward(x, coeff_numerator, coeff_denominator)
        return hulk_boost_rationals.forward(x, coeff_numerator, coeff_denominator)

    @staticmethod
    def backward(ctx, grad_output):
        import hulk_boost_rationals
        x, coeff_num, coeff_den = ctx.saved_tensors
        grad_x, grad_num, grad_den = hulk_boost_rationals.backward(
            grad_output.contiguous(), x.contiguous(),
            coeff_num.contiguous(), coeff_den.contiguous()
        )
        return grad_x, grad_num, grad_den
