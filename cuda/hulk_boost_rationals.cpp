// hulk_boost_rationals.cpp

#include <torch/extension.h>

// Declarations of CUDA functions
void hulk_boost_rational_forward_cuda(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor output);

void hulk_boost_rational_backward_cuda(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor grad_x,
    at::Tensor grad_coeff_numerator,
    at::Tensor grad_coeff_denominator);

// C++ interface (forward and backward functions)
at::Tensor hulk_boost_rational_forward(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator) {

    auto output = at::zeros_like(x);
    hulk_boost_rational_forward_cuda(x, coeff_numerator, coeff_denominator, output);
    return output;
}

std::vector<at::Tensor> hulk_boost_rational_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator) {

    auto grad_x = at::zeros_like(x);
    auto grad_coeff_numerator = at::zeros_like(coeff_numerator, coeff_numerator.options().dtype(at::kFloat));
    auto grad_coeff_denominator = at::zeros_like(coeff_denominator, coeff_denominator.options().dtype(at::kFloat));

    hulk_boost_rational_backward_cuda(
        grad_output,
        x,
        coeff_numerator,
        coeff_denominator,
        grad_x,
        grad_coeff_numerator,
        grad_coeff_denominator);

    return {grad_x, grad_coeff_numerator, grad_coeff_denominator};
}

// PyTorch binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hulk_boost_rational_forward, "Hulk Boost Rational forward");
    m.def("backward", &hulk_boost_rational_backward, "Hulk Boost Rational backward");
}
