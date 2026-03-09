#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> era_forward(
    at::Tensor x,
    float a,
    float b,
    at::Tensor c,
    at::Tensor d,
    at::Tensor e,
    at::Tensor f,
    int n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("era_forward", &era_forward, "ERA forward function");
}
