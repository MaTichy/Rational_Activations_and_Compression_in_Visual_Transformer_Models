#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
__global__ void era_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t a,
    const scalar_t b,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ d,
    const scalar_t* __restrict__ e,
    const scalar_t* __restrict__ f,
    scalar_t* __restrict__ output,
    int num_elements,
    int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        scalar_t x_val = x[idx];
        scalar_t sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            // Avoid very small denominators by adding a small constant
            scalar_t denominator = (x_val - e[i]) * (x_val - e[i]) + f[i] * f[i] + 1e-6;
            sum += (c[i] * x_val + d[i]) / denominator;
        }
        output[idx] = a * x_val + b + sum;
    }
}


std::vector<at::Tensor> era_forward(
    at::Tensor x,
    float a,
    float b,
    at::Tensor c,
    at::Tensor d,
    at::Tensor e,
    at::Tensor f,
    int n) {

    auto x_flat = x.contiguous().view(-1);
    int num_elements = x_flat.size(0);
    auto output = torch::empty_like(x_flat);

    int threads = 1024;
    int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "era_forward_kernel", ([&] {
        era_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            a, b,
            c.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            e.data_ptr<scalar_t>(),
            f.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements,
            n);
    }));

    return {output.view(x.sizes())};
}
