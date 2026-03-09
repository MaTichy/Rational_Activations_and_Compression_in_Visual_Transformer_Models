// hulk_boost_rationals_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <ATen/AccumulateType.h>  // Correct header for acc_type

#define M 5
#define N 4  // Since N = M - 1

// Forward declarations and helper functions
template <typename scalar_t>
__device__ __forceinline__ scalar_t fused_mul_add(scalar_t a, scalar_t b, scalar_t c) {
    return a * b + c;  // Default implementation
}

template <>
__device__ __forceinline__ at::Half fused_mul_add(at::Half a, at::Half b, at::Half c) {
    return at::Half(float(a) * float(b) + float(c));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t abs_val(scalar_t x) {
    return fabs(x);  // Default implementation
}

template <>
__device__ __forceinline__ at::Half abs_val(at::Half x) {
    return at::Half(fabs(float(x)));
}

// AtomicAdd wrapper to support float and double
template <typename T>
__device__ void atomicAddWrapper(T* address, T val);

template <>
__device__ void atomicAddWrapper<float>(float* address, float val) {
    atomicAdd(address, val);
}

template <>
__device__ void atomicAddWrapper<double>(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    printf("AtomicAdd on double is not supported on this device\n");
#endif
}

// Forward kernel
template <typename scalar_t>
__global__ void hulk_boost_rational_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ coeff_num,
    const scalar_t* __restrict__ coeff_den,
    scalar_t* __restrict__ output,
    int num_elements) {

    // Shared memory for coefficients
    __shared__ scalar_t shared_coeff_num[M];
    __shared__ scalar_t shared_coeff_den[N];

    // Load coefficients into shared memory (reverse order)
    if (threadIdx.x < M) {
        shared_coeff_num[threadIdx.x] = coeff_num[M - 1 - threadIdx.x];
    }
    if (threadIdx.x < N) {
        shared_coeff_den[threadIdx.x] = coeff_den[N - 1 - threadIdx.x];
    }
    __syncthreads();

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of threads
    int stride = blockDim.x * gridDim.x;

    // Loop over data with stride
    for (; idx < num_elements; idx += stride) {
        scalar_t x_val = x[idx];

        // Horner's method for numerator
        scalar_t numerator = shared_coeff_num[0];
        #pragma unroll
        for (int j = 1; j < M; ++j) {
            numerator = fused_mul_add(numerator, x_val, shared_coeff_num[j]);
        }

        // Horner's method for denominator
        scalar_t denominator = shared_coeff_den[0];
        #pragma unroll
        for (int j = 1; j < N; ++j) {
            denominator = fused_mul_add(denominator, x_val, shared_coeff_den[j]);
        }

        // Multiply denominator by x_val to account for starting at x^1
        denominator = denominator * x_val;

        // Denominator adjustment
        scalar_t denom = abs_val(denominator) + scalar_t(1.0);

        // Compute output
        output[idx] = numerator / denom;
    }
}

// Backward kernel
template <typename scalar_t>
__global__ void hulk_boost_rational_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ coeff_num,
    const scalar_t* __restrict__ coeff_den,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_coeff_num,
    scalar_t* __restrict__ grad_coeff_den,
    int num_elements) {

    using accscalar_t = at::acc_type<scalar_t, true>;

    // Shared memory for coefficients
    __shared__ scalar_t shared_coeff_num[M];
    __shared__ scalar_t shared_coeff_den[N];

    // Load coefficients into shared memory (reverse order)
    if (threadIdx.x < M) {
        shared_coeff_num[threadIdx.x] = coeff_num[M - 1 - threadIdx.x];
    }
    if (threadIdx.x < N) {
        shared_coeff_den[threadIdx.x] = coeff_den[N - 1 - threadIdx.x];
    }
    __syncthreads();

    // Local accumulators for gradients w.r.t coefficients
    accscalar_t local_grad_coeff_num[M] = {0};
    accscalar_t local_grad_coeff_den[N] = {0};

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Loop over data with stride
    for (; idx < num_elements; idx += stride) {
        scalar_t x_val = x[idx];
        scalar_t grad_out = grad_output[idx];

        // Compute numerator and its derivative
        scalar_t numerator = shared_coeff_num[0];
        scalar_t numerator_derivative = scalar_t(0);
        #pragma unroll
        for (int i = 1; i < M; ++i) {
            numerator_derivative = fused_mul_add(numerator_derivative, x_val, numerator);
            numerator = fused_mul_add(numerator, x_val, shared_coeff_num[i]);
        }

        // Compute denominator Q(x) and its derivative Q'(x)
        scalar_t Q = shared_coeff_den[0];
        scalar_t Q_prime = scalar_t(0);
        #pragma unroll
        for (int i = 1; i < N; ++i) {
            Q_prime = fused_mul_add(Q_prime, x_val, Q);
            Q = fused_mul_add(Q, x_val, shared_coeff_den[i]);
        }

        // Compute Q_tilde = Q(x) * x_val
        scalar_t Q_tilde = Q * x_val;

        // Compute Q_tilde_prime = Q'(x) * x_val + Q(x)
        scalar_t Q_tilde_prime = Q_prime * x_val + Q;

        // Compute denom and its derivative
        scalar_t denom = abs_val(Q_tilde) + scalar_t(1.0);
        scalar_t sign_Q_tilde = (Q_tilde >= scalar_t(0)) ? scalar_t(1.0) : scalar_t(-1.0);
        scalar_t denom_derivative = sign_Q_tilde * Q_tilde_prime;

        // Compute grad_input
        scalar_t denom_squared = denom * denom;
        scalar_t grad_input = (numerator_derivative * denom - numerator * denom_derivative) / denom_squared;

        // Store gradient w.r.t x[idx]
        grad_x[idx] = grad_out * grad_input;

        // Common factors for gradients w.r.t coefficients
        scalar_t inv_denom = scalar_t(1.0) / denom;
        scalar_t inv_denom_squared = inv_denom * inv_denom;
        scalar_t common_factor_num = grad_out * inv_denom;
        scalar_t common_factor_den = -grad_out * numerator * sign_Q_tilde * inv_denom_squared;

        // Accumulate gradients w.r.t numerator coefficients
        scalar_t accum = scalar_t(1.0);
        #pragma unroll
        for (int i = 0; i < M; ++i) {
            accscalar_t grad_coeff = accum * common_factor_num;
            local_grad_coeff_num[i] += grad_coeff;
            accum *= x_val;
        }

        // Accumulate gradients w.r.t denominator coefficients
        accum = x_val;  // Start from x^1
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            accscalar_t grad_coeff = accum * common_factor_den;
            local_grad_coeff_den[i] += grad_coeff;
            accum *= x_val;
        }
    }

    // Allocate shared memory for block-level reduction
    extern __shared__ char shared_grad_raw[];
    accscalar_t* shared_grad = reinterpret_cast<accscalar_t*>(shared_grad_raw);
    accscalar_t* shared_grad_coeff_num = shared_grad; // Size: M * blockDim.x
    accscalar_t* shared_grad_coeff_den = shared_grad + M * blockDim.x; // Size: N * blockDim.x

    // Copy local accumulators to shared memory
    for (int i = 0; i < M; ++i) {
        shared_grad_coeff_num[i * blockDim.x + threadIdx.x] = local_grad_coeff_num[i];
    }
    for (int i = 0; i < N; ++i) {
        shared_grad_coeff_den[i * blockDim.x + threadIdx.x] = local_grad_coeff_den[i];
    }
    __syncthreads();

    // Perform reduction within the block for numerator coefficients
    for (int i = 0; i < M; ++i) {
        accscalar_t* sdata = &shared_grad_coeff_num[i * blockDim.x];
        unsigned int tid = threadIdx.x;
        // Reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        // Write result to global memory
        if (tid == 0) {
            atomicAddWrapper(&grad_coeff_num[i], sdata[0]);
        }
        __syncthreads();
    }

    // Perform reduction within the block for denominator coefficients
    for (int i = 0; i < N; ++i) {
        accscalar_t* sdata = &shared_grad_coeff_den[i * blockDim.x];
        unsigned int tid = threadIdx.x;
        // Reduction
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        // Write result to global memory
        if (tid == 0) {
            atomicAddWrapper(&grad_coeff_den[i], sdata[0]);
        }
        __syncthreads();
    }
}

// Host function for forward pass
void hulk_boost_rational_forward_cuda(
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor output) {

    // Ensure tensors are contiguous
    x = x.contiguous();
    coeff_numerator = coeff_numerator.contiguous();
    coeff_denominator = coeff_denominator.contiguous();
    output = output.contiguous();

    // Flatten tensors for processing
    auto x_flat = x.view(-1);
    auto output_flat = output.view(-1);
    int num_elements = x_flat.size(0);

    // Launch parameters
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hulk_boost_rational_forward_cuda", ([&] {
        hulk_boost_rational_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_flat.data_ptr<scalar_t>(),
            coeff_numerator.data_ptr<scalar_t>(),
            coeff_denominator.data_ptr<scalar_t>(),
            output_flat.data_ptr<scalar_t>(),
            num_elements);

        // Error checking
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA error in hulk_boost_rational_forward_kernel: ", cudaGetErrorString(err));
        }
    }));
}

// Host function for backward pass
void hulk_boost_rational_backward_cuda(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor coeff_numerator,
    at::Tensor coeff_denominator,
    at::Tensor grad_x,
    at::Tensor grad_coeff_numerator,
    at::Tensor grad_coeff_denominator) {

    // Ensure tensors are contiguous
    grad_output = grad_output.contiguous();
    x = x.contiguous();
    coeff_numerator = coeff_numerator.contiguous();
    coeff_denominator = coeff_denominator.contiguous();
    grad_x = grad_x.contiguous();
    grad_coeff_numerator = grad_coeff_numerator.contiguous();
    grad_coeff_denominator = grad_coeff_denominator.contiguous();

    // Flatten tensors for processing
    auto x_flat = x.view(-1);
    auto grad_output_flat = grad_output.view(-1);
    auto grad_x_flat = grad_x.view(-1);
    int num_elements = x_flat.size(0);

    // Launch parameters
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "hulk_boost_rational_backward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        size_t shared_mem_size = (M + N) * threads * sizeof(accscalar_t);

        hulk_boost_rational_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            grad_output_flat.data_ptr<scalar_t>(),
            x_flat.data_ptr<scalar_t>(),
            coeff_numerator.data_ptr<scalar_t>(),
            coeff_denominator.data_ptr<scalar_t>(),
            grad_x_flat.data_ptr<scalar_t>(),
            grad_coeff_numerator.data_ptr<scalar_t>(),
            grad_coeff_denominator.data_ptr<scalar_t>(),
            num_elements);

        // Error checking
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            AT_ERROR("CUDA error in hulk_boost_rational_backward_kernel: ", cudaGetErrorString(err));
        }
    }));
}
