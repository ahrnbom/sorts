#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void add_self_cuda_kernel(scalar_t* __restrict__ input, scalar_t* __restrict__ output, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements)
        output[index] = input[index] + input[index] + 1;
}

torch::Tensor add_self_cuda_kernel_caller(torch::Tensor input) {
    const int elements = input.numel();
    const int threads = 1024;
    const int blocks = (elements + threads - 1) / threads;
    auto output = torch::zeros_like(input);
    AT_DISPATCH_ALL_TYPES(input.type(), "add_self_cuda_kernel_caller", ([&] {
        add_self_cuda_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(), output.data<scalar_t>(), elements);
    }));
    return output;
}

__global__ void pixel_nms_kernel(bool* __restrict__ input,  int size, int vectors) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < vectors) {
        bool mask = true;
        for (int i=0; i<size; i++) {
            bool v = input[i*vectors + index];
            v = v & mask;
            mask = mask ^ v;
            input[i*vectors + index] = v;
        }
    }
}


void pixel_nms_kernel_caller(torch::Tensor input) {
    const int size = input.size(0);
    const int vectors = input.numel() / size;
    const int threads = 1024;
    const int blocks = (vectors + threads - 1) / threads;
    pixel_nms_kernel<<<blocks, threads>>>(input.data<bool>(), size, vectors);
}

/*  Threads   ms - GeForce RTX 2080 Ti
       1    13.959
       8     1.426
      64     0.329
     512     0.268
    1024     0.256
*/