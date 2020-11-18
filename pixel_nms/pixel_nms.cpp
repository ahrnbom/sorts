#include <torch/extension.h>

#include <iostream>

torch::Tensor add_self(torch::Tensor input) {
    auto out = input + input;
    return out;
}

torch::Tensor add_self_cuda_kernel_caller(torch::Tensor input);
void pixel_nms_kernel_caller(torch::Tensor input);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor add_self_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    return add_self_cuda_kernel_caller(input);
}

void pixel_nms_cuda_(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dtype() == torch::ScalarType::Bool);
    pixel_nms_kernel_caller(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_self", &add_self, "Add self");
  m.def("add_self_cuda", &add_self_cuda, "Add self (cuda)");
  m.def("pixel_nms_cuda_", &pixel_nms_cuda_, "Add self (cuda)");
}
