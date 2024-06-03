#include <torch/extension.h>
#include "../include/wrapper.h"

void torch_launch_reduce(const torch::Tensor &h_x, float &h_y, const int n) {
    TORCH_CHECK(h_x.is_cuda(), "A must be a CUDA tensor");
    launch_reduce((float*)h_x.data_ptr(), &h_y, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_reduce", &torch_launch_reduce, "reduce kernel wrapper");
}