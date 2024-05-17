#include <torch/extension.h>
#include "ops.h"

void torch_launch_sgemm(const torch::Tensor &A, const torch::Tensor &B,
                        torch::Tensor &C, int64_t M, int64_t N, int64_t K) {
    launch_sgemm((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(),
                M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_sgemm",
    &torch_launch_sgemm,
    "sgemm kernel warpper");
}