#include <torch/extension.h>
#include "ops.h"

void torch_launch_sgemm(const torch::Tensor &A, const torch::Tensor &B,
                        torch::Tensor &C, int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    launch_sgemm((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(),
                M, N, K);
}

void torch_launch_sgemm_thread_tile_vec4(const torch::Tensor &A, const torch::Tensor &B,
                        torch::Tensor &C, int64_t M, int64_t N, int64_t K) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    launch_sgemm_thread_tile_vec4((float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(),
                M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_sgemm", &torch_launch_sgemm, "SGEMM kernel wrapper");
    m.def("torch_launch_sgemm_thread_tile_vec4", &torch_launch_sgemm_thread_tile_vec4, "SGEMM optimized kernel wrapper");
}