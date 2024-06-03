#include <torch/extension.h>
#include "../include/wrapper.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_1_fwd_f32", &flash_attn_1_fwd_f32, "FlashAttention1 forward (f32)");
    m.def("flash_attn_2_fwd_f32", &flash_attn_2_fwd_f32, "FlashAttention2 forward (f32)");
}
