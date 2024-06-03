#ifdef DUSE_DP
typedef double real;
#else
typedef float real;
#endif

// standard sgemm
void launch_sgemm(float* h_a, float* h_b, float* h_c, int M, int N, int K);

// optimized sgemm
void launch_sgemm_thread_tile_vec4(float* h_a, float* h_b, float* h_c, int M, int N, int K);

// reduce
real launch_reduce(const real *h_x, real *h_y, const int N);

// flash attention
torch::Tensor flash_attn_1_fwd_f32(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor flash_attn_2_fwd_f32(torch::Tensor q, torch::Tensor k, torch::Tensor v);