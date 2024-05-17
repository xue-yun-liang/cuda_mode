// standard sgemm
void launch_sgemm(float* h_a, float* h_b, float* h_c, int M, int N, int K);

// optimized sgemm
void launch_sgemm_thread_tile_vec4(float* h_a, float* h_b, float* h_c, int M, int N, int K);