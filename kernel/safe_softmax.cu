#include <iostream>
#include <cmath>
#include <cuda.h>

#define WARP_SIZE 32
constexpr int NUM_THREADS = 128;

template<const int kWarpSize>
__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
  }
  return val;
}

template<const int kWarpSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int NUM_THREADS>
__global__ void safe_softmax(float* x, float* y, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float max_smem[NUM_WARPS];
  __shared__ float sum_smem[NUM_WARPS];

  // Step 1: Find the max value
  float val = (idx < N) ? x[idx] : -INFINITY;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  float max_val = warp_reduce_max<WARP_SIZE>(val);
  if (lane_id == 0) max_smem[warp_id] = max_val;
  __syncthreads();

  if (warp_id == 0) {
    max_val = (tid < NUM_WARPS) ? max_smem[lane_id] : -INFINITY;
    max_val = warp_reduce_max<NUM_WARPS>(max_val);
    if (lane_id == 0) max_smem[0] = max_val;
  }
  __syncthreads();

  float max_value = max_smem[0];

  // Step 2: Compute the exponentials and their sum
  val = (idx < N) ? expf(x[idx] - max_value) : 0.0f;
  float sum_val = warp_reduce_sum<WARP_SIZE>(val);
  if (lane_id == 0) sum_smem[warp_id] = sum_val;
  __syncthreads();

  if (warp_id == 0) {
    sum_val = (tid < NUM_WARPS) ? sum_smem[lane_id] : 0.0f;
    sum_val = warp_reduce_sum<NUM_WARPS>(sum_val);
    if (lane_id == 0) sum_smem[0] = sum_val;
  }
  __syncthreads();

  float total_sum = sum_smem[0];

  // Step 3: Normalize the values
  if (idx < N) y[idx] = expf(x[idx] - max_value) / total_sum;
}

void run_safe_softmax() {
  int N = 32;
  float* h_x = new float[N];
  float* h_y = new float[N];
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(i);
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
  safe_softmax<NUM_THREADS><<<num_blocks, NUM_THREADS>>>(d_x, d_y, N);

  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Before softmax:" << std::endl;
  for (int i = 0; i < N; ++i) {
    std::cout << h_x[i] << " \n";
  }
  std::cout << std::endl;

  std::cout << "After softmax:" << std::endl;
  double total = 0.0;
  for (int i = 0; i < N; ++i) {
    std::cout << h_y[i] << " \n";
    total += h_y[i];
  }
  std::cout << std::endl;
  std::cout << "softmax after total is :" << total << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y;
}

int main() {
  run_safe_softmax();
  return 0;
}
