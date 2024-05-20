#include "error.cuh"
#include <stdio.h>
#include <float.h>

#ifdef DUSE_DP
typedef double real;
#else
typedef float real;
#endif

#define N_ 1024 // Example size, should be a power of 2 for simplicity
#define BLOCK_SIZE 128 // Example block size, can be tuned
#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


// Reduce in CPU
real reduce_cpu(const real *x, const int N){
  float sum = 0.0;
  for (int i = 0; i < N; i++)
  {
    sum += x[i];
  }
  return sum;
}

// a incorrect reduce in gpu
__global__ void recude_error(real *d_x, int N){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int offset = N/2; offset>0; offset /=2){
    if (tid < offset) d_x[tid] += d_x[tid + offset];
  }
}

// a reduce using global memory
__global__ void reduce_global(real *d_x, real *d_y) {
  const int tid = threadIdx.x;
  real *x = d_x + blockIdx.x * blockDim.x;
  for (int offset=blockDim.x >> 1; offset>0; offset>>=1) {
    if(tid < offset) x[tid] += x[tid + offset];
    __syncthreads();
  }

  if (tid == 0) {
    d_y[blockIdx.x] = x[0];
  }
}


// a reduce using static shared memory
__global__ void reduce_share(real* d_x, real *d_y){
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + tid;
  __shared__ real s_y[128];
  s_y[tid] = (n < N_) ? d_x[n] : 0.0;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) s_y[tid] = s_y[tid + offset];
    __syncthreads();
  }

  if (tid == 0) {
    d_y[tid] = s_y[0];
  }
}

// a reduce using dynmic shared memory
__global__ void reduce_share_(real* d_x, real *d_y){
  // call reduce_share_<<<grid_size, block_size, sizeof(real) * block>>>(x, y);
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int n = bid * blockDim.x + tid;
  extern __shared__ real s_y[];
  s_y[tid] = (n < N_) ? d_x[n] : 0.0;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) s_y[tid] = s_y[tid + offset];
    __syncthreads();
  }

  if (tid == 0) {
    d_y[tid] = s_y[0];
  }
}



// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_sum<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum<NUM_WARPS>(val);
  return val;
}

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  val = warp_reduce_max<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  val = warp_reduce_max<NUM_WARPS>(val);
  return val;
}

// Kernel to test block reduce sum and max
__global__ void test_reduce_kernels() {
  float val = static_cast<float>(threadIdx.x);
  
  float sum = block_reduce_sum<128>(val);
  float max = block_reduce_max<128>(val);
  
  if (threadIdx.x == 0) {
    printf("Block reduce sum: %f\n", sum);
    printf("Block reduce max: %f\n", max);
  }
}

// int main() {
//   // Launch kernel with 1 block of 128 threads
//   test_reduce_kernels<<<1, 128>>>();
//   cudaDeviceSynchronize();
  
//   return 0;
// }

void reduce_wrapper(real *d_x, real *d_y, int size) {
    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_global<<<gridSize, BLOCK_SIZE>>>(d_x, d_y);
}

int main() {
    real *h_x = new real[N_];
    real *h_result = new real[(N_ + BLOCK_SIZE - 1) / BLOCK_SIZE];

    // Initialize input data
    for (int i = 0; i < N_; i++) {
        h_x[i] = static_cast<real>(i + 1); // Example data
    }

    real *d_x, *d_y;
    cudaMalloc((void**)&d_x, N_ * sizeof(real));
    cudaMalloc((void**)&d_y, ((N_ + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(real));

    cudaMemcpy(d_x, h_x, N_ * sizeof(real), cudaMemcpyHostToDevice);

    reduce_wrapper(d_x, d_y, N_);

    cudaMemcpy(h_result, d_y, ((N_ + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(real), cudaMemcpyDeviceToHost);

    // Print the intermediate result for checking
    printf("Intermediate reduction results:\n");
    for (int i = 0; i < (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        printf("%f ", h_result[i]);
    }
    printf("\n");

    // To get the final result, reduce the remaining values on the host or launch another kernel
    real final_result = 0;
    for (int i = 0; i < (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        final_result += h_result[i];
    }

    printf("Final reduction result: %f\n", final_result);

    delete[] h_x;
    delete[] h_result;
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}