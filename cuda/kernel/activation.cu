#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "error.cuh"
#include "../include/kernel.cuh"

#define WARP_SIZE 32
constexpr int NUM_THREADS = 128;

template<const int kWarpSize>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS>
__global__ void softmax(float* x, float* y, float* total, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid; 
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
  
    // compute the e^x_i, and store them to y[idx]
    y[idx] = (idx < N) ? expf(x[idx]) : 0.0f;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    float sum = warp_reduce_sum<WARP_SIZE>(y[tid]);
    if (lane_id == 0) {
        reduce_smem[warp_id] = sum;
    }
    __syncthreads();
    // compute the final sum in each warp
    sum = (lane_id < NUM_WARPS) ? reduce_smem[lane_id] : 0.0f;
    sum = warp_reduce_sum<NUM_WARPS>(sum); // sum(e^x_0,...,e^x_n-1)
    // get the total sum of all blocks.
    if (tid == 0) atomicAdd(total, sum);
    __threadfence(); // grid level memory fence
    // e^x_i/sum(e^x_0,...,e^x_n-1) 
    if (idx < N) y[idx] /= (*total); 
}


// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax_v2(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid; 
  
  float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) y[idx] = exp_val / (*total); 
}

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template<const int NUM_THREADS = 128/4>
__global__ void softmax_v2_vec4(float* x, float* y, float* total, int N) {
  const int tid = threadIdx.x;
  const int idx = (blockIdx.x * blockDim.x + tid) * 4; 
  
  float4 reg_x = FLOAT4(x[idx]);
  float4 reg_exp;
  reg_exp.x = (idx < N) ? expf(reg_x.x) : 0.0f;
  reg_exp.y = (idx < N) ? expf(reg_x.y) : 0.0f;
  reg_exp.z = (idx < N) ? expf(reg_x.z) : 0.0f;
  reg_exp.w = (idx < N) ? expf(reg_x.w) : 0.0f;
  float exp_val = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
  float sum = block_reduce_sum<NUM_THREADS>(exp_val);
  // get the total sum of all blocks.
  if (tid == 0) atomicAdd(total, sum);
  __threadfence(); // grid level memory fence
  // e^x_i/sum(e^x_0,...,e^x_n-1) 
  if (idx < N) {
    float4 reg_y;
    reg_y.x = reg_exp.x / (*total);
    reg_y.y = reg_exp.y / (*total);
    reg_y.z = reg_exp.z / (*total);
    reg_y.w = reg_exp.w / (*total);
    FLOAT4(y[idx]) = reg_y; 
  }
}


// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/128), block(K=128) 
__global__ void sigmoid(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/128), block(128/4)
__global__ void sigmoid_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
  }
}

// Relu x: N, y: N y=max(0,x)
// grid(N/128), block(K=128) 
__global__ void relu(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = fmaxf(0.0f, x[idx]);
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/128/4), block(128/4) 
__global__ void relu_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = fmaxf(0.0f, reg_x.x);
    reg_y.y = fmaxf(0.0f, reg_x.y);
    reg_y.z = fmaxf(0.0f, reg_x.z);
    reg_y.w = fmaxf(0.0f, reg_x.w);
    FLOAT4(y[idx]) = reg_y;
  }
}


void print_results(float* data, int N) {
    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << " ";
        if (i % 20 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}


void run_softmax(){
  const int N = 32;

  float *x, *y, *total;
  cudaMallocManaged((void**)&x, N * sizeof(float));
  cudaMallocManaged((void**)&y, N * sizeof(float));
  cudaMallocManaged((void**)&total, sizeof(float));

  for (int i=0; i<N; i++) {
    x[i] = std::rand() % 5;
    y[i] = 0;
  }

  const int grid_size(N / 128);
  const int block_size(128);
  size_t shared_mem_size = block_size * sizeof(float);

  softmax<NUM_THREADS><<<grid_size, block_size, shared_mem_size>>>(x, y, total, N);
  
  print_results(y, N);
  std::cout<<*total<<std::endl;
  cudaFree(x);
  cudaFree(y);
  cudaFree(total);

}

void run(const std::vector<float>& input) {
    const int N = input.size();
    const int size = N * sizeof(float);

    // Allocate host memory for output
    std::vector<float> output(N);

    // Allocate device memory
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_total = nullptr;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_total, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_x, input.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_total, 0, sizeof(float));

    // Launch kernel
    int numBlocks = (N + NUM_THREADS - 1) / NUM_THREADS;
    softmax<NUM_THREADS><<<numBlocks, NUM_THREADS>>>(d_x, d_y, d_total, N);

    // Copy the result back to host
    cudaMemcpy(output.data(), d_y, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_total);

    // Print the result
    for (int i = 0; i < N; ++i) {
        std::cout << "y[" << i << "] = " << output[i] << std::endl;
    }
}

int main() {
    // Example input
    const int N = 128;
    std::vector<float> input(N);
    srand(time(0));
    for(int i=0;i<N;i++){
      input[i]=rand()%5;
    }

    for(int i=0;i<N;i++)
      std::cout<<input[i]<<" ";
    std::cout<<std::endl;

    // Run the softmax function
    run(input);

    return 0;
}