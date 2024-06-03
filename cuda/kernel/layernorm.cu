#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "error.cuh"
#include "../include/kernel.cuh"

// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128>
__global__ void layer_norm(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = bid * blockDim.x + threadIdx.x;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float value = (idx < N * K) ? x[idx] : 0.0f; // load once only
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float variance = (value - s_mean) * (value - s_mean);
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  if (idx < N * K) y[idx] = ((value - s_mean) * s_variance) * g + b;
}

// Layer Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128/4>
__global__ void layer_norm_vec4(float* x, float* y, float g, float b, int N, int K) {
  int tid = threadIdx.x; // 0..K-1
  int bid = blockIdx.x; // 0..N-1
  int idx = (bid * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean; // shared within block
  __shared__ float s_variance; // shared within block
  float4 reg_x = FLOAT4(x[idx])
  float value = (idx < N * K) ? (reg_x.x + reg_x.y 
                               + reg_x.z + reg_x.w) : 0.0f;
  float sum = block_reduce_sum<NUM_THREADS>(value);
  if (tid == 0) s_mean = sum / (float) K;
  // wait for s_mean in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y 
                 + reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum<NUM_THREADS>(variance);
  if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
  // wait for s_variance in shared memory to be ready for all threads
  __syncthreads();
  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K) FLOAT4(y[idx]) = reg_y;
}
