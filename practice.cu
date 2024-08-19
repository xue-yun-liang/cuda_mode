#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "./cuda/include/error.cuh"

#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
    for (int mask = kWarpSize; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, val);
    }
    return val;
}

// dot product
// a: N*1, b: N*1, y: sum(elementwise_mul(a,b))  shape->N*1,
template <const int NUM_THREADS=128>
__global__ void dot(float* a, float* b, float* y, const int N){
    int tid = threadIdx.x;
    int idx = tid + NUM_THREADS * blockIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    prod = warp_reduce_sum<WARP_SIZE>(prod);
    if (lane_id == 0) smem[lane_id] = prod;
    __syncthreads();
    prod = (lane_id < NUM_THREADS) ? smem[lane_id] : 0.0f;
    if(warp_id == 0) prod = warp_reduce_sum<WARP_SIZE>(prod);
    if(lane_id == 0) atomicAdd(prod, y);
}