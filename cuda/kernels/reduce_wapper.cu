#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "./cuda/include/error.cuh"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define WARP_SIZE 32

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val){
    #pragma unroll
    for(int mask = kWarpSize; mask >0; mask >>= 1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val){
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE -1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE; // which warp_id in block
    int lane = threadIdx.x % WARP_SIZE; // which thread_id in warp
    static __shared__ float smem[NUM_WARPS];
    
    val = warp_reduce_sum<WARP_SIZE>(val);  // got the sum in every warp
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

template <const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val){
    // assume, NUM_THREADS = 128, WARP_size = 32, NUM_WARPS = (128 + 32 -1) / 32 = 
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE; // which warp_id in block
    int lane = threadIdx.x % WARP_SIZE; // which thread_id in warp
    static __shared__ float smem[NUM_WARPS];

    val = warp_reduce_max<WARP_SIZE>(val);  // got the max in every warp
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : -1.0f;
    val = warp_reduce_max<NUM_WARPS>(val);
    return val;
}

// block_all_reduce
// a: Nx1, b: sum(a)
template <const int NUM_THREADS=128>
__global__ float block_all_reduce_sum(float* a, float* b, const int N){
    int tid = threadIdx.x;
    int idx = tid + NUM_THREADS * blockIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];
    // keep data in reg is enougth for warp reduce
    float sum = (idx < N) ? a[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    // warp leader store the result to smem
    if (lane == 0) smem[warp] = sum;
    // make sure all temp result in smem
    __syncthreads();
    sum = (lane < NUM_WARPS) smem[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (tid == 0) atomicAdd(b, sum);
}

// block_all_reduce + vec4
template <const int NUM_THREADS=128>
__global__ float block_all_reduce_vec4_sum(float* a, float* b, const int N){
    int tid = threadIdx.x;
    int idx = 4 * (tid + NUM_THREADS * blockIdx.x);
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w);
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane_id == 0) smem[lane_id] = sum;
    sum = (lane_id < NUM_WARPS)? smem[lane_id] : 0.0f;
    if (warp_id == 0) warp_reduce_sum<WARP_SIZE>(sum);
    if (tid == 0) atomicAdd(sum, b);
}


__global__ void test_block_reduce_sum(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    *output = block_reduce_sum(input[idx]);
}

__global__ void test_block_reduce_max(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    *output = block_reduce_max(input[idx]);
}

int main() {
    int n = 128; 
    float* h_input = new float[n];
    float h_output;

    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);


    int threadsPerBlock = 128;
    int blocksPerGrid = 1; 
    test_block_reduce_sum<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("sum:%f\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}
