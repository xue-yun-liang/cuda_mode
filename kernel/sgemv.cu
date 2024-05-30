#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include "error.cuh"

#define WARP_SIZE 32
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// SGEMV: Warp SGEMV K32
// assume K is a multiple of 32, each warp handles one row
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32(float *a, float *x, float *y, int M, int K)
{
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
    if (m < M)
    {
        float sum = 0.0f;
        int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            // if NUM_WARPS>=2，先将当前行的数据累加到第一个warp中
            int k = w * WARP_SIZE + lane;
            sum += a[m * K + k] * x[k];
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0)
            y[m] = sum;
    }
}

// SGEMV: Warp SGEMV K128 + Vec4
// assume K is a mutilple of 128, float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128(float *a, float *x, float *y, int M, int K)
{
    // each thread handle 4 elements, a warp cover 128 elements
    int tx = threadIdx.x;         // 0~31
    int ty = threadIdx.y;         // 0~3
    int bx = blockIdx.x;          // 0~M/4
    int lane = tx % WARP_SIZE;    // 0~31
    int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)

    if (m < M)
    {
        float sum = 0.0f;
        // process 4*WARP_SIZE elements per warp.
        int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            int k = (w * WARP_SIZE + lane) * 4;
            float4 reg_x = FLOAT4(x[k]);
            float4 reg_a = FLOAT4(a[m * K + k]);
            sum += (reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z + reg_a.w * reg_x.w);
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);
        if (lane == 0)
            y[m] = sum;
    }
}

// SGEMV: Warp SGEMV K16
// assume K is 16 < 32 and each warp handle 2 rows with 16 elements per row
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16(float *A, float *x, float *y, int M, int K)
{
    constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
    int tx = threadIdx.x;      // 0~31
    int ty = threadIdx.y;      // 0~NUM_WARPS
    int bx = blockIdx.x;       // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
    int lane = tx % WARP_SIZE; // 0~31

    int k = lane % K_WARP_SIZE; // 0~15
    // global row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
    int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
    if (m < M)
    {
        float sum = A[m * K + k] * x[k];
        sum = warp_reduce_sum<K_WARP_SIZE>(sum);
        // NOTE: k == 0，not lane == 0
        if (k == 0)
            y[m] = sum;
    }
}

void sgemv_cpu(float *a, float *x, float *y, int M, int K)
{
    for (int i = 0; i < M; ++i)
    {
        y[i] = 0.0f;
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            y[i] += a[i * K + j] * x[j];
        }
    }
}

// run_sgemv_k32 function definition
void run_sgemv_k32()
{
    // Matrix dimensions
    int M = 8;
    int K = 128;

    // Allocate unified memory
    float *a, *x, *y;
    CHECK(cudaMallocManaged(&a, M * K * sizeof(float)));
    CHECK(cudaMallocManaged(&x, K * sizeof(float)));
    CHECK(cudaMallocManaged(&y, M * sizeof(float)));

    float *a_, *x_;
    CHECK(cudaMallocManaged(&a_, M * K * sizeof(float)));
    CHECK(cudaMallocManaged(&x_, K * sizeof(float)));

    // Initialize data
    for (int i = 0; i < M * K; ++i)
    {
        a[i] = static_cast<float>(std::rand() % 10); // Example initialization with random values
    }
    for (int i = 0; i < K; ++i)
    {
        x[i] = static_cast<float>(std::rand() % 5); // Example initialization with random values
    }

    for (int i = 0; i < M * K; ++i)
    {
        a_[i] = a[i]; // Example initialization with random values
    }
    for (int i = 0; i < K; ++i)
    {
        x_[i] = x[i]; // Example initialization with random values
    }

    // Print input data
    printf("Matrix a:\n");
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            printf("%d ", (int)a[i * K + j]);
        }
        printf("\n");
    }

    printf("Vector x:\n");
    for (int i = 0; i < K; ++i)
    {
        printf("%d ", (int)x[i]);
    }
    printf("\n");

    // Define grid and block dimensions
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(M / 4);

    // Launch kernel
    sgemv_k128<<<gridDim, blockDim>>>(a, x, y, M, K);

    // Synchronize to ensure all kernels have finished
    CHECK(cudaDeviceSynchronize());

    // Print result
    printf("Result vector y:\n");
    for (int i = 0; i < M; ++i)
    {
        printf("%d ", (int)y[i]);
    }
    printf("\n");

    for (int i = 0; i < M * K; ++i)
    {
        a[i] = static_cast<float>(i % 10); // Example initialization
    }
    for (int i = 0; i < K; ++i)
    {
        x[i] = static_cast<float>(i % 5); // Example initialization
    }
    printf("cpu ver as follows: \n");
    sgemv_cpu(a_, x_, y, M, K);
    // Print result
    printf("Result vector y:\n");
    for (int i = 0; i < M; ++i)
    {
        printf("%d ", (int)y[i]);
    }
    printf("\n");

    // Free unified memory
    CHECK(cudaFree(a));
    CHECK(cudaFree(x));
    CHECK(cudaFree(y));
}

// Main function
int main()
{
    run_sgemv_k32();
    printf("finished!");
    return 0;
}