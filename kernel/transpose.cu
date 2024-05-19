#include <stdio.h>
#include <cuda_runtime.h>
#include "error.cuh"

const int TILE_DIM = 32;

__global__ void copy(const float *matrix_A, float* matrix_B, const int N){
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
    const int idx = ny * N + nx;
    if (nx < N && ny < N) {
        matrix_B[idx] = matrix_A[idx];
    }
}

__global__ void transpose(float* matrix, float* matrix_B, const int N){
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
    if (nx < N && ny < N) {
        // 合并读入，但非合并写入
        // matrix_B[ny * N + nx] = matrix[nx * N + ny];
        // 合并写入，但并非合并读入
        // 由于complier会识别到非合并读入并将这片数据使用_ldg()函数自动读取
        // 因此，2的实现更快
        matrix_B[nx * N + ny] = matrix[ny * N + nx];
    }
}


int main(){
    const int N = 32;              // the length of matric
    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    // Allocate host memory
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);

    // Initialize host matrix A
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_A, *d_B;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // exec the func
    transpose<<<grid_size, block_size>>>(d_A, d_B, N);

    // Copy the result from device to host
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Print the matrix B
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%2.0f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
    free(h_A);
    free(h_B);
    
    return 0;
}

