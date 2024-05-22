#include <stdio.h>
#include <cuda_runtime.h>
#include "error.cuh"

#ifdef DUSE_DP
typedef double real;
#else
typedef float real;
#endif

const int TILE_DIM = 32;


// version 0: copy
__global__ void copy(const float *matrix_A, float* matrix_B, const int N){
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
    const int idx = ny * N + nx;
    if (nx < N && ny < N) {
        matrix_B[idx] = matrix_A[idx];
    }
}

// version 1: coalesced read, but uncoalesced write
__global__ void transpose_coalesced_read(float* matrix, float* matrix_t, const int N){
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
    if (nx < N && ny < N) {
        matrix_t[ny * N + nx] = matrix[nx * N + ny];
    }
}


// version 2: coalesced write, but uncoalesced read
// As the uncoalesced read is trabsform by complier to 
// matrix_t[nx * N + ny] = __ldg(&matrix[ny * N + nx]);
__global__ void transpose_coalesced_write(float* matrix, float* matrix_t, const int N){
    const int nx = threadIdx.x + blockIdx.x * TILE_DIM;
    const int ny = threadIdx.y + blockIdx.y * TILE_DIM;
    if (nx < N && ny < N) {
        matrix_t[nx * N + ny] = matrix[ny * N + nx];
    }
}


// version 3: using share memory,but with bank conflict
__global__ void transpose_share_v3(float* matrix, float* matrix_t, const int N){
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    
    // copy date from global memo to share memo
    if(nx1 < N && ny1 < N){
        S[threadIdx.x][threadIdx.y] = matrix[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.x;
    int ny2 = by + threadIdx.y;
    if (nx2 < N && ny2 < N){
        matrix_t[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}


// version 4: using share memory,without bank conflict
__global__ void transpose_share_v4(float* matrix, float* matrix_t, const int N){
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    
    // copy date from global memo to share memo
    if(nx1 < N && ny1 < N){
        S[threadIdx.x][threadIdx.y] = matrix[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.x;
    int ny2 = by + threadIdx.y;
    if (nx2 < N && ny2 < N){
        matrix_t[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}


// version 5: using share memory, without bank conflict, by alloc more share memory
__global__ void transpose_share(float* matrix, float* matrix_t, const int N){
    __shared__ real S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    
    // copy date from global memo to share memo
    if(nx1 < N && ny1 < N){
        S[threadIdx.x][threadIdx.y] = matrix[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.x;
    int ny2 = by + threadIdx.y;
    if (nx2 < N && ny2 < N){
        matrix_t[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
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
    transpose_share<<<grid_size, block_size>>>(d_A, d_B, N);

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

