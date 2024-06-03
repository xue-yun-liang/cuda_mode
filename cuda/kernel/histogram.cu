#include <iostream>
#include <cuda_runtime.h>
#include "error.cuh"
#include "../include/kernel.cuh"


// Histogram
// grid(N/128), block(128)
// a: Nx1, y: count histogram
__global__ void histogram(int* a, int* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) atomicAdd(&(y[a[idx]]), 1);
}


// Histogram + Vec4
// grid(N/128), block(128/4)
// a: Nx1, y: count histogram
__global__ void histogram_vec4(int* a, int* y, int N) {
  // each thread handle 4 consistent element
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    int4 reg_a = INT4(a[idx]);
    atomicAdd(&(y[reg_a.x]), 1);
    atomicAdd(&(y[reg_a.y]), 1);
    atomicAdd(&(y[reg_a.z]), 1);
    atomicAdd(&(y[reg_a.w]), 1);
  }
}


// Function to print results
void print_results(const char* label, int* data, int N) {
    std::cout << label << ": ";
    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << " ";
        if (i % 20 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Function to run histogram kernel
void run_histogram(int* h_a, int* h_y, int N, int M) {
    int* d_a;
    int* d_y;

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_y, M * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(int));

    int blockSize = 128;
    int gridSize = (N + blockSize - 1) / blockSize;
    histogram<<<gridSize, blockSize>>>(d_a, d_y, N);

    cudaMemcpy(h_y, d_y, M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_y);
}

// Function to run histogram_vec4 kernel
void run_histogram_vec4(int* h_a, int* h_y, int N, int M) {
    int* d_a;
    int* d_y;

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_y, M * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, M * sizeof(int));

    int blockSize = 128 / 4;
    int gridSize = (N / 4 + blockSize - 1) / blockSize;
    histogram_vec4<<<gridSize, blockSize>>>(d_a, d_y, N);

    cudaMemcpy(h_y, d_y, M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_y);
}

int main() {
    const int N = 1024; // Number of elements in the input array
    const int M = 256;  // Number of bins in the histogram

    int h_a[N]; // Input array
    int h_y[M]; // Histogram output array

    for (int i = 0; i < N; ++i) {
        h_a[i] = rand() % M;
    }
    print_results("Input data", h_a, N);

    memset(h_y, 0, M * sizeof(int));  // Initialize output array
    run_histogram(h_a, h_y, N, M);
    print_results("Histogram", h_y, M);

    memset(h_y, 0, M * sizeof(int));
    run_histogram_vec4(h_a, h_y, N, M);
    print_results("Histogram Vec4", h_y, M);

    return 0;
}
