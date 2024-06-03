#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include "error.cuh"

#ifdef USE_DP
typedef double real;
#else
typedef float real;
#endif

void check_real_type() {
    if (sizeof(real) == sizeof(float)) {
        printf("real is of type float.\n");
    } else if (sizeof(real) == sizeof(double)) {
        printf("real is of type double.\n");
    } else {
        printf("Unknown type.\n");
    }
}

void arithmetic(real *x, const real x0, const int N) {
    for (int i=0; i<N; i++) {
        real x_tmp = x[i];
        while (sqrt(x_tmp) < x0) {
            ++x_tmp;
        }
        x[i] = x_tmp;
    }
}


void __global__ arithmetic_(real *d_x, const real x0, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        real x_tmp = d_x[tid];
        while (sqrt(x_tmp) < x0) {
            ++x_tmp;
        }
        d_x[tid] = x_tmp;
    }
}

int main() {
    const int N = 100000000;
    const real x0 = 10.0;
    real *h_x = new real[N];
    real *h_x_copy = new real[N];

    check_real_type();

    // Initialize host array with some values
    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<real>(i % 10);
    }

    // Copy the host array to avoid modifying the original data for comparison
    std::copy(h_x, h_x + N, h_x_copy);

    // Measure the time for the CPU version
    cudaEvent_t start, stop;
    float milliseconds = 0;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    arithmetic(h_x, x0, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "CPU function arithmetic() time: " << milliseconds << " ms" << std::endl;

    // Allocate device memory and copy data to the device
    real *d_x;
    CHECK(cudaMalloc(&d_x, N * sizeof(real)));
    CHECK(cudaMemcpy(d_x, h_x_copy, N * sizeof(real), cudaMemcpyHostToDevice));

    // Define the number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Measure the time for the GPU version
    CHECK(cudaEventRecord(start));
    arithmetic_<<<blocksPerGrid, threadsPerBlock>>>(d_x, x0, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU function arithmetic_() time: " << milliseconds << " ms" << std::endl;

    // Clean up
    delete[] h_x;
    delete[] h_x_copy;
    CHECK(cudaFree(d_x));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}