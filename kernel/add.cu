#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "error.cuh"

void __global__ hello(){
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("hello from %d bid and (%d, %d)!\n", bx, tx, ty);
}

const double error = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *d_x, const double *d_y, double *d_z, int64_t N) {
    const int tid = blockDim.x * blockIdx.x  + threadIdx.x;
    if (tid < N) {
        d_z[tid] = d_x[tid] + d_y[tid];
    }
}

void check(const double *z, const int64_t N) {
    bool has_error = false;
    for (int i=0; i<N; i++) {
        if (fabs(z[i]-c) > error){
            // printf("error is %f\n", z[i]-c);
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}



int main(){
    // dim3 grid_size = 2;
    // dim3 block_size(2,4);   
    // blockDim.x = 2, blockDim.y = 4
    // threadIdx.x -> (0, 1), threadIdx.y -> (0, 3)

    // NOTE: blockIdx.x * blockIdx.y * blockIdx.z <= 1024
    // hello<<<grid_size, block_size>>>();

    const int64_t N = 16;
    const int64_t M = sizeof(double)*N;
    double *h_x = (double*)malloc(M);
    double *h_y = (double*)malloc(M);
    double *h_z = (double*)malloc(M);

    for (int i=0; i<N; i++){
        h_x[i] = a;
        h_y[i] = b;
    }

    // alloc the memo for device
    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    // copy the data from host to device
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice));

    const int block_size = 4;
    const int grid_size = N / 4;

    // execution the add compute
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // copy the result from host to device
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);
    
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);


    // cudaDeviceSynchronize();
    return 0;
}