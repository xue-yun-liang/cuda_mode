#include <stdio.h>
#include <cuda_runtime.h>
#include "error.cuh"


__device__ int d_x = 1;     // single static var
__device__ int d_y[2];      // static int array

void __global__ kernel(void){
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d\n", d_x, d_y[0], d_y[1]);  
}

int main(void){
    int h_y[2] = {10, 20};
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int)*2));

    kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int)*2));
    printf("h_y[0] = %d, h_y[1] = %d\n", h_y[0], h_y[1]);

    return 0;
}