#include <stdio.h>


__global__ void vectorAddition(int *a, int *b, int *c, int n) {
    // 计算当前线程的全局索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程索引在向量范围内
    if (tid < n) {
        // 执行向量加法
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // vector's length
    int n = 1000;

    // alloc the memo
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(n * sizeof(int));
    h_b = (int*)malloc(n * sizeof(int));
    h_c = (int*)malloc(n * sizeof(int));

    // init the vector's values
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 分配设备内存
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

    // 定义 CUDA 核函数的线程块大小
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 调用 CUDA 核函数
    vectorAddition<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Vector Addition Result:\n");
    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

