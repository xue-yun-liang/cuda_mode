#include <iostream>
#include <cmath>
#include <cuda.h>

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/128), block(K=128) 
__global__ void sigmoid(float* x, float* y, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/128), block(128/4)
__global__ void sigmoid_vec4(float* x, float* y, int N) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (idx < N) {
    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_y;
    reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
    reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
    reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
    reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));
    FLOAT4(y[idx]) = reg_y;
  }
}

void run_sigmoid() {
  int N = 32;
  float* h_x = new float[N];
  float* h_y = new float[N];
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(i % 100 - 50) / 10.0f; // [-5.0, 5.0]
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (N + 128 - 1) / 128;
  sigmoid<<<num_blocks, 128>>>(d_x, d_y, N);

  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Sigmoid results:" << std::endl;
  for (int i = 0; i < N; ++i) {
    std::cout << h_y[i] << " ";
  }
  std::cout << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y;
}

void run_sigmoid_vec4() {
  int N = 32;
  float* h_x = new float[N];
  float* h_y = new float[N];
  for (int i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(i % 100 - 50) / 10.0f; // [-5.0, 5.0]
  }

  float *d_x, *d_y;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (N + 128 - 1) / 128;
  sigmoid_vec4<<<num_blocks, 128 / 4>>>(d_x, d_y, N);

  cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Sigmoid Vec4 results:" << std::endl;
  for (int i = 0; i < N; ++i) {
    std::cout << h_y[i] << " ";
  }
  std::cout << std::endl;

  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_x;
  delete[] h_y;
}

int main() {
  run_sigmoid();
  run_sigmoid_vec4();
  return 0;
}
