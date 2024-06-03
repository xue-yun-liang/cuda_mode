/*
this project include all ops as follows:
    elementwise_add
    reduce(warp_reduce, block_reduce, block_all_reduce)
    sgemm
    sgemv
    dot product
    histogram
    softmax
    sigmoid
    relu
    layernorm
*/
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef INT4
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#endif

#ifndef FLOAT4
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#endif


// ElementWise Add  
// grid(N/128), block(128)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add(float* a, float* b, float* c, int N);

// ElementWise Add + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_vec4(float* a, float* b, float* c, int N);

// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val)
{
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
  {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/128), block(128)
template <const int NUM_THREADS = 128>
__device__ __forceinline__ float block_reduce_sum(float val);

template <const int NUM_THREADS = 128>
__device__ __forceinline__ float block_reduce_max(float val);

// Block All Reduce Sum
// grid(N/128), block(128)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 128>
__global__ void block_all_reduce_sum(float *a, float *y, int N);

// Block All Reduce Sum + vec4
// grid(N/128), block(128/4)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 128 / 4>
__global__ void block_all_reduce_sum_vec4(float *a, float *y, int N);

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
__global__ void sgemm(float* a, float* b, float* c, int M, int N, int K);

// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)
__global__ void sgemm_thread_tile_vec4(
  float* a, float* b, float* c, int M, int N, int K);

// SGEMV: Warp SGEMV K32
// assume K is a multiple of 32, each warp handles one row
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32(float *a, float *x, float *y, int M, int K);

// SGEMV: Warp SGEMV K128 + Vec4
// assume K is a mutilple of 128, float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k128(float *a, float *x, float *y, int M, int K);

// SGEMV: Warp SGEMV K16
// assume K is 16 < 32 and each warp handle 2 rows with 16 elements per row
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16(float *A, float *x, float *y, int M, int K);

// Dot Product
// grid(N/128), block(128)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 128>
__global__ void dot(float* a, float* b, float* y, int N);

// Dot Product + Vec4
// grid(N/128), block(128/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template<const int NUM_THREADS = 128/4>
__global__ void dot_vec4(float* a, float* b, float* y, int N);

// Histogram
// grid(N/128), block(128)
// a: Nx1, y: count histogram
__global__ void histogram(int* a, int* y, int N);

// Histogram + Vec4
// grid(N/128), block(128/4)
// a: Nx1, y: count histogram
__global__ void histogram_vec4(int* a, int* y, int N);

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax(float* x, float* y, float* total, int N);

// Softmax x: N, y: N
// grid(N/128), block(K=128)
template<const int NUM_THREADS = 128>
__global__ void softmax_v2(float* x, float* y, float* total, int N);

// Softmax Vec4 x: N, y: N
// grid(N/128), block(128/4)
template<const int NUM_THREADS = 128/4>
__global__ void softmax_v2_vec4(float* x, float* y, float* total, int N);

// Sigmoid x: N, y: N y=1/(1+exp(-x))
// grid(N/128), block(K=128) 
__global__ void sigmoid(float* x, float* y, int N);

// Sigmoid x: N, y: N y=1/(1+exp(-x)) Vec4
// grid(N/128), block(128/4)
__global__ void sigmoid_vec4(float* x, float* y, int N);

// Relu x: N, y: N y=max(0,x)
// grid(N/128), block(K=128) 
__global__ void relu(float* x, float* y, int N);

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/128/4), block(128/4) 
__global__ void relu_vec4(float* x, float* y, int N);

// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128>
__global__ void layer_norm(float* x, float* y, float g, float b, int N, int K);

// Layer Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128/4>
__global__ void layer_norm_vec4(float* x, float* y, float g, float b, int N, int K);