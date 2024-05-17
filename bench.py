import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
import time
import numpy as np

m, n, k = 64, 64, 64
a = torch.rand((m, k), device="cuda:0")
b = torch.rand((k, n), device="cuda:0")
c = torch.zeros((m, n), device="cuda:0")
ntest = 20

def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    torch.cuda.empty_cache()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time-start_time)*1e6)
    return times, res

def run_torch():
    torch.matmul(input=a, other=b, out=c)
    return c

def run_cuda():
    sgemm.torch_launch_sgemm(a, b, c, m, n, k)
    return c

def run_cuda_tile():
    sgemm.torch_launch_sgemm_thread_tile_vec4(a, b, c, m, n, k)
    return c

sgemm = load(name="sgemm",
             extra_include_paths=["include"],
             sources=["kernel/sgemm.cu", "kernel/sgemm.cc"],
             verbose=True)

if __name__ == '__main__':
    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    torch.allclose(cuda_res, torch_res)
    print("Kernel test passed.")