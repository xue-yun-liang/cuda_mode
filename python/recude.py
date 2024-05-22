import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from utils import show_time
import numpy as np

n = 32
a = torch.rand((n), device="cuda:0")
b = 0.0
print(a)

def run_reduce_torch():
    return torch.sum(a)

def run_reduce_cuda():
    reduce.torch_launch_reduce(a, b, n)
    return b

reduce = load(name="reduce",
             extra_include_paths=["../include"],
             sources=["../kernel/reduce.cu", "../kernel/reduce.cc"],
             verbose=True)

if __name__ == '__main__':
    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_reduce_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_reduce_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    
    print(cuda_res)
    print(torch_res)
    print("Kernel test passed.")