import torch

import triton
import triton.language as tl


"""
the layernorm's compute logic as follows:
Y = (X - mean(X)) * W / (std(X) - epsilon) + B
X: input for this layer
W: weight for this layer
B: bias ...
epsilon: hyperparam, avoid divide by zero
"""

def layernorm_torch(x: torch.Tensor, epsilon, w, b):
    return (x - x.mean()) * w / (x.std() - epsilon) + b


@triton.jit
def layernorm_fwd_triton(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride, # how much to increase the pointer when moving by 1 row
    N,      # the number of columns in X
    eps,    # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    X += row * stride
    Y += row * stride
    
    # compute mean
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols += off + tl.arange(0, BLOCK_SIZE)
        xi = tl.load(X + cols, mask=cols<N, other=0).to(tl.float32)
        _mean += xi
    mean = tl.sum(_mean, axis=0) / N
    
    # compute std
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    
    # normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x -mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)        
    
    
    
    

if __name__ == '__main__':
    print()