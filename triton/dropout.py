import torch
import tabulate

import triton
import triton.language as tl


@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    
    output = tl.where(x_keep, x / (1 - p), 0)
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


def run_dropout():
    print("call run_dropout")
    # Input tensor
    x = torch.randn(size=(10, )).cuda()
    # Dropout mask
    p = 0.5
    x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
    # Output tensor
    output = dropout(x, x_keep=x_keep, p=p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))


@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


def run_seeded_dropout():
    print("call run_seeded_dropout")
    # Input tensor
    x = torch.randn(size=(10, )).cuda()
    # Dropout mask
    p = 0.5
    seed = 1234
    # Output tensor
    output = seeded_dropout(x, p=0.5, seed=123)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output"] + output.tolist(),
    ]))


# Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
#version 1: x is matrix, mask is matrix
@triton.jit
def dropout_matrix_kernel(x_ptr, mask_ptr, output_ptr, p, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # get the data ptr in each rows
    x_row_ptrs = x_ptr + row_idx * n_cols + col_offsets
    mask_row_ptrs = mask_ptr + row_idx * n_cols + col_offsets
    output_row_ptrs = output_ptr + row_idx * n_cols + col_offsets
    # set the mask in each rows
    mask = col_offsets < n_cols
    # load x and mask data
    x_vals = tl.load(x_row_ptrs, mask=mask, other=0.0)
    mask_vals = tl.load(mask_row_ptrs, mask=mask, other=0.0)
    # compute and return the dropout algo's result
    output = tl.where(mask_vals, x_vals / (1 - p), 0.0)
    tl.store(output_row_ptrs, output, mask=mask)


def dropout_matrix(x, mask, p=0.5, block_size=128):
    assert x.shape == mask.shape, "Input and mask must have the same shape"
    output = torch.empty_like(x)
    grid = (x.shape[0],)
    n_cols = x.shape[1]
    dropout_matrix_kernel[grid](x, mask, output, p, n_cols, BLOCK_SIZE=block_size)
    
    return output
    
    
def run_dropout_matrix():
    print("call run_dropout_matrix")
    x = torch.randn(8, 4, device='cuda')
    mask = (torch.rand_like(x) > 0.5).float()
    output = dropout_matrix(x, mask)
    print("input:")
    print(x)
    print("mask:")
    print(mask)
    print("output:")
    print(output)

#version 2: x is matrix, mask is vector
@triton.jit
def dropout_matrix_kernel_v2(x_ptr, mask_ptr, output_ptr, p, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # load mask
    # NOTE: if one row has only one value, it can be load directly, don't need mask
    mask_value = tl.load(mask_ptr + row_idx)
    
    # get the data ptr in each rows
    x_row_ptrs = x_ptr + row_idx * n_cols + col_offsets

    output_row_ptrs = output_ptr + row_idx * n_cols + col_offsets
    # set the mask in each rows
    mask = col_offsets < n_cols
    # load x and mask data
    x_vals = tl.load(x_row_ptrs, mask=mask, other=0.0)
    # mask_vals = tl.load(mask_row_ptr, mask=mask, other=0.0)
    # compute and return the dropout algo's result
    output = tl.where(mask_value, x_vals / (1 - p), 0.0)
    tl.store(output_row_ptrs, output, mask=mask)


def dropout_matrix_v2(x, mask, p=0.5, block_size=128):
    assert x.shape[0] == mask.shape[0], "Input and mask must have the same shape"
    output = torch.empty_like(x)
    grid = (x.shape[0],)
    n_cols = x.shape[1]
    dropout_matrix_kernel_v2[grid](x, mask, output, p, n_cols, BLOCK_SIZE=block_size)
    
    return output

def run_dropout_matrix_v2():
    print("call run_dropout_matrix_v2")
    x = torch.randn(8, 4, device='cuda')
    mask = (torch.rand(x.shape[0], device='cuda') > 0.5).float()
    output = dropout_matrix_v2(x, mask)
    print("input:")
    print(x)
    print("mask:")
    print(mask)
    print("output:")
    print(output)
    
if __name__ == '__main__':
    run_dropout()
    run_seeded_dropout()
    run_dropout_matrix()
    run_dropout_matrix_v2()
