import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == '__main__':
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, dtype=torch.float32, device='cuda')
    y = torch.rand(size, dtype=torch.float32, device='cuda')
    output_triton = add(x, y)
    print(output_triton)


'''
How to test compilation and get Triton IR:
Execute `python3 -m triton.tools.compile ./compile_test.py -n "add_kernel" -s "*fp32, *fp32, *fp32, i32, 1024" -on "add_kernel_cmp"`
Open `/nfshome/tchen307/triton/.venv/lib/python3.8/site-packages/triton/compiler/compiler.py` and follow the instructions at L475
ttir: Triton IR, llir: LLVM IR, ttgir: Triton GPU IR
'''
