import os
import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({'B_N': 16}, num_warps=nw, num_stages=ns)
#         for nw in [2, 4, 8]
#         for ns in [3, 4, 5]
#     ],
#     key=['N']
# )
@triton.jit
def matmul_kernel(x_ptr, y_ptr, z_ptr, N, B_N: tl.constexpr):
    i_offset = B_N * tl.program_id(0) + tl.arange(0, B_N)
    j_offset = B_N * tl.program_id(1) + tl.arange(0, B_N)

    z = tl.zeros([B_N, B_N], tl.float32)

    for k in tl.range(0, N, B_N):
        k_offset = k + tl.arange(0, B_N)

        x_blk_offset = N * i_offset[:, None] + k_offset[None, :]
        x_blk_mask = (i_offset < N)[:, None] & (k_offset < N)[None, :]
        x = tl.load(x_ptr + x_blk_offset, x_blk_mask, 0.0)

        y_blk_offset = N * k_offset[:, None] + j_offset[None, :]
        y_blk_mask = (k_offset < N)[:, None] & (j_offset < N)[None, :]
        y = tl.load(y_ptr + y_blk_offset, y_blk_mask, 0.0)

        z = tl.dot(x, y, acc=z)

    z_blk_offset = N * i_offset[:, None] + j_offset[None, :]
    z_blk_mask = (i_offset < N)[:, None] & (j_offset < N)[None, :]
    tl.store(z_ptr + z_blk_offset, z, z_blk_mask)


def triton_matmul(x: torch.Tensor, y: torch.Tensor):
    assert x.size() == y.size()
    N = x.size(0)

    z = torch.empty([N, N], dtype=torch.float16, device=x.device)
    grid = lambda meta: ( triton.cdiv(N, meta['B_N']), triton.cdiv(N, meta['B_N']) )

    # Best config: B_N: 16, num_warps: 2, num_ctas: 1, num_stages: 5
    out = matmul_kernel[grid](x, y, z, N, 16)

    for ext in ['ttir', 'ttgir', 'ptx']:
        with open(f'matmul32.{ext}', 'w') as f:
            f.write(out.asm[ext])

    # os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
    # matmul_kernel[grid](x, y, z, N)

    return z


def main(N):
    torch.manual_seed(3985)
    x = torch.rand([N, N], dtype=torch.float16, device='cuda')
    y = torch.rand([N, N], dtype=torch.float16, device='cuda')
    z_tt = triton_matmul(x, y)
    assert torch.allclose(z_tt, x @ y)


if __name__ == '__main__':
    main(32)
