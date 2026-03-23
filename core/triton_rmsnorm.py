from __future__ import annotations

from typing import Final

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only when Triton is installed
    triton = None
    tl = None


TRITON_RMSNORM_MAX_N: Final[int] = 4096


def has_triton_rmsnorm() -> bool:
    return triton is not None and tl is not None


def supports_triton_rmsnorm_shape(x: Tensor) -> bool:
    return x.ndim >= 2 and x.size(-1) <= TRITON_RMSNORM_MAX_N


def rmsnorm_reference(x: Tensor, eps: float = 1e-6) -> Tensor:
    return torch.nn.functional.rms_norm(x, (x.size(-1),), eps=eps)


if triton is not None and tl is not None:  # pragma: no branch

    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        y_ptr,
        x_stride,
        y_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        tl.static_assert(BLOCK_SIZE >= n_cols)
        row = tl.program_id(axis=0)
        cols = tl.arange(0, BLOCK_SIZE)
        x_row_ptr = x_ptr + row * x_stride
        y_row_ptr = y_ptr + row * y_stride
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
        y = x * inv_rms
        tl.store(y_row_ptr + cols, y, mask=mask)


def triton_rmsnorm(x: Tensor, eps: float = 1e-6, block_size: int | None = None) -> Tensor:
    if not has_triton_rmsnorm():
        raise RuntimeError("Triton is not available in this environment")
    if not x.is_cuda:
        raise ValueError("triton_rmsnorm requires a CUDA tensor")
    if x.ndim < 2:
        raise ValueError(f"triton_rmsnorm expects at least 2 dims, got {tuple(x.shape)}")
    if not supports_triton_rmsnorm_shape(x):
        raise ValueError(
            f"triton_rmsnorm only supports hidden sizes <= {TRITON_RMSNORM_MAX_N}, got {x.size(-1)}"
        )
    if x.stride(-1) != 1:
        x = x.contiguous()

    orig_shape = x.shape
    x2d = x.reshape(-1, x.size(-1))
    rows, n_cols = x2d.shape

    y = torch.empty_like(x2d)
    bs = block_size or triton.next_power_of_2(n_cols)
    if bs > TRITON_RMSNORM_MAX_N:
        raise ValueError(f"block_size must be <= {TRITON_RMSNORM_MAX_N}, got {bs}")
    num_warps = min(max(bs // 256, 1), 8)
    _rmsnorm_fwd_kernel[(rows,)](
        x2d,
        y,
        x2d.stride(0),
        y.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=bs,
        num_warps=num_warps,
        num_ctas=1,
    )
    return y.reshape(orig_shape)
