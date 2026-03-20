from __future__ import annotations

import math

import torch
from torch import Tensor


def compute_token_bytes(
    prev_ids: Tensor,
    tgt_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    """Return byte counts for target tokens under the repo's boundary/space rules."""
    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
    return token_bytes


def compute_loss_byte_deltas(
    losses: Tensor,
    prev_ids: Tensor,
    tgt_ids: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Accumulate raw loss sum, raw byte sum, and token count for a scored slice."""
    loss_sum = losses.to(torch.float64).sum()
    byte_sum = compute_token_bytes(
        prev_ids,
        tgt_ids,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    ).to(torch.float64).sum()
    return loss_sum, byte_sum, int(tgt_ids.numel())


def compute_val_bpb(loss_sum: float | int, byte_count: float | int) -> float:
    if byte_count <= 0:
        raise ValueError(f"byte_count must be positive, got {byte_count}")
    return float((float(loss_sum) / math.log(2.0)) / float(byte_count))
