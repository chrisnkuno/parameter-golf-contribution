from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EvalResult:
    val_loss: float
    val_bpb: float
    loss_sum: float
    token_count: float
    byte_count: float


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


def finalize_eval_result(
    loss_sum: Tensor | float | int,
    token_count: Tensor | float | int,
    byte_count: Tensor | float | int,
) -> EvalResult:
    loss_sum_value = float(loss_sum.item()) if isinstance(loss_sum, Tensor) else float(loss_sum)
    token_count_value = float(token_count.item()) if isinstance(token_count, Tensor) else float(token_count)
    byte_count_value = float(byte_count.item()) if isinstance(byte_count, Tensor) else float(byte_count)
    if token_count_value <= 0:
        raise ValueError(f"token_count must be positive, got {token_count_value}")
    if byte_count_value <= 0:
        raise ValueError(f"byte_count must be positive, got {byte_count_value}")
    return EvalResult(
        val_loss=loss_sum_value / token_count_value,
        val_bpb=compute_val_bpb(loss_sum_value, byte_count_value),
        loss_sum=loss_sum_value,
        token_count=token_count_value,
        byte_count=byte_count_value,
    )
