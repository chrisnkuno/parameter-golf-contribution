from __future__ import annotations

import hashlib
import math
import os
from typing import TypedDict

import torch
from torch import Tensor

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
DEFAULT_INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS = "tok_emb.weight"
INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS",
        DEFAULT_INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS,
    ).split(",")
    if pattern
)
SUPPORTED_INT8_PRECONDITIONERS = ("none", "hadamard", "hadamard_sign")
INT8_PRECONDITIONER = os.environ.get("INT8_PRECONDITIONER", "none").strip().lower() or "none"
INT8_PRECONDITIONER_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get("INT8_PRECONDITIONER_NAME_PATTERNS", "").split(",") if pattern
)

if INT8_PRECONDITIONER not in SUPPORTED_INT8_PRECONDITIONERS:
    raise ValueError(
        f"Unsupported INT8_PRECONDITIONER={INT8_PRECONDITIONER!r}; "
        f"expected one of {SUPPORTED_INT8_PRECONDITIONERS}"
    )


class QuantMetaEntry(TypedDict, total=False):
    scheme: str
    axis: int
    preconditioner: str
    precondition_dim: int


class QuantizedStateDict(TypedDict, total=False):
    __quant_format__: str
    quantized: dict[str, Tensor]
    scales: dict[str, Tensor]
    dtypes: dict[str, str]
    passthrough: dict[str, Tensor]
    qmeta: dict[str, QuantMetaEntry]
    passthrough_orig_dtypes: dict[str, str]


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def should_keep_large_float_tensor(name: str) -> bool:
    return any(pattern in name for pattern in INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS)


def should_precondition_tensor(name: str, t: Tensor) -> bool:
    return (
        INT8_PRECONDITIONER != "none"
        and t.ndim == 2
        and bool(INT8_PRECONDITIONER_NAME_PATTERNS)
        and any(pattern in name for pattern in INT8_PRECONDITIONER_NAME_PATTERNS)
        and is_power_of_two(int(t.shape[-1]))
    )


def hadamard_transform_last_dim(t: Tensor) -> Tensor:
    width = int(t.shape[-1])
    if not is_power_of_two(width):
        raise ValueError(f"Hadamard preconditioner requires a power-of-two last dim, got {tuple(t.shape)}")
    x = t.to(dtype=torch.float32).reshape(-1, width).clone()
    h = 1
    while h < width:
        x = x.view(-1, width // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.view(-1, width)
        h *= 2
    return (x / math.sqrt(width)).reshape_as(t).contiguous()


def deterministic_sign_vector(name: str, width: int) -> Tensor:
    values: list[float] = []
    counter = 0
    while len(values) < width:
        digest = hashlib.sha256(f"{name}:{width}:{counter}".encode("utf-8")).digest()
        for byte in digest:
            for bit_idx in range(8):
                values.append(1.0 if ((byte >> bit_idx) & 1) else -1.0)
                if len(values) == width:
                    break
            if len(values) == width:
                break
        counter += 1
    return torch.tensor(values, dtype=torch.float32)


def apply_structured_preconditioner(name: str, t: Tensor, preconditioner: str) -> Tensor:
    if preconditioner == "none":
        return t.detach().clone().contiguous()
    if t.ndim != 2:
        raise ValueError(f"Structured preconditioner requires a 2D tensor, got {tuple(t.shape)}")
    base = t.to(dtype=torch.float32)
    if preconditioner == "hadamard":
        return hadamard_transform_last_dim(base)
    if preconditioner == "hadamard_sign":
        sign = deterministic_sign_vector(name, int(base.shape[-1])).to(device=base.device)
        return hadamard_transform_last_dim(base * sign.view(1, -1))
    raise ValueError(f"Unsupported structured preconditioner: {preconditioner}")


def invert_structured_preconditioner(name: str, t: Tensor, preconditioner: str) -> Tensor:
    if preconditioner == "none":
        return t.detach().clone().contiguous()
    original_dtype = t.dtype
    base = t.to(dtype=torch.float32)
    if preconditioner == "hadamard":
        return hadamard_transform_last_dim(base).to(dtype=original_dtype).contiguous()
    if preconditioner == "hadamard_sign":
        sign = deterministic_sign_vector(name, int(base.shape[-1])).to(device=base.device)
        return (hadamard_transform_last_dim(base) * sign.view(1, -1)).to(dtype=original_dtype).contiguous()
    raise ValueError(f"Unsupported structured preconditioner: {preconditioner}")


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    min_scale = torch.finfo(torch.float32).tiny
    scale = torch.tensor(max(clip_abs / 127.0, min_scale) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]) -> tuple[QuantizedStateDict, dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, QuantMetaEntry] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "num_large_float_passthrough_tensors",
            "large_float_passthrough_bytes",
            "num_preconditioned_tensors",
            "preconditioned_tensor_bytes",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if should_keep_large_float_tensor(name):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["num_large_float_passthrough_tensors"] += 1
            stats["large_float_passthrough_bytes"] += tensor_nbytes(kept)
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        to_quantize = t.float()
        meta: QuantMetaEntry = {}
        if should_precondition_tensor(name, t):
            to_quantize = apply_structured_preconditioner(name, to_quantize, INT8_PRECONDITIONER)
            meta["preconditioner"] = INT8_PRECONDITIONER
            meta["precondition_dim"] = int(t.shape[-1])
            stats["num_preconditioned_tensors"] += 1
            stats["preconditioned_tensor_bytes"] += tensor_nbytes(t)
        q, s = quantize_float_tensor(to_quantize)
        if s.ndim > 0:
            meta["scheme"] = "per_row"
            meta["axis"] = 0
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: QuantizedStateDict = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: QuantizedStateDict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        meta = qmeta.get(name, {})
        s = obj["scales"][name]
        if meta.get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            dequantized = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            dequantized = (q.float() * scale).to(dtype=dtype).contiguous()
        preconditioner = meta.get("preconditioner")
        if isinstance(preconditioner, str) and preconditioner != "none":
            expected_dim = int(meta.get("precondition_dim", dequantized.shape[-1]))
            if int(dequantized.shape[-1]) != expected_dim:
                raise ValueError(
                    f"Preconditioned tensor {name} expected last dim {expected_dim}, got {tuple(dequantized.shape)}"
                )
            dequantized = invert_structured_preconditioner(name, dequantized, preconditioner)
        out[name] = dequantized
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out
