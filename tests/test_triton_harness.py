from __future__ import annotations

import torch

from core.triton_harness import BenchmarkSpec, KernelCase, has_triton
from core.triton_rmsnorm import TRITON_RMSNORM_MAX_N, has_triton_rmsnorm, supports_triton_rmsnorm_shape


def test_kernel_case_fields_roundtrip() -> None:
    case = KernelCase("demo", (256, 512), "bf16", "example")
    assert case.name == "demo"
    assert case.shape == (256, 512)
    assert case.dtype == "bf16"
    assert case.description == "example"


def test_benchmark_spec_defaults() -> None:
    spec = BenchmarkSpec()
    assert spec.warmup == 25
    assert spec.rep == 100
    assert spec.quantiles == (0.5, 0.2, 0.8)


def test_has_triton_returns_bool() -> None:
    assert isinstance(has_triton(), bool)


def test_has_triton_rmsnorm_returns_bool_and_max_hidden_positive() -> None:
    assert isinstance(has_triton_rmsnorm(), bool)
    assert TRITON_RMSNORM_MAX_N > 0


def test_supports_triton_rmsnorm_shape_tracks_last_dimension_limit() -> None:
    assert supports_triton_rmsnorm_shape(torch.zeros(4, 512))
    assert not supports_triton_rmsnorm_shape(torch.zeros(4))
    assert not supports_triton_rmsnorm_shape(torch.zeros(2, TRITON_RMSNORM_MAX_N + 1))
