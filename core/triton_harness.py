from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


def has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class KernelCase:
    name: str
    shape: tuple[int, ...]
    dtype: str
    description: str = ""


@dataclass(frozen=True)
class BenchmarkSpec:
    warmup: int = 25
    rep: int = 100
    quantiles: tuple[float, float, float] = (0.5, 0.2, 0.8)


def assert_triton_close(
    actual: Any,
    expected: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    equal_nan: bool = False,
) -> None:
    import torch

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=equal_nan)


def do_bench(
    fn: Callable[[], Any],
    *,
    warmup: int = 25,
    rep: int = 100,
    quantiles: tuple[float, float, float] = (0.5, 0.2, 0.8),
) -> tuple[float, float, float]:
    import triton

    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, quantiles=quantiles)


def maybe_perf_report(*args: Any, **kwargs: Any) -> Any:
    import triton

    return triton.testing.perf_report(*args, **kwargs)

