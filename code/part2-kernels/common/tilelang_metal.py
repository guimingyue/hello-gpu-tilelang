"""TileLang Metal 路线共用的主机工具：MPS 张量、墙钟计时、环境记录与正确性检查。

计时口径说明
------------
TileLang 的 Metal 后端目前不暴露 command buffer GPU 时间戳接口，因此本教程统一
采用**端到端墙钟**（``time.perf_counter`` 包住一次完整算子调用 + ``torch.mps.synchronize``），
口径标记为 ``wall-clock-full-operator``。它与 hello-gpu-metal 中 MLX 路线的墙钟同口径，
可以直接横向比较；但**不**与 Metal 版 MSL 的 GPU 时间戳同口径，跨口径只比各自相对改进。
"""
from __future__ import annotations

import platform
import statistics
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def gpu_name() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "Apple Silicon"


def environment(runtime: str = "tilelang") -> dict[str, str]:
    return {
        "runtime": runtime,
        "gpu": gpu_name(),
        "architecture": "applegpu_g13g",
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "tilelang": _pkg("tilelang"),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mps_available": str(torch.mps.is_available()),
    }


def to_mps(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to("mps")


def readback(tensor: torch.Tensor) -> np.ndarray:
    torch.mps.synchronize()
    return tensor.detach().cpu().numpy()


def wall_samples(op, warmup: int, repeat: int) -> list[float]:
    """对完整算子 ``op()``（内部已含同步）做 warmup 次预热 + repeat 次墙钟采样，返回 ms。"""
    for _ in range(warmup):
        op()
    samples = []
    for _ in range(repeat):
        torch.mps.synchronize()
        start = time.perf_counter()
        op()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def median(values: list[float]) -> float:
    return statistics.median(values)


def finite_abs_check(output: np.ndarray, reference: np.ndarray,
                     tolerance: float) -> tuple[bool, float]:
    """全量 finite + 最大绝对误差检查（与 Metal 版同容差）。"""
    out = np.asarray(output)
    ref = np.asarray(reference)
    if out.shape != ref.shape or not np.isfinite(out).all():
        return False, float("inf")
    error = float(np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64)))) if out.size else 0.0
    return error <= tolerance, error