"""第 9 章共用：输入生成、FP64 参考、误差上界与有效带宽（与 hello-gpu-metal 同口径）。

ones / weighted / random 都是 1/32 的整数倍，部分和远小于 2^19，任何相加顺序都不产生
舍入，因此与 FP64 参考严格相等；uniform 是 [0,1) 一般浮点数，用 Higham 误差上界作容差。
主实验输入为 random（与 Metal 版一致）。
"""
from __future__ import annotations

import numpy as np

OPERATOR = "sum-reduction"
MAX_SIZE = 1 << 24
FP32_UNIT_ROUNDOFF = 2.0 ** -24
EXACT_INPUTS = ("ones", "weighted", "random")
INPUTS = (*EXACT_INPUTS, "uniform")
MAIN_INPUT = "random"


def make_input(kind: str, n: int, seed: int) -> np.ndarray:
    if not 0 < n <= MAX_SIZE:
        raise ValueError("require 0 < N <= 2^24")
    i = np.arange(n, dtype=np.int64)
    if kind == "ones":
        return np.ones(n, np.float32)
    if kind == "weighted":
        return (((i * 17 + (i // 251) * 7 + seed) % 31) - 15).astype(np.float32) / np.float32(32)
    if kind == "random":
        return np.random.default_rng(seed).integers(-16, 17, size=n).astype(np.float32) / np.float32(32)
    if kind == "uniform":
        return np.random.default_rng(seed).random(n, dtype=np.float32)
    raise ValueError(f"unknown input: {kind}")


def reference_fp64(x: np.ndarray) -> float:
    return float(np.sum(x, dtype=np.float64))


def error_bound(x: np.ndarray, depth: int) -> float:
    k = depth * FP32_UNIT_ROUNDOFF
    return k / (1.0 - k) * float(np.sum(np.abs(x), dtype=np.float64))


def check(kind: str, actual: float, expected: float, tolerance: float) -> tuple[bool, float]:
    if not np.isfinite(actual):
        return False, float("inf")
    err = abs(float(actual) - float(expected))
    if kind in EXACT_INPUTS:
        return actual == expected, err
    return err <= tolerance, err


def bandwidth_gbs(shape: str, median_ms: float) -> float:
    n = int(shape)
    return (4 * n + 4) / (median_ms * 1e6)