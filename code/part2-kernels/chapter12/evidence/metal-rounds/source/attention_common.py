"""第 12 章共用：教学版 Attention 输入、FP64 参考、混合容差（与 hello-gpu-metal 同口径）。

Q/K/V 为 dyadic（1/64 或 1/32 的整数倍，FP32 精确）。参考为 FP64 稳定 softmax 注意力。
容差 atol=2e-5、rtol=2e-4（逐元素 |err| <= atol + rtol·|ref|）。
"""
from __future__ import annotations

import math

import numpy as np

OPERATOR = "attention"
ATOL = 2.0e-5
RTOL = 2.0e-4


def make_input(seq: int, dim: int, seed: int, kind: str = "dyadic"):
    r = np.arange(seq, dtype=np.int64)[:, None]
    c = np.arange(dim, dtype=np.int64)[None, :]
    q = (((17 * r + 13 * c + seed) % 65) - 32) / 64.0
    k = (((29 * r + 7 * c + seed + 11) % 65) - 32) / 64.0
    v = (((11 * r + 19 * c + seed + 23) % 65) - 32) / 32.0
    if kind == "rising-max":
        q = np.where(c == 0, 8.0, 0.0) + 0 * r
        k = np.where(c == 0, (r - seq // 2) / 8.0, 0.0)
    elif kind != "dyadic":
        raise ValueError(f"unknown input {kind!r}")
    return tuple(np.ascontiguousarray(x, dtype=np.float32) for x in (q, k, v))


def reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q64, k64, v64 = (x.astype(np.float64) for x in (q, k, v))
    scores = (q64 @ k64.T) / math.sqrt(q.shape[1])
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights @ v64).astype(np.float32)


def validate(actual: np.ndarray, reference: np.ndarray) -> tuple[bool, float, float]:
    actual = np.asarray(actual, dtype=np.float64)
    ref = reference.astype(np.float64)
    if actual.shape != ref.shape or not np.isfinite(actual).all():
        return False, float("inf"), float("inf")
    error = np.abs(actual - ref)
    tolerance = ATOL + RTOL * np.abs(ref)
    ok = bool((error <= tolerance).all())
    return ok, float(error.max(initial=0.0)), float((error / tolerance).max(initial=0.0))