"""第 13 章共用：Fused RMSNorm 输入、FP64 参考、混合容差与有效带宽（同 Metal 版）。

y[i,j] = x[i,j] / sqrt(mean(x[i,:]^2) + epsilon) * w[j]。输入 dyadic（1/128 整数倍）。
容差 atol=2e-5、rtol=2e-5；epsilon=1e-5。
"""
from __future__ import annotations

import numpy as np

OPERATOR = "rmsnorm"
ATOL = 2.0e-5
RTOL = 2.0e-5
EPSILON = 1e-5


def make_inputs(rows: int, cols: int, seed: int, input_mode: str = "normal"):
    indices = np.arange(rows * cols, dtype=np.int64)
    values = (((17 * (indices % 257) + seed % 257) % 257 - 128).astype(np.float32) / np.float32(128.0))
    columns = np.arange(cols, dtype=np.int64)
    weight = (((29 * (columns % 251) + 3 * (seed % 251)) % 251 - 125).astype(np.float32) / np.float32(128.0))
    if input_mode == "zero":
        values[:] = 0
    elif input_mode == "zero-weight":
        weight[:] = 0
    elif input_mode == "signed-weight":
        signs = np.where(columns % 2 == 0, -1.0, 1.0).astype(np.float32)
        weight = signs * (np.float32(0.5) + (columns % 9).astype(np.float32) / np.float32(16.0))
    return values.reshape(rows, cols), weight.astype(np.float32)


def reference_rmsnorm(x: np.ndarray, w: np.ndarray, epsilon: float) -> np.ndarray:
    values = x.astype(np.float64)
    inverse_rms = 1.0 / np.sqrt(np.square(values).sum(axis=1, keepdims=True) / values.shape[1] + epsilon)
    return (values * inverse_rms * w.astype(np.float64)).astype(np.float32)


def validate(actual: np.ndarray, reference: np.ndarray) -> tuple[bool, float, float]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = reference.astype(np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return False, float("inf"), float("inf")
    difference = np.abs(actual - expected)
    acceptable = difference <= ATOL + RTOL * np.abs(expected)
    relative = difference / np.maximum(np.abs(expected), 1e-6)
    return bool(acceptable.all()), float(difference.max(initial=0.0)), float(relative.max(initial=0.0))


def bandwidth_gbs(shape: str, median_ms: float) -> float:
    rows, cols = (int(v) for v in shape.split("x"))
    return 8 * rows * cols / (median_ms * 1e6)