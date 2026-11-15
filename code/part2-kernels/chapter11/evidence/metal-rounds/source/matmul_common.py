"""第 11 章共用：dyadic 矩阵乘输入、FP64 参考、Higham FP32 上界与 TFLOPS（同 Metal 版）。

C = A @ B，A M×K、B K×N、C M×N，FP32 行优先。输入都是 1/128 的整数倍（FP32 精确）。
逐元素容差 |err| <= (gamma_K + u)·(|A|@|B|)，gamma_K=K·u/(1-K·u)，u=2^-24。
"""
from __future__ import annotations

import numpy as np

OPERATOR = "matmul"
UNIT_ROUNDOFF = 2.0 ** -24
INPUT_FORMULA = ("a[i,j]=((17*(i*k+j)+seed)%2048-1024)/128; "
                 "b[i,j]=((29*(i*n+j)+3*seed)%4096-2048)/128")


def make_inputs(m: int, n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    a_index = np.arange(m * k, dtype=np.int64)
    a = (((17 * a_index + seed) % 2048) - 1024).astype(np.float32) / np.float32(128)
    b_index = np.arange(k * n, dtype=np.int64)
    b = (((29 * b_index + 3 * seed) % 4096) - 2048).astype(np.float32) / np.float32(128)
    return a.reshape(m, k), b.reshape(k, n)


def reference_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)


def error_bound(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = a.shape[1]
    u = UNIT_ROUNDOFF
    gamma = k * u / (1.0 - k * u)
    return (gamma + u) * (np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64)))


def validate(output: np.ndarray, reference: np.ndarray, bound: np.ndarray) -> tuple[bool, float, float]:
    actual = np.asarray(output)
    if actual.shape != reference.shape or not np.isfinite(actual).all():
        return False, float("inf"), float("inf")
    error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    ratio = np.zeros_like(error)
    np.divide(error, bound, out=ratio, where=bound > 0)
    ratio[(bound == 0) & (error > 0)] = np.inf
    max_error = float(error.max()) if error.size else 0.0
    max_ratio = float(ratio.max()) if ratio.size else 0.0
    return max_ratio <= 1.0, max_error, max_ratio


def tflops(shape: str, median_ms: float) -> float:
    m, n, k = (int(v) for v in shape.split("x"))
    return 2 * m * n * k / (median_ms * 1e9)