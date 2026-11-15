"""第 10 章共用：稳定 Softmax 输入、FP64 参考、容差与有效带宽（与 hello-gpu-metal 同口径）。

输入为 shifted-dyadic：值都是 1/128 的整数倍（FP32 精确），按行 +1000/-1000/0 平移，
直接 exp 会溢出/下溢，必须先减行最大值。容差 abs=2e-5、row_sum=2e-5。
"""
from __future__ import annotations

import numpy as np

OPERATOR = "softmax"
ABSOLUTE_TOLERANCE = 2.0e-5
ROW_SUM_TOLERANCE = 2.0e-5
INPUT_KIND = "shifted-dyadic"


def make_input(rows: int, cols: int, seed: int) -> np.ndarray:
    row_ids = np.arange(rows, dtype=np.int64).reshape(rows, 1)
    col_ids = np.arange(cols, dtype=np.int64).reshape(1, cols)
    mixed = col_ids * 17 + row_ids * 131 + seed
    centered = np.remainder(mixed, 4096) - 2048
    shifts = np.select([row_ids % 3 == 0, row_ids % 3 == 1], [1000.0, -1000.0], 0.0)
    return (centered.astype(np.float32) / np.float32(128.0) + shifts.astype(np.float32)).astype(np.float32)


def reference_softmax(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    p = np.exp(x64 - x64.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def validate(output: np.ndarray, reference: np.ndarray) -> tuple[bool, float, float]:
    actual = np.asarray(output, dtype=np.float32)
    finite = bool(np.isfinite(actual).all())
    if not finite:
        return False, float("inf"), float("inf")
    abs_err = float(np.max(np.abs(actual.astype(np.float64) - reference)))
    row_err = float(np.abs(actual.astype(np.float64).sum(axis=1) - 1.0).max())
    ok = abs_err <= ABSOLUTE_TOLERANCE and row_err <= ROW_SUM_TOLERANCE
    return ok, abs_err, row_err


def bandwidth_gbs(shape: str, median_ms: float) -> float:
    rows, cols = (int(v) for v in shape.split("x"))
    return 8 * rows * cols / (median_ms * 1e6)