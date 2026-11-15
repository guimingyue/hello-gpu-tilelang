"""第 8 章 TileLang / MLX 两条路线共用的输入、校验与输出格式。

输入公式与 hello-gpu-metal 完全一致：所有值都是 2 的负幂次的整数倍，FP32 加法精确，
正确实现的最大绝对误差应为 0；判定阈值仍取 1e-6。
"""
from __future__ import annotations

import numpy as np

TOLERANCE = 1e-6
OPERATOR = "vector-add"
INPUT_FORMULA = "a=((17*i+seed)%2048-1024)/1024; b=((29*i+3*seed)%4096-2048)/2048"
BYTES_PER_ELEMENT = 12  # 读 A、读 B、写 C，各 4 Byte


def make_inputs(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(size, dtype=np.int64)
    a = (((17 * index + seed) % 2048) - 1024).astype(np.float32) / np.float32(1024)
    b = (((29 * index + 3 * seed) % 4096) - 2048).astype(np.float32) / np.float32(2048)
    return a, b, a + b


def bandwidth_gbs(shape: str, median_ms: float) -> float:
    return BYTES_PER_ELEMENT * int(shape) / (median_ms * 1e6)