"""第 6 课：global → shared → fragment；可选 tile 内反转。

验证：2026-11-19，Apple M1，TileLang 0.1.15，target="metal"。
从仓库根运行：.venv/bin/python code/language/memory_copy.py
"""
from __future__ import annotations

import numpy as np
import torch
import tilelang
import tilelang.language as T

BLOCK = 32


def make_copy_kernel(n: int, reverse: bool = False):
    """返回 kernel(A, B)；A/B 是长度 ceil(n/32)*32 的 FP32 MPS 张量。"""
    if n < 1:
        raise ValueError("n must be positive")
    padded = ((n + BLOCK - 1) // BLOCK) * BLOCK

    @tilelang.jit(target="metal")
    def copy_tile(A, B):
        A: T.Tensor((padded,), T.float32)
        B: T.Tensor((padded,), T.float32)
        with T.Kernel(padded // BLOCK, threads=32) as bx:
            shared = T.alloc_shared((BLOCK,), T.float32)
            local = T.alloc_fragment((BLOCK,), T.float32)
            T.copy(A[bx * BLOCK], shared)
            T.copy(shared, local)
            if reverse:
                # 寄存器片段不是任意线程都能索引的共享数组。
                # 先放回 shared，再让所有线程到齐，才能跨线程取值。
                T.copy(local, shared)
                T.sync_threads()
                for i in T.Parallel(BLOCK):
                    if bx * BLOCK + i < n:
                        B[bx * BLOCK + i] = shared[T.min(BLOCK, n - bx * BLOCK) - 1 - i]
                    else:
                        B[bx * BLOCK + i] = 0.0
            else:
                T.copy(local, B[bx * BLOCK])

    return copy_tile


def copy_reference(x: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    """CPU 数学参考：每个至多 32 个元素的有效小段独立反转，不反转 padding。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite 1-D FP32 array")
    result = x.copy()
    if reverse:
        for start in range(0, x.size, BLOCK):
            stop = min(start + BLOCK, x.size)
            result[start:stop] = x[start:stop][::-1]
    return result


def run_copy(x: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    """接收一维有限数值数组；返回同形状 CPU np.float32 数组，不修改输入。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite 1-D FP32 array")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This lesson requires an Apple GPU with PyTorch MPS")
    n = x.size
    padded = ((n + BLOCK - 1) // BLOCK) * BLOCK
    host = np.zeros(padded, dtype=np.float32)
    host[:n] = x
    a = torch.from_numpy(host).to("mps")
    b = torch.full((padded,), float("nan"), dtype=torch.float32, device="mps")
    make_copy_kernel(n, reverse)(a, b)
    torch.mps.synchronize()
    return b.cpu().numpy()[:n].copy()


def main() -> None:
    print(f"TileLang {tilelang.__version__}; target=metal; FP32; BLOCK={BLOCK}")
    for n in (1, 7, 32, 33, 64, 65):
        x = np.arange(n, dtype=np.float32) - 3
        for reverse in (False, True):
            actual = run_copy(x, reverse=reverse)
            np.testing.assert_array_equal(actual, copy_reference(x, reverse=reverse))
            print(f"n={n:2d} reverse={str(reverse):5s}: OK (exact)")
    x = np.arange(35, dtype=np.float32)
    print("reverse first 4 / last 3:", run_copy(x, reverse=True)[:4],
          run_copy(x, reverse=True)[-3:])


if __name__ == "__main__":
    main()
