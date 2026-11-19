"""第 7 课：小矩阵的行求和与稳定行 Softmax，手动 shared 归约。

验证：2026-11-19，Apple M1，TileLang 0.1.15，target="metal"。
此安装的 T.reduce_sum 编译报 no reduce implementation is registered；
这里显式使用每线程局部结果 + shared + 线程 0 串行合并，不是性能基线。
从仓库根运行：.venv/bin/python code/language/reductions.py
"""
from __future__ import annotations

import numpy as np
import torch
import tilelang
import tilelang.language as T

THREADS = 32


def make_sum_kernel(rows: int, padded_cols: int):
    """返回 kernel(A, B)；A[rows,padded_cols]，B[rows]，尾列须补 0。"""
    if rows < 1 or padded_cols < THREADS or padded_cols % THREADS:
        raise ValueError("rows > 0; padded_cols must be a positive multiple of 32")

    @tilelang.jit(target="metal")
    def row_sum(A, B):
        A: T.Tensor((rows, padded_cols), T.float32)
        B: T.Tensor((rows,), T.float32)
        with T.Kernel(rows, threads=THREADS) as bx:
            tid = T.get_thread_binding()
            local = T.alloc_fragment((1,), T.float32)
            T.clear(local)
            for j in T.serial(padded_cols // THREADS):
                local[0] = local[0] + A[bx, tid + j * THREADS]
            shared = T.alloc_shared((THREADS,), T.float32)
            shared[tid] = local[0]
            T.sync_threads()
            if tid == 0:
                total = T.alloc_fragment((1,), T.float32)
                T.clear(total)
                for i in T.serial(THREADS):
                    total[0] = total[0] + shared[i]
                B[bx] = total[0]

    return row_sum


def make_softmax_kernel(rows: int, padded_cols: int):
    """返回 kernel(A, B)；均为 [rows,padded_cols]，A 尾列须补 -inf。"""
    if rows < 1 or padded_cols < THREADS or padded_cols % THREADS:
        raise ValueError("rows > 0; padded_cols must be a positive multiple of 32")

    @tilelang.jit(target="metal")
    def row_softmax(A, B):
        A: T.Tensor((rows, padded_cols), T.float32)
        B: T.Tensor((rows, padded_cols), T.float32)
        with T.Kernel(rows, threads=THREADS) as bx:
            tid = T.get_thread_binding()
            maxima = T.alloc_shared((THREADS,), T.float32)
            row_max = T.alloc_shared((1,), T.float32)
            sums = T.alloc_shared((THREADS,), T.float32)
            row_total = T.alloc_shared((1,), T.float32)

            # 1. 每个线程找局部最大值；线程 0 合并，发布行最大值。
            local_max = T.alloc_fragment((1,), T.float32)
            local_max[0] = -T.infinity(T.float32)
            for j in T.serial(padded_cols // THREADS):
                local_max[0] = T.max(local_max[0], A[bx, tid + j * THREADS])
            maxima[tid] = local_max[0]
            T.sync_threads()
            if tid == 0:
                maximum = T.alloc_fragment((1,), T.float32)
                maximum[0] = -T.infinity(T.float32)
                for i in T.serial(THREADS):
                    maximum[0] = T.max(maximum[0], maxima[i])
                row_max[0] = maximum[0]
            T.sync_threads()

            # 2. exp(x - max) 求和；单独保存 row_max，避免复用缓冲区的竞态。
            local_sum = T.alloc_fragment((1,), T.float32)
            T.clear(local_sum)
            for j in T.serial(padded_cols // THREADS):
                local_sum[0] = local_sum[0] + T.exp(A[bx, tid + j * THREADS] - row_max[0])
            sums[tid] = local_sum[0]
            T.sync_threads()
            if tid == 0:
                total = T.alloc_fragment((1,), T.float32)
                T.clear(total)
                for i in T.serial(THREADS):
                    total[0] = total[0] + sums[i]
                row_total[0] = total[0]
            T.sync_threads()

            # 3. 每个线程写回自己负责的列；padding 对应 exp(-inf) = 0。
            for j in T.serial(padded_cols // THREADS):
                B[bx, tid + j * THREADS] = T.exp(A[bx, tid + j * THREADS] - row_max[0]) / row_total[0]

    return row_softmax


def _input(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite 2-D FP32 array")
    return x


def softmax_reference(x: np.ndarray) -> np.ndarray:
    """以实际 FP32 输入为起点的 FP64 CPU 稳定 Softmax 参考。"""
    values = _input(x).astype(np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _run(x: np.ndarray, *, softmax: bool) -> np.ndarray:
    x = _input(x)
    if not torch.backends.mps.is_available():
        raise RuntimeError("This lesson requires an Apple GPU with PyTorch MPS")
    rows, cols = x.shape
    padded_cols = ((cols + THREADS - 1) // THREADS) * THREADS
    host = np.full((rows, padded_cols), -np.inf if softmax else 0.0, dtype=np.float32)
    host[:, :cols] = x
    a = torch.from_numpy(host).to("mps")
    shape = (rows, padded_cols) if softmax else (rows,)
    b = torch.full(shape, float("nan"), dtype=torch.float32, device="mps")
    factory = make_softmax_kernel if softmax else make_sum_kernel
    factory(rows, padded_cols)(a, b)
    torch.mps.synchronize()
    result = b.cpu().numpy()
    return result[:, :cols].copy() if softmax else result.copy()


def run_sum(x: np.ndarray) -> np.ndarray:
    """逐行求和：[rows,cols] 有限输入 → [rows] CPU np.float32。"""
    return _run(x, softmax=False)


def run_softmax(x: np.ndarray) -> np.ndarray:
    """逐行稳定 Softmax：非空有限输入 → 同形状 CPU np.float32。"""
    return _run(x, softmax=True)


def main() -> None:
    print(f"TileLang {tilelang.__version__}; target=metal; FP32; manual reduction")
    for cols in (1, 7, 32, 33, 64):
        base = np.linspace(-3, 3, cols, dtype=np.float32)
        # 混合正负、全负、全零、大正偏移、大负偏移。
        x = np.stack((base, base - 5, np.zeros_like(base), base + 10000, base - 10000))
        summed = run_sum(x)
        probabilities = run_softmax(x)
        ref_sum = x.astype(np.float64).sum(axis=1)
        ref_softmax = softmax_reference(x)
        assert np.isfinite(summed).all() and np.isfinite(probabilities).all()
        np.testing.assert_allclose(summed, ref_sum, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(probabilities, ref_softmax, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(probabilities.sum(axis=1, dtype=np.float64), 1, atol=2e-6, rtol=0)
        assert (probabilities >= 0).all()
        print(f"cols={cols:2d}, 5 rows: sum OK max_abs={np.max(np.abs(summed - ref_sum)):.3g}; "
              f"softmax OK max_abs={np.max(np.abs(probabilities - ref_softmax)):.3g}")
    hand = np.array([[1, 2, 3]], dtype=np.float32)
    print("[1, 2, 3]: sum=", run_sum(hand), "softmax=", run_softmax(hand))


if __name__ == "__main__":
    main()
