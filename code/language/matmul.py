"""第 8 课：FP32 分块矩阵乘，16×16 输出 tile，BK=32，128 线程。

验证：2026-11-19，Apple M1，TileLang 0.1.15，target="metal"。
从仓库根运行：.venv/bin/python code/language/matmul.py
"""
from __future__ import annotations

import numpy as np
import torch
import tilelang
import tilelang.language as T

TILE = 16
BK = 32
THREADS = 128


def make_matmul_kernel(m: int, n: int, k: int):
    """返回 kernel(A, B, C)；FP32 MPS 输入尺寸须为 16/16/32 的正倍数。"""
    if min(m, n, k) < 1 or m % TILE or n % TILE or k % BK:
        raise ValueError("m/n must be positive multiples of 16; k of 32")

    @tilelang.jit(target="metal")
    def tiled_matmul(A, B, C):
        A: T.Tensor((m, k), T.float32)
        B: T.Tensor((k, n), T.float32)
        C: T.Tensor((m, n), T.float32)
        with T.Kernel(n // TILE, m // TILE, threads=THREADS) as (bx, by):
            As = T.alloc_shared((TILE, BK), T.float32)
            Bs = T.alloc_shared((BK, TILE), T.float32)
            Cl = T.alloc_fragment((TILE, TILE), T.float32)
            T.clear(Cl)
            for ko in T.serial(k // BK):
                T.copy(A[by * TILE, ko * BK], As)
                T.copy(B[ko * BK, bx * TILE], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * TILE, bx * TILE])

    return tiled_matmul


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """有限二维 a[M,K] @ b[K,N] → CPU np.float32[M,N]；主机补零并裁剪。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or min(*a.shape, *b.shape) < 1:
        raise ValueError("a and b must be nonempty 2-D arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("a and b must contain finite FP32 values")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This lesson requires an Apple GPU with PyTorch MPS")
    m, k = a.shape
    n = b.shape[1]
    mp = ((m + TILE - 1) // TILE) * TILE
    npad = ((n + TILE - 1) // TILE) * TILE
    kp = ((k + BK - 1) // BK) * BK
    ap = np.zeros((mp, kp), dtype=np.float32)
    bp = np.zeros((kp, npad), dtype=np.float32)
    ap[:m, :k] = a
    bp[:k, :n] = b
    ta = torch.from_numpy(ap).to("mps")
    tb = torch.from_numpy(bp).to("mps")
    tc = torch.full((mp, npad), float("nan"), dtype=torch.float32, device="mps")
    make_matmul_kernel(mp, npad, kp)(ta, tb, tc)
    torch.mps.synchronize()
    return tc.cpu().numpy()[:m, :n].copy()


def main() -> None:
    print(f"TileLang {tilelang.__version__}; target=metal; FP32; TILE={TILE}, BK={BK}, threads={THREADS}")
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    result = matmul(a, b)
    np.testing.assert_array_equal(result, np.array([[19, 22], [43, 50]], dtype=np.float32))
    print("2x2 hand calculation: OK\n", result)
    rng = np.random.default_rng(8)
    for m, n, k in ((1, 1, 1), (16, 16, 32), (3, 5, 7), (17, 19, 33), (31, 9, 65)):
        a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
        b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
        actual = matmul(a, b)
        reference = a.astype(np.float64) @ b.astype(np.float64)
        assert actual.shape == (m, n) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        print(f"M={m:2d}, N={n:2d}, K={k:2d}: OK max_abs={np.max(np.abs(actual - reference)):.3g}")
    zero = matmul(np.zeros((2, 3), dtype=np.float32), np.ones((3, 5), dtype=np.float32))
    np.testing.assert_array_equal(zero, np.zeros((2, 5), dtype=np.float32))
    print("zero matrix: OK")


if __name__ == "__main__":
    main()
