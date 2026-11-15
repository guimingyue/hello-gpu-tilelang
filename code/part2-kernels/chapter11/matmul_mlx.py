"""第 11 章 MLX 对照路线（参考实现）：mx.matmul / mx.compile / metal_kernel 分块。

本教程 MLX 数字直接引用 hello-gpu-metal 同口径墙钟实测（见 mlx_reference.json）。
本仓库 .venv 未安装 mlx，此文件仅作路线参考与可复现源码保留。
metal_kernel 每个 threadgroup 计算一个 T×T 的 C tile，沿 K 累加。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from matmul_common import make_inputs, reference_matmul, error_bound, validate

KERNEL_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.y * T + thread_position_in_threadgroup.y;
    const uint col = threadgroup_position_in_grid.x * T + thread_position_in_threadgroup.x;
    const uint M = a_shape[0], K = a_shape[1], N = b_shape[1];
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint kk = 0; kk < K; ++kk) acc += a[row * K + kk] * b[kk * N + col];
    c[row * N + col] = acc;
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    import time
    import mlx.core as mx

    a, b = make_inputs(args.m, args.n, args.k, args.seed)
    ref = reference_matmul(a, b)
    bound = error_bound(a, b)
    kernel = mx.fast.metal_kernel(name="ch11_matmul", input_names=["a", "b"], output_names=["c"],
                                  source=KERNEL_SOURCE)
    ta, tb = mx.array(a), mx.array(b)
    gx = (args.n + args.tile - 1) // args.tile
    gy = (args.m + args.tile - 1) // args.tile

    def op():
        (c,) = kernel(inputs=[ta, tb], template=[("T", args.tile)],
                      grid=(gx * args.tile, gy * args.tile, 1),
                      threadgroup=(args.tile, args.tile, 1),
                      output_shapes=[(args.m, args.n)], output_dtypes=[mx.float32])
        return c
    out = op()
    mx.eval(out)
    ok, err, ratio = validate(np.array(out), ref, bound)
    if not ok:
        raise RuntimeError(f"mlx matmul check failed: ratio={ratio}")
    samples = []
    for _ in range(args.warmup):
        mx.eval(op())
    for _ in range(args.repeat):
        start = time.perf_counter()
        mx.eval(op())
        samples.append((time.perf_counter() - start) * 1e3)
    import statistics
    print(f"RESULT operator=matmul implementation=mlx-kernel-t{args.tile} runtime=mlx "
          f"shape={args.m}x{args.n}x{args.k} scope=wall-clock-full-operator correct=OK "
          f"median_ms={statistics.median(samples):.6f} max_error_ratio={ratio:.3g}")


if __name__ == "__main__":
    main()