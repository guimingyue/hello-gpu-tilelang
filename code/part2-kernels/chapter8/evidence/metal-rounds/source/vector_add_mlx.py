"""第 8 章 MLX 对照路线（参考实现）：内置 a+b 与 mx.fast.metal_kernel 自定义 tile。

说明
----
本教程的 MLX 对照数字**直接引用 hello-gpu-metal 的同口径（端到端墙钟）实测**，
见 ``mlx_reference.json`` 与 ``evidence/metal-rounds/summary/summary.csv`` 中
``provenance=cited-hello-gpu-metal`` 的行；本仓库 ``.venv`` 未安装 mlx，此文件仅作
路线参考与可复现源码保留（安装 mlx 后可运行）。MLX 惰性求值，``mx.eval`` 时才提交 GPU，
计时为墙钟，与 TileLang 墙钟同口径，但与 Metal 版 MSL 的 GPU 时间戳不同口径。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vector_add_common import TOLERANCE, make_inputs

# 第 group 个 threadgroup 负责 [group*E*tg, (group+1)*E*tg)；第 j 轮相邻线程读相邻地址。
M1_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    uint tg = threads_per_threadgroup.x;
    uint n = uint(a_shape[0]);
    for (uint j = 0; j < E; ++j) {
        uint i = (group * E + j) * tg + tid;
        float x = i < n ? a[i] : 0.0f;
        float y = i < n ? b[i] : 0.0f;
        if (i < n) { c[i] = x + y; }
    }
"""


def check(output: np.ndarray, reference: np.ndarray) -> tuple[bool, float]:
    if output.shape != reference.shape or not np.isfinite(output).all():
        return False, float("inf")
    error = float(np.max(np.abs(output - reference)))
    return error <= TOLERANCE, error


def run_mlx(args, host_a, host_b, reference):
    import mlx.core as mx

    a, b = mx.array(host_a), mx.array(host_b)
    mx.eval(a, b)
    if args.version == "m0":
        def op():
            return a + b
    else:
        kernel = mx.fast.metal_kernel(name="vector_add_tile", input_names=["a", "b"],
                                      output_names=["c"], source=M1_SOURCE)
        groups = -(-args.size // (args.threadgroup * args.elements))

        def op():
            (out,) = kernel(inputs=[a, b], template=[("E", args.elements)],
                            grid=(groups * args.threadgroup, 1, 1),
                            threadgroup=(args.threadgroup, 1, 1),
                            output_shapes=[a.shape], output_dtypes=[a.dtype])
            return out
    out = op()
    mx.eval(out)
    ok, error = check(np.array(out), reference)
    if not ok:
        raise RuntimeError(f"mlx precheck failed: {error}")
    import time
    samples = []
    for _ in range(args.warmup):
        mx.eval(op())
    for _ in range(args.repeat):
        start = time.perf_counter()
        mx.eval(op())
        samples.append((time.perf_counter() - start) * 1e3)
    return samples, error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("m0", "m1"), default="m0")
    parser.add_argument("--size", type=int, default=16777216)
    parser.add_argument("--elements", type=int, default=1)
    parser.add_argument("--threadgroup", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    host_a, host_b, reference = make_inputs(args.size, args.seed)
    samples, error = run_mlx(args, host_a, host_b, reference)
    import statistics
    print(f"RESULT operator=vector-add implementation=mlx-{args.version} runtime=mlx "
          f"shape={args.size} scope=wall-clock-full-operator correct=OK "
          f"median_ms={statistics.median(samples):.6f} max_abs_error={error:.3g}")


if __name__ == "__main__":
    main()