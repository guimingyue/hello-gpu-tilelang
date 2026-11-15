"""第 10 章 MLX 对照路线（参考实现）：算子组合 / mx.compile / metal_kernel / 内置 softmax。

本教程 MLX 数字直接引用 hello-gpu-metal 同口径墙钟实测（见 mlx_reference.json）。
本仓库 .venv 未安装 mlx，此文件仅作路线参考与可复现源码保留。
metal_kernel 让一个 threadgroup 处理 W 行、每行 T 线程，行内用 simd_max/simd_sum 归约。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softmax_common import make_input, reference_softmax, validate

KERNEL_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x * W + threadgroup_index_in_threadgroup.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint cols = x_shape[1];
    if (row >= x_shape[0]) return;
    float m = -INFINITY;
    for (uint j = lane; j < cols; j += T) m = max(m, x[row * cols + j]);
    m = simd_max(m);
    threadgroup float smax[32];
    if (thread_index_in_simdgroup == 0) smax[simdgroup_index_in_threadgroup] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        m = simd_max(thread_index_in_simdgroup < T/32 ? smax[thread_index_in_simdgroup] : -INFINITY);
        if (thread_index_in_simdgroup == 0) smax[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = smax[0];
    float s = 0.0f;
    for (uint j = lane; j < cols; j += T) s += exp(x[row * cols + j] - m);
    s = simd_sum(s);
    threadgroup float ssum[32];
    if (thread_index_in_simdgroup == 0) ssum[simdgroup_index_in_threadgroup] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        s = simd_sum(thread_index_in_simdgroup < T/32 ? ssum[thread_index_in_simdgroup] : 0.0f);
        if (thread_index_in_simdgroup == 0) ssum[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / ssum[0];
    for (uint j = lane; j < cols; j += T) y[row * cols + j] = exp(x[row * cols + j] - m) * inv;
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    import time
    import mlx.core as mx

    x = make_input(args.rows, args.cols, args.seed)
    ref = reference_softmax(x)
    kernel = mx.fast.metal_kernel(name="ch10_softmax", input_names=["x"], output_names=["y"],
                                  source=KERNEL_SOURCE)
    tx = mx.array(x)
    groups = (args.rows + args.width - 1) // args.width

    def op():
        (y,) = kernel(inputs=[tx], template=[("W", args.width), ("T", args.threads)],
                      grid=(groups * args.threads, 1, 1), threadgroup=(args.threads, 1, 1),
                      output_shapes=[(args.rows, args.cols)], output_dtypes=[mx.float32])
        return y
    out = op()
    mx.eval(out)
    ok, abs_err, row_err = validate(np.array(out), ref)
    if not ok:
        raise RuntimeError(f"mlx softmax check failed: {abs_err} {row_err}")
    samples = []
    for _ in range(args.warmup):
        mx.eval(op())
    for _ in range(args.repeat):
        start = time.perf_counter()
        mx.eval(op())
        samples.append((time.perf_counter() - start) * 1e3)
    import statistics
    print(f"RESULT operator=softmax implementation=mlx-kernel-w{args.width}-t{args.threads} "
          f"runtime=mlx shape={args.rows}x{args.cols} scope=wall-clock-full-operator correct=OK "
          f"median_ms={statistics.median(samples):.6f} max_abs_error={abs_err:.3g}")


if __name__ == "__main__":
    main()