"""第 13 章 MLX 对照路线（参考实现）：算子组合 / mx.compile / metal_kernel / mx.fast.rms_norm。

本教程 MLX 数字直接引用 hello-gpu-metal 同口径墙钟实测（见 mlx_reference.json）。
本仓库 .venv 未安装 mlx，此文件仅作路线参考与可复现源码保留。
metal_kernel 一个 threadgroup 处理一行，simd_sum 求平方和，rsqrt 缩放乘权重。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rmsnorm_common import make_inputs, reference_rmsnorm, validate, EPSILON

KERNEL_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_position_in_threadgroup.x;
    const uint cols = x_shape[1];
    const uint T = threads_per_threadgroup.x;
    float ss = 0.0f;
    for (uint j = lane; j < cols; j += T) { float xv = x[row * cols + j]; ss += xv * xv; }
    ss = simd_sum(ss);
    threadgroup float red[32];
    if (thread_index_in_simdgroup == 0) red[simdgroup_index_in_threadgroup] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        ss = simd_sum(thread_index_in_simdgroup < T/32 ? red[thread_index_in_simdgroup] : 0.0f);
        if (thread_index_in_simdgroup == 0) red[0] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(red[0] / (float)cols + eps);
    for (uint j = lane; j < cols; j += T) y[row * cols + j] = x[row * cols + j] * inv * w[j];
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    import time
    import mlx.core as mx

    x, w = make_inputs(args.rows, args.cols, args.seed)
    ref = reference_rmsnorm(x, w, EPSILON)
    kernel = mx.fast.metal_kernel(name="ch13_rmsnorm", input_names=["x", "w"], output_names=["y"],
                                  source=KERNEL_SOURCE)
    tx, tw = mx.array(x), mx.array(w)

    def op():
        (y,) = kernel(inputs=[tx, tw], template=[("eps", EPSILON)],
                      grid=(args.rows * args.threads, 1, 1), threadgroup=(args.threads, 1, 1),
                      output_shapes=[(args.rows, args.cols)], output_dtypes=[mx.float32])
        return y
    out = op()
    mx.eval(out)
    ok, abs_err, rel_err = validate(np.array(out), ref)
    if not ok:
        raise RuntimeError(f"mlx rmsnorm check failed: {abs_err}")
    samples = []
    for _ in range(args.warmup):
        mx.eval(op())
    for _ in range(args.repeat):
        start = time.perf_counter()
        mx.eval(op())
        samples.append((time.perf_counter() - start) * 1e3)
    import statistics
    print(f"RESULT operator=rmsnorm implementation=mlx-kernel runtime=mlx "
          f"shape={args.rows}x{args.cols} scope=wall-clock-full-operator correct=OK "
          f"median_ms={statistics.median(samples):.6f} max_abs_error={abs_err:.3g}")


if __name__ == "__main__":
    main()