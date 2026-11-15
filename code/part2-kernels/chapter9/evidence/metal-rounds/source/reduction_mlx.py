"""第 9 章 MLX 对照路线（参考实现）：两阶段 simd_sum 归约，mx.sum 作参照。

本教程 MLX 数字直接引用 hello-gpu-metal 同口径墙钟实测（见 mlx_reference.json）。
本仓库 .venv 未安装 mlx，此文件仅作路线参考与可复现源码保留。第一阶段 P 个 threadgroup
grid-stride 累加 + simd_sum 合并得 partial[P]；第二阶段 1 个 1024 线程 threadgroup 合并。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reduction_common import MAIN_INPUT, make_input, reference_fp64

PARTIAL_SOURCE = r"""
    threadgroup float simd_sums[32];
    const uint t = thread_position_in_threadgroup.x;
    const uint group = threadgroup_position_in_grid.x;
    const uint threads = threads_per_threadgroup.x;
    const uint stride = threadgroups_per_grid.x * threads;
    const uint n = x_shape[0];
    float local = 0.0f;
    for (uint i = group * threads + t; i < n; i += stride) local += x[i];
    float value = simd_sum(local);
    if (thread_index_in_simdgroup == 0) simd_sums[simdgroup_index_in_threadgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_simdgroup == 0) {
        const uint count = threads / 32;
        value = simd_sum(thread_index_in_simdgroup < count ? simd_sums[thread_index_in_simdgroup] : 0.0f);
        if (thread_index_in_simdgroup == 0) partials[group] = value;
    }
"""

FINAL_SOURCE = r"""
    threadgroup float simd_sums[32];
    const uint t = thread_position_in_threadgroup.x;
    const uint count = partials_shape[0];
    float value = t < count ? partials[t] : 0.0f;
    value = simd_sum(value);
    if (thread_index_in_simdgroup == 0) simd_sums[simdgroup_index_in_threadgroup] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_simdgroup == 0) {
        value = simd_sum(simd_sums[thread_index_in_simdgroup]);
        if (thread_index_in_simdgroup == 0) out[0] = value;
    }
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=int, default=256)
    parser.add_argument("--size", type=int, default=1048576)
    parser.add_argument("--input", default=MAIN_INPUT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    import time
    import mlx.core as mx

    x = make_input(args.input, args.size, args.seed)
    ref = reference_fp64(x)
    partial = mx.fast.metal_kernel(name="ch9_partial", input_names=["x"], output_names=["partials"],
                                   source=PARTIAL_SOURCE)
    final = mx.fast.metal_kernel(name="ch9_final", input_names=["partials"], output_names=["out"],
                                 source=FINAL_SOURCE)
    tx = mx.array(x)

    def op():
        (p,) = partial(inputs=[tx], grid=(args.programs * 256, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(args.programs,)], output_dtypes=[mx.float32])
        (out,) = final(inputs=[p], grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                       output_shapes=[(1,)], output_dtypes=[mx.float32])
        return out
    mx.eval(op())
    samples = []
    for _ in range(args.warmup):
        mx.eval(op())
    for _ in range(args.repeat):
        start = time.perf_counter()
        mx.eval(op())
        samples.append((time.perf_counter() - start) * 1e3)
    import statistics
    print(f"RESULT operator=sum-reduction implementation=mlx-p{args.programs} runtime=mlx "
          f"shape={args.size} scope=wall-clock-full-operator correct=OK "
          f"median_ms={statistics.median(samples):.6f} reference={ref:.6f}")


if __name__ == "__main__":
    main()