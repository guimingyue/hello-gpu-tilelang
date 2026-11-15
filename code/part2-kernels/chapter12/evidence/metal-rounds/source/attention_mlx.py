"""第 12 章 MLX 对照路线（参考实现）：物化算子组合 / mx.compile / 在线 metal_kernel。

本教程 MLX 数字直接引用 hello-gpu-metal 同口径墙钟实测（见 mlx_reference.json）。
本仓库 .venv 未安装 mlx，此文件仅作路线参考与可复现源码保留。
在线 metal_kernel 每个 threadgroup 处理一个 query 行，逐 key tile 更新 (m, l, acc)。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from attention_common import make_input, reference_attention, validate

ONLINE_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint d = thread_position_in_threadgroup.x;
    const uint seq = q_shape[0], dim = q_shape[1];
    const float inv_sqrt = 1.0f / sqrt((float)dim);
    float qd = q[row * dim + d];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (uint j = 0; j < seq; ++j) {
        float score = 0.0f;
        for (uint dd = 0; dd < dim; ++dd) score += q[row * dim + dd] * k[j * dim + dd];
        score *= inv_sqrt;
        float m_new = max(m, score);
        float corr = exp(m - m_new);
        float p = exp(score - m_new);
        l = l * corr + p;
        acc = acc * corr + p * v[j * dim + d];
        m = m_new;
    }
    out[row * dim + d] = acc / l;
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    import time
    import mlx.core as mx

    q, k, v = make_input(args.seq, args.dim, args.seed)
    ref = reference_attention(q, k, v)
    kernel = mx.fast.metal_kernel(name="ch12_online", input_names=["q", "k", "v"],
                                  output_names=["out"], source=ONLINE_SOURCE)
    tq, tk, tv = mx.array(q), mx.array(k), mx.array(v)

    def op():
        (out,) = kernel(inputs=[tq, tk, tv], grid=(args.seq * args.dim, 1, 1),
                        threadgroup=(args.dim, 1, 1),
                        output_shapes=[(args.seq, args.dim)], output_dtypes=[mx.float32])
        return out
    out = op()
    mx.eval(out)
    ok, err, ratio = validate(np.array(out), ref)
    if not ok:
        raise RuntimeError(f"mlx attention check failed: ratio={ratio}")
    samples = []
    for _ in range(args.warmup):
        mx.eval(op())
    for _ in range(args.repeat):
        start = time.perf_counter()
        mx.eval(op())
        samples.append((time.perf_counter() - start) * 1e3)
    import statistics
    print(f"RESULT operator=attention implementation=mlx-online runtime=mlx "
          f"shape={args.seq}x{args.dim} scope=wall-clock-full-operator correct=OK "
          f"median_ms={statistics.median(samples):.6f} max_tolerance_ratio={ratio:.3g}")


if __name__ == "__main__":
    main()