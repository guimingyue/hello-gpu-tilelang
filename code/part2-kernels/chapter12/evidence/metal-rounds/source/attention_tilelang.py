"""第 12 章 TileLang 路线：教学版 Attention 前向（target="metal"）。

一个 threadgroup 负责一个 query 行，threads=dim（每线程一个 dim 分量）。
Q/K/V 打包进张量 A[3,seq,dim]（TileLang eager Metal 仅 A/B 名读取可靠），输出 B[seq,dim]。
- online：逐 key 在线更新 (m, l, acc)，免物化 scores；
- materialized：先算整行 scores 并 softmax，再加权求和（物化中间数组）。
score 的点积用「每线程乘积 + thread0 serial 合并」（Metal 无 tl.reduce/simd_sum）。
计时为端到端墙钟，与 MLX 墙钟同口径。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import tilelang
import tilelang.language as T

from common import rounds
from common.tilelang_metal import environment, to_mps, readback, wall_samples
from attention_common import make_input, reference_attention, validate


def make_online(seq: int, dim: int):
    inv_sqrt = 1.0 / math.sqrt(dim)

    @tilelang.jit(target="metal")
    def attn(A, B):
        A: T.Tensor((3, seq, dim), T.float32)
        B: T.Tensor((seq, dim), T.float32)
        with T.Kernel(seq, threads=dim) as bx:
            tid = T.get_thread_binding()
            qf = T.alloc_fragment((1,), T.float32)
            qf[0] = A[0, bx, tid]
            m = T.alloc_fragment((1,), T.float32)
            m[0] = -1e30
            l = T.alloc_fragment((1,), T.float32)
            T.clear(l)
            acc = T.alloc_fragment((1,), T.float32)
            T.clear(acc)
            s = T.alloc_shared((dim,), T.float32)
            for j in T.serial(seq):
                s[tid] = qf[0] * A[1, j, tid]
                T.sync_threads()
                if tid == 0:
                    a = T.alloc_fragment((1,), T.float32)
                    T.clear(a)
                    for d in T.serial(dim):
                        a[0] = a[0] + s[d]
                    s[0] = a[0]
                T.sync_threads()
                score = s[0] * inv_sqrt
                m_new = T.max(m[0], score)
                corr = T.exp(m[0] - m_new)
                p = T.exp(score - m_new)
                l[0] = l[0] * corr + p
                acc[0] = acc[0] * corr + p * A[2, j, tid]
                m[0] = m_new
            B[bx, tid] = acc[0] / l[0]
    return attn


def make_materialized(seq: int, dim: int):
    inv_sqrt = 1.0 / math.sqrt(dim)

    @tilelang.jit(target="metal")
    def attn(A, B):
        A: T.Tensor((3, seq, dim), T.float32)
        B: T.Tensor((seq, dim), T.float32)
        with T.Kernel(seq, threads=dim) as bx:
            tid = T.get_thread_binding()
            qf = T.alloc_fragment((1,), T.float32)
            qf[0] = A[0, bx, tid]
            scores = T.alloc_shared((seq,), T.float32)
            s = T.alloc_shared((dim,), T.float32)
            mb = T.alloc_shared((1,), T.float32)
            # 1) 整行 scores
            for j in T.serial(seq):
                s[tid] = qf[0] * A[1, j, tid]
                T.sync_threads()
                if tid == 0:
                    a = T.alloc_fragment((1,), T.float32)
                    T.clear(a)
                    for d in T.serial(dim):
                        a[0] = a[0] + s[d]
                    scores[j] = a[0] * inv_sqrt
                T.sync_threads()
            # 2) 行最大值（用独立 shared 单元广播，避免覆盖 scores[0]）
            if tid == 0:
                mx = T.alloc_fragment((1,), T.float32)
                mx[0] = -1e30
                for j in T.serial(seq):
                    mx[0] = T.max(mx[0], scores[j])
                mb[0] = mx[0]
            T.sync_threads()
            mx = mb[0]
            # 3) exp 与和（每个线程负责一段 key，再 thread0 合并）
            lsum = T.alloc_fragment((1,), T.float32)
            T.clear(lsum)
            for j in T.serial(T.ceildiv(seq, dim)):
                if tid + j * dim < seq:
                    scores[tid + j * dim] = T.exp(scores[tid + j * dim] - mx)
                    lsum[0] = lsum[0] + scores[tid + j * dim]
            s[tid] = lsum[0]
            T.sync_threads()
            if tid == 0:
                a = T.alloc_fragment((1,), T.float32)
                T.clear(a)
                for d in T.serial(dim):
                    a[0] = a[0] + s[d]
                s[0] = a[0]
            T.sync_threads()
            inv = 1.0 / s[0]
            # 4) 加权求和
            acc = T.alloc_fragment((1,), T.float32)
            T.clear(acc)
            for j in T.serial(seq):
                acc[0] = acc[0] + scores[j] * inv * A[2, j, tid]
            B[bx, tid] = acc[0]
    return attn


KERNELS = {"online": make_online, "materialized": make_materialized}


def build_meta(args) -> dict:
    return {
        "config_id": args.config_id, "implementation": f"tilelang-{args.version}",
        "runtime": "tilelang", "shape": f"{args.seq}x{args.dim}", "dtype": "float32",
        "threadgroup": args.dim, "block": 0, "threads": args.seq * args.dim,
        "dispatch": "tile", "variant": args.version, "scope": rounds.WALL_SCOPE,
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("online", "materialized"), default="online")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--config-id", default="tilelang-online")
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()

    q, k, v = make_input(args.seq, args.dim, args.seed)
    ref = reference_attention(q, k, v)
    packed = np.stack([q, k, v], axis=0)
    kernel = KERNELS[args.version](args.seq, args.dim)
    ta = to_mps(packed)
    tb = torch.full((args.seq, args.dim), float("nan"), dtype=torch.float32, device="mps")

    def op():
        kernel(ta, tb)
        torch.mps.synchronize()

    op()
    pre_ok, pre_err, pre_ratio = validate(readback(tb), ref)
    tb.fill_(float("nan"))
    samples = wall_samples(op, args.warmup, args.repeat) if args.repeat > 0 else []
    post_ok, post_err, post_ratio = validate(readback(tb), ref)

    meta = build_meta(args)
    print("ENV " + " ".join(f"{k}={v}" for k, v in environment().items()))
    correct = pre_ok and post_ok
    fields = {"operator": "attention", **meta, "process": args.process,
              "correct": "OK" if correct else "FAIL",
              "precheck": "OK" if pre_ok else "FAIL", "postcheck": "OK" if post_ok else "FAIL",
              "max_abs_error": f"{max(pre_err, post_err):.3g}",
              "max_tolerance_ratio": f"{max(pre_ratio, post_ratio):.3g}", "timed": int(bool(samples))}
    if samples:
        import statistics
        med = statistics.median(samples)
        fields.update(median_ms=f"{med:.6f}", min_ms=f"{min(samples):.6f}", max_ms=f"{max(samples):.6f}")
    print("RESULT " + " ".join(f"{k}={v}" for k, v in fields.items()))
    if not correct:
        raise SystemExit("correctness check failed")
    if args.samples is not None and samples:
        import csv
        if args.samples.exists():
            raise FileExistsError(f"refusing to overwrite: {args.samples}")
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        columns = [*rounds.GROUP_KEYS, "process", "sample", "ms"]
        with args.samples.open("x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for index, value in enumerate(samples):
                writer.writerow({**{k: meta[k] for k in rounds.GROUP_KEYS},
                                 "process": args.process, "sample": index, "ms": f"{value:.6f}"})


if __name__ == "__main__":
    main()