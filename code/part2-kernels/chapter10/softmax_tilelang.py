"""第 10 章 TileLang 路线：行级稳定 Softmax（target="metal"）。

一个 threadgroup 负责一行，threads=T。三次行内归约（max、exp 和）都用
「每线程 serial 扫描 + thread0 serial 合并」（Metal 无 tl.reduce/simd_sum）。
张量参数命名 A/B（TileLang eager Metal 后端对其它名字的读取存在缺陷）。
计时为端到端墙钟，与 MLX 墙钟同口径。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import tilelang
import tilelang.language as T

from common import rounds
from common.tilelang_metal import environment, to_mps, readback, wall_samples
from softmax_common import make_input, reference_softmax, validate, bandwidth_gbs


def make_kernel(rows: int, cols: int, threads: int):
    @tilelang.jit(target="metal")
    def softmax(A, B):
        A: T.Tensor((rows, cols), T.float32)
        B: T.Tensor((rows, cols), T.float32)
        with T.Kernel(rows, threads=threads) as bx:
            tid = T.get_thread_binding()
            # 1) 行最大值
            lmax = T.alloc_fragment((1,), T.float32)
            lmax[0] = -1e30
            for j in T.serial(T.ceildiv(cols, threads)):
                if tid + j * threads < cols:
                    lmax[0] = T.max(lmax[0], A[bx, tid + j * threads])
            s = T.alloc_shared((threads,), T.float32)
            s[tid] = lmax[0]
            T.sync_threads()
            if tid == 0:
                acc = T.alloc_fragment((1,), T.float32)
                acc[0] = -1e30
                for i in T.serial(threads):
                    acc[0] = T.max(acc[0], s[i])
                s[0] = acc[0]
            T.sync_threads()
            mx = s[0]
            # 2) exp 和
            lsum = T.alloc_fragment((1,), T.float32)
            T.clear(lsum)
            for j in T.serial(T.ceildiv(cols, threads)):
                if tid + j * threads < cols:
                    lsum[0] = lsum[0] + T.exp(A[bx, tid + j * threads] - mx)
            s[tid] = lsum[0]
            T.sync_threads()
            if tid == 0:
                acc = T.alloc_fragment((1,), T.float32)
                T.clear(acc)
                for i in T.serial(threads):
                    acc[0] = acc[0] + s[i]
                s[0] = acc[0]
            T.sync_threads()
            inv = 1.0 / s[0]
            # 3) 归一化写回
            for j in T.serial(T.ceildiv(cols, threads)):
                if tid + j * threads < cols:
                    B[bx, tid + j * threads] = T.exp(A[bx, tid + j * threads] - mx) * inv
    return softmax


def build_meta(args) -> dict:
    return {
        "config_id": args.config_id, "implementation": f"tilelang-t{args.threads}",
        "runtime": "tilelang", "shape": f"{args.rows}x{args.cols}", "dtype": "float32",
        "threadgroup": args.threads, "block": 1, "threads": args.rows * args.threads,
        "dispatch": "tile", "variant": f"t{args.threads}", "scope": rounds.WALL_SCOPE,
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--config-id", default="tilelang-t256")
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()

    x = make_input(args.rows, args.cols, args.seed)
    ref = reference_softmax(x)
    kernel = make_kernel(args.rows, args.cols, args.threads)
    ta = to_mps(x)
    tb = torch.full((args.rows, args.cols), float("nan"), dtype=torch.float32, device="mps")

    def op():
        kernel(ta, tb)
        torch.mps.synchronize()

    op()
    pre_ok, pre_abs, pre_row = validate(readback(tb), ref)
    tb.fill_(float("nan"))
    samples = wall_samples(op, args.warmup, args.repeat) if args.repeat > 0 else []
    post_ok, post_abs, post_row = validate(readback(tb), ref)

    meta = build_meta(args)
    print("ENV " + " ".join(f"{k}={v}" for k, v in environment().items()))
    correct = pre_ok and post_ok
    fields = {"operator": "softmax", **meta, "process": args.process,
              "correct": "OK" if correct else "FAIL",
              "precheck": "OK" if pre_ok else "FAIL", "postcheck": "OK" if post_ok else "FAIL",
              "max_abs_error": f"{max(pre_abs, post_abs):.3g}",
              "max_row_sum_error": f"{max(pre_row, post_row):.3g}", "timed": int(bool(samples))}
    if samples:
        import statistics
        med = statistics.median(samples)
        fields.update(median_ms=f"{med:.6f}", min_ms=f"{min(samples):.6f}",
                      max_ms=f"{max(samples):.6f}",
                      effective_bandwidth_gbs=f"{bandwidth_gbs(meta['shape'], med):.2f}")
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