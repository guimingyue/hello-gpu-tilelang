"""第 11 章 TileLang 路线：分块矩阵乘 C=A@B（target="metal"，T.gemm 内建）。

T.gemm 在 Metal 后端可用（实测）；用 alloc_shared 暂存 A/B tile，fragment 累加 C tile，
T.copy 搬入/写回。tile 尺寸 BM=BN∈{16,32,64}，BK=32，threads=128。
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
from matmul_common import make_inputs, reference_matmul, error_bound, validate, tflops

BK = 32
THREADS = 128


def make_kernel(m: int, n: int, k: int, tile: int):
    @tilelang.jit(target="metal")
    def matmul(A, B, C):
        A: T.Tensor((m, k), T.float32)
        B: T.Tensor((k, n), T.float32)
        C: T.Tensor((m, n), T.float32)
        with T.Kernel(T.ceildiv(n, tile), T.ceildiv(m, tile), threads=THREADS) as (bx, by):
            As = T.alloc_shared((tile, BK), T.float32)
            Bs = T.alloc_shared((BK, tile), T.float32)
            Cl = T.alloc_fragment((tile, tile), T.float32)
            T.clear(Cl)
            for ko in T.serial(T.ceildiv(k, BK)):
                T.copy(A[by * tile, ko * BK], As)
                T.copy(B[ko * BK, bx * tile], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * tile, bx * tile])
    return matmul


def build_meta(args) -> dict:
    return {
        "config_id": args.config_id, "implementation": f"tilelang-tile{args.tile}",
        "runtime": "tilelang", "shape": f"{args.m}x{args.n}x{args.k}", "dtype": "float32",
        "threadgroup": THREADS, "block": args.tile, "threads": 0,
        "dispatch": "tile", "variant": f"tile{args.tile}", "scope": rounds.WALL_SCOPE,
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--config-id", default="tilelang-tile64")
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()

    a, b = make_inputs(args.m, args.n, args.k, args.seed)
    ref = reference_matmul(a, b)
    bound = error_bound(a, b)
    kernel = make_kernel(args.m, args.n, args.k, args.tile)
    ta, tb = to_mps(a), to_mps(b)
    tc = torch.full((args.m, args.n), float("nan"), dtype=torch.float32, device="mps")

    def op():
        kernel(ta, tb, tc)
        torch.mps.synchronize()

    op()
    pre_ok, pre_err, pre_ratio = validate(readback(tc), ref, bound)
    tc.fill_(float("nan"))
    samples = wall_samples(op, args.warmup, args.repeat) if args.repeat > 0 else []
    post_ok, post_err, post_ratio = validate(readback(tc), ref, bound)

    meta = build_meta(args)
    print("ENV " + " ".join(f"{k}={v}" for k, v in environment().items()))
    correct = pre_ok and post_ok
    fields = {"operator": "matmul", **meta, "process": args.process,
              "correct": "OK" if correct else "FAIL",
              "precheck": "OK" if pre_ok else "FAIL", "postcheck": "OK" if post_ok else "FAIL",
              "max_abs_error": f"{max(pre_err, post_err):.3g}",
              "max_error_ratio": f"{max(pre_ratio, post_ratio):.3g}", "timed": int(bool(samples))}
    if samples:
        import statistics
        med = statistics.median(samples)
        fields.update(median_ms=f"{med:.6f}", min_ms=f"{min(samples):.6f}",
                      max_ms=f"{max(samples):.6f}", tflops=f"{tflops(meta['shape'], med):.3f}")
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