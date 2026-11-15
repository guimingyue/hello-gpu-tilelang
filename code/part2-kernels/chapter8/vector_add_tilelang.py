"""第 8 章 TileLang 路线：逐元素向量加法（target="metal"）+ 主机校验/墙钟计时。

kernel 用 tile-level DSL 表达：一个 T.Kernel 网格，每 threadgroup ``block`` 个线程，
每线程处理 ``elems`` 个元素（同一轮相邻线程访问相邻地址，与 Metal 版 MLX m1 同映射）。

计时口径：端到端墙钟（含一次完整 kernel 调用 + torch.mps.synchronize）。TileLang Metal
后端不暴露 GPU 时间戳，故 scope=wall-clock-full-operator，与 MLX 墙钟同口径。
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
from common.tilelang_metal import environment, to_mps, readback, wall_samples, finite_abs_check
from vector_add_common import TOLERANCE, OPERATOR, make_inputs, bandwidth_gbs


def make_kernel(n: int, block: int, elems: int):
    @tilelang.jit(target="metal")
    def vadd(A, B, C):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        C: T.Tensor((n,), T.float32)
        with T.Kernel(T.ceildiv(n, block * elems), threads=block) as bx:
            tid = T.get_thread_binding()
            for j in T.serial(elems):
                idx = bx * block * elems + j * block + tid
                if idx < n:
                    C[idx] = A[idx] + B[idx]
    return vadd


def build_meta(args, block: int, elems: int) -> dict:
    groups = (args.size + block * elems - 1) // (block * elems)
    return {
        "config_id": args.config_id, "implementation": f"tilelang-{args.version}",
        "runtime": "tilelang", "shape": str(args.size), "dtype": "float32",
        "threadgroup": block, "block": elems, "threads": groups * block,
        "dispatch": "threads", "variant": f"b{block}e{elems}",
        "scope": rounds.WALL_SCOPE, "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v0", choices=("v0", "v1"))
    parser.add_argument("--size", type=int, default=16777216)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--elements", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--config-id", default="tilelang-v0")
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()

    a, b, ref = make_inputs(args.size, args.seed)
    kernel = make_kernel(args.size, args.block, args.elements)
    ta, tb = to_mps(a), to_mps(b)
    tc = torch.full((args.size,), float("nan"), dtype=torch.float32, device="mps")

    def op():
        kernel(ta, tb, tc)
        torch.mps.synchronize()

    op()
    pre_ok, pre_err = finite_abs_check(readback(tc), ref, TOLERANCE)
    tc.fill_(float("nan"))
    samples = wall_samples(op, args.warmup, args.repeat) if args.repeat > 0 else []
    post_ok, post_err = finite_abs_check(readback(tc), ref, TOLERANCE)

    meta = build_meta(args, args.block, args.elements)
    env = environment()
    print("ENV " + " ".join(f"{k}={v}" for k, v in env.items()))
    correct = pre_ok and post_ok
    fields = {"operator": OPERATOR, **meta, "process": args.process,
              "correct": "OK" if correct else "FAIL",
              "precheck": "OK" if pre_ok else "FAIL", "postcheck": "OK" if post_ok else "FAIL",
              "max_abs_error": f"{max(pre_err, post_err):.3g}", "timed": int(bool(samples))}
    if samples:
        import statistics
        med = statistics.median(samples)
        fields.update(median_ms=f"{med:.6f}", min_ms=f"{min(samples):.6f}",
                      max_ms=f"{max(samples):.6f}",
                      effective_bandwidth_gbs=f"{bandwidth_gbs(str(args.size), med):.2f}")
    print("RESULT " + " ".join(f"{k}={v}" for k, v in fields.items()))
    if not correct:
        raise SystemExit("correctness check failed")

    if args.samples is not None and samples:
        rounds_write_samples(args.samples, meta, args.process, samples)


def rounds_write_samples(path: Path, meta: dict, process: int, samples: list[float]) -> None:
    import csv
    if path.exists():
        raise FileExistsError(f"refusing to overwrite samples: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [*rounds.GROUP_KEYS, "process", "sample", "ms"]
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, value in enumerate(samples):
            writer.writerow({**{k: meta[k] for k in rounds.GROUP_KEYS},
                             "process": process, "sample": index, "ms": f"{value:.6f}"})


if __name__ == "__main__":
    main()