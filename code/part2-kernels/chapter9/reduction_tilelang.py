"""第 9 章 TileLang 路线：两阶段 grid-stride 求和归约（target="metal"）。

Metal 后端限制（实测）：
- ``T.reduce_sum`` 未注册 metal 实现，改用「每线程 serial 累加 + thread0 serial 合并」；
- float32 ``T.atomic_add`` 不支持（仅 int32/uint32），故采用 exclusive partial 两阶段，
  与 Metal 版 MLX 的 partial+final 同结构；
- 并行跨步 shared 树形归约触发 LayoutInference 缺陷，本实现回避之。

第一阶段 P 个 threadgroup（各 256 线程）grid-stride 累加得到 partial[P]；
第二阶段 1 个 1024 线程 threadgroup 合并 partial（不足补 0）得到标量。
计时为端到端墙钟（两次 kernel 调用 + 同步），与 MLX 墙钟同口径。
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
from reduction_common import MAIN_INPUT, make_input, reference_fp64, check, bandwidth_gbs

THREADS = 256
FINAL_THREADS = 1024


def make_partial(n: int, programs: int, threads: int = THREADS):
    # n 已补零到 stride 的整数倍；无条件读取（serial 循环内 guard 会触发 Metal
    # LayoutInference 缺陷），越界补零位置贡献 0。索引内联进张量访问。
    stride = programs * threads

    @tilelang.jit(target="metal")
    def partial(A, partials):
        A: T.Tensor((n,), T.float32)
        partials: T.Tensor((programs,), T.float32)
        with T.Kernel(programs, threads=threads) as bx:
            tid = T.get_thread_binding()
            local = T.alloc_fragment((1,), T.float32)
            T.clear(local)
            for j in T.serial(n // stride):
                local[0] = local[0] + A[bx * threads + tid + j * stride]
            s = T.alloc_shared((threads,), T.float32)
            s[tid] = local[0]
            T.sync_threads()
            if tid == 0:
                acc = T.alloc_fragment((1,), T.float32)
                T.clear(acc)
                for i in T.serial(threads):
                    acc[0] = acc[0] + s[i]
                partials[bx] = acc[0]
    return partial


def make_final(programs: int, threads: int = FINAL_THREADS):
    # partial 缓冲区补零到 threads 的整数倍，final 无条件读取（避免 serial 循环内 guard
    # 触发 Metal LayoutInference 缺陷）；越界位置贡献 0，与 Metal 版 MLX final 同语义。
    padded = ((programs + threads - 1) // threads) * threads

    @tilelang.jit(target="metal")
    def final(A, C):
        A: T.Tensor((padded,), T.float32)
        C: T.Tensor((1,), T.float32)
        with T.Kernel(1, threads=threads) as bx:
            tid = T.get_thread_binding()
            local = T.alloc_fragment((1,), T.float32)
            T.clear(local)
            for j in T.serial(padded // threads):
                local[0] = local[0] + A[tid + j * threads]
            s = T.alloc_shared((threads,), T.float32)
            s[tid] = local[0]
            T.sync_threads()
            if tid == 0:
                acc = T.alloc_fragment((1,), T.float32)
                T.clear(acc)
                for i in T.serial(threads):
                    acc[0] = acc[0] + s[i]
                C[0] = acc[0]
    return final


def build_meta(args) -> dict:
    return {
        "config_id": args.config_id, "implementation": f"tilelang-p{args.programs}",
        "runtime": "tilelang", "shape": str(args.size), "dtype": "float32",
        "threadgroup": THREADS, "block": args.programs, "threads": args.programs * THREADS,
        "dispatch": "tile", "variant": f"p{args.programs}", "scope": rounds.WALL_SCOPE,
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--programs", type=int, default=256)
    parser.add_argument("--size", type=int, default=1048576)
    parser.add_argument("--input", default=MAIN_INPUT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--config-id", default="tilelang-p256")
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()

    x = make_input(args.input, args.size, args.seed)
    ref = reference_fp64(x)
    stride = args.programs * THREADS
    padded_n = ((args.size + stride - 1) // stride) * stride
    xpad = np.zeros(padded_n, np.float32)
    xpad[:args.size] = x
    partial = make_partial(padded_n, args.programs)
    final = make_final(args.programs)
    tx = to_mps(xpad)
    padded = ((args.programs + FINAL_THREADS - 1) // FINAL_THREADS) * FINAL_THREADS
    tp = torch.zeros(padded, dtype=torch.float32, device="mps")
    tc = torch.full((1,), float("nan"), dtype=torch.float32, device="mps")

    def op():
        partial(tx, tp)
        final(tp, tc)
        torch.mps.synchronize()

    op()
    pre_ok, pre_err = check(args.input, float(readback(tc)[0]), ref, 0.0)
    tc.fill_(float("nan"))
    samples = wall_samples(op, args.warmup, args.repeat) if args.repeat > 0 else []
    post_ok, post_err = check(args.input, float(readback(tc)[0]), ref, 0.0)

    meta = build_meta(args)
    print("ENV " + " ".join(f"{k}={v}" for k, v in environment().items()))
    correct = pre_ok and post_ok
    fields = {"operator": "sum-reduction", **meta, "process": args.process, "input": args.input,
              "correct": "OK" if correct else "FAIL",
              "precheck": "OK" if pre_ok else "FAIL", "postcheck": "OK" if post_ok else "FAIL",
              "max_abs_error": f"{max(pre_err, post_err):.3g}", "timed": int(bool(samples))}
    if samples:
        import statistics
        med = statistics.median(samples)
        fields.update(median_ms=f"{med:.6f}", min_ms=f"{min(samples):.6f}",
                      max_ms=f"{max(samples):.6f}",
                      logical_bandwidth_gbs=f"{bandwidth_gbs(str(args.size), med):.2f}")
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