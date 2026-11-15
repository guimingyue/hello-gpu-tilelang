"""第 11 章边界正确性：非整除 M/N/K 与尾部，Higham FP32 上界。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.tilelang_metal import environment, to_mps, readback
from matmul_common import make_inputs, reference_matmul, error_bound, validate
from matmul_tilelang import make_kernel, BK

SHAPES = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (129, 131, 127)]
TILES = [16, 64]


def ceil_mul(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    for tile in TILES:
        for m, n, k in SHAPES:
            a, b = make_inputs(m, n, k, args.seed)
            ref = reference_matmul(a, b)
            bound = error_bound(a, b)
            # T.gemm 分块 kernel 要求 M/N 为 tile 倍数、K 为 BK 倍数；非整除形状由主机
            # 补零到对齐尺寸，再取左上 M×N 校验（kernel 内不做边界 guard，见限制清单）。
            mp, np_, kp = ceil_mul(m, tile), ceil_mul(n, tile), ceil_mul(k, BK)
            ap = np.zeros((mp, kp), np.float32); ap[:m, :k] = a
            bp = np.zeros((kp, np_), np.float32); bp[:k, :n] = b
            kernel = make_kernel(mp, np_, kp, tile)
            tc = torch.full((mp, np_), float("nan"), dtype=torch.float32, device="mps")
            kernel(to_mps(ap), to_mps(bp), tc)
            ok, err, ratio = validate(readback(tc)[:m, :n], ref, bound)
            cases.append({"m": m, "n": n, "k": k, "tile": tile,
                          "correct": "OK" if ok else "FAIL", "max_abs_error": err,
                          "max_error_ratio": ratio})
            if not ok:
                raise SystemExit(f"boundary failed: {m}x{n}x{k} tile{tile} ratio={ratio}")
    result = {"schema_version": 1, "kind": "chapter11-tilelang-boundary", "shapes": SHAPES,
              "seed": args.seed, "correct": "OK", "environment": environment(), "cases": cases}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT operator=matmul boundary=OK cases={len(cases)}")


if __name__ == "__main__":
    main()