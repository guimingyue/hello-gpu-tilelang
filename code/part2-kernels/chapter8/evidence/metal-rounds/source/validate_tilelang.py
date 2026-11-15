"""第 8 章边界正确性：非整除长度 + 尾部，全量 finite + abs error（与计时分开）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.tilelang_metal import environment, to_mps, readback, finite_abs_check
from vector_add_common import TOLERANCE, make_inputs
from vector_add_tilelang import make_kernel

SHAPES = [1, 3, 4, 5, 31, 32, 33, 255, 256, 257, 1027, 4099]
TAIL = 16777219
CONFIGS = [(256, 1), (512, 1), (256, 2), (256, 4), (256, 8)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    shapes = SHAPES + [TAIL]
    for block, elems in CONFIGS:
        for shape in shapes:
            a, b, ref = make_inputs(shape, args.seed)
            kernel = make_kernel(shape, block, elems)
            tc = torch.full((shape,), float("nan"), dtype=torch.float32, device="mps")
            kernel(to_mps(a), to_mps(b), tc)
            ok, error = finite_abs_check(readback(tc), ref, TOLERANCE)
            cases.append({"shape": shape, "block": block, "elements": elems,
                          "correct": "OK" if ok else "FAIL", "max_abs_error": error})
            if not ok:
                raise SystemExit(f"boundary failed: shape={shape} block={block} e={elems} err={error}")
    result = {"schema_version": 1, "kind": "chapter8-tilelang-boundary", "shapes": SHAPES,
              "tail_shape": TAIL, "seed": args.seed, "correct": "OK",
              "environment": environment(), "cases": cases}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT operator=vector-add boundary=OK cases={len(cases)} shapes={len(shapes)} configs={len(CONFIGS)}")


if __name__ == "__main__":
    main()