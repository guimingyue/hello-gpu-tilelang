"""第 9 章边界正确性：非整除长度 + 尾部，exact（random）与误差上界（uniform）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.tilelang_metal import environment, to_mps, readback
from reduction_common import make_input, reference_fp64, check, error_bound
from reduction_tilelang import make_partial, make_final

SHAPES = [1, 3, 31, 32, 33, 255, 256, 257, 1027, 4099, 262147]
PROGRAMS = [128, 256, 1024]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    for programs in PROGRAMS:
        for shape in SHAPES:
            for kind in ("random", "uniform"):
                x = make_input(kind, shape, args.seed)
                ref = reference_fp64(x)
                tol = 0.0 if kind == "random" else error_bound(x, depth=32)
                stride = programs * 256
                padded_n = ((shape + stride - 1) // stride) * stride
                xpad = np.zeros(padded_n, np.float32)
                xpad[:shape] = x
                partial = make_partial(padded_n, programs)
                final = make_final(programs)
                padded = ((programs + 1023) // 1024) * 1024
                tp = torch.zeros(padded, dtype=torch.float32, device="mps")
                tc = torch.full((1,), float("nan"), dtype=torch.float32, device="mps")
                partial(to_mps(xpad), tp)
                final(tp, tc)
                ok, err = check(kind, float(readback(tc)[0]), ref, tol)
                cases.append({"shape": shape, "programs": programs, "input": kind,
                              "correct": "OK" if ok else "FAIL", "max_abs_error": err})
                if not ok:
                    raise SystemExit(f"boundary failed: {shape} p{programs} {kind} err={err}")
    result = {"schema_version": 1, "kind": "chapter9-tilelang-boundary", "shapes": SHAPES,
              "seed": args.seed, "correct": "OK", "environment": environment(), "cases": cases}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT operator=sum-reduction boundary=OK cases={len(cases)}")


if __name__ == "__main__":
    main()