"""第 13 章边界正确性：含非整除列宽与 zero/signed-weight 输入，混合容差。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.tilelang_metal import environment, to_mps, readback
from rmsnorm_common import make_inputs, reference_rmsnorm, validate, EPSILON
from rmsnorm_tilelang import make_kernel

SHAPES = [(8, 64), (64, 256), (256, 1024), (64, 1025)]
THREADS = [256, 1024]
MODES = ["normal", "zero", "signed-weight"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    for threads in THREADS:
        for rows, cols in SHAPES:
            eff = min(threads, cols)
            for mode in MODES:
                x, w = make_inputs(rows, cols, args.seed, mode)
                ref = reference_rmsnorm(x, w, EPSILON)
                kernel = make_kernel(rows, cols, eff)
                tc = torch.full((rows, cols), float("nan"), dtype=torch.float32, device="mps")
                kernel(to_mps(x), to_mps(w), tc)
                ok, abs_err, rel_err = validate(readback(tc), ref)
                cases.append({"rows": rows, "cols": cols, "threads": eff, "input_mode": mode,
                              "correct": "OK" if ok else "FAIL", "max_abs_error": abs_err,
                              "max_rel_error": rel_err})
                if not ok:
                    raise SystemExit(f"boundary failed: {rows}x{cols} t{eff} {mode} abs={abs_err}")
    result = {"schema_version": 1, "kind": "chapter13-tilelang-boundary", "shapes": SHAPES,
              "seed": args.seed, "correct": "OK", "environment": environment(), "cases": cases}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT operator=rmsnorm boundary=OK cases={len(cases)}")


if __name__ == "__main__":
    main()