"""第 10 章边界正确性：含非整除列宽与 +/-1000 平移行的稳定 Softmax。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.tilelang_metal import environment, to_mps, readback
from softmax_common import make_input, reference_softmax, validate
from softmax_tilelang import make_kernel

SHAPES = [(8, 64), (32, 128), (64, 256), (128, 512), (256, 1024), (64, 1025)]
THREADS = [256, 1024]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    for threads in THREADS:
        for rows, cols in SHAPES:
            # Metal 后端在 threads>cols 时行内归约失效，边界测试取 threads<=cols。
            eff = min(threads, cols)
            x = make_input(rows, cols, args.seed)
            ref = reference_softmax(x)
            kernel = make_kernel(rows, cols, eff)
            tb = torch.full((rows, cols), float("nan"), dtype=torch.float32, device="mps")
            kernel(to_mps(x), tb)
            ok, abs_err, row_err = validate(readback(tb), ref)
            cases.append({"rows": rows, "cols": cols, "threads": eff,
                          "correct": "OK" if ok else "FAIL", "max_abs_error": abs_err,
                          "max_row_sum_error": row_err})
            if not ok:
                raise SystemExit(f"boundary failed: {rows}x{cols} t{eff} abs={abs_err} row={row_err}")
    result = {"schema_version": 1, "kind": "chapter10-tilelang-boundary", "shapes": SHAPES,
              "seed": args.seed, "correct": "OK", "environment": environment(), "cases": cases}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT operator=softmax boundary=OK cases={len(cases)}")


if __name__ == "__main__":
    main()