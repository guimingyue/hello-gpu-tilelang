"""第 12 章边界正确性：不同 seq/dim 与 rising-max 输入，混合容差。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.tilelang_metal import environment, to_mps, readback
from attention_common import make_input, reference_attention, validate
from attention_tilelang import KERNELS

SHAPES = [(128, 64), (64, 64), (256, 64), (32, 32)]
VERSIONS = ["online", "materialized"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = []
    for version in VERSIONS:
        for seq, dim in SHAPES:
            for kind in ("dyadic", "rising-max"):
                q, k, v = make_input(seq, dim, args.seed, kind)
                ref = reference_attention(q, k, v)
                kernel = KERNELS[version](seq, dim)
                tb = torch.full((seq, dim), float("nan"), dtype=torch.float32, device="mps")
                kernel(to_mps(np.stack([q, k, v], axis=0)), tb)
                ok, err, ratio = validate(readback(tb), ref)
                cases.append({"seq": seq, "dim": dim, "version": version, "input": kind,
                              "correct": "OK" if ok else "FAIL", "max_abs_error": err,
                              "max_tolerance_ratio": ratio})
                if not ok:
                    raise SystemExit(f"boundary failed: {seq}x{dim} {version} {kind} ratio={ratio}")
    result = {"schema_version": 1, "kind": "chapter12-tilelang-boundary", "shapes": SHAPES,
              "seed": args.seed, "correct": "OK", "environment": environment(), "cases": cases}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"RESULT operator=attention boundary=OK cases={len(cases)}")


if __name__ == "__main__":
    main()