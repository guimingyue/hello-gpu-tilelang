"""第 11 章汇总：三进程 median 的中位数 + 引用 MLX 对照。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import rounds
from matmul_common import tflops

SOURCE = Path(__file__).resolve().parent


def summarize(root: Path, out: Path) -> list[dict]:
    ref = json.loads((SOURCE / "mlx_reference.json").read_text())

    def metric_fn(shape, median_ms):
        return {"tflops": tflops(shape, median_ms)}

    return rounds.summarize(root, out, group_keys=rounds.GROUP_KEYS,
                            metric_keys=ref["metric_keys"], metric_fn=metric_fn, cited_rows=ref["rows"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summarize(args.run_dir, args.out or args.run_dir / "summary")