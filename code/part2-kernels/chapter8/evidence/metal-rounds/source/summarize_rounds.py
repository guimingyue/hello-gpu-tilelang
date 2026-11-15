"""第 8 章汇总：三进程 median 的中位数 + 引用 MLX 对照，产出 metal-rounds 风格 summary。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import rounds
from vector_add_common import bandwidth_gbs

SOURCE = Path(__file__).resolve().parent


def summarize(root: Path, out: Path) -> list[dict]:
    ref = json.loads((SOURCE / "mlx_reference.json").read_text())

    def metric_fn(shape: str, median_ms: float) -> dict:
        return {"effective_bandwidth_gbs": bandwidth_gbs(shape, median_ms)}

    return rounds.summarize(root, out, group_keys=rounds.GROUP_KEYS,
                            metric_keys=ref["metric_keys"], metric_fn=metric_fn,
                            cited_rows=ref["rows"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summarize(args.run_dir, args.out or args.run_dir / "summary")