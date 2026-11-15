"""第 12 章：只读重绘归档 summary。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.round_plot import plot_summary

LABELS = {
    "tilelang-online": "TileLang 在线融合",
    "tilelang-materialized": "TileLang 物化三阶段",
    "mlx-materialized": "MLX 物化（引用）",
    "mlx-compiled": "MLX mx.compile（引用）",
    "mlx-online-k32": "MLX 在线 k32（引用）",
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path,
                        default=Path(__file__).parent / "evidence/metal-rounds/summary/summary.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "evidence/metal-rounds/summary/round-summary")
    args = parser.parse_args()
    plot_summary(args.summary, args.out,
                 title="第 12 章 Attention 前向：TileLang Metal 与 MLX 对照",
                 subtitle="128×64 · FP32 · 端到端墙钟（同口径）· Apple M1",
                 labels=LABELS,
                 notes=["TileLang 为本机实测；MLX 行引用 hello-gpu-metal 同口径墙钟实验。",
                        "score 点积用 thread0 serial 合并（Metal 无 simd_sum），是主要开销。"])


if __name__ == "__main__":
    main()