"""第 9 章：只读重绘归档 summary。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.round_plot import plot_summary

LABELS = {
    "tilelang-p128": "TileLang 两阶段 P=128",
    "tilelang-p256": "TileLang 两阶段 P=256",
    "tilelang-p512": "TileLang 两阶段 P=512",
    "tilelang-p1024": "TileLang 两阶段 P=1024",
    "mlx-p128": "MLX simd_sum P=128（引用）",
    "mlx-p256": "MLX simd_sum P=256（引用）",
    "mlx-p512": "MLX simd_sum P=512（引用）",
    "mlx-p1024": "MLX simd_sum P=1024（引用）",
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
                 title="第 9 章 求和归约：TileLang Metal 与 MLX 对照",
                 subtitle="N=1,048,576 · FP32 · 端到端墙钟（同口径）· Apple M1",
                 labels=LABELS,
                 notes=["TileLang 为本机实测；MLX 行引用 hello-gpu-metal 同口径墙钟实验。",
                        "逻辑带宽 = (4·N+4) / t；float32 atomic 不支持，采用 exclusive partial 两阶段。"])


if __name__ == "__main__":
    main()