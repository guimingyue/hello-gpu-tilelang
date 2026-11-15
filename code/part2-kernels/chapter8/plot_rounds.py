"""第 8 章：只读重绘归档 summary（不重跑实验）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.round_plot import plot_summary

LABELS = {
    "tilelang-v0-b256": "TileLang v0 · 256 线程",
    "tilelang-v0-b512": "TileLang v0 · 512 线程",
    "tilelang-v1-e2": "TileLang v1 · E=2",
    "tilelang-v1-e4": "TileLang v1 · E=4",
    "tilelang-v1-e8": "TileLang v1 · E=8",
    "mlx-m0": "MLX 内置 a+b（引用）",
    "mlx-m1-e1": "MLX metal_kernel E=1（引用）",
    "mlx-m1-e2": "MLX metal_kernel E=2（引用）",
    "mlx-m1-e4": "MLX metal_kernel E=4（引用）",
    "mlx-m1-e8": "MLX metal_kernel E=8（引用）",
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
                 title="第 8 章 向量加法：TileLang Metal 与 MLX 对照",
                 subtitle="N=16,777,216 · FP32 · 端到端墙钟（同口径）· Apple M1",
                 labels=LABELS,
                 notes=["TileLang 为本机实测；MLX 行引用 hello-gpu-metal 同口径墙钟实验。",
                        "有效带宽 = 12·N / t（逻辑字节，非硬件流量）。"])


if __name__ == "__main__":
    main()