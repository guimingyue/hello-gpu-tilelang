"""第 13 章：只读重绘归档 summary。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.round_plot import plot_summary

LABELS = {
    "tilelang-t256": "TileLang 行级 T=256",
    "tilelang-t512": "TileLang 行级 T=512",
    "tilelang-t1024": "TileLang 行级 T=1024",
    "mlx-compose": "MLX 算子组合（引用）",
    "mlx-compile": "MLX mx.compile（引用）",
    "mlx-kernel": "MLX metal_kernel（引用）",
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
                 title="第 13 章 Fused RMSNorm：TileLang Metal 与 MLX 对照",
                 subtitle="4096×1024 · FP32 · 端到端墙钟（同口径）· Apple M1",
                 labels=LABELS,
                 notes=["TileLang 为本机实测；MLX 行引用 hello-gpu-metal 同口径墙钟实验。",
                        "有效带宽 = 8·rows·cols / t；平方和用 thread0 serial 合并（Metal 无 simd_sum）。"])


if __name__ == "__main__":
    main()