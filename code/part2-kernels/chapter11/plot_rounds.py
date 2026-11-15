"""第 11 章：只读重绘归档 summary。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.round_plot import plot_summary

LABELS = {
    "tilelang-tile16": "TileLang T.gemm tile=16",
    "tilelang-tile32": "TileLang T.gemm tile=32",
    "tilelang-tile64": "TileLang T.gemm tile=64",
    "mlx-compile": "MLX mx.compile matmul（引用）",
    "mlx-kernel-t8": "MLX metal_kernel t8（引用）",
    "mlx-kernel-t16": "MLX metal_kernel t16（引用）",
    "mlx-kernel-t32": "MLX metal_kernel t32（引用）",
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
                 title="第 11 章 矩阵乘：TileLang Metal 与 MLX 对照",
                 subtitle="1024×1024×1024 · FP32 · 端到端墙钟（同口径）· Apple M1",
                 labels=LABELS,
                 notes=["TileLang 为本机实测；MLX 行引用 hello-gpu-metal 同口径墙钟实验。",
                        "TFLOPS = 2·M·N·K / t；T.gemm Metal 后端未用 SIMD-group 矩阵内建，明显慢于 MLX。"])


if __name__ == "__main__":
    main()