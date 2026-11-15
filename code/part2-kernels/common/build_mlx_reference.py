"""从 hello-gpu-metal 的 metal-rounds 证据中提取 MLX 墙钟实测，生成各章 mlx_reference.json。

这些数字来自 Metal 版**同口径（端到端墙钟）**实验，本教程直接引用作为对照，不重跑 MLX。
provenance 标注来源；metric 用与各章 TileLang 相同的公式从引用的 median_ms 重新折算。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

METAL = Path("/Users/guimingyue/github/hello-gpu-metal/code/part2-kernels")
HERE = Path(__file__).resolve().parent.parent
SEED = 20260920
SCOPE = "wall-clock-full-operator"

# chapter -> (main_shape, warmup, repeat, metric_kind, [(metal_config_id, launch...)])
# launch = (threadgroup, block, threads, dispatch, variant)
CHAPTERS = {
    8: dict(shape="16777216", warmup=10, repeat=50, metric="bw12", configs=[
        ("mlx-m0", (0, 0, 0, "op", "builtin")),
        ("mlx-m1-e1", (256, 1, 0, "threads", "E1")),
        ("mlx-m1-e2", (256, 2, 0, "threads", "E2")),
        ("mlx-m1-e4", (256, 4, 0, "threads", "E4")),
        ("mlx-m1-e8", (256, 8, 0, "threads", "E8")),
    ]),
    9: dict(shape="1048576", warmup=5, repeat=10, metric="bw4", configs=[
        ("mlx-p128", (0, 128, 0, "op", "p128")),
        ("mlx-p256", (0, 256, 0, "op", "p256")),
        ("mlx-p512", (0, 512, 0, "op", "p512")),
        ("mlx-p1024", (0, 1024, 0, "op", "p1024")),
    ]),
    10: dict(shape="4096x1024", warmup=10, repeat=50, metric="bw8", configs=[
        ("mlx-ops", (0, 0, 0, "op", "ops")),
        ("mlx-compile", (0, 0, 0, "op", "compile")),
        ("mlx-kernel-w1-t128", (128, 1, 0, "threads", "w1t128")),
        ("mlx-kernel-w1-t256", (256, 1, 0, "threads", "w1t256")),
        ("mlx-kernel-w2-t128", (128, 2, 0, "threads", "w2t128")),
        ("mlx-kernel-w2-t256", (256, 2, 0, "threads", "w2t256")),
    ]),
    11: dict(shape="1024x1024x1024", warmup=10, repeat=50, metric="tflops", configs=[
        ("mlx-compile", (0, 0, 0, "op", "compile")),
        ("mlx-kernel-t8", (0, 8, 0, "tile", "t8")),
        ("mlx-kernel-t16", (0, 16, 0, "tile", "t16")),
        ("mlx-kernel-t32", (0, 32, 0, "tile", "t32")),
    ]),
    12: dict(shape="128x64", warmup=10, repeat=50, metric="none", configs=[
        ("mlx-materialized", (0, 0, 0, "op", "materialized")),
        ("mlx-compiled", (0, 0, 0, "op", "compiled")),
        ("mlx-online-k32", (0, 32, 0, "op", "online-k32")),
    ]),
    13: dict(shape="4096x1024", warmup=10, repeat=50, metric="bw8", configs=[
        ("mlx-compose", (0, 0, 0, "op", "compose")),
        ("mlx-compile", (0, 0, 0, "op", "compile")),
        ("mlx-kernel", (256, 0, 0, "threads", "kernel")),
    ]),
}


def metric_value(kind: str, shape: str, ms: float) -> dict:
    if kind == "none":
        return {}
    if kind == "bw12":
        n = int(shape)
        return {"effective_bandwidth_gbs": 12 * n / (ms * 1e6)}
    if kind == "bw4":
        n = int(shape)
        return {"logical_bandwidth_gbs": (4 * n + 4) / (ms * 1e6)}
    if kind == "bw8":
        rows, cols = (int(x) for x in shape.split("x"))
        return {"effective_bandwidth_gbs": 8 * rows * cols / (ms * 1e6)}
    if kind == "tflops":
        m, n, k = (int(x) for x in shape.split("x"))
        return {"tflops": 2 * m * n * k / (ms * 1e9)}
    raise ValueError(kind)


def metric_keys(kind: str) -> list[str]:
    return {"none": [], "bw12": ["effective_bandwidth_gbs"], "bw4": ["logical_bandwidth_gbs"],
            "bw8": ["effective_bandwidth_gbs"], "tflops": ["tflops"]}[kind]


def main() -> None:
    for chapter, spec in CHAPTERS.items():
        rows = list(csv.DictReader(open(METAL / f"chapter{chapter}/evidence/metal-rounds/summary/summary.csv")))
        by_id = {r["config_id"]: r for r in rows if r["shape"] == spec["shape"] and r["runtime"] == "mlx"}
        cited = []
        for cid, (tg, block, threads, dispatch, variant) in spec["configs"]:
            src = by_id[cid]
            med = float(src["median_ms"])
            row = {
                "config_id": cid, "implementation": cid, "runtime": "mlx", "shape": spec["shape"],
                "dtype": "float32", "threadgroup": tg, "block": block, "threads": threads,
                "dispatch": dispatch, "variant": variant, "scope": SCOPE,
                "warmup": spec["warmup"], "repeat": spec["repeat"], "seed": SEED,
                "run_count": int(src["run_count"]), "correct": "OK",
                "median_ms": med, "median_ms_run_min": float(src["median_ms_run_min"]),
                "median_ms_run_max": float(src["median_ms_run_max"]),
                "provenance": "cited-hello-gpu-metal",
                **metric_value(spec["metric"], spec["shape"], med),
            }
            cited.append(row)
        out = HERE / f"chapter{chapter}/mlx_reference.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metric_keys": metric_keys(spec["metric"]), "rows": cited},
                                  indent=2, ensure_ascii=False) + "\n")
        print(f"chapter{chapter}: cited {len(cited)} MLX rows -> {out.name}")


if __name__ == "__main__":
    main()