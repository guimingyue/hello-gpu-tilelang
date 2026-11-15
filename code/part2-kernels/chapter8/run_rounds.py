"""第 8 章三进程受控实验入口（TileLang Metal 路线 + 引用 MLX 对照）。

用法：
    .venv/bin/python chapter8/run_rounds.py --output chapter8/evidence/my-run
默认主阶段：边界正确性 + 5 个 TileLang 配置 × 3 个独立进程（GPU 串行）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import rounds
from common.tilelang_metal import environment

SOURCE = Path(__file__).resolve().parent
SCRIPT = SOURCE / "vector_add_tilelang.py"
SOURCES = ["vector_add_tilelang.py", "vector_add_common.py", "vector_add_mlx.py",
           "run_rounds.py", "summarize_rounds.py", "validate_tilelang.py"]

CONFIGS = [
    {"id": "tilelang-v0-b256", "version": "v0", "block": 256, "elements": 1},
    {"id": "tilelang-v0-b512", "version": "v0", "block": 512, "elements": 1},
    {"id": "tilelang-v1-e2", "version": "v1", "block": 256, "elements": 2},
    {"id": "tilelang-v1-e4", "version": "v1", "block": 256, "elements": 4},
    {"id": "tilelang-v1-e8", "version": "v1", "block": 256, "elements": 8},
]


def command(config: dict, b: dict, process: int, samples: Path) -> list[str]:
    return [sys.executable, str(SCRIPT), "--version", config["version"],
            "--block", str(config["block"]), "--elements", str(config["elements"]),
            "--size", str(b["shape"]), "--warmup", str(b["warmup"]), "--repeat", str(b["repeat"]),
            "--seed", str(b["seed"]), "--config-id", config["id"], "--process", str(process),
            "--samples", str(samples)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=16777216)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    root = args.output.resolve()
    benchmark = {"shape": args.size, "shape_tag": f"n{args.size}", "warmup": args.warmup,
                 "repeat": args.repeat, "seed": args.seed, "dtype": "float32", "processes": 3,
                 "scope": rounds.WALL_SCOPE,
                 "timing": "wall-clock (TileLang Metal 无 GPU 时间戳接口)",
                 "correctness": "full finite + abs error <= 1e-6, NaN-prefilled output"}
    manifest = rounds.initialize(root, experiment="chapter8-tilelang-vector-add-rounds",
                                 configs=CONFIGS, benchmark=benchmark, sources=SOURCES,
                                 source_dir=SOURCE)
    # 边界正确性（独立于计时）
    rounds.run([sys.executable, str(SOURCE / "validate_tilelang.py"), "--seed", str(args.seed),
                "--out", str(root / "boundary.json")], root / "logs" / "boundary.log", root)
    rounds.benchmark(root, manifest, command)
    manifest["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    rounds.save_manifest(root, manifest)
    from summarize_rounds import summarize
    summarize(root, root / "summary")


if __name__ == "__main__":
    main()