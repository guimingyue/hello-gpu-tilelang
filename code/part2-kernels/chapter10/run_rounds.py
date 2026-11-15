"""第 10 章三进程受控实验入口（TileLang 行级 Softmax + 引用 MLX 对照）。"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import rounds

SOURCE = Path(__file__).resolve().parent
SCRIPT = SOURCE / "softmax_tilelang.py"
SOURCES = ["softmax_tilelang.py", "softmax_common.py", "softmax_mlx.py",
           "run_rounds.py", "summarize_rounds.py", "validate_tilelang.py"]
CONFIGS = [{"id": f"tilelang-t{t}", "threads": t} for t in (256, 512, 1024)]


def command(config, b, process, samples):
    rows, cols = (int(v) for v in b["shape"].split("x"))
    return [sys.executable, str(SCRIPT), "--rows", str(rows), "--cols", str(cols),
            "--threads", str(config["threads"]), "--warmup", str(b["warmup"]),
            "--repeat", str(b["repeat"]), "--seed", str(b["seed"]), "--config-id", config["id"],
            "--process", str(process), "--samples", str(samples)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", default="4096x1024")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    root = args.output.resolve()
    benchmark = {"shape": args.shape, "shape_tag": f"n{args.shape}", "warmup": args.warmup,
                 "repeat": args.repeat, "seed": args.seed, "dtype": "float32", "processes": 3,
                 "scope": rounds.WALL_SCOPE, "timing": "wall-clock (TileLang Metal 无 GPU 时间戳接口)",
                 "correctness": "FP64 softmax reference; abs<=2e-5 and row_sum<=2e-5; NaN-prefilled"}
    manifest = rounds.initialize(root, experiment="chapter10-tilelang-softmax-rounds",
                                 configs=CONFIGS, benchmark=benchmark, sources=SOURCES, source_dir=SOURCE)
    rounds.run([sys.executable, str(SOURCE / "validate_tilelang.py"), "--seed", str(args.seed),
                "--out", str(root / "boundary.json")], root / "logs" / "boundary.log", root)
    rounds.benchmark(root, manifest, command)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    rounds.save_manifest(root, manifest)
    from summarize_rounds import summarize
    summarize(root, root / "summary")


if __name__ == "__main__":
    main()