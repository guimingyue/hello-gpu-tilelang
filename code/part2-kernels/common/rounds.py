"""三进程受控实验框架（TileLang Metal 版），结构与 hello-gpu-metal 的 metal-rounds 一致。

方法学（与 Metal 版同口径）：
- 每个配置在 3 个**独立进程**中各跑一遍，配置顺序在进程间轮换；GPU 串行（子进程顺序执行）。
- 进程内：warmup 次预热 + repeat 次墙钟采样；柱长 = 3 个进程 median 的中位数，
  误差线 = 这 3 个 median 的最小—最大范围（不是置信区间）。
- 正确性：计时前后各做一次全量 finite + abs error 检查；输出先填 NaN。
- MLX 对照不重跑，直接引用 hello-gpu-metal 同口径（墙钟）实测，provenance 标注来源。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common.tilelang_metal import environment

SCHEMA_VERSION = 2

# 统一的样本/汇总分组键：所有章、两条路线共用同一套 launch 字段，避免不同口径混算。
# threadgroup=每 threadgroup 线程数；block=tile/分块参数；threads=总线程数(0=框架算子)；
# dispatch=threads/tile/op；variant=路线特定标签(每线程元素数/宽度/tile/block_k 等)。
GROUP_KEYS = ["config_id", "implementation", "runtime", "shape", "dtype", "threadgroup",
              "block", "threads", "dispatch", "variant", "scope", "warmup", "repeat", "seed"]
WALL_SCOPE = "wall-clock-full-operator"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(root: Path, manifest: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def initialize(root: Path, *, experiment: str, configs: list[dict], benchmark: dict,
               sources: list[str], source_dir: Path, extra: dict | None = None) -> dict:
    if root.exists():
        raise FileExistsError(f"choose a new --output directory: {root}")
    root.mkdir(parents=True)
    for part in ("logs", "samples", "source"):
        (root / part).mkdir()
    for name in sources:
        shutil.copy2(source_dir / name, root / "source" / name)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "source_sha256": {name: sha(root / "source" / name) for name in sources},
        "benchmark": benchmark,
        "configurations": configs,
        "execution_order": [],
        **(extra or {}),
    }
    save_manifest(root, manifest)
    return manifest


def run(argv: list[str], log: Path, root: Path) -> None:
    if log.exists():
        raise FileExistsError(f"refusing to overwrite a run: {log}")
    print(f"running {log.stem}", flush=True)
    with (root / "commands.jsonl").open("a") as file:
        file.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "argv": argv,
                               "log": str(log.relative_to(root))}) + "\n")
    with log.open("x") as file:
        subprocess.run(argv, stdout=file, stderr=subprocess.STDOUT, check=True, cwd=root)


def benchmark(root: Path, manifest: dict, command_fn, *, processes: int = 3) -> None:
    configs = manifest["configurations"]
    b = manifest["benchmark"]
    for process in range(1, processes + 1):
        offset = (process - 1) * max(1, len(configs) // processes)
        order = configs[offset:] + configs[:offset]
        manifest["execution_order"].append({"process": process, "configs": [c["id"] for c in order]})
        save_manifest(root, manifest)
        for config in order:
            stem = f"{b['shape_tag']}-{config['id']}-p{process}"
            samples = root / "samples" / f"{stem}.csv"
            if samples.exists():
                raise FileExistsError(f"refusing to append another run: {samples}")
            argv = command_fn(config, b, process, samples)
            run(argv, root / "logs" / f"{stem}.log", root)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, out: Path, *, group_keys: list[str], metric_keys: list[str],
              metric_fn, cited_rows: list[dict], shape_key: str = "shape") -> list[dict]:
    """聚合 TileLang 三进程样本，并并入引用的 MLX 对照行，产出 metal-rounds 风格 summary。"""
    manifest = json.loads((root / "manifest.json").read_text())
    processes = manifest["benchmark"]["processes"]
    records = []
    sample_files = {}
    for path in sorted((root / "samples").glob("*.csv")):
        rows = _read_csv(path)
        if not rows:
            raise ValueError(f"empty sample file: {path}")
        first = rows[0]
        if len(rows) != int(first["repeat"]):
            raise ValueError(f"incomplete samples: {path}")
        values = [float(row["ms"]) for row in rows]
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError(f"nonpositive/nonfinite sample: {path}")
        meta = {key: first[key] for key in group_keys}
        records.append({**meta, "process": int(first["process"]),
                        "sample_count": len(values), "median_ms": statistics.median(values),
                        "min_ms": min(values), "max_ms": max(values),
                        "sample_file": str(path.relative_to(root))})
        sample_files[str(path.relative_to(root))] = sha(path)

    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary = []
    for key, group in sorted(grouped.items()):
        if sorted(row["process"] for row in group) != list(range(1, processes + 1)):
            raise ValueError(f"missing/duplicate processes: {key}")
        medians = [row["median_ms"] for row in group]
        meta = dict(zip(group_keys, key, strict=True))
        med = statistics.median(medians)
        row = {**meta, "run_count": len(group), "correct": "OK",
               "median_ms": med, "median_ms_run_min": min(medians), "median_ms_run_max": max(medians),
               "provenance": "measured-tilelang-metal"}
        row.update(metric_fn(meta[shape_key], med))
        summary.append(row)

    summary.extend(cited_rows)
    if not summary:
        raise ValueError("no samples")

    columns = [*group_keys, "run_count", "correct", "median_ms", "median_ms_run_min",
               "median_ms_run_max", *metric_keys, "provenance"]
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "summary.csv", summary, columns)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_csv(out / "process-summary.csv", records,
               [*group_keys, "process", "sample_count", "median_ms", "min_ms", "max_ms", "sample_file"])
    manifest["sample_sha256"] = sample_files
    manifest["statistics"] = {
        "within_process": "median of wall-clock samples",
        "across_processes": "median of process medians; min/max of process medians",
        "range_is_confidence_interval": False,
        "scope": "wall-clock-full-operator (TileLang Metal 无 GPU 时间戳接口)",
        "mlx_provenance": "cited from hello-gpu-metal same-caliber wall-clock experiments",
    }
    save_manifest(out, manifest)
    print(f"summarized {len(summary)} rows ({len(records)} processes) -> {out}")
    return summary


def load_summary(path: Path) -> list[dict]:
    return _read_csv(path)