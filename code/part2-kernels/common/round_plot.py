"""从归档 summary.csv 重绘逐轮比较图（复用 metal 版 plot_style 思路）。"""
from __future__ import annotations

import csv
from pathlib import Path

from common.plot_style import TimingPanel, render_timing_figure


def plot_summary(summary_csv: Path, output: Path, *, title: str, subtitle: str,
                 labels: dict[str, str], notes: list[str],
                 formats=("png", "svg")) -> None:
    rows = list(csv.DictReader(open(summary_csv)))
    if not rows:
        raise ValueError(f"empty summary: {summary_csv}")
    # 按 runtime 分组排序：TileLang 在前，MLX 引用在后，各自按 median 升序。
    order = {"tilelang": 0, "mlx": 1, "framework": 2}
    rows.sort(key=lambda r: (order.get(r["runtime"], 9), float(r["median_ms"])))
    render_timing_figure(panels=[TimingPanel(rows=rows)], title=title, subtitle=subtitle,
                         notes=notes, output=output, labels=labels, formats=formats)