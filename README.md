# hello-gpu-tilelang

> 用 **TileLang** 学 GPU：与 [hello-gpu-metal](../hello-gpu-metal) 同结构的第三份教程，目标后端为 **Metal（Apple Silicon）**。

本仓库是 hello-gpu 课程的 TileLang 版：保留原课程 20 章 + 附录 + Notebook 的结构，把两条实现路线改为 **TileLang（tile-level DSL）** 与 **MSL/MLX 对照**。TileLang 通过 `target="metal"` 把 tile 级 kernel lower 成 Metal Shading Language，在 Apple M1 上实测。

## 与 Metal 版的关系

- 数学语义、正确性标准、计时口径（GPU 时间戳 vs 端到端墙钟）、三进程受控实验方法完全一致；
- 每个算子给出 TileLang 实现，并与 hello-gpu-metal 的 MSL/MLX 实测数字横向对比（同口径才比）；
- TileLang Metal 后端的限制（如不支持的 tile 原语、threadgroup 内存上限）会明确标注，不隐瞒负面结果。

## 环境

```bash
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python tilelang numpy torch
.venv/bin/python code/part2-kernels/chapter8/smoke_metal.py
```

## 结构

```text
code/
  part0-intro/      # 第 0–4 章
  part1-profiling/  # 第 5–7 章
  part2-kernels/    # 第 8–13 章（TileLang + MSL/MLX 对照）
  part3-agent/      # 第 14–17 章
  part4-models-agent/ # 第 18–19 章
docs/               # VitePress 站点
notebooks/          # 交互式 Notebook
```

## 状态

建设中：先完成第 8 章 TileLang Metal 冒烟与逐轮实测，再按 Metal 版结构逐章迁移。