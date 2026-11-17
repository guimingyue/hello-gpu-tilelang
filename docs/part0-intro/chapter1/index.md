---
title: "第1章 环境准备与第一个 TileLang kernel"
description: "Hello GPU TileLang 第1章 · Apple Silicon 上安装 TileLang（Metal 后端），跑通第一个逐元素加法 kernel"
---

# 第1章 环境准备与第一个 TileLang kernel

## 本章导读

> 本教程用 **TileLang** 在 Apple Silicon 上写 GPU kernel。TileLang 是 tile 级 DSL，通过 `target="metal"` 把 kernel lower 成 Metal Shading Language。本章装好环境并跑通第一个逐元素加法，建立「写 kernel → 校验 → 计时」的最小闭环。

## 1.1 环境要求

- Apple Silicon Mac（M1/M2/M3/M4），macOS 12+；
- Python 3.12；`uv`；
- TileLang ≥ 0.1.15（含 Metal 后端）、numpy、torch（仅用于把张量放到 MPS）。

```bash
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python \
  --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple tilelang numpy torch
```

> 默认 PyPI 下载 wheel 可能超时；本仓库 `pyproject.toml` 已配置 BFSU 镜像。

## 1.2 第一个 kernel：逐元素加法

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="metal")
def vadd(A, B, C):
    A: T.Tensor((n,), T.float32)
    B: T.Tensor((n,), T.float32)
    C: T.Tensor((n,), T.float32)
    with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
        for i in T.Parallel(256):
            idx = bx * 256 + i
            if idx < n:
                C[idx] = A[idx] + B[idx]
```

要点：

- `@tilelang.jit(target="metal")` 选择 Metal 后端；
- **eager 风格**需把输出张量 `C` 作为参数传入（`out_idx` 仅 lazy 模式可用）；
- `T.Kernel(grid, threads=...)` 声明 threadgroup 划分；`T.Parallel` 展开线程内循环；
- 边界用 `if idx < n` 保护，与 MSL 版一致。

## 1.3 校验与运行

```python
import numpy as np
import torch

n = 1 << 20
a = np.arange(n, dtype=np.float32)
b = 2 * a
ta = torch.from_numpy(a).to("mps")
tb = torch.from_numpy(b).to("mps")
tc = torch.empty(n, dtype=torch.float32, device="mps")
vadd(ta, tb, tc)
torch.mps.synchronize()
assert np.max(np.abs(tc.cpu().numpy() - (a + b))) == 0.0
```

冒烟脚本见 `code/part2-kernels/chapter8/smoke_metal.py`，本机 M1 实测 `max_abs_error=0.0`。

## 1.4 计时口径

TileLang Metal 后端**不暴露 GPU 时间戳**，本教程统一用**端到端墙钟**（含提交与 `torch.mps.synchronize`），与 MLX 同口径对比；不要与 MSL 的 command-buffer GPU 时间戳混比。详见[第 5 章](../../part1-profiling/chapter5/index.md)。

## 1.5 练习

1. 把 `threads` 从 256 改为 512，观察墙钟变化；
2. 去掉 `if idx < n`，用非整除长度运行，观察错误；
3. 用 `T.empty` + lazy 模式重写，比较两种风格。