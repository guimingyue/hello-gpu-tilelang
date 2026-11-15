---
title: "第5章 可信计时：TileLang Metal 的口径"
description: "Hello GPU TileLang 第5章 · 端到端墙钟 vs GPU 时间戳，三进程受控实验与范围表达"
---

# 第5章 可信计时：TileLang Metal 的口径

## 本章导读

> TileLang 的 Metal 后端**不暴露 command-buffer GPU 时间戳**，因此本教程统一用**端到端墙钟**：从提交到 `torch.mps.synchronize()` 返回。本章建立与 Metal 版一致的受控实验方法，并明确口径边界。

## 5.1 两种口径

| 口径 | 包含什么 | 谁提供 |
| --- | --- | --- |
| GPU 时间戳 | 仅 GPU 执行（GPUStartTime→GPUEndTime） | MSL / hello_metal |
| 端到端墙钟 | 图构建/编码/提交/同步 | TileLang Metal、MLX、PyTorch MPS |

TileLang 与 MLX 同属墙钟口径，可以互比；**不要**与 MSL 的 GPU 时间戳混比。

## 5.2 三进程受控实验

- 固定输入（dyadic，FP32 精确）、seed、warmup、repeat；
- 3 个独立进程，轮换配置顺序；
- 柱长 = 三进程 median 的中位数；误差线 = 进程 median 的最小—最大范围（**不是置信区间**）；
- 每次计时前后全量校验输出（finite + abs error）。

## 5.3 计时模板

```python
import time, torch
def wall(fn, warmup=10, repeat=50):
    for _ in range(warmup): fn()
    torch.mps.synchronize()
    samples=[]
    for _ in range(repeat):
        t0=time.perf_counter(); fn(); torch.mps.synchronize()
        samples.append((time.perf_counter()-t0)*1e3)
    return samples
```

## 5.4 边界

- 墙钟含主机开销，小 kernel 时占比高；
- 无 GPU 时间戳 ⇒ 无法单独剥离主机开销，如实标注；
- Instruments / Xcode GPU Capture 可补硬件视角，但**本教程未在本机实测**。

## 5.5 练习

1. 对第 8 章 v0 用 `wall()` 复测，与 summary.csv 对比；
2. 把 repeat 从 50 降到 5，观察范围变宽；
3. 解释为什么不能把 TileLang 墙钟与 MSL GPU 时间戳直接相减。