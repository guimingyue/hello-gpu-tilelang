---
title: "第4章 从 kernel 到实验：校验、计时与受控复测"
description: "Hello GPU TileLang 第4章 · 正确性先于性能，三进程受控实验与证据归档"
---

# 第4章 从 kernel 到实验：校验、计时与受控复测

## 本章导读

> 写出 kernel 只是开始。本章建立「校验 → 计时 → 受控复测 → 归档」的闭环，让每个性能数字都有出处、可复跑。

## 4.1 正确性先于性能

每次改动先全量校验：输出有限 + 与 FP64 参考的绝对误差在容差内；边界用 guard 区检查越界写。`validate_tilelang.py` 对小形状矩阵做这件事，失败即停，不进入计时。

## 4.2 计时口径

TileLang Metal 无 GPU 时间戳，用端到端墙钟：

```python
def wall(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out
```

预热排除首次编译；repeat 取 median 抗抖动。

## 4.3 三进程受控复测

`run_rounds.py` 按配置矩阵跑 3 个独立进程、轮换顺序，写 `evidence/metal-rounds/samples/*.csv` 与 manifest（源码 SHA-256、环境、命令）。`summarize_rounds.py` 校验矩阵完整、口径不混算后产出 summary.csv；`plot_rounds.py` 只读归档重绘。

## 4.4 读图三参照

每轮对比图同时看：路线基线、本轮修改前、本轮修改后。微小差异若落在进程范围内，结论是「尚未观察到稳定提速」。

## 4.5 练习

1. 跑 `run_rounds.py --output evidence/my-run`，核对 manifest 的 source_sha256；
2. 把 repeat 从 50 降 5，观察范围变宽；
3. 解释为什么不能把 TileLang 墙钟与 MSL GPU 时间戳相减。