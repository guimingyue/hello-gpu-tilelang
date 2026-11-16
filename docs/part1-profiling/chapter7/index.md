---
title: "第7章 Roofline：算术强度与 Apple M1 参考线"
description: "Hello GPU TileLang 第7章 · 用实测 copy/GEMM 参考线把 TileLang 算子映射为工作点"
---

# 第7章 Roofline：算术强度与 Apple M1 参考线

## 本章导读

> Roofline 用「算术强度 AI = FLOP / Byte」把算子放到带宽屋顶与算力屋顶之下。本章用 M1 实测参考线，把 TileLang 算子映射为工作点，定性瓶颈。

## 7.1 逻辑算术强度

向量加法：F = N，Q = 12N ⇒ AI = 1/12 ≈ 0.083 FLOP/Byte，远低于拐点 ⇒ 带宽受限。矩阵乘 AI 随 K 增大 ⇒ 趋向算力受限。

## 7.2 参考线（实测，非硬件峰值）

- 带宽参考：MSL copy 的实测有效带宽（M1 约 43–60 GB/s，见 Metal 版第 7 章）；
- 算力参考：MLX FP32 GEMM 实测 TFLOPS；
- 拐点 = 算力参考 / 带宽参考。

**算法字节 ≠ 物理流量**；参考线是独立 workload 的实测，不是标称峰值。

## 7.3 工作点

把第 8–13 章 TileLang 实测 (P, BW) 画到 log-log Roofline：加法贴在带宽屋顶下；归约/Softmax 因额外归约流量略低；矩阵乘随 tile 复用向右移动。负面结果（tile64 变慢）表现为工作点下移而非右移。

## 7.4 练习

1. 计算第 9 章归约的 AI（读 N、写 1，F=N）；
2. 解释为什么融合（第 12/13 章）能减少 Q 从而上移工作点；
3. 用 `plot_rounds.py` 的带宽列核对工作点横坐标。