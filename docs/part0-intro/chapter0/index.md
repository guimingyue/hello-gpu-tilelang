---
title: "第0章 课程地图：用 TileLang 在 Apple GPU 上学算子优化"
description: "Hello GPU TileLang 第0章 · 课程结构、两条路线、Apple M1 基线与学习方法"
---

# 第0章 课程地图：用 TileLang 在 Apple GPU 上学算子优化

## 本章导读

> 本教程是 [hello-gpu-metal](https://github.com/datawhalechina/hello-gpu) 的 **TileLang 版**：同样的 20 章结构与受控实验方法，但 kernel 用 **TileLang**（`target="metal"`）书写，在 **Apple M1** 上实测。本章给出地图与约定，不写代码。

## 0.1 为什么是 TileLang + Metal

- **TileLang** 是 tile 级 DSL：用 Python 表达 `T.Kernel` / `T.Parallel` / `T.copy` / `T.gemm`，lower 到 MSL/CUDA；
- **Metal** 是 Apple Silicon 的底层 GPU API；TileLang 的 Metal 后端把 tile 逻辑编译成 MSL；
- 统一内存省去显式拷贝，但**内存流量与带宽瓶颈依然存在**——优化的核心问题不变。

## 0.2 两条路线

| 路线 | 抽象 | 何时用 |
| --- | --- | --- |
| TileLang | tile / threadgroup 显式分工 | 想控制分块、复用、融合 |
| MLX | 数组算子，框架自动分块 | 快速基线、工业参照 |

两路线共用同一数学问题与正确性标准；每章用标签页对照。MLX 数字引用 Metal 版同口径实验。

## 0.3 基线与口径

- 硬件基线 **Apple M1（8 核 GPU、16 GiB 统一内存、applegpu_g13g）**；
- TileLang Metal 后端**不暴露 GPU 时间戳**，统一用**端到端墙钟**（含提交与同步）；
- 三进程受控实验：柱长 = 三进程 median 的中位数，误差线 = 进程范围（**不是置信区间**）；
- 未实测项（Instruments/Xcode 硬件计数器）明确标注。

## 0.4 怎样读

第一次接触 TileLang，可以先读独立的[语言入门教程](../../language/index.md)：从历史、技术关系和十个数的加法开始，用 11 课建立语言基础，再回来读算子优化主线。语言课与原有章节独立编号。

按 `part0 → part1 → part2 → part3/4` 递进；附录 A–D 随时查。每章八节：数据依赖 → 语义正确性 → 成本模型 → 实现双路线 → 正确性基准剖析 → 两路线比较 → 负面结果边界 → 复现练习。

下一步：[第 1 章 环境准备与第一个 TileLang kernel](../chapter1/index.md)。