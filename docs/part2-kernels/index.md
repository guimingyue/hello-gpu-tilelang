---
title: "第 2 篇：经典算子与 Kernel 实战（TileLang Metal 版）"
description: "从向量加法到 Fused RMSNorm：六章 TileLang Metal 受控实验，端到端墙钟，与 hello-gpu-metal 的 MLX 同口径对照"
---

# 第 2 篇：经典算子与 Kernel 实战

在第 1 篇的方法论之上（可信计时、受控对照、Roofline 定性），本篇亲手解剖六类经典 GPU 算子。
与 Metal 版不同，这里的两条实现路线是 **TileLang（tile 级 DSL，`target="metal"`）** 与
**MLX（数组算子 / `mx.fast.metal_kernel`，数字引用 hello-gpu-metal 同口径墙钟实验）**。

六类算子按**数据依赖复杂度**递进：输出互不相干的逐元素加法 → 需要跨线程合作的求和归约 →
归约与逐元素结合的行级 Softmax → 二维分块复用的矩阵乘 → 消除中间物化的 Attention 在线融合 →
收束全部方法的 Fused RMSNorm。

## 六章一览

| 章节 | 算子 | 主输入 | TileLang 最快配置（本机墙钟中位数） | 核心攻坚点 |
| --- | --- | --- | --- | --- |
| [第 8 章](./chapter8/) | Vector Add | N=16,777,216 | v0 · 512 线程 · 3.638 ms | threadgroup 宽度、每线程元素数 |
| [第 9 章](./chapter9/) | Sum Reduction | N=1,048,576 | P=128 · 0.490 ms | 两阶段 partial+final、float32 atomic 缺失 |
| [第 10 章](./chapter10/) | 行级 Softmax | 4096×1024 | T=256 · 2.175 ms | 三遍扫描、thread0 serial 合并 |
| [第 11 章](./chapter11/) | Matmul | 1024³ | tile=32 · 1.751 ms | `T.gemm` 分块、tile64 的 46 倍负面结果 |
| [第 12 章](./chapter12/) | Attention 前向 | 128×64 | materialized · 1.196 ms | 在线融合 vs 物化三阶段 |
| [第 13 章](./chapter13/) | Fused RMSNorm | 4096×1024 | T=256 · 1.395 ms | 平方和归约 + rsqrt 缩放、T 扫描 |

## 怎样阅读

每章固定 8 节：

1. **数据依赖**：一个输出需要哪些输入？能否并行？
2. **语义与正确性**：固定数学定义、参考实现、容差与边界用例。
3. **成本模型**：逻辑字节 / FLOP、算术强度、瓶颈假设。
4. **实现双路线**：`<ImplementationTabs>` 的 TileLang / MLX 标签页，各自保留自然基线与逐轮改动。
5. **正确性、基准与剖析**：三进程受控实验字段、边界结果、工具边界。
6. **两路线比较**：同口径墙钟的逐轮比较图与决策表。
7. **负面结果与边界**：没有收益或变慢的改动、Metal 后端限制，如实记录。
8. **复现与练习**：`.venv/bin/python` 命令、证据归档位置、手算练习。

## 一轮优化怎样进行

1. **建立基线**：逻辑最纯粹的正确实现，不刻意写慢来制造提速比。
2. **观察表现**：墙钟中位数 + 进程范围 + 逻辑带宽 / TFLOPS。
3. **提出假设**：明确下一项改动的技术动机。
4. **实施改动**：一轮只改一个核心设计选择。
5. **校验与复测**：正确性先于性能；计时前后各检查一次，输出先填 NaN。
6. **比较与决策**：收益超过进程范围才保留；否则撤回或继续排查。

读逐轮比较图时同时定位三个参照点：**路线基线、本轮修改前、本轮修改后**。
柱长 = 3 个独立进程 median 的中位数；误差线 = 这 3 个 median 的最小—最大范围，**不是置信区间**。
若两次测量差异与进程范围相当，结论是「尚未观察到稳定提速」，而不是优化成果。

## 三种对照各回答什么

| 对照 | 回答的问题 | 形式 |
| --- | --- | --- |
| 正确性参考 | 输出在数学上对吗？在容差内吗？ | CPU FP64 参考 / 解析误差上界 |
| 路线基线 | 同语言内这次改动有净收益吗？ | 本路线最先建立的正确版本 |
| 跨路线参照 | 与另一条路线 / 工业库差多少？ | 同口径墙钟的 MLX 引用行 |

## 计时与平台边界

- **口径**：端到端墙钟（`wall-clock-full-operator`），含一次完整算子调用与 `torch.mps.synchronize`。
  TileLang Metal 后端无 GPU 时间戳接口；该口径与 hello-gpu-metal 的 MLX 墙钟一致，与 MSL GPU 时间戳**不一致**。
- **平台**：Apple M1（applegpu_g13g）、macOS 27.0、Python 3.12.11、TileLang 0.1.15、NumPy 2.5.4、PyTorch 2.15.0。
- **工具**：Instruments / Xcode GPU Capture 为官方 profiling 工具，**未在本机实测**；各章只说明方法，不给出伪造的 trace 数字。
- **MLX 行**：`provenance=cited-hello-gpu-metal`，引用 hello-gpu-metal 同 shape、同 seed、同 warmup/repeat 的墙钟实验；本仓库 `.venv` 未安装 mlx，`*_mlx.py` 仅作路线参考源码保留。