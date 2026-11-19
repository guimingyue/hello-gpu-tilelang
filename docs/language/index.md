---
title: TileLang 语言入门
description: 11 课从零认识 TileLang：历史与目的、技术关系、环境、Tile 与执行模型，再用可运行 demo 学会加法、归约、矩阵乘与融合。
---

# TileLang 语言入门

> 面向有 Python 基础、还没写过 GPU kernel 的读者。先知道这门语言为什么存在、和已有工具什么关系，再通过每次只加一点新概念的 demo 学会用它。本机动手主线为 Apple Silicon + Metal；语言概念与特定后端能力分开讲。

## 这条路线怎样走

| 课 | 内容 | 关键 demo |
| --- | --- | --- |
| [第 0 课](./00-history.md) | TileLang 的来历、目的与用途 | 不写代码：读时间线与应用证据 |
| [第 1 课](./01-ecosystem.md) | 语言、框架与编译器的关系 | 不写代码：画清技术栈位置 |
| [第 2 课](./02-environment.md) | 环境准备：从 Python 到 GPU | 环境自检 + MPS 张量加法 |
| [第 3 课](./03-first-kernel.md) | 第一个 kernel：十个数相加 | `vector_add.py`：定义→调用→校验 |
| [第 4 课](./04-tiles.md) | Tile、索引与尾部保护 | 交互分块演示 + 改长度复跑 |
| [第 5 课](./05-control-flow.md) | 控制流与编译时参数 | `control_flow.py`：条件 + 串行递推 |
| [第 6 课](./06-memory.md) | 内存、搬运与线程协作 | `memory_copy.py`：shared/fragment/copy |
| [第 7 课](./07-reduction.md) | 从行求和到稳定 Softmax | `reductions.py`：手动归约 |
| [第 8 课](./08-matmul.md) | 从点积到分块矩阵乘 | `matmul.py`：T.gemm 分块 |
| [第 9 课](./09-fusion.md) | 融合：少写一次中间结果 | `fusion.py`：加法+ReLU 两版对照 |
| [第 10 课](./10-performance.md) | 正确之后：计时、调参与生成代码 | `benchmark.py`：预热/同步/看中端码 |

每课固定结构：**一个问题 → 必要基础 → 最小 demo → 逐段解释 → 校验结果 → 改一个条件 → 练习**。demo 全部在仓库 `code/language/` 下，可独立运行；正文通过代码引用展示同一份源码，不会出现“文档与代码两份各改各的”。

## 学完之后去哪里

- 想系统做性能实验：进入[第 0 篇课程地图](../part0-intro/chapter0/index.md)，再按 `part0 → part1 → part2` 主线走；第 8–13 章会把本课的加法、归约、Softmax、矩阵乘、融合做成受控优化实验。
- 想查环境与平台边界：[第 1 章环境准备](../part0-intro/chapter1/index.md)与[附录 A](../appendix/appendix-a-env-install/index.md)。
- 想看 TileLang 之下发生了什么：[附录 C](../appendix/appendix-c-metal-low-level/index.md)。

## 三条阅读约定

1. **本机实测与上游能力分开。** 正文标注的实测结果来自本机 Apple M1（TileLang 0.1.15，验证日期 2026-11-18–19）；上游文档里的新硬件特性（如 M5 协作张量）不当作本机能力。
2. **正确先于性能。** 每个 demo 先与独立 CPU 参考比对，再谈快慢；第 10 课才正式引入计时。
3. **不承诺自动最优。** TileLang 把调度细节交给编译器，但性能结论必须由测量给出，语言本身不保证“写了就快”。
