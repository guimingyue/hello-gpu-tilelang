---
title: 第 1 课 语言、框架与编译器的关系
description: 分清 Python、PyTorch/MLX、TileLang/Triton、TVM/TensorIR/TIRX、CUDA/HIP/Metal 各自在哪一层，避免把不同抽象混为一谈。
---

# 第 1 课 语言、框架与编译器的关系

> 学新工具最容易晕的不是语法，而是“它和我已经会的东西是什么关系”。本课不写 GPU 代码，只把名字放回各自的位置。读完后你应该能画出本课那张分层图，并说出每一层负责什么。

## 1.1 这些名字不全是“语言”

| 名称 | 它是什么 | 在本课中的角色 |
| --- | --- | --- |
| Python | 通用编程语言 | 主机逻辑：准备数据、选参数、调用 kernel |
| PyTorch / MLX | 张量与机器学习框架 | 提供张量存储、参考计算、模型层能力 |
| TileLang | 嵌入 Python 的 kernel DSL + 编译器 | 描述 tile 级计算，编译成设备代码 |
| Triton | 另一个 kernel DSL + 编译器 | 同类工具，不是 TileLang 的底层 |
| TVM | 编译器基础设施 | TileLang 的 IR 与编译管线建立在它上面 |
| TensorIR / TIRX | TVM 的中间表示（IR） | 编译过程中的程序表示，不是用户日常书写层 |
| CUDA / HIP / Metal | GPU 平台与工具链 | 各自的设备编程生态；MSL 是 Metal 的着色语言 |

三个常见误解先纠正：

1. **“TileLang 是 Python 在 GPU 上跑”——不是。** kernel 函数体看起来像 Python，但被解析、编译成设备代码；任意 Python 代码（比如 `print` 一个列表、调用 requests）不会因此能在 GPU 上执行。
2. **“TileLang 基于 Triton”——不是。** 官方说明与论文都把它建立在 TVM 上；Triton 是并列的另一种 DSL。2025 年论文中与 Triton 的对比是当时的设计比较，不能当作对后来版本的永久优劣结论。
3. **“用了 TileLang 就不用 PyTorch”——不是。** 本课所有 demo 都用 PyTorch 的 MPS 张量做输入输出与参考校验；TileLang 只接管你写的那个 kernel。

## 1.2 一次调用经过的完整路径

以本课环境（Apple M1）为例：

```text
Python 主机程序
  │  准备 MPS 张量、选择 block 等编译时参数
  ▼
TileLang kernel（Python 语法书写）
  │  @tilelang.jit(target="metal")
  ▼
TileLang / TVM 编译管线
  │  解析 → IR（TensorIR 传统接口；上游 2026 年起迁移到 TIRX）
  │  → 调度与 lowering → 代码生成
  ▼
Metal Shading Language（MSL）源码
  │  torch.mps.compile_shader 编译
  ▼
Apple GPU 执行
```

两个要点：

- **IR 是编译器的内部表示**，你平时不写它；但调试时看 IR 和生成代码能回答“编译器到底把我的意图变成了什么”。第 10 课会真的打印生成的 MSL。
- **TIRX 迁移**（上游 PR #2216，2026-05-20 合入）说明 IR 接口在演进：读旧博客里的 `tvm.tir` 例子时，要对照自己安装的版本，不能混用。

## 1.3 与 MLX 的关系：同硬件，不同层

MLX 是 Apple 的数组/机器学习框架，抽象层级与 PyTorch 类似；TileLang 是 kernel DSL。它们都面向 Apple GPU，但：

- TileLang **不是**建立在 MLX 之上；本课 Metal 路线经由 TVM 代码生成与 `torch.mps.compile_shader`。
- 共享 Metal 硬件**不等于**张量自动互通；本课不做 MLX↔TileLang 的零拷贝互操作承诺。
- 本仓库第 8–13 章把 MLX 作为**同口径对照基线**：同一数学问题、同一计时口径，比较两条路线。

## 1.4 与 CUDA/HIP/MSL 的关系：目标不同，概念相通

TileLang 可按后端生成 CUDA、HIP 或 Metal 设备代码。选 `target="metal"` 时走 MSL；`threadIdx`、`threadgroup` 这些名字会出现在生成代码里（第 10 课可见）。

概念对应关系（帮助迁移，不是逐 API 翻译）：

| 概念 | CUDA | Metal | 本课 TileLang 写法 |
| --- | --- | --- | --- |
| 线程组 | thread block | threadgroup | `T.Kernel(..., threads=N)` |
| 组内线程编号 | `threadIdx.x` | `thread_position_in_threadgroup` | `T.get_thread_binding()` |
| 组内共享内存 | `__shared__` | `threadgroup` 地址空间 | `T.alloc_shared` |
| 线程私有累加 | 寄存器/local | thread 地址空间 | `T.alloc_fragment` |
| 组内同步 | `__syncthreads()` | `threadgroup_barrier` | `T.sync_threads()` |

**warp（32 线程）与 SIMD-group（Apple 通常 32 lane）不是同一个硬件概念的同义词**，本课只在需要时提 SIMD-group，不假设 warp 语义可平移。

## 1.5 练习与自检

1. 为什么“我会 Python”不等于“我会写 TileLang kernel”？还缺哪两块知识？
2. 同事说“把这段 Triton kernel 逐行翻译成 TileLang 就行”，这个思路的风险是什么？
3. 在 M1 上跑 `target="cuda"` 会发生什么层面的问题？

<details>
<summary>参考答案</summary>

1. 还缺：GPU 执行模型（线程组/索引/同步）与编译执行心智（哪些代码在主机、哪些被编译到设备）。语法相似掩盖了执行模型差异。
2. 两者执行抽象与内存表达不同，逐行翻译容易把 Triton 的 program/mask 语义错当成 TileLang 的线程组语义；应按数学问题重写，再分别校验。
3. 目标后端与硬件不匹配：没有 NVIDIA GPU 与 CUDA 运行时，编译/执行路径不可用；应选 `metal`。

</details>

**本课带走：** Python 管主机，框架管张量，TileLang 管 kernel 描述，TVM 管编译，Metal 管执行。下一课把环境装好并逐层验证。

---

*主要出处：[官方 README](https://github.com/tile-ai/tilelang)、[官方概览](https://tilelang.com/get_started/overview.html)、[TIRX 迁移 PR #2216](https://github.com/tile-ai/tilelang/pull/2216)、[MLX](https://github.com/ml-explore/mlx)。核对日期：2026-11-18。*
