---
title: 第 0 课 TileLang 的来历、目的与用途
description: 用可核对的时间线理解 TileLang 为什么被创造出来、现在用在哪些场景、什么时候不该用它。
---

# 第 0 课 TileLang 的来历、目的与用途

> 本课不写代码。目标是在动手前回答三个问题：这门语言解决什么别人没解决好的问题？它从哪里来？今天谁在用它、用来做什么？所有历史节点都给出可核对的出处。

## 0.1 一个具体的痛点

在 PyTorch 里写 `C = A @ B`，框架替你选好矩阵乘实现，你不必安排线程和内存。但如果要做的是“反量化之后立刻做矩阵乘，再顺手加一个偏置”这样的**融合算子**，现成接口往往不够：拆成三个算子，中间结果就要在显存里写一遍、读一遍，而这正是很多推理 kernel 的主要开销。

直接写 CUDA、HIP 或 Metal Shading Language 能拿到全部控制权，但线程分工、共享内存、同步都要自己管，一个 kernel 几百行，换一代硬件还要重调。

**TileLang 想缩短这段距离**：用接近 Python 的语法描述“一块数据怎样加载、计算、写回”，把线程映射、布局、指令选择、流水线这些调度问题尽量交给编译器，同时保留需要时下沉到低层的通道。论文把这件事概括为：**把数据流（dataflow）与调度空间（scheduling space）解耦**。

注意它不是什么：

- 不是“不懂硬件也自动最快”的魔法——官方文档明确说，完全硬件无关的 Beginner Level 接口**尚未完整实现**；本课教的是现有 tile 接口里适合入门的子集。
- 不是训练框架——它不替代 PyTorch/MLX 的模型、自动求导与训练循环。
- 不是 Triton 的新版本——两者是并列的 kernel DSL，编译路径不同（第 1 课展开）。

## 0.2 一条可核对的时间线

| 日期 | 事件 | 出处 |
| --- | --- | --- |
| 2025-01-20 | 官方 README 记录项目开源 | [官方 README](https://github.com/tile-ai/tilelang) |
| 2025-02-12 | 发布 v0.1.0（首个 v0.1 公开版本；此前已有 v0.0.1） | [v0.1.0 release](https://github.com/tile-ai/tilelang/releases/tag/v0.1.0) |
| 2025-04-24 | 论文首次提交 arXiv（4-27 更新 v2） | [arXiv:2504.17577](https://arxiv.org/abs/2504.17577) |
| 2025-10-07 | Apple Metal 后端合入（PR #799） | [PR #799](https://github.com/tile-ai/tilelang/pull/799) |
| 2026 年 | 多后端重构、TIRX 迁移、调试工具与更多硬件路径持续演进 | 官方 README 新闻与 [PR #2216](https://github.com/tile-ai/tilelang/pull/2216) |

论文全名 *TileLang: A Composable Tiled Programming Model for AI Systems*。**论文日期不是项目诞生日期**——开源在前，论文在后。

**谁做的？** 官方致谢：早期实现主要由 LeiWang1999、chengyupku、nox-410 完成，受北京大学 Zhi Yang 教授指导；部分工作在微软研究院实习期间开展，得到 Lingxiao Ma、Yuqing Xia、Jilong Xue、Fan Yang 的建议与支持。论文署名单位包括北京大学、微软研究院与帝国理工学院。所以它是学界与工业界研究者的合作成果，**不宜简单称为“微软的产品”**。编译基础设施建立在 [Apache TVM](https://tvm.apache.org/) 之上。

## 0.3 今天在哪些场景使用

要区分两类证据：**官方示例**证明“能用来写这类 kernel”，**模型源码集成**证明“真的被接进了某个公开实现”。

**官方示例覆盖的工作负载**（见[官方 examples](https://github.com/tile-ai/tilelang/tree/main/examples)）：

- GEMM、分组 GEMM、FP8/反量化 GEMM、块缩放 GEMM；
- FlashAttention、Flash Decoding、稀疏/线性注意力；
- DeepSeek MLA 解码等模型算子；
- 自动调优、布局可视化、编译过程调试等研究工具。

**一个可核对的源码级集成**：DeepSeek-V3.2-Exp 的公开参考推理实现里，`inference/kernel.py` 用 TileLang JIT 写了激活量化、FP8 GEMM 与 FP8 索引打分 kernel，`inference/model.py` 在模型前向路径中调用它们（[kernel.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/kernel.py)、[model.py](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py)，固定 commit 核对）。这说明 TileLang 被用于真实模型代码；但**不能据此推断生产部署规模，也不能推断这些 FP8 kernel 能在 M1 上跑**——它们面向支持 FP8 的 NVIDIA 硬件。

对本课读者更贴近的场景：**在 Apple Silicon 上学习、实验自定义算子**——这正是本仓库第 8–13 章在做的事。

## 0.4 什么时候不必用它

| 情况 | 更省力的选择 |
| --- | --- |
| 常规模型训练/推理，框架算子够快 | PyTorch / MLX 直接写 |
| 只需要一次性脚本 | NumPy / 框架算子 |
| 需要极致控制且团队已有 CUDA 专家 | 手写 CUDA/HIP/MSL 仍是合理选项 |
| 想学 GPU 编程、需要控制分块与融合、又希望少写样板 | TileLang 这类 tile DSL |

判断标准是**有没有具体需求或测出来的瓶颈**，而不是“新工具一定更好”。

## 0.5 练习与自检

1. “TileLang 2025 年 4 月诞生”这句话错在哪？
2. 看到“某模型官方仓库用了 TileLang”，能推出“该模型线上服务跑在 TileLang 上”吗？
3. 你的任务只是把两个现成算子串起来跑通，应该先做什么？

<details>
<summary>参考答案</summary>

1. 开源记录是 2025-01-20，4 月 24 日是论文首次提交日期；且更早还有 v0.0.1。
2. 不能。公开参考实现 ≠ 生产部署；还要看部署硬件与后端支持。
3. 先用框架算子跑通并测量；只有发现具体瓶颈或表达缺口，才值得引入手写 kernel 的维护成本。

</details>

**本课带走：** TileLang 是 2025 年初开源、建立在 TVM 上的 tile 级 kernel DSL，目标是让数据流与调度解耦；它有真实的模型源码集成案例，但能力随后端与硬件而变。下一课把它放进你已知的技术栈里。

---

*资料核对日期：2026-11-18。上游持续演进，引用链接以官方仓库与论文页为准。*
