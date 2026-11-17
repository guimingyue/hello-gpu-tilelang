---
layout: page
sidebar: false
aside: false
title: hello-gpu-tilelang
description: 用 TileLang（Metal 后端）在 Apple M1 上学 GPU 算子优化：六章受控实验、端到端墙钟、与 hello-gpu-metal 的 MLX 同口径对照。
---

<div class="hg-home vp-doc">

# hello-gpu-tilelang

<p class="hg-lede">
用 <strong>TileLang</strong> 的 tile 级 DSL 写 GPU 算子，<code>target="metal"</code> 直接 lower 成
Metal Shading Language，在 Apple M1 上实测。本仓库是
<a href="https://github.com/datawhalechina/hello-gpu">hello-gpu</a> 课程的第三份实现：
数学语义、正确性标准与三进程受控实验方法与
<a href="https://github.com/datawhalechina/hello-gpu">hello-gpu-metal</a> 完全一致，
两条实现路线改为 <strong>TileLang</strong> 与 <strong>MLX（引用对照）</strong>。
</p>

<div class="hg-cards">
  <a class="hg-card" href="./part2-kernels/">
    <strong>第 2 篇：经典算子与 Kernel 实战</strong>
    <span>第 8–13 章：逐元素、归约、Softmax、矩阵乘、Attention、Fused RMSNorm。每章 8 节、双路线标签页、逐轮比较图。</span>
  </a>
  <a class="hg-card" href="./part2-kernels/chapter8/">
    <strong>从第 8 章开始</strong>
    <span>向量加法：threadgroup 宽度与每线程元素数的受控扫描，建立「先正确、再计时、后归因」的闭环。</span>
  </a>
  <a class="hg-card" href="./part2-kernels/chapter13/">
    <strong>直接看综合实战</strong>
    <span>第 13 章 Fused RMSNorm：归约 + 逐元素融合，T=256/512/1024 扫描与 MLX 三配置对照。</span>
  </a>
</div>

<div class="hg-note">

**计时口径。** TileLang 的 Metal 后端不暴露 command buffer GPU 时间戳，本仓库统一采用
**端到端墙钟**（`time.perf_counter` 包住一次完整算子调用 + `torch.mps.synchronize`），
口径标记 `wall-clock-full-operator`。它与 hello-gpu-metal 中 **MLX 路线的墙钟同口径**，可以直接横向比较；
但**不**与 Metal 版 MSL 的 GPU 时间戳同口径。每章图中 TileLang 柱为本机实测，MLX 柱注明引用 hello-gpu-metal 同口径实验。

**负面结果不隐瞒。** Metal 后端的限制（无 float32 `atomic_add`、无 `simd_sum`/`tl.reduce`、
`T.gemm` 未用 SIMD-group 矩阵内建、threadgroup 宽度上限等）与实测到的 slowdown 都写进正文，
例如第 11 章 `tile=64` 的 46 倍变慢、第 10/13 章 `T=1024` 的行内归约退化。

</div>

## 怎样读这一篇

每章固定 8 节：**数据依赖 → 语义与正确性 → 成本模型 → 实现双路线 → 正确性/基准/剖析 → 两路线比较 → 负面结果与边界 → 复现与练习**。
第 4 节用 `<ImplementationTabs>` 提供 TileLang / MLX 两个标签页，切换会保留各自阅读位置；
其余各节两条路线共用。所有命令在仓库根目录用 `.venv/bin/python` 执行。

## 环境

```bash
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python tilelang numpy torch
.venv/bin/python code/part2-kernels/chapter8/smoke_metal.py
```

实验基线：Apple M1（applegpu_g13g）、macOS 27.0、Python 3.12.11、TileLang 0.1.15、
NumPy 2.5.4、PyTorch 2.15.0（MPS 可用）。Instruments / Xcode GPU Capture 为官方工具，
**未在本机实测**，各章只作方法说明。

</div>