---
title: "附录 C · 低层视角：TileLang 之下的 Metal 提交"
description: "Hello GPU TileLang 附录 C · 理解 TileLang lower 出的 MSL 如何经 command buffer 提交"
---

# 附录 C · 低层视角：TileLang 之下的 Metal 提交

## C.1 TileLang → MSL → Metal

TileLang 把 `T.Kernel/T.Parallel/T.copy/T.gemm` lower 成 MSL kernel；运行时经 `MTLDevice → MTLLibrary → MTLComputePipelineState → MTLCommandBuffer → compute encoder → dispatchThreadgroups` 提交。理解这条链，才能读懂第 6 章 KernelTrace 与第 4 章墙钟口径。

## C.2 一次提交的状态

`makeCommandBuffer(notEnqueued) → encode → endEncoding → enqueue(enqueued) → commit(committed) → scheduled → completed`。墙钟计到 `waitUntilCompleted` 返回；GPU 时间戳（GPUEndTime−GPUStartTime）是另一口径，TileLang 后端不暴露。

## C.3 五种提交方式（Metal 版实测）

sync / async / batched(serial) / batched(concurrent) / ICB 的每 dispatch 墙钟依次下降（M1 实测约 175→37→3.3→1.3→1.2 µs）。小 kernel 合并提交收益大；有数据依赖时仍需 serial 或事件同步。详见 Metal 版附录 C 的 Swift 示例。

## C.4 练习

1. 用 `k.get_kernel_source()` 查看 TileLang lower 出的 MSL；
2. 解释为什么 batched concurrent 比 sync 快两个数量级；
3. 说明 ICB 适用场景与限制。