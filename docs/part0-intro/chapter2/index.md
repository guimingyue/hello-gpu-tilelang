---
title: "第2章 执行层次：thread / SIMD-group / threadgroup / grid"
description: "Hello GPU TileLang 第2章 · 用 T.Kernel 与 T.Parallel 理解 Apple GPU 的四级执行层次"
---

# 第2章 执行层次：thread / SIMD-group / threadgroup / grid

## 本章导读

> TileLang 把 Apple GPU 的执行层次暴露为 `T.Kernel(grid, threads=...)` 与 `T.Parallel(...)`。本章把四级层次和 M1 的真实参数对上，建立「当前线程算哪部分数据」的图景。

## 2.1 四级层次

| 层 | TileLang 表达 | Apple M1 参数 |
| --- | --- | --- |
| grid | `T.Kernel(T.ceildiv(n, T), threads=T)` 的组数 | 由 n 与 T 决定 |
| threadgroup | `threads=T` | ≤1024 线程 |
| SIMD-group | 硬件锁步执行 | 宽度 32（threadExecutionWidth） |
| thread | `T.Parallel` 展开的单个迭代 | 一个 lane |

```python
with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
    for i in T.Parallel(256):
        idx = bx * 256 + i
        if idx < n:
            C[idx] = A[idx] + B[idx]
```

`bx` 是 threadgroup 编号；`T.Parallel(256)` 把组内 256 个线程展开；`idx` 是全局线程下标。

## 2.2 与 MSL 的对应

TileLang lower 出的 MSL 用 `threadgroup_position_in_grid`（=bx）、`thread_position_in_threadgroup`（=i）、`thread_position_in_grid`（=idx）。理解这层映射，才能读懂第 8–13 章的负面结果（如 threadgroup 过大导致并发组数锐减）。

## 2.3 边界

`if idx < n` 保护最后一组越界线程。去掉它，非整除长度会越界写——第 8 章用 guard 区实测验证。

## 2.4 练习

1. 把 `threads` 从 256 改 512/1024，观察墙钟与并发组数；
2. 用 `hello_metal.device_report()` 核对 threadExecutionWidth 与 maxTotalThreadsPerThreadgroup；
3. 解释为什么 SIMD-group 宽度 32 决定了「同轮相邻 lane 访问相邻地址」最重要。