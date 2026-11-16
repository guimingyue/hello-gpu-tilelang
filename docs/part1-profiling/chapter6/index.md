---
title: "第6章 剖析：KernelTrace 与 Instruments 的边界"
description: "Hello GPU TileLang 第6章 · 用 KernelTrace 核对启动规模，明确 TileLang 墙钟口径与硬件计数器的边界"
---

# 第6章 剖析：KernelTrace 与 Instruments 的边界

## 本章导读

> 计时回答「多久」，剖析回答「为什么」。本章用 `hello_metal.KernelTrace` 核对 kernel 的启动规模与时间窗口，并明确 TileLang 墙钟口径下哪些结论能下、哪些不能。

## 6.1 KernelTrace 能看什么

`trace_tilelang.py` 用 stage-boundary counter sampling，每个 dispatch 一个 compute encoder，记录 grid/threadgroup/起止时间。它能核对：

- 启动的 threadgroup 数与线程数是否符合 `T.Kernel` 声明；
- 各 dispatch 的时间窗口与顺序（warmup 与 run 分离）。

它**不能**给出占用率、缓存命中率、物理内存流量——那些需要硬件计数器。

## 6.2 口径边界

TileLang 的 bench 是墙钟；KernelTrace 是另一套采样窗口。两者**不能相减**估算主机开销。第 8–13 章的 trace 表只用于核对启动规模，不用于性能归因。

## 6.3 Instruments / Xcode

Apple 的 Instruments（Metal System Trace / GPU Counters）与 Xcode GPU Capture 能提供硬件视角。**本教程未在本机实测**，只作方法说明；需要硬件归因时请在你的机器上采集并标注。

## 6.4 练习

1. 跑 `trace_tilelang.py`，核对 grid 与 `T.ceildiv(n, T)` 一致；
2. 解释为什么 trace 中位数与 bench 中位数不同口径；
3. 列出「跨步访存变慢」这一结论目前有的证据与缺的证据。