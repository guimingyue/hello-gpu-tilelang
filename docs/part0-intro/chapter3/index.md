---
title: "第3章 内存层次：统一内存、threadgroup 内存与寄存器"
description: "Hello GPU TileLang 第3章 · Apple 统一内存下的数据搬运与 T.alloc_shared / T.alloc_fragment"
---

# 第3章 内存层次：统一内存、threadgroup 内存与寄存器

## 本章导读

> Apple Silicon 用统一内存：CPU/GPU 共享物理内存，没有独立显存拷贝。但 GPU 内部仍有寄存器 → threadgroup 内存 → 系统缓存 → 统一内存的层次。本章讲清数据在哪、搬运代价几何。

## 3.1 层次与代价

| 层 | TileLang | 特点 |
| --- | --- | --- |
| 寄存器 | `T.alloc_fragment` | 每线程私有，最快 |
| threadgroup 内存 | `T.alloc_shared` | 组内共享，≤32 KB（M1），需屏障 |
| 系统缓存 | 硬件管理 | 跨组复用 |
| 统一内存 | 输入/输出张量 | 带宽 ~68 GB/s（M1 标称） |

## 3.2 分块复用

矩阵乘把 A/B 的 tile 先搬进 threadgroup 内存再复用，减少统一内存往返：

```python
a_shared = T.alloc_shared((T, T), dtype)
b_shared = T.alloc_shared((T, T), dtype)
acc = T.alloc_fragment((T, T), accum_dtype)
for k in T.Pipelined(T.ceildiv(K, T)):
    T.copy(A[by * T, k * T], a_shared)
    T.copy(B[k * T, bx * T], b_shared)
    T.gemm(a_shared, b_shared, acc)
```

`T.copy` 编排全局→shared 搬运；`T.gemm` 在 shared 上累加。第 11 章实测 tile=16/32/64 的复用收益与 tile64 的负面结果。

## 3.3 统一内存的误区

统一内存**不等于零搬运**：GPU 仍要把数据读进寄存器/shared。有效带宽 = 逻辑字节 / 时间，是算法口径，不是硬件流量。

## 3.4 练习

1. 用 `T.alloc_shared` 改写第 8 章加法（无复用收益，观察开销）；
2. 解释为什么 threadgroup 内存上限 32 KB 约束了 tile 尺寸；
3. 对照第 7 章 Roofline，说明加法为何带宽受限。