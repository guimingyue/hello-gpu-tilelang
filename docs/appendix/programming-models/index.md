---
title: "附录 D · TileLang 与 MLX 的编程范式"
description: "用同一个向量加法理解 tile 级 DSL 与数组算子的分工差异"
---

# 附录 D · TileLang 与 MLX 的编程范式

## D.1 同一道题

`C[i] = A[i] + B[i]`，N=6，启动覆盖 8 个位置，最后两个越界。

## D.2 TileLang：tile 级显式分工

```python
with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
    for i in T.Parallel(256):
        idx = bx * 256 + i
        if idx < n:
            C[idx] = A[idx] + B[idx]
```

- 你显式声明 threadgroup 划分（`T.Kernel`）与线程内循环（`T.Parallel`）；
- 边界自己用 `if` 保护；
- lower 到 MSL 后与手写 MSL 同构，但用 Python 表达 tile 逻辑。

## D.3 MLX：数组算子交给框架

```python
c = a + b   # 惰性表达式，mx.eval 时框架决定分块与边界
```

- 不暴露 thread/threadgroup；框架自动分块与保护边界；
- 需要自定义时用 `mx.fast.metal_kernel` 回到线程视角。

## D.4 对照表

| 维度 | TileLang | MLX |
| --- | --- | --- |
| 主语 | tile / threadgroup | 整数组表达式 |
| 索引 | 显式 `idx` | 框架内部 |
| 边界 | 自己 `if` | 框架保证 |
| 启动 | `T.Kernel(grid, threads)` | `mx.eval` |
| 计时 | 墙钟（Metal 无 GPU 时间戳） | 墙钟 |

## D.5 接下来

回到[第 8 章](../../part2-kernels/chapter8/index.md)用受控实验比较两种路线。