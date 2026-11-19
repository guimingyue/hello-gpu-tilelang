---
title: 第 3 课 第一个 kernel：十个数相加
description: 从一维张量出发，在 Apple GPU 上执行第一个 TileLang 加法 kernel，并逐段理解定义、编译、调用与校验。
---

# 第 3 课 第一个 kernel：十个数相加

> 本课不追求速度。结束时，你要能回答：数据在哪里创建？哪段程序交给 GPU？怎样证明每个输出都算对了？先完成[环境检查](./02-environment.md)。

## 3.1 先写出正确答案

输入两组数，逐个相加：

```text
A = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
B = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
C = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

每个 `C[i]` 只依赖 `A[i]` 和 `B[i]`，不需要等待另一个输出。这是适合并行处理的最小问题，也方便我们手工校验。

普通 Python 可以写成：

```python
a = list(range(10))
b = [10] * 10
c = [a[i] + b[i] for i in range(10)]
print(c)
```

这段代码由 Python 在 CPU 上执行。下面换成 GPU kernel，但数学问题不变。

## 3.2 先认识三个名词

| 名词 | 本例的意思 | 不要混淆 |
| --- | --- | --- |
| Tensor（张量） | 存放十个数的一维数组 | 本课不要求懂深度学习 |
| shape / dtype | `(10,)` 表示形状，`float32` 表示元素类型 | 形状不是元素的值 |
| kernel（核函数） | 交给 GPU 执行的计算程序 | 不是整个 Python 脚本 |

`float32` 每个元素占 4 字节；这里每个数组的数据部分是 40 字节。`(10,)` 中的逗号表示 Python 的单元素元组，不是二维矩阵。

## 3.3 跑通一次完整程序

在仓库根目录运行：

```bash
.venv/bin/python code/language/vector_add.py --n 10 --block 4
```

本机 Apple M1、TileLang 0.1.15 实测的关键输出（2026-11-18，省略编译日志）：

```text
n=10, block=4, groups=3, skipped=2
output (first 16): [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
Vector add: PASS
```

`PASS` 来自逐元素断言，不是“程序没有报错”就打印成功。输出先填 `NaN`，遗漏写入的位置不会悄悄成为零而蒙混过关。

<details>
<summary>完整可运行源码：与命令使用同一份文件</summary>

<<< @/../code/language/vector_add.py

</details>

## 3.4 读懂 kernel 定义

先看外层：

```python
def make_add(n, block=4):
    @tilelang.jit(target="metal")
    def add(A, B, C):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        C: T.Tensor((n,), T.float32)
```

- `make_add` 是普通 Python 函数，它创建一个针对长度 `n`、块大小 `block` 的 kernel。
- `@tilelang.jit` 把函数交给 TileLang 编译，`target="metal"` 选择 Apple GPU 路线。
- 函数体里的 `T.Tensor` 声明让编译器知道输入输出的形状和类型，不是在这里申请三个数组。
- 本例采用显式输出：调用前准备 `C`，kernel 写入它；不要把返回值当作计算结果。

接着看分工：

```python
with T.Kernel(T.ceildiv(n, block), threads=32) as bx:
    for i in T.Parallel(block):
        idx = bx * block + i
        if idx < n:
            C[idx] = A[idx] + B[idx]
```

先把它读成一句话：**把数组按块分给若干线程组；每组处理自己负责的一段；有效位置逐个相加。**

`bx` 是线程组编号，`i` 是块内的逻辑迭代索引，`idx` 才是整个数组中的位置。`T.Parallel` 声明可并行的迭代，由编译器映射到执行线程，不等于 Python 在 CPU 上执行 `range`。

本课只需理解分工。为什么用 3 组？为什么有 2 个候选位置要跳过？下一课会手算。4 个元素、32 个线程的设置为方便观察，并非性能建议。

## 3.5 主机怎样准备并调用它

```python
a = torch.arange(10, dtype=torch.float32, device="mps")
b = torch.full_like(a, 10)
c = torch.full_like(a, float("nan"))
make_add(10)(a, b, c)
torch.mps.synchronize()
```

`torch.arange` 创建张量，`device="mps"` 选择 PyTorch 的 Apple GPU 设备。这里 PyTorch 负责张量存储和外围调用，TileLang 负责我们写的加法 kernel；两者不是谁取代谁。

首次调用可能包含编译，后续可复用编译结果。主机发出 GPU 工作后，并不意味着工作已经全部完成；`synchronize()` 等待完成，方便明确校验和计时的边界。

```python
torch.testing.assert_close(c.cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)
```

`.cpu()` 把结果变成 CPU 侧张量。参考答案在 CPU 上用同一数学公式计算；本例小整数及其结果能被 FP32 精确表示，所以用零容差。**这不是后续所有浮点算子的默认规则**，归约、指数和矩阵乘会单独讨论误差。

## 3.6 改一个条件

只把长度改成 33，不修改公式：

```bash
.venv/bin/python code/language/vector_add.py --n 33 --block 4
```

先预测：应有 9 组，36 个候选位置，跳过 3 个；33 个有效输出仍应全部正确。脚本只显示前 16 个输出，但断言检查整个数组，不是只验打印出来的部分。

不要为了演示越界而删除保护并真的访问非法内存；下一课用图示观察没有保护时会访问哪些位置。

## 3.7 练习与自检

1. 为什么 `T.Tensor((n,), T.float32)` 不是 `torch.empty` 的替代品？
2. 为什么 `C` 先填 `NaN` 比填零更容易发现漏算？
3. 如果输出正确，是否已经证明这个 kernel 比 PyTorch 快？

<details>
<summary>参考答案</summary>

1. 前者描述传入缓冲区的类型和形状，后者在运行时创建存储；本例的存储由 PyTorch 事先准备。
2. 零可能恰好就是某个位置的正确答案，`NaN` 遗留会使当前断言失败。它不能代替完整的边界检查，但能发现未写输出。
3. 没有。我们尚未计时，而且只有十个元素，提交开销可能比计算本身更显著。

</details>

**本课带走：** 数据由主机准备，kernel 交给编译器与 GPU，正确性由独立参考检查。下一课拆开这段程序最关键的索引公式。
