---
title: 第 6 课 内存、搬运与线程协作
description: 用 global→shared→fragment 的拷贝与 tile 内反转，理解三种存储、T.copy 的块语义与同步的必要性。
---

# 第 6 课 内存、搬运与线程协作

> 前面的 kernel 都是“读一个、算一个、写一个”。本课的问题升级：**把每 32 个元素的小段在段内反转**。反转需要线程读到别的线程负责的数据——这逼出共享内存与同步两个概念。

## 6.1 三种存储，三种角色

| TileLang 写法 | 对应 Metal 概念 | 角色 | 谁能访问 |
| --- | --- | --- | --- |
| 函数参数 `T.Tensor` | `device` 缓冲 | 全局内存，容量大、延迟高 | 所有线程组 |
| `T.alloc_shared` | `threadgroup` 内存 | 组内共享的暂存区，快但小 | 本组线程 |
| `T.alloc_fragment` | 线程私有寄存/本地存储 | 每线程自己的累加器 | 仅所属线程（布局由编译器管） |

关键直觉：**fragment 不是“小数组”那么简单**。它的元素到线程的映射由编译器决定，你不能假设“线程 5 能直接读 fragment[7]”。需要跨线程交换数据时，走 shared。

## 6.2 先看不需要交换的版本：纯拷贝

```python
shared = T.alloc_shared((BLOCK,), T.float32)
local = T.alloc_fragment((BLOCK,), T.float32)
T.copy(A[bx * BLOCK], shared)      # global → shared
T.copy(shared, local)              # shared → fragment
T.copy(local, B[bx * BLOCK])       # fragment → global
```

`T.copy(A[bx * BLOCK], shared)` 里的 `A[bx * BLOCK]` 是**块首地址**，不是读一个标量：拷贝范围由目标 `shared` 的形状 `(BLOCK,)` 推断。这是 TileLang 的块语义——你描述“搬哪一块”，编译器安排线程怎样搬。

这个版本数学上等于恒等拷贝，但完整走过了三级存储，是理解后续一切的脚手架。

## 6.3 需要交换的版本：段内反转

反转要求位置 `i` 读到位置 `BLOCK-1-i` 的值，而那个值可能属于别的线程。做法：

```python
T.copy(local, shared)              # 先把各自的数据放回共享区
T.sync_threads()                   # 等全组都写完
for i in T.Parallel(BLOCK):
    if bx * BLOCK + i < n:
        B[bx * BLOCK + i] = shared[T.min(BLOCK, n - bx * BLOCK) - 1 - i]
```

两个要点：

1. **`T.sync_threads()` 是组内屏障**：所有线程到齐才继续。少了它，线程 0 可能在线程 31 还没写完时就去读——结果随调度变化，是典型的竞态。屏障只同步**本线程组**，不同组之间没有这种保证。
2. **尾部用主机补零 + 段内有效长度**：输入补零到 32 的倍数，设备端无条件读合法位置；反转范围用 `min(BLOCK, n - bx*BLOCK)` 限制在有效元素内，补零位置写 0。这是第 4 课 `if` 保护之外的另一种尾部策略。

## 6.4 跑起来

```bash
.venv/bin/python code/language/memory_copy.py
```

本机 M1 实测（2026-11-19）：长度 1、7、32、33、64、65 × 拷贝/反转共 12 个用例全部精确通过；长度 35 反转后前 4 个为 `[31, 30, 29, 28]`、末 3 个为 `[34, 33, 32]`（第二段只有 3 个有效元素，段内反转）。

<details>
<summary>完整源码</summary>

<<< @/../code/language/memory_copy.py

</details>

## 6.5 一个真实的编译坑：MSL 保留字

这份代码最初把 fragment 缓冲命名为 `fragment`，结果 Metal 编译报错：

```text
error: expected unqualified-id
  thread float fragment[1];
```

`fragment` 是 MSL 保留字（用于片元函数修饰）。TileLang 把缓冲名带进生成代码，于是冲突。**教训：kernel 里的标识符会出现在目标语言里，避开目标语言关键字**；改名为 `local` 后一次通过。这也是第 2 课说的“第三层失败”的真实样本。

## 6.6 改一个条件

把 `reverse=True` 改成 `False` 复跑，确认恒等拷贝路径；再试着**删掉 `T.sync_threads()`** 之前先回答：结果可能怎样？（不建议真跑——竞态结果不可复现，观察一次“碰巧正确”没有意义。）正确做法是理解屏障保护的是“写共享区 → 读共享区”这条跨线程依赖。

## 6.7 练习与自检

1. 为什么反转不能直接在 fragment 上跨线程索引？
2. `T.copy(A[bx*BLOCK], shared)` 搬了多少个元素？谁决定的？
3. 主机补零策略下，设备端为什么可以不写 `if idx < n` 的读保护？写保护还需要吗？

<details>
<summary>参考答案</summary>

1. fragment 的元素-线程布局由编译器决定，跨线程直读没有语义保证；shared 才是组内共享的交换区。
2. `BLOCK=32` 个，由目标缓冲形状推断；源表达式只给首地址。
3. 补零后所有候选位置都是合法内存，读不会越界；但**写回全局输出仍要区分有效/补零位置**（本例补零位置写 0，由调用方裁剪），否则会把填充值当成结果。

</details>

**本课带走：** global 大、shared 快而共享、fragment 私有；跨线程交换走 shared 加屏障；标识符会进入生成代码。下一课用这套协作机制做归约。
