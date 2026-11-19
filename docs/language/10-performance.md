---
title: 第 10 课 正确之后：计时、调参与生成代码
description: 用预编译、预热、同步墙钟建立可信计时，打印生成的 MSL 理解编译器映射，并知道何时进入实战篇。
---

# 第 10 课 正确之后：计时、调参与生成代码

> 前九课每个 demo 都先校验正确性。本课才问“快不快”，并回答两个一直悬着的问题：编译器把我的 `T.Parallel` 变成了什么？为什么第一次调用特别慢？

## 10.1 可信计时的四个动作

```python
compiled = kernel.compile(a, b, c)   # ① 显式编译，把编译开销移出计时
compiled(a, b, c); torch.mps.synchronize()   # ② 先跑一次并校验
for _ in range(warmup):              # ③ 预热：让缓存/调度进入稳态
    compiled(a, b, c)
torch.mps.synchronize()
for _ in range(repeat):              # ④ 重复采样，取中位数
    t0 = time.perf_counter()
    compiled(a, b, c)
    torch.mps.synchronize()
    samples.append((time.perf_counter() - t0) * 1e3)
```

四个动作各挡一类假结论：

| 动作 | 挡掉的假结论 |
| --- | --- |
| 预编译 | “第一次 148ms，这 kernel 真慢”——那是编译 |
| 先校验 | 计时一个算错的 kernel 毫无意义 |
| 预热 | 冷缓存/首次调度的偶发慢 |
| 同步 + 中位数 | 只测了提交没测执行；单次样本被噪声带偏 |

**计时口径**：本课测的是“预编译 kernel 调用 + MPS 同步”的端到端墙钟，不含张量分配与 `.cpu()` 读回。本仓库实战篇统一用这个口径（`wall-clock-full-operator`），因为 TileLang Metal 路线不直接暴露 command buffer GPU 时间戳。**不同口径的数字不能互比。**

## 10.2 跑起来

```bash
.venv/bin/python code/language/benchmark.py --n 65536 --block 128
.venv/bin/python code/language/benchmark.py --n 65536 --block 128 --source
```

本机 M1 实测（2026-11-19，10 次采样）：

```text
compile_or_cache_ms ≈ 148      ← 首次编译，不计入运行时间
median_ms ≈ 0.203              ← 65536 个元素加法 + 同步
min_ms ≈ 0.192, max_ms ≈ 0.583 ← 单次噪声可见，中位数更稳
correct: true
```

0.2ms 量级里包含提交与同步开销——65536 个 FP32 加法本身远小于此。**小算子的墙钟主要测的是开销，不是算力**；这也是为什么实战篇要用更大的形状与三进程受控实验。

<details>
<summary>完整源码</summary>

<<< @/../code/language/benchmark.py

</details>

## 10.3 看生成代码：编译器做了什么

`--source` 打印的 MSL 关键部分（本机 0.1.15 实测，节选）：

```cpp
[[kernel, max_total_threads_per_threadgroup(32)]] void add_kernel(
    const device float4* __restrict A [[ buffer(0) ]],
    const device float4* __restrict B [[ buffer(1) ]],
    device float4* __restrict C [[ buffer(2) ]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint3 threadIdx [[thread_position_in_threadgroup]], ...) {
  C[(blockIdx.x * 128 + threadIdx.x * 4) / 4] =
      A[(blockIdx.x * 128 + threadIdx.x * 4) / 4] +
      B[(blockIdx.x * 128 + threadIdx.x * 4) / 4];
}
```

三个值得注意的事实：

1. **`float4` 向量化**：你写的是标量加法，编译器把 4 个连续元素打包成 `float4` 一次运算。block=128、threads=32 → 每线程 4 个元素，正好一个 `float4`。
2. **边界检查消失了**：n=65536 能被 128 整除，编译器在编译期证明 `idx < n` 恒真，直接删掉分支。第 4 课的护栏没有白写——它让编译器**有机会**做这个证明。
3. **`threadIdx`/`blockIdx` 出现了**：第 1 课的对应表在这里兑现——TileLang 的 `bx`/线程绑定 lowering 成 Metal 的内建变量。

这就是“声明意图，编译器决定映射”的实证。想深入编译管线，见[附录 C](../appendix/appendix-c-metal-low-level/index.md)与官方 [IR Lower Trace 工具](https://tilelang.com/tools/lower_trace.html)。

## 10.4 调参的正确姿势

```bash
.venv/bin/python code/language/benchmark.py --n 65536 --block 32
.venv/bin/python code/language/benchmark.py --n 65536 --block 256
```

一次只改一个参数，记录形状、block、口径、样本数与中位数。本机单次实测（2026-11-19，各 10 样本）：block=32 → 0.248ms，block=128 → 0.203ms，block=256 → 0.240ms。block=128 看起来最好，但**这是单进程、各一轮的结果**，max 样本到 0.58ms 说明噪声明显；要下“128 更优”的结论，需要实战篇的三进程受控复测。**不要**拿不同配置各一次运行直接比——噪声可能大于差异。差异不显著时，诚实结论是“未观察到收益”，而不是挑一个好看的数。

本课的加法是访存受限的轻算子，block 扫描的差异很可能落在噪声里——这本身就是重要一课：**不是每个参数都值得调，先判断瓶颈在哪**（实战篇第 7 章 Roofline）。

## 10.5 语言入门到此为止

你现在会：声明张量与 kernel、分块与索引、条件与串行/并行循环、三级存储与同步、归约、分块矩阵乘、融合、可信计时与读生成代码。接下来：

- [第 0 篇课程地图](../part0-intro/chapter0/index.md)：进入 20 章主线；
- [第 8 章 Element-Wise](../part2-kernels/chapter8/index.md)：把本课加法做成逐轮受控实验；
- [第 5 章 可信计时](../part1-profiling/chapter5/index.md)：三进程、中位数与误差线的完整方法。

## 10.6 练习与自检

1. 为什么 `compile_or_cache_ms` 要单独报告而不是混进 samples？
2. 生成代码里 `if idx < n` 消失，是编译器 bug 吗？什么条件下它会保留？
3. 两次运行 median 差 5%，能说新配置快 5% 吗？

<details>
<summary>参考答案</summary>

1. 编译是一次性成本（且有缓存），运行时间是重复成本；混在一起会高估 kernel 本身耗时，也无法比较缓存命中后的真实开销。
2. 不是。n 能被 block 整除时条件编译期恒真，删除是合法优化；n 不整除（如 33）时保护会保留。
3. 不能。需要看样本分布、重复实验与口径一致性；5% 可能在噪声范围内，应报告“未观察到显著差异”或增加受控重复。

</details>

**本课带走：** 预编译、预热、同步、中位数是可信计时的底线；生成代码是理解编译器的窗口；调参一次一个变量，结论由证据给出。语言入门完成，去实战篇把每个算子从正确做到快。
