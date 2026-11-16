---
title: "第17章 优化报告与独立复测：接受之后还要确认"
description: "Hello GPU TileLang 第17章 · 配对改进、独立复测冻结基线、旧复测不可归到当前实现"
---

# 第17章 优化报告与独立复测：接受之后还要确认

## 本章导读

> 接受候选只是单次配对比较的结论。本章用独立复测（fresh 进程、冻结基线）确认收益，并保证报告不把旧复测归到已改变的实现。

## 17.1 两种改进口径

- `improvementFraction`：单次配对（candidate vs incumbent）；
- `medianPairedImprovementFraction`：每次运行内配对改进中位数，再跨独立运行取中位数。

两者回答不同问题，不能互推。小幅收益要结合样本波动与重复次数解释。

## 17.2 独立复测

`final_retest(runs≥3)` 冻结 task/reference/baseline/candidate 到新目录，调 compare.py 跑独立进程。报告带四个 SHA-256；`report()` 校验它们与当前实现一致，否则报「旧复测」拒绝。

## 17.3 历史数据纪律

本仓库第 17 章引用的 AMD 历史搜索记录（RX 9070 XT + Triton）**仅作方法示例，非本机 Metal 实测**；TileLang 候选的收益以 `evidence/metal-rounds/` 与独立复测为准。

## 17.4 练习

1. 跑 final_retest 后改动 best.py，再 report()，观察「旧复测」拒绝；
2. 解释为什么复测要冻结基线而不是用当前 best；
3. 用 trajectory 与 retest 报告写一段「接受但收益不显著」的诚实结论。