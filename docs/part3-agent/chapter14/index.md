---
title: "第14章 Agent 入门：模型提修改，工具给证据"
description: "Hello GPU TileLang 第14章 · 把算子优化变成「提出候选→校验计时→证据裁决」的闭环"
---

# 第14章 Agent 入门：模型提修改，工具给证据

## 本章导读

> 前 two 篇你手动做「改 kernel → 校验 → 计时 → 决策」。本章把这个闭环交给 Agent：模型负责提出候选，工具负责给出可信证据，裁决由确定性规则完成。

## 14.1 闭环

模型 → `compile_kernel` / `bench_kernel`（权威计时）→ `accept_candidate`（按阈值晋升）→ 轨迹归档。模型不直接碰 GPU，也不自己报性能数字。

## 14.2 TileLang 候选合同

候选是一个 Python 文件：顶层只有 `MSL` 源码字符串（或 TileLang 生成入口）与 `launch(lib, ...)`；只允许 import numpy/math。评测器先 AST 白名单检查，再子进程隔离运行。详见第 15 章。

## 14.3 证据先于结论

`accept_candidate` 无论成败都写轨迹；只有接受才更新 best.py。错误候选（wrong_answer）不产生性能数字，更不会被画成「提速」。

## 14.4 练习

1. 用 `hello_gpu.agentlab.Lab` 跑一个故意写错的候选，观察 wrong_answer；
2. 解释为什么「编译通过」不等于「正确」；
3. 读 trajectory.jsonl，区分 accepted 与 status。