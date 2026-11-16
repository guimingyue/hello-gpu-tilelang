---
title: "第16章 交互式 Agent：手动循环与在线规划"
description: "Hello GPU TileLang 第16章 · step/call 手动闭环与 run_online 的模型规划，证据状态 truthful"
---

# 第16章 交互式 Agent：手动循环与在线规划

## 本章导读

> 同一套工具，两种驱动：手动 `step()` 教学循环，或 `run_online()` 让模型规划。本章讲两者如何共享 pending-source 守卫与 truthful 事件状态。

## 16.1 手动循环

`lab.step("bench_kernel", source=...)` 产生 pending 候选；`step("accept_candidate", source=相同字节)` 才裁决。字节不一致会被 pending_accept 拒绝，防止「偷换候选」。

## 16.2 在线规划

`run_online()` 把工具表交给模型，但只暴露 compile/bench/accept/profile 等权威工具，不暴露 setup_task/convert_kernel；每次模型调用计数，事件状态记录 pendingCandidate 的真实变化。

## 16.3 truthful 状态

finish() 的 complete 要求「有裁决轮次且无 pending、无未裁决活动」。错误候选被拒绝也算完成裁决，但不声称改进。

## 16.4 练习

1. 用 step 跑「bench 错候选 → accept」，观察 accepted=False；
2. 用 run_online（fixture 模型）核对 llmCalls 与轨迹一致；
3. 解释为什么 accept 的 source 必须与 bench 的字节一致。