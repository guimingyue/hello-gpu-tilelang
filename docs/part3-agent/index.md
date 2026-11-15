---
title: "第 3 篇 · 算子优化 Agent（对照）"
description: "Hello GPU TileLang 第3篇 · Agent 闭环方法与 Metal 版对照"
---

# 第 3 篇 · 算子优化 Agent（对照）

Agent 的「模型提出修改 → 工具校验/计时 → 证据决定接受」闭环与具体 kernel 语言无关。本篇方法与 [hello-gpu-metal 第 14–17 章](https://github.com/datawhalechina/hello-gpu) 一致；把候选 kernel 换成 TileLang（`target="metal"`）即可复用同一评测器合同（`MSL` 字面量 + `launch(lib,…)`）。

> 本仓库第 2 篇已给出 TileLang 候选的实测与负面结果，可直接作为 Agent 的候选池。完整 Agent 章节见 Metal 版。