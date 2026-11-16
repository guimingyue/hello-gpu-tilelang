---
title: "第15章 工具与合同：评测器只认 MSL 与 launch(lib,…)"
description: "Hello GPU TileLang 第15章 · 候选源码白名单、隔离运行与权威计时"
---

# 第15章 工具与合同：评测器只认 MSL 与 launch(lib,…)

## 本章导读

> Agent 的可信度来自工具的确定性。本章讲评测器的静态合同与隔离运行，确保候选「能不能跑」在 GPU 计时之前就有可读答案。

## 15.1 静态合同（source_policy）

- 只允许 `import numpy / math`；禁止 from-import、exec/eval/open、下划线属性、类；
- 顶层只有文档串、import、字面量常量（`MSL`、`COMPILE_OPTIONS`）与函数定义；
- 必须有 `MSL` 字符串与 `launch(lib, <task.launchArguments>)` 入口。

## 15.2 隔离运行

评测在子进程跑冻结的 candidate/task/reference，超时杀进程组；正确性用固定非均匀输入全量核对，计时用 warmup + 多次重复。结果权威，模型不可篡改。

## 15.3 TileLang 的位置

TileLang 生成的 MSL 可直接作为 `MSL` 常量提交；`launch` 用 `lib.<kernel>(...)` 把调度写进评测器准备好的 command buffer。第 8–13 章的 TileLang 实现即候选池。

## 15.4 练习

1. 提交一个带 `from hello_metal import Device` 的候选，观察白名单拒绝；
2. 把 launch 签名改错，观察签名报错；
3. 解释为什么隔离运行不是安全沙箱而是「可读拒绝 + 进程隔离」。