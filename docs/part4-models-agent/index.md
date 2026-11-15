---
title: "第 4 篇 · 模型实战（对照）"
description: "Hello GPU TileLang 第4篇 · Core ML / MLX YOLO 与 MLX-LM，与 Metal 版对照"
---

# 第 4 篇 · 模型实战（对照）

模型层（YOLO / LLM）走 Core ML / MLX / MLX-LM，与 kernel DSL 选择解耦。本篇与 [hello-gpu-metal 第 18–19 章](https://github.com/datawhalechina/hello-gpu) 一致；其中自定义算子可替换为 TileLang Metal kernel。

> 完整模型实战章节见 Metal 版；本仓库聚焦 TileLang kernel 路线（第 2 篇）。