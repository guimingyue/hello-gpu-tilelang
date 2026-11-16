---
title: "第18章 模型实战：Core ML / MLX 跑 YOLO"
description: "Hello GPU TileLang 第18章 · 在 Apple Silicon 上用 Core ML / MLX 部署 YOLO，自定义算子可换 TileLang"
---

# 第18章 模型实战：Core ML / MLX 跑 YOLO

## 本章导读

> 模型层与 kernel DSL 解耦。本章在 Apple Silicon 上用 **Core ML / MLX** 跑 YOLO 推理；其中自定义算子可替换为 TileLang Metal kernel。完整流程与 [hello-gpu-metal 第 18 章](https://github.com/datawhalechina/hello-gpu) 一致。

## 18.1 路线

- 导出：PyTorch YOLO → ONNX → Core ML `.mlpackage`（coremltools），或 MLX 权重；
- 推理：Core ML（`compute_units=.all`）或 MLX；fp32/fp16 对比；
- 校验：与 PyTorch CPU 参考输出比对（容差内），再计时。

## 18.2 TileLang 的位置

检测头里的自定义算子（如 NMS 前的逐元素/归约）可用 TileLang 写并 lower 到 Metal，作为 Core ML 自定义层或 MLX `metal_kernel`。第 8–13 章的方法直接适用。

## 18.3 口径与边界

推理计时用端到端墙钟；Core ML 的 ANE/GPU 调度不透明，不做硬件归因。未在本机实测的数字明确标注。

## 18.4 练习

1. 用 `prepare_models.py` 导出 fp16 `.mlpackage` 并核对输出；
2. 对比 fp32/fp16 墙钟与精度；
3. 把一个逐元素后处理换成 TileLang kernel，复测。