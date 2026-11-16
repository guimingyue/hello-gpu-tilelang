---
title: "第19章 LLM 解码：MLX-LM 与量化"
description: "Hello GPU TileLang 第19章 · 用 MLX-LM 在 Apple Silicon 上做 decode，4bit/8bit 量化与逐 token 计时"
---

# 第19章 LLM 解码：MLX-LM 与量化

## 本章导读

> 把 Agent 与算子方法用到 LLM 解码。本章用 **MLX-LM** 在 Apple Silicon 上跑 decode，比较 4bit/8bit 量化，逐 token 计时。完整流程与 [hello-gpu-metal 第 19 章](https://github.com/datawhalechina/hello-gpu) 一致。

## 19.1 路线

- 转换：`python -m mlx_lm.convert --hf-path <model> -q --q-bits 4/8`；
- 生成：`mlx_lm.generate` / `stream_generate`，或 `mlx_lm.server`；
- 计时：prefill 与 decode 分离，TPOT（每 token 墙钟）与 decode t/s。

## 19.2 量化与带宽

decode 是带宽受限：每 token 读全部权重。4bit 权重字节减半 ⇒ 理论 t/s 提升，但反量化开销与 kernel 效率决定实际收益。第 7 章 Roofline 直接适用。

## 19.3 TileLang 的位置

反量化/逐元素/归约等 decode kernel 可用 TileLang 写并 lower 到 Metal，替换 MLX 默认实现做受控对比。历史 AMD（llama.cpp ROCm）数据仅作方法示例，非本机实测。

## 19.4 练习

1. 用 `bench.py` 跑 4bit/8bit，记录 TPOT 与 t/s；
2. 解释为什么 decode 带宽受限而 prefill 算力受限；
3. 把一个反量化 kernel 换成 TileLang，复测收益是否显著。