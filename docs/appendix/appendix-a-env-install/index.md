---
title: "附录 A · 环境安装细节与常见坑"
description: "Hello GPU TileLang 附录 · Apple Silicon 上 TileLang/Metal 的安装、镜像与常见坑"
---

# 附录 A · 环境安装细节与常见坑

## A.1 依赖与平台

- Apple Silicon（arm64）+ macOS 12+；Python 3.12；
- `tilelang>=0.1.15`（Metal 后端）、`numpy`、`torch`（MPS 张量）；
- 本仓库 `pyproject.toml` 默认 BFSU 镜像，避免 PyPI wheel 下载超时。

```bash
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python \
  --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple tilelang numpy torch
```

## A.2 常见坑

1. **网络超时**：tilelang/torch wheel 较大，默认 30s 超时易失败；提高 `UV_HTTP_TIMEOUT` 或换镜像。
2. **eager vs lazy**：`out_idx` 仅 lazy 模式可用；eager 风格把输出张量作为参数传入。
3. **`kernel` 保留字**：MSL 中 `kernel` 是关键字，不能作函数名；教学默认 entry 已自动重命名。
4. **无 GPU 时间戳**：Metal 后端不暴露 GPUStartTime/GPUEndTime，统一用墙钟。
5. **threadgroup 上限**：`threads` 超过 `maxTotalThreadsPerThreadgroup`（M1 为 1024）会失败。

## A.3 验证

```bash
.venv/bin/python code/part2-kernels/chapter8/smoke_metal.py   # max_abs_error=0.0
```