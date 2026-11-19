---
title: 第 2 课 环境准备：从 Python 到 GPU
description: 在 Apple Silicon 上准备 TileLang Metal 环境，分层验证 Python、MPS、TileLang 编译执行，并理解每层失败意味着什么。
---

# 第 2 课 环境准备：从 Python 到 GPU

> 本课只做一件事：让你有底气回答“我的机器能不能跑后面的 demo”。验证分三层，一层失败就停下来修，不要带着坏环境往后学。不需要懂 ROCm、CUDA 或编译器原理。

## 2.1 本课的实验基线

| 项目 | 基线 |
| --- | --- |
| 硬件 | Apple M1（8 核 GPU、16 GiB 统一内存） |
| 系统 | macOS 27.0（arm64） |
| Python | 3.12.11 |
| TileLang | 0.1.15 |
| PyTorch | 2.15.0（MPS 可用） |
| NumPy | 2.5.4 |

这些是本教程实测使用的组合（2026-11-18 核对）。**不是“只有这些版本能跑”的声明**：其他 Apple Silicon 代际与更新版本可能可用，但性能数字与个别 API 行为需要自行复测。M2/M3/M4/M5 的读者可参考[附录 B](../appendix/appendix-b-switch-gpu/index.md)；上游新增的 M5 协作张量等特性不属于本课基线。

## 2.2 安装

在仓库根目录：

```bash
uv venv --python 3.12 .venv
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python tilelang numpy torch
```

仓库 `pyproject.toml` 已配置 BFSU PyPI 镜像，默认 PyPI 超时的网络环境可直接用上述配置。想严格复现实测组合，可固定版本：

```bash
UV_HTTP_TIMEOUT=600 uv pip install --python .venv/bin/python \
  tilelang==0.1.15 numpy==2.5.4 torch==2.15.0
```

不需要 sudo，不需要改系统 Python，也不需要安装 Xcode 完整工具链来跑本课 demo（`torch.mps.compile_shader` 使用系统 Metal 编译器）。

## 2.3 三层验证

**第一层：Python 与包版本。**

```bash
.venv/bin/python code/language/check_env.py
```

本机实测输出（2026-11-18）：

```json
{
  "python": "3.12.11",
  "platform": "macOS-27.0-arm64-arm-64bit",
  "machine": "arm64",
  "tilelang": "0.1.15",
  "numpy": "2.5.4",
  "torch": "2.15.0",
  "mps_built": true,
  "mps_available": true
}
MPS tensor: PASS; 下一步运行 vector_add.py 检查 TileLang 编译与执行。
```

脚本同时做了一次 MPS 张量加法并校验——**包能 import 不等于 GPU 能用**，所以版本检查后必须真的算一次。

**第二层：MPS 张量计算。** 即上面的 `MPS tensor: PASS`。失败说明 PyTorch 的 Apple GPU 路径有问题（系统版本、torch 安装），与 TileLang 无关，先修这层。

**第三层：TileLang 编译并执行 kernel。**

```bash
.venv/bin/python code/language/vector_add.py --n 10 --block 4
```

看到 `Vector add: PASS` 才算三层全过。这一层失败而前两层通过，问题在 TileLang 安装或 Metal 编译路径。

<details>
<summary>环境自检脚本源码</summary>

<<< @/../code/language/check_env.py

</details>

## 2.4 失败时先收集什么

| 症状 | 先看什么 |
| --- | --- |
| `mps_available: false` | macOS 版本、torch 是否官方 arm64 轮子、是否 Rosetta 下的 x86 Python |
| import tilelang 报错 | Python 是否 3.12、是否装进了别的解释器（用 `.venv/bin/python -m pip show tilelang` 核对） |
| 编译期 SyntaxError / Metal 报错 | 完整报错文本、TileLang 版本；第 6 课会见到一个真实例子（MSL 保留字冲突） |
| 结果不对但不报错 | 先确认输出张量在 MPS 上、调用后做了 `torch.mps.synchronize()` |

排错时保留**完整报错 + 版本信息 + 最小复现脚本**，比“跑不起来”三个字有用得多。更细的安装坑见[附录 A](../appendix/appendix-a-env-install/index.md)。

## 2.5 两个心智准备

1. **主机与设备是两台“机器”。** Python 在 CPU 上跑；`device="mps"` 的张量在 GPU 可访问的内存里；kernel 在 GPU 上执行。数据要显式搬运（`.to("mps")` / `.cpu()`），计算完成要显式等待（`synchronize()`）。
2. **首次调用含编译。** 第一次调用 kernel 可能花几百毫秒编译，之后复用缓存。所以“第一次好慢”不是性能结论——第 10 课专门处理这件事。

## 2.6 练习与自检

1. 为什么版本检查之后还要做一次 MPS 加法？
2. `check_env.py` 通过但 `vector_add.py` 失败，问题最可能在哪一层？
3. 统一内存的 Mac 上，为什么还需要 `.to("mps")` 和 `.cpu()`？

<details>
<summary>参考答案</summary>

1. import 成功只说明包装上了；MPS 可用性依赖系统与运行时，必须实际算一次才知道。
2. TileLang 编译/执行层（第三层）：前两层已证明 Python、torch、MPS 正常。
3. 统一内存省去的是物理拷贝的必要性，但张量仍有设备归属与框架管理；显式声明设备让框架知道在哪执行、何时同步，`.cpu()` 还承担把结果拿回主机校验的职责。

</details>

**本课带走：** 环境分三层验证，每层失败含义不同；主机/设备分离与首次编译是后续所有课的前提。下一课写第一个真正的 kernel。
