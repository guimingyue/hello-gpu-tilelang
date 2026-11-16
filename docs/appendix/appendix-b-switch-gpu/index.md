---
title: "附录 B · 换一台 Mac：Apple Silicon 代际与 GPU family"
description: "Hello GPU TileLang 附录 B · 查 GPU family/架构、统一内存，换机后重测"
---

# 附录 B · 换一台 Mac：Apple Silicon 代际与 GPU family

## B.1 代际对照（以本机查询为准）

| 芯片 | GPU family | 架构 |
| --- | --- | --- |
| M1 系列 | apple7 | applegpu_g13g |
| M2 系列 | apple8 | applegpu_g14g |
| M3 系列 | apple9 | applegpu_g15g |
| M4 系列 | apple9 | applegpu_g16g |

```python
import mlx.core as mx
from hello_metal import default_device
print(default_device().name, mx.device_info().get("architecture"))
```

## B.2 换机后必须重测

本教程基线为 M1。其他代际的绝对时间、最优 threadgroup/tile、带宽/算力参考都不同；换机后重跑各章 `run_rounds.py`，不要套用 M1 数字。

## B.3 统一内存

内存容量决定能放多大的模型/矩阵；GPU 核心数影响并行吞吐，但逐元素算子受带宽限制，核心数增加未必线性提速。

> 验证脚本见 `code/appendix/appendix-b-switch-gpu/check_gpu.py`（Metal 版仓库）。