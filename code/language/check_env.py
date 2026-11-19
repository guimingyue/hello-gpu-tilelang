import json
import platform
import sys

import numpy as np
import tilelang
import torch


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tilelang": tilelang.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }


def main():
    info = environment()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if not info["mps_available"]:
        sys.exit("本课程的运行主线需要可用的 Apple Metal / PyTorch MPS 环境。")
    a = torch.arange(4, dtype=torch.float32, device="mps")
    out = (a + 10).cpu()
    torch.testing.assert_close(out, torch.tensor([10., 11., 12., 13.]))
    print("MPS tensor: PASS; 下一步运行 vector_add.py 检查 TileLang 编译与执行。")


if __name__ == "__main__":
    main()
