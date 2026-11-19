import argparse
import json
import statistics
import time

import torch

from check_env import environment
from vector_add import make_add


def measure(n=65536, block=128, warmup=3, repeat=10, show_source=False):
    a = torch.arange(n, dtype=torch.float32, device="mps") / n
    b = torch.full_like(a, 2)
    c = torch.full_like(a, float("nan"))
    kernel = make_add(n, block)
    torch.mps.synchronize()
    start = time.perf_counter()
    compiled = kernel.compile(a, b, c)
    compile_or_cache_ms = (time.perf_counter() - start) * 1000
    if show_source:
        print(compiled.get_kernel_source())
    compiled(a, b, c)
    torch.mps.synchronize()
    ref = a.cpu() + b.cpu()
    torch.testing.assert_close(c.cpu(), ref, rtol=0, atol=0)
    for _ in range(warmup):
        compiled(a, b, c)
    torch.mps.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        compiled(a, b, c)
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    torch.testing.assert_close(c.cpu(), ref, rtol=0, atol=0)
    return {
        "environment": environment(),
        "n": n, "block": block, "threads": 32,
        "warmup": warmup, "repeat": repeat,
        "scope": "precompiled-kernel-call-plus-mps-synchronize",
        "compile_or_cache_ms": compile_or_cache_ms,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples), "max_ms": max(samples),
        "correct": True,
    }


def main():
    parser = argparse.ArgumentParser(description="预编译、预热、同步墙钟计时；不含分配与读回。")
    parser.add_argument("--n", type=int, default=65536)
    parser.add_argument("--block", type=int, choices=[32, 64, 128, 256], default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--source", action="store_true")
    args = parser.parse_args()
    if args.n < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("n、repeat 必须为正整数，warmup 不能为负数")
    print(json.dumps(measure(args.n, args.block, args.warmup, args.repeat, args.source),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
