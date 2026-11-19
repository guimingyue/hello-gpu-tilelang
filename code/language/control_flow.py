import argparse

import tilelang
import tilelang.language as T
import torch


def make_transform(n, steps=2, block=32):
    @tilelang.jit(target="metal")
    def transform(A, B):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        with T.Kernel(T.ceildiv(n, block), threads=32) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    if A[idx] > 0:
                        B[idx] = A[idx]
                    else:
                        B[idx] = 0
                    for step in T.serial(steps):
                        B[idx] = 0.5 * B[idx] + 1
    return transform


def reference(a, steps):
    result = a.clamp_min(0)
    for _ in range(steps):
        result = 0.5 * result + 1
    return result


def main():
    parser = argparse.ArgumentParser(description="并行处理不同元素，串行推进每个元素的递推。")
    parser.add_argument("--steps", type=int, choices=range(9), default=2)
    args = parser.parse_args()
    a = torch.tensor([-4., -2., 0., 2., 4.], device="mps")
    b = torch.full_like(a, float("nan"))
    make_transform(a.numel(), args.steps)(a, b)
    torch.mps.synchronize()
    torch.testing.assert_close(b.cpu(), reference(a.cpu(), args.steps), rtol=0, atol=0)
    print("output:", b.cpu().tolist())
    print(f"Control flow: PASS; steps={args.steps}")


if __name__ == "__main__":
    main()
