import argparse

import tilelang
import tilelang.language as T
import torch


def make_add(n, block=4):
    @tilelang.jit(target="metal")
    def add(A, B, C):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        C: T.Tensor((n,), T.float32)
        with T.Kernel(T.ceildiv(n, block), threads=32) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]
    return add


def main():
    parser = argparse.ArgumentParser(description="给每个元素加上 10，观察分块与尾部。")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--block", type=int, choices=[4, 8, 32, 64, 128, 256], default=4)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n 必须为正整数")
    a = torch.arange(args.n, dtype=torch.float32, device="mps")
    b = torch.full_like(a, 10)
    c = torch.full_like(a, float("nan"))
    make_add(args.n, args.block)(a, b, c)
    torch.mps.synchronize()
    result = c.cpu()
    torch.testing.assert_close(result, a.cpu() + b.cpu(), rtol=0, atol=0)
    groups = (args.n + args.block - 1) // args.block
    print(f"n={args.n}, block={args.block}, groups={groups}, skipped={groups * args.block - args.n}")
    print("output (first 16):", result[:16].tolist())
    print("Vector add: PASS")


if __name__ == "__main__":
    main()
