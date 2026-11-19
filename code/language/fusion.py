import tilelang
import tilelang.language as T
import torch

from vector_add import make_add


def make_relu(n, block=32):
    @tilelang.jit(target="metal")
    def relu(A, B):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        with T.Kernel(T.ceildiv(n, block), threads=32) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    B[idx] = T.max(A[idx], 0)
    return relu


def make_add_relu(n, block=32):
    @tilelang.jit(target="metal")
    def add_relu(A, B, C):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        C: T.Tensor((n,), T.float32)
        with T.Kernel(T.ceildiv(n, block), threads=32) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    C[idx] = T.max(A[idx] + B[idx], 0)
    return add_relu


def main():
    a = torch.tensor([-4., -1., 0., 2., 4.], device="mps")
    b = torch.tensor([1., -1., 0., -1., 2.], device="mps")
    n = a.numel()
    temporary = torch.full_like(a, float("nan"))
    separate = torch.full_like(a, float("nan"))
    fused = torch.full_like(a, float("nan"))
    make_add(n, 32)(a, b, temporary)
    make_relu(n)(temporary, separate)
    make_add_relu(n)(a, b, fused)
    torch.mps.synchronize()
    ref = (a.cpu() + b.cpu()).clamp_min(0)
    torch.testing.assert_close(separate.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(fused.cpu(), ref, rtol=0, atol=0)
    print("separate:", separate.cpu().tolist())
    print("fused:   ", fused.cpu().tolist())
    print("Fusion: PASS; 2 launches + intermediate tensor vs 1 launch")


if __name__ == "__main__":
    main()
