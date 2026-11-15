"""TileLang Metal 后端冒烟：编译并运行一个逐元素加法 kernel（Apple M1）。"""
import numpy as np
import tilelang
import tilelang.language as T


def make_kernel(n, block=256):
    @tilelang.jit(target="metal")
    def vadd(A, B, C):
        A: T.Tensor((n,), T.float32)
        B: T.Tensor((n,), T.float32)
        C: T.Tensor((n,), T.float32)
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]
    return vadd


def main():
    n = 1 << 20
    k = make_kernel(n)
    a = np.arange(n, dtype=np.float32)
    b = (2 * np.arange(n, dtype=np.float32))
    import torch
    ta = torch.from_numpy(a).to("mps")
    tb = torch.from_numpy(b).to("mps")
    tc = torch.empty(n, dtype=torch.float32, device="mps")
    k(ta, tb, tc)
    torch.mps.synchronize()
    out = tc.cpu().numpy()
    ref = a + b
    err = float(np.max(np.abs(out - ref)))
    print("max_abs_error:", err)
    assert err == 0.0, err
    print("SMOKE OK")


if __name__ == "__main__":
    main()