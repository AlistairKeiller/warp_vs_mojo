import torch
import time

def bench(size):
    M = N = K = size
    dtype = torch.float32
    a = torch.randn(M, K, dtype=dtype, device="mps")
    b = torch.randn(K, N, dtype=dtype, device="mps")

    # Warmup
    c = a @ b
    torch.mps.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        c = a @ b
    torch.mps.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / 100.0
    gflops = 2.0 * M * N * K / (elapsed * 1e9)
    print(f"PyTorch MPS {size}x{size}: {elapsed:.6f}s  {gflops:.1f} GFLOPS/s")

if __name__ == "__main__":
    print("--- PyTorch MPS Matmul (float32) ---")
    for s in [256, 512, 1024, 2048]:
        bench(s)
