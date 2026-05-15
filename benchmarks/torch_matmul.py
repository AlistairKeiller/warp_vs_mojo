import time

import torch


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bench(size, device):
    M = N = K = size
    dtype = torch.float32
    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    c = a @ b
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(100):
        c = a @ b
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / 100.0
    gflops = 2.0 * M * N * K / (elapsed * 1e9)
    print(f"PyTorch {device} {size}x{size}: {elapsed:.6f}s  {gflops:.1f} GFLOPS/s")


if __name__ == "__main__":
    device = get_best_device()
    device_name = {"cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}[device]
    print(f"--- PyTorch Matmul (float32, {device_name}) ---")
    for s in [256, 512, 1024, 2048]:
        bench(s, device)
