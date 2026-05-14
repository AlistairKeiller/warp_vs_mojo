import time
import os
import sys
import numpy as np
import warp as wp

wp.config.quiet = True
wp.init()

if not wp.is_cuda_available():
    print("\n=== Warp GPU Matmul ===")
    print("ERROR: No CUDA GPU available — Warp needs NVIDIA CUDA")
    print()
    print("Fix: recreate the pixi environment ON this GPU machine:")
    print()
    print("  rm -rf .pixi pixi.lock")
    print("  pixi install")
    print()
    print("This forces pixi to download the Linux/CUDA-enabled warp-lang.")
    print("(The current environment was built on macOS where CUDA isn't available.)")
    print()
    print("Alternative: pip install --upgrade warp-lang")
    print()
    sys.stdout.flush()
    exit(0)

TILE_M = wp.constant(64)
TILE_N = wp.constant(64)
TILE_K = wp.constant(64)
BLOCK_DIM = 128

wp.set_module_options({
    "fast_math": True,
    "enable_backward": False,
})


@wp.kernel
def matmul_gpu(
    A: wp.array(dtype=float, ndim=2),
    B: wp.array(dtype=float, ndim=2),
    C: wp.array(dtype=float, ndim=2),
):
    i, j = wp.tid()
    K = A.shape[1]
    acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    count = int(K / TILE_K)
    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
        wp.tile_matmul(a, b, acc)

    wp.tile_store(C, acc, offset=(i * TILE_M, j * TILE_N))


def benchmark(M, N, K, num_iters=100):
    print(f"\n--- Size {M} ---")

    rng = np.random.default_rng(42)
    A_np = rng.random((M, K), dtype=np.float32)
    B_np = rng.random((K, N), dtype=np.float32)

    A = wp.array(A_np, dtype=wp.float32, device="cuda")
    B = wp.array(B_np, dtype=wp.float32, device="cuda")
    C = wp.zeros((M, N), dtype=wp.float32, device="cuda")

    dim = ((M + TILE_M - 1) // TILE_M, (N + TILE_N - 1) // TILE_N)

    cmd = wp.launch_tiled(
        kernel=matmul_gpu,
        dim=dim,
        inputs=[A, B, C],
        block_dim=BLOCK_DIM,
        record_cmd=True,
    )
    wp.synchronize_device("cuda")

    t0 = time.perf_counter()
    for _ in range(num_iters):
        cmd.launch()
    wp.synchronize_device("cuda")
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / num_iters

    flops = 2.0 * M * N * K
    gflops = flops / (elapsed * 1e9)

    print(f"  GPU Tile (record_cmd): {elapsed:.6f}s  {gflops:.1f} GFLOPS/s")
    return gflops


def verify():
    print("\n--- Correctness check (256x256) ---")
    M = 256
    rng = np.random.default_rng(42)
    A_np = rng.random((M, M), dtype=np.float32)
    B_np = rng.random((M, M), dtype=np.float32)

    A = wp.array(A_np, dtype=wp.float32, device="cuda")
    B = wp.array(B_np, dtype=wp.float32, device="cuda")
    C = wp.zeros((M, M), dtype=wp.float32, device="cuda")

    dim = (M // 64, M // 64)
    wp.launch_tiled(kernel=matmul_gpu, dim=dim, inputs=[A, B, C], block_dim=BLOCK_DIM)
    wp.synchronize_device("cuda")

    C_np = C.numpy()
    C_ref = A_np @ B_np
    max_err = float(np.max(np.abs(C_np - C_ref)))
    print(f"Max error: {max_err:.6f}")
    print(f"Correctness: {max_err < 0.1}")


if __name__ == "__main__":
    device = wp.get_cuda_device()
    print(f"\n=== Warp GPU Matmul ===")
    print(f"Device: {device} (CUDA)")

    for size in [256, 512, 1024, 2048]:
        benchmark(size, size, size)

    verify()
