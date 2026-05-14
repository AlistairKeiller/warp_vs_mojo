import time
import numpy as np
import warp as wp

wp.init()
wp.config.mode = "release"

TILE_M = 16
TILE_N = 16
TILE_K = 16
TILE_THREADS = 64


@wp.kernel
def matmul_naive(
    A: wp.array(dtype=float, ndim=2),
    B: wp.array(dtype=float, ndim=2),
    C: wp.array(dtype=float, ndim=2),
):
    i, j = wp.tid()
    s = float(0.0)
    for k in range(A.shape[1]):
        s = s + A[i, k] * B[k, j]
    C[i, j] = s


@wp.kernel
def matmul_tile(
    A: wp.array(dtype=float, ndim=2),
    B: wp.array(dtype=float, ndim=2),
    C: wp.array(dtype=float, ndim=2),
):
    i, j = wp.tid()
    K = A.shape[1]
    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    for k in range(0, int(K / TILE_K)):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
        wp.tile_matmul(a, b, sum)
    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


def benchmark(M, N, K):
    rng = np.random.default_rng(42)
    A_np = rng.random((M, K), dtype=np.float32)
    B_np = rng.random((K, N), dtype=np.float32)
    C_np = np.zeros((M, N), dtype=np.float32)

    A_wp = wp.array(A_np, dtype=wp.float32)
    B_wp = wp.array(B_np, dtype=wp.float32)
    C_wp = wp.array(C_np, dtype=wp.float32)

    # warmup naive
    wp.launch(
        kernel=matmul_naive,
        dim=(M, N),
        inputs=[A_wp, B_wp],
        outputs=[C_wp],
        block_dim=64,
    )
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(3):
        wp.launch(
            kernel=matmul_naive,
            dim=(M, N),
            inputs=[A_wp, B_wp],
            outputs=[C_wp],
            block_dim=64,
        )
    wp.synchronize()
    t1 = time.perf_counter()
    naive_time = (t1 - t0) / 3

    if M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0:
        wp.launch_tiled(
            kernel=matmul_tile,
            dim=(M // TILE_M, N // TILE_N),
            inputs=[A_wp, B_wp],
            outputs=[C_wp],
            block_dim=TILE_THREADS,
        )
        wp.synchronize()

        t0 = time.perf_counter()
        for _ in range(50):
            wp.launch_tiled(
                kernel=matmul_tile,
                dim=(M // TILE_M, N // TILE_N),
                inputs=[A_wp, B_wp],
                outputs=[C_wp],
                block_dim=TILE_THREADS,
            )
        wp.synchronize()
        t1 = time.perf_counter()
        tile_time = (t1 - t0) / 50
    else:
        tile_time = None

    flops = 2.0 * M * N * K
    naive_gflops = flops / (naive_time * 1e9)

    print(f"Naive:          {naive_time:.4f}s  {naive_gflops:.1f} GFLOPS/s")
    if tile_time is not None:
        tile_gflops = flops / (tile_time * 1e9)
        print(f"Tile:           {tile_time:.4f}s  {tile_gflops:.1f} GFLOPS/s")
        return tile_gflops
    return naive_gflops


if __name__ == "__main__":
    sizes = [256, 512, 1024]
    for size in sizes:
        print(f"\n--- Size {size} ---")
        benchmark(size, size, size)

    print("\n--- Correctness check ---")
    M = 256
    rng = np.random.default_rng(42)
    A_np = rng.random((M, M), dtype=np.float32)
    B_np = rng.random((M, M), dtype=np.float32)
    A_wp = wp.array(A_np, dtype=wp.float32)
    B_wp = wp.array(B_np, dtype=wp.float32)
    C_wp = wp.zeros((M, M), dtype=wp.float32)

    wp.launch_tiled(
        kernel=matmul_tile,
        dim=(M // TILE_M, M // TILE_N),
        inputs=[A_wp, B_wp],
        outputs=[C_wp],
        block_dim=TILE_THREADS,
    )
    wp.synchronize()

    C_np = C_wp.numpy()
    C_ref = A_np @ B_np
    max_err = np.max(np.abs(C_np - C_ref))
    print(f"Max error: {max_err:.6f}")
    print(f"Correctness: {max_err < 0.1}")
