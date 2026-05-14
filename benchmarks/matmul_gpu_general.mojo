from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from std.gpu.memory import AddressSpace
from layout import TileTensor, stack_allocation, row_major, Coord, Idx
from std.time import perf_counter

comptime float_dtype = DType.float32
comptime TILE_M = 16
comptime TILE_N = 16
comptime TILE_K = 16

@parameter
def bench[size_m: Int, size_n: Int, size_k: Int]() raises:
    comptime M = size_m
    comptime N = size_n
    comptime K = size_k
    comptime a_layout = row_major[M, K]()
    comptime b_layout = row_major[K, N]()
    comptime c_layout = row_major[M, N]()

    def tiled_matmul_kernel(
        A: TileTensor[float_dtype, type_of(a_layout), MutAnyOrigin],
        B: TileTensor[float_dtype, type_of(b_layout), MutAnyOrigin],
        C: TileTensor[float_dtype, type_of(c_layout), MutAnyOrigin],
    ):
        var tx = thread_idx.x
        var ty = thread_idx.y
        var bx = block_idx.x
        var by = block_idx.y

        var global_row = by * TILE_M + ty
        var global_col = bx * TILE_N + tx

        var tile_row_start = by * TILE_M
        var tile_col_start = bx * TILE_N

        var a_tile = stack_allocation[
            float_dtype, address_space=AddressSpace.SHARED
        ](row_major(Coord(Idx(TILE_M), Idx(TILE_K))))
        var b_tile = stack_allocation[
            float_dtype, address_space=AddressSpace.SHARED
        ](row_major(Coord(Idx(TILE_K), Idx(TILE_N))))

        var accumulator: C.ElementType = 0.0

        comptime for k_tile in range(0, K, TILE_K):
            var a_global_row = tile_row_start + ty
            var a_global_col = k_tile + tx
            var b_global_row = k_tile + ty
            var b_global_col = tile_col_start + tx

            var load_a_valid = (a_global_row < M) and (a_global_col < K)
            var load_b_valid = (b_global_row < K) and (b_global_col < N)

            if load_a_valid:
                a_tile[ty, tx] = A[a_global_row, a_global_col]
            else:
                a_tile[ty, tx] = 0.0

            if load_b_valid:
                b_tile[ty, tx] = B[b_global_row, b_global_col]
            else:
                b_tile[ty, tx] = 0.0

            barrier()

            comptime for k in range(TILE_K):
                accumulator += a_tile[ty, k] * b_tile[k, tx]

            barrier()

        if (global_row < M) and (global_col < N):
            C[global_row, global_col] = accumulator

    print("\n--- Size ", M, " ---")

    var ctx = DeviceContext()
    var buf_a = ctx.enqueue_create_buffer[float_dtype](M * K)
    var buf_b = ctx.enqueue_create_buffer[float_dtype](K * N)
    var buf_c = ctx.enqueue_create_buffer[float_dtype](M * N)
    var ha = ctx.enqueue_create_host_buffer[float_dtype](M * K)
    var hb = ctx.enqueue_create_host_buffer[float_dtype](K * N)
    var hc = ctx.enqueue_create_host_buffer[float_dtype](M * N)
    ctx.synchronize()

    var ha_t = TileTensor(ha, a_layout)
    var hb_t = TileTensor(hb, b_layout)
    for i in range(M):
        for j in range(K):
            ha_t[i, j] = Float32(Int(i * K + j) % 100) / 100.0
    for i in range(K):
        for j in range(N):
            hb_t[i, j] = Float32(Int(i * N + j * 7) % 100) / 100.0

    ctx.enqueue_copy(src_buf=ha, dst_buf=buf_a)
    ctx.enqueue_copy(src_buf=hb, dst_buf=buf_b)

    var da = TileTensor(buf_a, a_layout)
    var db = TileTensor(buf_b, b_layout)
    var dc = TileTensor(buf_c, c_layout)

    var num_blocks_x = ceildiv(N, TILE_N)
    var num_blocks_y = ceildiv(M, TILE_M)
    ctx.enqueue_function[tiled_matmul_kernel](
        da, db, dc,
        grid_dim=(num_blocks_x, num_blocks_y),
        block_dim=(TILE_N, TILE_M),
    )
    ctx.synchronize()

    var t0 = perf_counter()
    for _ in range(100):
        ctx.enqueue_function[tiled_matmul_kernel](
            da, db, dc,
            grid_dim=(num_blocks_x, num_blocks_y),
            block_dim=(TILE_N, TILE_M),
        )
    ctx.synchronize()
    var t1 = perf_counter()
    var elapsed = (t1 - t0) / 100.0

    var gflops = 2.0 * Float64(M) * Float64(N) * Float64(K) / (elapsed * 1e9)
    print("GPU Matmul: ", elapsed, "s  ", gflops, " GFLOPS/s")

    ctx.enqueue_copy(src_buf=buf_c, dst_buf=hc)
    ctx.synchronize()
    var hc_t = TileTensor(hc, c_layout)

    var max_err: Float32 = 0.0
    for i in range(M):
        for j in range(N):
            var expected: Float32 = 0.0
            for k in range(K):
                expected += ha_t[i, k] * hb_t[k, j]
            var err = abs(hc_t[i, j] - expected)
            if err > max_err:
                max_err = err
    print("Max error: ", max_err)
    print("Correctness: ", max_err < 1.0)


def main() raises:
    comptime if not has_accelerator():
        print("No GPU accelerator detected")
        return

    print("--- General-Purpose Mojo GPU GEMM (tiled, cross-platform) ---")
    print("Square tile-aligned matrices:")
    bench[256, 256, 256]()
    bench[512, 512, 512]()
    bench[1024, 1024, 1024]()
    bench[2048, 2048, 2048]()
    print("\nNon-standard rectangular matrices (correctness test):")
    bench[100, 200, 128]()
    bench[256, 123, 100]()
    bench[1023, 1025, 256]()
    bench[333, 444, 555]()
