from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from std.gpu.memory import AddressSpace
from layout import TileTensor, stack_allocation, row_major, Coord, Idx
from std.time import perf_counter

comptime float_dtype = DType.float32
comptime BM = 32
comptime BN = 32
comptime BK = 32

comptime tile_a_layout = row_major(Coord(Idx(BM), Idx(BK)))
comptime tile_b_layout = row_major(Coord(Idx(BK), Idx(BN)))

@parameter
def bench[size: Int]() raises:
    comptime M = size
    comptime N = size
    comptime K = size
    comptime a_layout = row_major[M, K]()
    comptime b_layout = row_major[K, N]()
    comptime c_layout = row_major[M, N]()

    comptime grid_x = ceildiv(N, BN)
    comptime grid_y = ceildiv(M, BM)

    def kernel(
        A: TileTensor[float_dtype, type_of(a_layout), MutAnyOrigin],
        B: TileTensor[float_dtype, type_of(b_layout), MutAnyOrigin],
        C: TileTensor[float_dtype, type_of(c_layout), MutAnyOrigin],
    ):
        var tx = thread_idx.x
        var ty = thread_idx.y
        var bx = block_idx.x
        var by = block_idx.y

        var tile_a = stack_allocation[float_dtype, address_space=AddressSpace.SHARED](tile_a_layout)
        var tile_b = stack_allocation[float_dtype, address_space=AddressSpace.SHARED](tile_b_layout)

        var acc: Float32 = 0.0

        for kt_idx in range(K // BK):
            var kt = kt_idx * BK
            var row = by * BM + ty
            var col = kt + tx
            if row < M and col < K:
                tile_a[ty, tx] = A[row, col]
            else:
                tile_a[ty, tx] = 0.0

            row = kt + ty
            col = bx * BN + tx
            if row < K and col < N:
                tile_b[ty, tx] = B[row, col]
            else:
                tile_b[ty, tx] = 0.0

            barrier()

            comptime for ki in range(BK):
                acc += tile_a[ty, ki] * tile_b[ki, tx]

            barrier()

        row = by * BM + ty
        col = bx * BN + tx
        if row < M and col < N:
            C[row, col] = acc

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

    ctx.enqueue_function[kernel](da, db, dc, grid_dim=(grid_x, grid_y), block_dim=(BN, BM))
    ctx.synchronize()

    var t0 = perf_counter()
    for _ in range(10):
        ctx.enqueue_function[kernel](da, db, dc, grid_dim=(grid_x, grid_y), block_dim=(BN, BM))
    ctx.synchronize()
    var t1 = perf_counter()
    var elapsed = (t1 - t0) / 10.0

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
    print("Correctness: ", max_err < 0.1)


def main() raises:
    comptime if not has_accelerator():
        print("No GPU accelerator detected — requires Metal (macOS) or CUDA (NVIDIA)")
    else:
        print("--- Mojo GPU Matmul (", BM, "x", BN, " tile, BK=", BK, ") ---")
        print("Device: auto (Metal on macOS, CUDA on NVIDIA)")
        bench[256]()
        bench[512]()
        bench[1024]()
        bench[2048]()
