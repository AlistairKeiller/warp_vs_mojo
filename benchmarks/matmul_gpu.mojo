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
comptime BK = 64
comptime RT = 2
comptime TH_Y = BM // RT
comptime TH_X = BN

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

        var acc0: Float32 = 0.0
        var acc1: Float32 = 0.0

        for kt_idx in range(K // BK):
            var kt = kt_idx * BK

            var a_row0 = by * BM + ty * RT
            var a_row1 = a_row0 + 1
            var a_col0 = kt + tx
            var a_col1 = kt + tx + TH_X

            if a_row0 < M and a_col0 < K:
                tile_a[ty * RT, tx] = A[a_row0, a_col0]
            else:
                tile_a[ty * RT, tx] = 0.0
            if a_row0 < M and a_col1 < K:
                tile_a[ty * RT, tx + TH_X] = A[a_row0, a_col1]
            else:
                tile_a[ty * RT, tx + TH_X] = 0.0
            if a_row1 < M and a_col0 < K:
                tile_a[ty * RT + 1, tx] = A[a_row1, a_col0]
            else:
                tile_a[ty * RT + 1, tx] = 0.0
            if a_row1 < M and a_col1 < K:
                tile_a[ty * RT + 1, tx + TH_X] = A[a_row1, a_col1]
            else:
                tile_a[ty * RT + 1, tx + TH_X] = 0.0

            var b_col = bx * BN + tx

            var b_row0 = kt + ty
            if b_row0 < K and b_col < N:
                tile_b[ty, tx] = B[b_row0, b_col]
            else:
                tile_b[ty, tx] = 0.0

            var b_row1 = kt + ty + TH_Y
            if b_row1 < K and b_col < N:
                tile_b[ty + TH_Y, tx] = B[b_row1, b_col]
            else:
                tile_b[ty + TH_Y, tx] = 0.0

            var b_row2 = kt + ty + 2 * TH_Y
            if b_row2 < K and b_col < N:
                tile_b[ty + 2 * TH_Y, tx] = B[b_row2, b_col]
            else:
                tile_b[ty + 2 * TH_Y, tx] = 0.0

            var b_row3 = kt + ty + 3 * TH_Y
            if b_row3 < K and b_col < N:
                tile_b[ty + 3 * TH_Y, tx] = B[b_row3, b_col]
            else:
                tile_b[ty + 3 * TH_Y, tx] = 0.0

            barrier()

            comptime for ki in range(BK):
                var a0 = tile_a[ty * RT, ki]
                var a1 = tile_a[ty * RT + 1, ki]
                var b = tile_b[ki, tx]
                acc0 += a0 * b
                acc1 += a1 * b

            barrier()

        var row0 = by * BM + ty * RT
        var row1 = row0 + 1
        var col = bx * BN + tx
        if row0 < M and col < N:
            C[row0, col] = acc0
        if row1 < M and col < N:
            C[row1, col] = acc1

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

    ctx.enqueue_function[kernel](da, db, dc, grid_dim=(grid_x, grid_y), block_dim=(TH_X, TH_Y))
    ctx.synchronize()

    var t0 = perf_counter()
    for _ in range(100):
        ctx.enqueue_function[kernel](da, db, dc, grid_dim=(grid_x, grid_y), block_dim=(TH_X, TH_Y))
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
    print("Correctness: ", max_err < 0.1)


def main() raises:
    comptime if not has_accelerator():
        print("No GPU accelerator detected — requires Metal (macOS) or CUDA (NVIDIA)")
    else:
        print("--- Mojo GPU Matmul (BK=64, 2x1 reg-tiled) ---")
        print("Device: auto (Metal on macOS, CUDA on NVIDIA)")
        bench[256]()
        bench[512]()
        bench[1024]()
        bench[2048]()
