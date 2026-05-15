from std.math import ceildiv, min, fma
from std.sys import has_accelerator
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx
from std.gpu.memory import AddressSpace
from layout import TileTensor, stack_allocation, row_major, Coord, Idx
from std.time import perf_counter
from std.collections import InlineArray

comptime float_dtype = DType.float32
comptime BM = 64
comptime BN = 64
comptime BN_PAD = BN + 1
comptime BK = 32
comptime BK_PAD = BK + 1
comptime VEC_W = 4
comptime THREADS_M = 16
comptime THREADS_N = 16
comptime THREADS = THREADS_M * THREADS_N
comptime ELEMS_M = BM // THREADS_M
comptime ELEMS_N = BN // THREADS_N

comptime a_smem_layout = row_major(Coord(Idx(BM), Idx(BK_PAD)))
comptime b_smem_layout = row_major(Coord(Idx(BK), Idx(BN_PAD)))

@parameter
def bench[size_m: Int, size_n: Int, size_k: Int]() raises:
    comptime M = size_m
    comptime N = size_n
    comptime K = size_k
    comptime a_layout = row_major[M, K]()
    comptime b_layout = row_major[K, N]()
    comptime c_layout = row_major[M, N]()

    def kernel(
        A: TileTensor[float_dtype, type_of(a_layout), MutAnyOrigin],
        B: TileTensor[float_dtype, type_of(b_layout), MutAnyOrigin],
        C: TileTensor[float_dtype, type_of(c_layout), MutAnyOrigin],
    ):
        var tid = thread_idx.x
        var ty = tid // THREADS_N
        var tx = tid % THREADS_N
        var by = block_idx.y
        var bx = block_idx.x

        var m_start = by * BM
        var n_start = bx * BN

        var a_smem = stack_allocation[float_dtype, address_space=AddressSpace.SHARED](
            a_smem_layout
        )
        var b_smem = stack_allocation[float_dtype, address_space=AddressSpace.SHARED](
            b_smem_layout
        )

        comptime a_elems = BM * BK // THREADS
        comptime b_elems = BK * BN // THREADS

        var accum = InlineArray[SIMD[float_dtype, ELEMS_N], ELEMS_M](
            uninitialized=True
        )
        comptime for i in range(ELEMS_M):
            accum[i] = SIMD[float_dtype, ELEMS_N](0)

        for kt in range(0, K, BK):
            # Load A tile: global → shared (scalar stores = no bank conflict)
            comptime for j in range(a_elems):
                var idx = tid * a_elems + j
                var r = idx // BK
                var c = idx % BK
                var gr = m_start + r
                var gc = kt + c
                if gr < M and gc < K:
                    a_smem[r, c] = A[gr, gc]
                else:
                    a_smem[r, c] = 0.0

            # Load B tile: global → shared (scalar stores = no bank conflict)
            comptime for j in range(b_elems):
                var idx = tid * b_elems + j
                var r = idx // BN
                var c = idx % BN
                var gr = kt + r
                var gc = n_start + c
                if gr < K and gc < N:
                    b_smem[r, c] = B[gr, gc]
                else:
                    b_smem[r, c] = 0.0

            barrier()

            for kk in range(BK):
                var b_vals = b_smem.raw_load[width=ELEMS_N](
                    kk * BN_PAD + tx * ELEMS_N
                )
                for i in range(ELEMS_M):
                    var a_val = a_smem[ty * ELEMS_M + i, kk]
                    for j in range(ELEMS_N):
                        accum[i][j] += a_val * b_vals[j]

            barrier()

        for i in range(ELEMS_M):
            var crow = m_start + ty * ELEMS_M + i
            for j in range(ELEMS_N):
                var ccol = n_start + tx * ELEMS_N + j
                if crow < M and ccol < N:
                    C[crow, ccol] = accum[i][j]

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

    var grid_x = ceildiv(N, BN)
    var grid_y = ceildiv(M, BM)
    ctx.enqueue_function[kernel](
        da, db, dc,
        grid_dim=(grid_x, grid_y),
        block_dim=(THREADS, 1),
    )
    ctx.synchronize()

    var t0 = perf_counter()
    for _ in range(100):
        ctx.enqueue_function[kernel](
            da, db, dc,
            grid_dim=(grid_x, grid_y),
            block_dim=(THREADS, 1),
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

    print("--- General-Purpose Mojo GPU GEMM (BM=64, BN=64, BK=32, runtime loops) ---")
    print("Tile-aligned square matrices:")
    bench[256, 256, 256]()
    bench[512, 512, 512]()
    bench[1024, 1024, 1024]()
    bench[2048, 2048, 2048]()
    print("\nNon-standard rectangular matrices (correctness test):")
    bench[100, 200, 128]()
    bench[256, 123, 100]()
    bench[1023, 1025, 256]()
    bench[333, 444, 555]()
