from std.math import ceildiv, min
from std.sys import has_accelerator, llvm_intrinsic
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, lane_id
from std.gpu.memory import AddressSpace
from layout import TileTensor, stack_allocation, row_major, Coord, Idx
from std.time import perf_counter
from std.memory import UnsafePointer
from std.collections import InlineArray

comptime float_dtype = DType.float32
comptime BM = 128
comptime BN = 128
comptime BK = 32
comptime VEC_W = 16

comptime WARP_TILES_M = 4
comptime WARP_TILES_N = 2
comptime WARPS_M = BM // (WARP_TILES_M * 16)
comptime WARPS_N = BN // (WARP_TILES_N * 16)
comptime WARPS = WARPS_M * WARPS_N
comptime THREADS = WARPS * 32
comptime N_ACC = WARP_TILES_M * WARP_TILES_N

comptime a_elems = BM * BK // THREADS
comptime b_elems = BK * BN // THREADS
comptime a_groups = a_elems // VEC_W
comptime a_rem = a_elems % VEC_W
comptime b_groups = b_elems // VEC_W
comptime b_rem = b_elems % VEC_W

@always_inline
def _frag_rc(tid: Int) -> Tuple[Int, Int]:
    return (
        ((tid & 7) // 2) + ((tid & 16) >> 2),
        ((tid & 1) << 2) + (tid & 8),
    )

@always_inline
def _load_frag(
    base: UnsafePointer[SIMD[float_dtype, 1], ...],
    stride: Int,
) -> SIMD[float_dtype, 8]:
    var tid = lane_id()
    var r, c = _frag_rc(Int(tid))
    var lo = (base + r * stride + c).load[width=4]()
    var hi = (base + (r + 8) * stride + c).load[width=4]()
    return lo.join(hi)

@always_inline
def _store_frag(
    base: UnsafePointer[mut=True, SIMD[float_dtype, 1], ...],
    stride: Int,
    frag: SIMD[float_dtype, 8],
):
    var tid = lane_id()
    var r, c = _frag_rc(Int(tid))
    (base + r * stride + c).store(frag.slice[4, offset=0]())
    (base + (r + 8) * stride + c).store(frag.slice[4, offset=4]())

@always_inline
def _store_frag_clamped(
    base: UnsafePointer[mut=True, SIMD[float_dtype, 1], ...],
    stride: Int,
    frag: SIMD[float_dtype, 8],
    limit_row: Int,
    limit_col: Int,
    base_row: Int,
    base_col: Int,
):
    var tid = lane_id()
    var r, c = _frag_rc(Int(tid))
    var row0 = base_row + r
    var row1 = base_row + r + 8
    var col0 = base_col + c
    var cols_valid = min(4, limit_col - col0)
    if row0 < limit_row and cols_valid > 0:
        for cc in range(cols_valid):
            base[r * stride + c + cc] = frag[cc]
    if row1 < limit_row and cols_valid > 0:
        for cc in range(cols_valid):
            base[(r + 8) * stride + c + cc] = frag[4 + cc]

@always_inline
def _apple_mma(
    mut d: SIMD[float_dtype, 8],
    a: SIMD[float_dtype, 8],
    b: SIMD[float_dtype, 8],
    c: SIMD[float_dtype, 8],
):
    d = rebind[type_of(d)](
        llvm_intrinsic[
            "llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate",
            SIMD[float_dtype, 8],
        ](a, False, b, False, c)
    )

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
        var warp_id = tid // 32
        var warp_y = warp_id // WARPS_N
        var warp_x = warp_id % WARPS_N
        var by = block_idx.y
        var bx = block_idx.x

        var m_start = by * BM
        var n_start = bx * BN

        comptime a_smem_layout = row_major(Coord(Idx(BM), Idx(BK)))
        comptime b_smem_layout = row_major(Coord(Idx(BK), Idx(BN)))
        var a_smem = stack_allocation[float_dtype, address_space=AddressSpace.SHARED](
            a_smem_layout
        )
        var b_smem = stack_allocation[float_dtype, address_space=AddressSpace.SHARED](
            b_smem_layout
        )

        var a_ptr = a_smem.ptr
        var b_ptr = b_smem.ptr

        var accum = InlineArray[SIMD[float_dtype, 8], N_ACC](
            uninitialized=True
        )
        comptime for i in range(N_ACC):
            accum[i] = SIMD[float_dtype, 8](0)

        for kt in range(0, K, BK):
            var kb = min(BK, K - kt)

            if kb < BK:
                comptime for j in range(a_groups):
                    a_smem.raw_store[width=VEC_W](
                        tid * a_elems + j * VEC_W, SIMD[float_dtype, VEC_W](0)
                    )
                comptime for j in range(b_groups):
                    b_smem.raw_store[width=VEC_W](
                        tid * b_elems + j * VEC_W, SIMD[float_dtype, VEC_W](0)
                    )

            comptime for j in range(a_groups):
                var idx = tid * a_elems + j * VEC_W
                var r = idx // BK
                var c = idx % BK
                var gr = m_start + r
                var gc = kt + c
                if gr < M and gc + VEC_W <= K:
                    a_smem.raw_store[width=VEC_W](
                        r * BK + c,
                        A.raw_load[width=VEC_W](gr * K + gc),
                    )
                else:
                    var v = SIMD[float_dtype, VEC_W](0)
                    for k in range(VEC_W):
                        if gr < M and gc + k < K:
                            v[k] = A[gr, gc + k]
                    a_smem.raw_store[width=VEC_W](r * BK + c, v)
            comptime for j in range(a_rem):
                var idx = tid * a_elems + a_groups * VEC_W + j
                var r = idx // BK
                var c = idx % BK
                var gr = m_start + r
                var gc = kt + c
                if gr < M and gc < K:
                    a_smem[r, c] = A[gr, gc]
                else:
                    a_smem[r, c] = 0.0

            comptime for j in range(b_groups):
                var idx = tid * b_elems + j * VEC_W
                var r = idx // BN
                var c = idx % BN
                var gr = kt + r
                var gc = n_start + c
                if gr < K and gc + VEC_W <= N:
                    b_smem.raw_store[width=VEC_W](
                        r * BN + c,
                        B.raw_load[width=VEC_W](gr * N + gc),
                    )
                else:
                    var v = SIMD[float_dtype, VEC_W](0)
                    for k in range(VEC_W):
                        if gr < K and gc + k < N:
                            v[k] = B[gr, gc + k]
                    b_smem.raw_store[width=VEC_W](r * BN + c, v)
            comptime for j in range(b_rem):
                var idx = tid * b_elems + b_groups * VEC_W + j
                var r = idx // BN
                var c = idx % BN
                var gr = kt + r
                var gc = n_start + c
                if gr < K and gc < N:
                    b_smem[r, c] = B[gr, gc]
                else:
                    b_smem[r, c] = 0.0

            barrier()

            comptime for ki in range(BK // 16):
                var kk = ki * 16
                comptime for tm in range(WARP_TILES_M):
                    var a_frag = _load_frag(
                        a_ptr + (warp_y * WARP_TILES_M + tm) * 16 * BK + kk, BK
                    )
                    comptime for tn in range(WARP_TILES_N):
                        var b_frag = _load_frag(
                            b_ptr + kk * BN + (warp_x * WARP_TILES_N + tn) * 16, BN
                        )
                        _apple_mma(
                            accum[tm * WARP_TILES_N + tn],
                            a_frag,
                            b_frag,
                            accum[tm * WARP_TILES_N + tn],
                        )

            barrier()

        comptime for tm in range(WARP_TILES_M):
            comptime for tn in range(WARP_TILES_N):
                var tile_row = m_start + (warp_y * WARP_TILES_M + tm) * 16
                var tile_col = n_start + (warp_x * WARP_TILES_N + tn) * 16
                var c_base = C.ptr + tile_row * N + tile_col
                if tile_row + 16 <= M and tile_col + 16 <= N:
                    _store_frag(c_base, N, accum[tm * WARP_TILES_N + tn])
                else:
                    _store_frag_clamped(
                        c_base, N, accum[tm * WARP_TILES_N + tn],
                        M, N, tile_row, tile_col,
                    )

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
    else:
        print("--- General-Purpose Mojo GPU GEMM (M5 Pro, simdgroup MMA) ---")
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
