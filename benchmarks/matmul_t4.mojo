from std.math import ceildiv
from std.sys import has_accelerator, _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.gpu.sync import barrier
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, lane_id
from std.gpu.memory import AddressSpace
from layout import TileTensor, stack_allocation, row_major, Coord, Idx
from std.time import perf_counter

comptime float_dtype = DType.float32

# m8n8k4: each warp produces 8×8 output, each thread: 1 F16 (a), 1 F16 (b), 2 F32 (d)
comptime BM = 64
comptime BN = 32
comptime BK = 32

comptime WARPS_M = BM // 8
comptime WARPS_N = BN // 8
comptime WARPS = WARPS_M * WARPS_N
comptime THREADS = WARPS * 32

# Per-thread elements for global→shared loads
comptime A_ELEMS = BM * BK // THREADS
comptime B_ELEMS = BK * BN // THREADS

comptime a_smem_layout = row_major(Coord(Idx(BM), Idx(BK)))
comptime b_smem_layout = row_major(Coord(Idx(BK), Idx(BN)))

@always_inline
def _f32_to_f16(x: Float32) -> Float16:
    return x.cast[DType.float16]()

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
        var tid = thread_idx.x
        var warp_id = tid // 32
        var warp_y = warp_id // WARPS_N
        var warp_x = warp_id % WARPS_N
        var lane = lane_id()
        var by = block_idx.y
        var bx = block_idx.x

        var m_start = by * BM
        var n_start = bx * BN

        # Shared memory for A and B tiles (F16)
        var a_smem = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
            a_smem_layout
        )
        var b_smem = stack_allocation[DType.float16, address_space=AddressSpace.SHARED](
            b_smem_layout
        )

        # Accumulator: 2 F32 per thread (m8n8k4 output)
        var accum = SIMD[DType.float32, 2](0)

        for kt in range(0, K, BK):
            # Load A tile: global F32 → shared F16
            comptime for j in range(A_ELEMS):
                var idx = tid * A_ELEMS + j
                var r = idx // BK
                var c = idx % BK
                var gr = m_start + r
                var gc = kt + c
                if gr < M and gc < K:
                    a_smem[r, c] = _f32_to_f16(A[gr, gc])
                else:
                    a_smem[r, c] = 0.0

            # Load B tile: global F32 → shared F16
            comptime for j in range(B_ELEMS):
                var idx = tid * B_ELEMS + j
                var r = idx // BN
                var c = idx % BN
                var gr = kt + r
                var gc = n_start + c
                if gr < K and gc < N:
                    b_smem[r, c] = _f32_to_f16(B[gr, gc])
                else:
                    b_smem[r, c] = 0.0

            barrier()

            # MMA loop: (BK/4) iterations × K=4 = BK
            comptime for ki in range(BK // 4):
                var kk = ki * 4
                var lane_row = lane // 4
                var lane_col = lane % 4

                # A fragment: 1 F16 per thread (row-major, 8×4)
                # row = lane//4 (0-7), col = lane%4 (0-3)
                var a_val = a_smem[warp_y * 8 + lane_row, kk + lane_col]

                # B fragment: 1 F16 per thread (column-major, 4×8)
                # row = lane%4 (0-3), col = lane//4 (0-7)
                var b_val = b_smem[kk + lane_col, warp_x * 8 + lane_row]

                # Tensor core MMA: D = A × B + C (m8n8k4, F16×F16+F32→F32)
                # Use inline PTX for sm_75 (T4) compatibility.
                # The stdlib mma() m8n8k4 path uses an LLVM intrinsic that
                # returns 8×F32, incompatible with this LLVM version.
                var r = inlined_assembly[
                    "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 " +
                    "{$0, $1}, {$2}, {$3}, {$4, $5};",
                    _RegisterPackType[Float32, Float32],
                    constraints="=f,=f,h,h,f,f",
                ](
                    a_val, b_val,
                    accum[0], accum[1],
                )
                accum = SIMD[DType.float32, 2](r[0], r[1])

            barrier()

        # Write accum to global memory (2 F32 per thread)
        # Output mapping: meta_row = lane//4 (0-7), meta_col = lane%4 (0-3)
        # d[0] = D[meta_row, meta_col], d[1] = D[meta_row, meta_col+4]
        var meta_row = lane // 4
        var meta_col = lane % 4
        var crow = m_start + warp_y * 8 + meta_row
        var ccol = n_start + warp_x * 8 + meta_col

        if crow < M and ccol < N:
            C[crow, ccol] = accum[0]
        if crow < M and ccol + 4 < N:
            C[crow, ccol + 4] = accum[1]

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
        print("--- Mojo GPU Matmul (T4, m8n8k4 tensor-core MMA, F16→F32) ---")
        bench[256]()
        bench[512]()
        bench[1024]()
        bench[2048]()
