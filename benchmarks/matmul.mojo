from std.algorithm import sync_parallelize
from std.math import ceildiv
from std.memory import alloc
from std.sys.info import simd_width_of
from std.time import perf_counter

comptime float_type = DType.float32
comptime simd = simd_width_of[float_type]()
comptime TM = 64
comptime TN = 64
comptime TK = 8
comptime NUM_TASKS = 8


def matmul_naive(
    A: UnsafePointer[Scalar[float_type], MutExternalOrigin],
    B: UnsafePointer[Scalar[float_type], MutExternalOrigin],
    C: UnsafePointer[Scalar[float_type], MutExternalOrigin],
    M: Int, N: Int, K: Int,
):
    for m in range(M):
        for n in range(N):
            var s: Scalar[float_type] = 0.0
            for k in range(K):
                s += A[m * K + k] * B[k * N + n]
            C[m * N + n] = s


def matmul_tiled_simd(
    A: UnsafePointer[Scalar[float_type], MutExternalOrigin],
    B: UnsafePointer[Scalar[float_type], MutExternalOrigin],
    C: UnsafePointer[Scalar[float_type], MutExternalOrigin],
    M: Int, N: Int, K: Int,
):
    @parameter
    def worker(task_id: Int):
        var chunk = ceildiv(M, NUM_TASKS)
        var m0 = task_id * chunk
        var m1 = min(m0 + chunk, M)

        for nt in range(0, N, TN):
            var n_end = min(nt + TN, N)
            for mt in range(m0, m1, TM):
                var m_end_tile = min(mt + TM, m1)

                for mm in range(mt, m_end_tile):
                    for nn in range(nt, n_end):
                        C[mm * N + nn] = 0.0

                for kt in range(0, K, TK):
                    var k_end = min(kt + TK, K)
                    for mm in range(mt, m_end_tile):
                        for nn in range(nt, n_end, simd):
                            var c_offset = mm * N + nn
                            var c_vec = C.load[width=simd](c_offset)
                            for k in range(kt, k_end):
                                var a_val = A[mm * K + k]
                                var b_vec = B.load[width=simd](k * N + nn)
                                c_vec = a_val * b_vec + c_vec
                            C.store[width=simd](c_offset, c_vec)

    sync_parallelize[worker](NUM_TASKS)


def run_benchmark(size: Int):
    var M = size
    var N = size
    var K = size
    print("\n--- Size ", M, " ---")

    var A = alloc[Scalar[float_type]](M * K)
    var B = alloc[Scalar[float_type]](K * N)
    var C = alloc[Scalar[float_type]](M * N)

    for i in range(M * K):
        A[i] = Float32(Int(i) % 100) / 100.0
    for i in range(K * N):
        B[i] = Float32(Int(i * 7) % 100) / 100.0

    if size <= 512:
        matmul_naive(A, B, C, M, N, K)
        var t0 = perf_counter()
        matmul_naive(A, B, C, M, N, K)
        var t1 = perf_counter()
        var elapsed = (t1 - t0)
        var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
        var gflops = flops / (elapsed * 1e9)
        print("Naive:           ", elapsed, "s  ", gflops, " GFLOPS/s")

    matmul_tiled_simd(A, B, C, M, N, K)
    var t0 = perf_counter()
    for _ in range(5):
        matmul_tiled_simd(A, B, C, M, N, K)
    var t1 = perf_counter()
    var elapsed = (t1 - t0) / 5.0
    var flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    var gflops = flops / (elapsed * 1e9)

    print("Tiled+SIMD+Par:   ", elapsed, "s  ", gflops, " GFLOPS/s")

    A.free()
    B.free()
    C.free()


def verify():
    var M = 256
    var A = alloc[Scalar[float_type]](M * M)
    var B = alloc[Scalar[float_type]](M * M)
    var C_tiled = alloc[Scalar[float_type]](M * M)
    var C_ref = alloc[Scalar[float_type]](M * M)

    for i in range(M * M):
        A[i] = Float32(Int(i) % 100) / 100.0
        B[i] = Float32(Int(i * 7) % 100) / 100.0

    matmul_naive(A, B, C_ref, M, M, M)
    matmul_tiled_simd(A, B, C_tiled, M, M, M)

    var max_err: Float32 = 0.0
    for i in range(M * M):
        var err = abs(C_tiled[i] - C_ref[i])
        if err > max_err:
            max_err = err
    print("Max error: ", max_err)
    print("Correctness: ", max_err < 0.1)

    A.free()
    B.free()
    C_tiled.free()
    C_ref.free()


def main():
    run_benchmark(256)
    run_benchmark(512)
    run_benchmark(1024)
    verify()
