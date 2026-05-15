# ===----------------------------------------------------------------------=== #
# Blackwell (sm_120a) tcgen05+TMA+TMEM GEMM kernel for RTX Pro 6000
#
# Uses NVIDIA's 5th-gen tensor cores via tcgen05.mma, TMA for async
# global->shared copies, and Tensor Memory (TMEM) for accumulators.
#
# Block tile: BM=128, BN=256, BK=64, with UMMA shape 64x256x16 (F16 kind).
# All data in BF16, accumulated in F32.
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv
from std.sys import size_of

from std.gpu import WARP_SIZE, barrier
from std.gpu import thread_idx, block_idx, lane_id, warp_id
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import external_memory
from std.gpu.compute.arch.mma_nvidia_sm100 import *
from std.gpu.compute.arch.tcgen05 import *
from layout import IntTuple, Layout, LayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

# --------------------------------------------------------------------------- #
# Block & MMA tile dimensions
# --------------------------------------------------------------------------- #
comptime BM = 128
comptime BN = 256
comptime BK = 64
comptime MMA_M = 64
comptime MMA_N = 256
comptime MMA_K = 16
comptime num_m_mmas = BM // MMA_M  # 2
comptime num_n_mmas = BN // MMA_N  # 1
comptime num_k_mmas = BK // MMA_K  # 4

comptime num_threads = 128
comptime num_warps = num_threads // WARP_SIZE  # 4

comptime a_type = DType.bfloat16
comptime b_type = DType.bfloat16
comptime c_type = DType.bfloat16
comptime accum_type = DType.float32

comptime a_swizzle = TensorMapSwizzle.SWIZZLE_128B
comptime b_swizzle = TensorMapSwizzle.SWIZZLE_128B

comptime transpose_b = True

# C fragment per thread after tcgen05_ld
comptime c_frag_size = MMA_M * MMA_N // num_threads  # 64*256/128 = 128

# Shared memory layouts
comptime a_smem_layout = tile_layout_k_major[
    a_type, BM, BK, swizzle_mode=a_swizzle
]()
comptime b_smem_layout = tile_layout_k_major[
    b_type, BN, BK, swizzle_mode=b_swizzle
]() if transpose_b else tile_layout_mn_major[
    b_type, BN, BK, swizzle_mode=b_swizzle
]()

comptime sub_a_smem_layout = tile_layout_k_major[
    a_type, BM, 64, swizzle_mode=a_swizzle
]()
comptime sub_b_smem_layout = tile_layout_k_major[
    b_type, BN, 64, swizzle_mode=b_swizzle
]() if transpose_b else tile_layout_mn_major[
    b_type, BN, 64, swizzle_mode=b_swizzle
]()

# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
def kernel_blackwell[
    a_tma_rank: Int,
    b_tma_rank: Int,
    a_tile_shape: IndexList[a_tma_rank],
    b_tile_shape: IndexList[b_tma_rank],
    c_layout: Layout,
    a_desc_shape: IndexList[a_tma_rank],
    b_desc_shape: IndexList[b_tma_rank],
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
](
    a_tma_op: TMATensorTile[a_type, a_tma_rank, a_tile_shape, a_desc_shape],
    b_tma_op: TMATensorTile[b_type, b_tma_rank, b_tile_shape, b_desc_shape],
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    num_iters: Int,
):
    # --- shared memory allocation ---
    comptime a_smem_tile_t = LayoutTensor[
        a_type, a_smem_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED, alignment=128,
    ]
    comptime b_smem_tile_t = LayoutTensor[
        b_type, b_smem_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED, alignment=128,
    ]
    comptime sub_a_smem_tile_t = LayoutTensor[
        a_type, sub_a_smem_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED, alignment=128,
    ]
    comptime sub_b_smem_tile_t = LayoutTensor[
        b_type, sub_b_smem_layout, MutAnyOrigin,
        address_space=AddressSpace.SHARED, alignment=128,
    ]

    a_smem = rebind[
        UnsafePointer[
            Scalar[a_type],
            address_space=AddressSpace.SHARED,
            ExternalOrigin[mut=True],
        ]
    ](
        external_memory[
            Scalar[a_type], address_space=AddressSpace.SHARED,
            alignment=128, name="blackwell_dynamic_shared",
        ]()
    )

    comptime a_size = a_smem_layout.size()
    comptime b_size = b_smem_layout.size()

    var b_smem = (a_smem + a_size).bitcast[Scalar[b_type]]()

    var a_smem_tile = a_smem_tile_t(a_smem)
    var b_smem_tile = b_smem_tile_t(b_smem)

    # --- C fragment registers ---
    var c_frag: InlineArray[Scalar[accum_type], c_frag_size]

    # --- barriers ---
    comptime a_expected_bytes = a_size * size_of[a_type]()
    comptime b_expected_bytes = b_size * size_of[b_type]()
    comptime expected_bytes = a_expected_bytes + b_expected_bytes

    tma_mbar = (b_smem + b_size).bitcast[SharedMemBarrier]()
    mma_mbar = tma_mbar + 1
    var ptr_tmem_addr = (mma_mbar + 1).bitcast[UInt32]()

    if thread_idx.x == 0:
        tma_mbar[0].init()
        mma_mbar[0].init()

    var tma_phase: UInt32 = 0
    var mma_phase: UInt32 = 0

    comptime max_tmem_cols = 512

    # allocate tensor memory
    if warp_id() == 0:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    barrier()

    tmem_addr = ptr_tmem_addr[0]

    # --- MMA descriptors ---
    comptime a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    comptime b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    comptime aSBO = a_canonical_layout[0].stride[1].value() * size_of[a_type]()
    comptime aLBO = a_canonical_layout[1].stride[1].value() * size_of[a_type]()
    comptime b_stride01 = b_canonical_layout[0].stride[1].value()
    comptime b_stride11 = b_canonical_layout[1].stride[1].value()
    comptime bSBO = (b_stride01 if transpose_b else b_stride11) * size_of[b_type]()
    comptime bLBO = (b_stride11 if transpose_b else b_stride01) * size_of[b_type]()

    adesc = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem_tile.ptr)
    bdesc = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem_tile.ptr)

    idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type, a_type, b_type,
        Index[dtype=DType.uint32](MMA_M, MMA_N),
        transpose_b=transpose_b,
    ]()

    # --- main loop over K tiles ---
    for i in range(num_iters):
        # --- TMA: copy A and B tiles ---
        if thread_idx.x == 0:
            tma_mbar[0].expect_bytes(Int32(expected_bytes))

            comptime for j in range(BK // 64):
                comptime k = 64 * j
                comptime a_offset = a_smem_layout(IntTuple(0, k))
                comptime b_offset = b_smem_layout(IntTuple(0, k))
                sub_a_smem_tile = sub_a_smem_tile_t(a_smem + a_offset)
                a_tma_op.async_copy(
                    sub_a_smem_tile, tma_mbar[0],
                    (i * BK + k, block_idx.y * BM),
                )
                sub_b_smem_tile = sub_b_smem_tile_t(b_smem + b_offset)
                b_tma_op.async_copy(
                    sub_b_smem_tile, tma_mbar[0],
                    (
                        i * BK + k,
                        block_idx.x * BN,
                    ) if transpose_b else (
                        block_idx.x * BN,
                        i * BK + k,
                    ),
                )

        tma_mbar[0].wait(tma_phase)
        tma_phase ^= 1

        # --- tcgen05 MMA ---
        if thread_idx.x == 0:
            comptime for j in range(num_k_mmas):
                comptime idx = IntTuple(0, MMA_K * j)
                comptime a_offset = a_smem_layout(idx) * size_of[a_type]()
                comptime b_offset = b_smem_layout(idx) * size_of[b_type]()

                var c_scale_value: UInt32 = UInt32(
                    0 if (i == 0 and j == 0) else 1
                )
                mma(
                    adesc + a_offset,
                    bdesc + b_offset,
                    tmem_addr,
                    idesc,
                    c_scale=c_scale_value,
                )

        mma_arrive(mma_mbar)
        mma_mbar[0].wait(mma_phase)
        mma_phase ^= 1

    # --- read C from TMEM into registers ---
    c_frag = tcgen05_ld[
        datapaths=16, bits=256, repeat=BN // 8,
        dtype=accum_type, pack=False, width=c_frag_size,
    ](tmem_addr)

    tcgen05_load_wait()

    # --- release TMEM ---
    if warp_id() == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, max_tmem_cols)

    # --- write C to global memory ---
    ctile = c.tile[BM, BN](block_idx.y, block_idx.x)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma
            c_gmem_warp_tile = ctile.tile[MMA_M // num_warps, MMA_N](
                4 * m_mma + warp_id(), n_mma
            )
            c_gmem_frag = c_gmem_warp_tile.vectorize[1, 2]().distribute[
                Layout.row_major(8, 4)
            ](lane_id())

            comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
            comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

            comptime for n_vec in range(num_vecs_n):
                comptime for m_vec in range(num_vecs_m):
                    comptime i_vec = n_vec * num_vecs_m + m_vec
                    c_gmem_frag[m_vec, n_vec] = rebind[
                        c_gmem_frag.element_type
                    ](
                        SIMD[accum_type, 2](
                            c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                        ).cast[c_type]()
                    )


# --------------------------------------------------------------------------- #
# Host launcher
# --------------------------------------------------------------------------- #
def benchmark_blackwell[
    M: Int, N: Int, K: Int,
](ctx: DeviceContext) raises:
    print("  M=", M, " N=", N, " K=", K)

    comptime a_layout = Layout.row_major(M, K)
    comptime b_layout = Layout.row_major(N, K) if transpose_b else Layout.row_major(K, N)
    comptime c_layout = Layout.row_major(M, N)

    # allocate host + device buffers
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var a_host = LayoutTensor[a_type, a_layout](a_host_ptr)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var b_host = LayoutTensor[b_type, b_layout](b_host_ptr)

    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var a_device_lt = LayoutTensor[a_type, a_layout](a_device.unsafe_ptr())
    var b_device = ctx.enqueue_create_buffer[b_type](N * K)
    var b_device_lt = LayoutTensor[b_type, b_layout](b_device.unsafe_ptr())
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_lt = LayoutTensor[c_type, c_layout](c_device.unsafe_ptr())

    # init A with sequential values, B with identity-like pattern
    for m_idx in range(M):
        for k_idx in range(K):
            a_host[m_idx, k_idx] = Float32(k_idx).cast[a_type]()
    for n_idx in range(N):
        for k_idx in range(K):
            b_host[n_idx, k_idx] = Float32(1 if n_idx == k_idx else 0).cast[b_type]()
    for i in range(M * N):
        (c_device_lt.ptr + i).store(Scalar[c_type](0))

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    # create TMA tiles
    a_tma_op = create_tensor_tile[
        Index(BM, 64), swizzle_mode=a_swizzle
    ](ctx, a_device_lt)
    b_tma_op = create_tensor_tile[
        Index(BN, 64) if transpose_b else Index(64, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b_device_lt)

    # launch kernel
    comptime block_tile_shape = Index(BM, BN, BK)
    comptime umma_shape = Index(MMA_M, MMA_N, MMA_K)

    comptime kernel = kernel_blackwell[
        type_of(a_tma_op).rank,
        type_of(b_tma_op).rank,
        type_of(a_tma_op).tile_shape,
        type_of(b_tma_op).tile_shape,
        c_layout,
        type_of(a_tma_op).desc_shape,
        type_of(b_tma_op).desc_shape,
    ]

    comptime smem_use = (BM * size_of[a_type]() + BN * size_of[b_type]()) * BK + 24

    ctx.enqueue_function[kernel](
        a_tma_op, b_tma_op, c_device_lt, K // BK,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(num_threads),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )

    ctx.synchronize()

    # benchmark
    comptime num_warmup = 10
    comptime num_runs = 100

    @always_inline
    @parameter
    def run_kernel(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            a_tma_op, b_tma_op, c_device_lt, K // BK,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(num_threads),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )

    for _ in range(num_warmup):
        run_kernel(ctx)
    ctx.synchronize()
    print("  warmup done")

    var nstime = Float64(ctx.execution_time[run_kernel](num_runs)) / num_runs
    var sectime = nstime * 1e-9
    var TFlop = 2.0 * Float64(M) * Float64(N) * Float64(K) * 1e-12

    print("  Avg time: ", sectime * 1000, " ms")
    print("  Performance: ", TFlop / sectime, " TFLOPS")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() raises:
    with DeviceContext() as ctx:
        print("Blackwell tcgen05+TMA+TMEM GEMM benchmark (RTX Pro 6000)")
        print("  BM=", BM, " BN=", BN, " BK=", BK)
        print("  UMMA shape: ", MMA_M, "x", MMA_N, "x", MMA_K)
        print("  a_type=", a_type, " b_type=", b_type, " c_type=", c_type)
        print("  accum_type =", accum_type)
        print("  swizzle: a=", a_swizzle, " b=", b_swizzle)
        print()

        benchmark_blackwell[4096, 4096, 4096](ctx)
        benchmark_blackwell[8192, 8192, 8192](ctx)
        benchmark_blackwell[16384, 16384, 16384](ctx)
