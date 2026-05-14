#!/bin/bash
set -e

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "==============================================="
echo "  GPU Matrix Multiplication — Mojo vs PyTorch"
echo "==============================================="
echo ""

echo "--- Mojo GPU Matmul (F32, simdgroup MMA) ---"
pixi run mojo benchmarks/matmul_gpu.mojo 2>&1 | tee "$TMPDIR/mojo.txt"
echo ""

echo "--- Mojo GPU Matmul (F16 input, F32 accum) ---"
pixi run mojo benchmarks/matmul_gpu_f16.mojo 2>&1 | tee "$TMPDIR/mojo_f16.txt"
echo ""

echo "--- Mojo GPU Matmul (general-purpose, FMA tiled) ---"
pixi run mojo benchmarks/matmul_gpu_general.mojo 2>&1 | tee "$TMPDIR/mojo_general.txt"
echo ""

echo "--- PyTorch Matmul ---"
pixi run python benchmarks/torch_matmul_mps.py 2>&1 | tee "$TMPDIR/torch.txt"
echo ""

echo "--- Warp Matmul (CUDA only) ---"
pixi run python benchmarks/warp_matmul_gpu.py 2>&1 | tee "$TMPDIR/warp.txt"
echo ""

echo "==============================================="
echo "  SUMMARY (GFLOPS/s)"
echo "==============================================="
echo ""

MOJO_VALS=($(grep "GPU Matmul:" "$TMPDIR/mojo.txt" | awk '{print $(NF-1)}'))
MOJO_F16_VALS=($(grep "GPU Matmul:" "$TMPDIR/mojo_f16.txt" | awk '{print $(NF-1)}'))
MOJO_GEN_VALS=($(grep "GPU Matmul:" "$TMPDIR/mojo_general.txt" | awk '{print $(NF-1)}'))
TORCH_VALS=($(grep -E "^PyTorch (mps|cuda|cpu) [0-9]" "$TMPDIR/torch.txt" | awk '{print $(NF-1)}'))
WARP_VALS=($(grep "GPU Tile" "$TMPDIR/warp.txt" | awk '{print $(NF-1)}'))
SIZES=(256 512 1024 2048)

printf "  %-7s | %-12s | %-12s | %-12s | %-12s | %s\n" "Size" "Mojo F32" "Mojo F16" "Mojo Gen" "PyTorch" "Warp"
printf "  %-7s | %-12s | %-12s | %-12s | %-12s | %s\n" "-------" "------------" "------------" "------------" "------------" "------------"
for i in "${!SIZES[@]}"; do
    m="${MOJO_VALS[$i]:----}"
    mf="${MOJO_F16_VALS[$i]:----}"
    mg="${MOJO_GEN_VALS[$i]:----}"
    t="${TORCH_VALS[$i]:----}"
    w="${WARP_VALS[$i]:-N/A}"
    printf "  %-7s | %-12s | %-12s | %-12s | %-12s | %s\n" "${SIZES[$i]}" "$m" "$mf" "$mg" "$t" "$w"
done
echo ""
