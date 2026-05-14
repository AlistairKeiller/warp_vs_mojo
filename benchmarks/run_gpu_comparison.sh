#!/bin/bash
set -e

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "==============================================="
echo "  Mojo vs Warp — GPU Matrix Multiplication"
echo "  Auto: Mojo(Metal/CUDA)  Warp(CUDA only)"
echo "==============================================="
echo ""

echo "--- Mojo GPU Matmul ---"
pixi run mojo benchmarks/matmul_gpu.mojo 2>&1 | tee "$TMPDIR/mojo.txt"
echo ""

echo "--- Warp GPU Matmul ---"
pixi run python benchmarks/warp_matmul_gpu.py 2>&1 | tee "$TMPDIR/warp.txt"
echo ""

echo "==============================================="
echo "  SUMMARY (GFLOPS/s)"
echo "==============================================="
echo ""

MOJO_VALS=($(grep "GPU Matmul:" "$TMPDIR/mojo.txt" | awk '{print $(NF-1)}'))
SIZES=(256 512 1024 2048)

if grep -q "No CUDA" "$TMPDIR/warp.txt"; then
    printf "  %-7s | %-25s | %s\n" "Size" "Mojo (Metal/CUDA)" "Warp"
    printf "  %-7s | %-25s | %s\n" "-------" "-------------------------" "---------------------------"
    for i in "${!SIZES[@]}"; do
        printf "  %-7s | %-25s | %s\n" "${SIZES[$i]}" "${MOJO_VALS[$i]:-.} GFLOPS/s" "No NVIDIA GPU"
    done
else
    WARP_VALS=($(grep "GPU Tile (record_cmd):" "$TMPDIR/warp.txt" | awk '{print $(NF-1)}'))
    printf "  %-7s | %-25s | %-20s | %s\n" "Size" "Mojo (Metal/CUDA)" "Warp (CUDA)" ""
    printf "  %-7s | %-25s | %-20s | %s\n" "-------" "-------------------------" "--------------------" "--------"
    for i in "${!SIZES[@]}"; do
        mojo="${MOJO_VALS[$i]:-.}"
        warp="${WARP_VALS[$i]:-.}"
        if [ "$(echo "$mojo > $warp" | bc -l 2>/dev/null)" = "1" ]; then
            win="Mojo"
        else
            win="Warp"
        fi
        printf "  %-7s | %-25s | %-20s | %s\n" "${SIZES[$i]}" "$mojo GFLOPS/s" "$warp GFLOPS/s" "$win"
    done
fi
echo ""
