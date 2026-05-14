#!/bin/bash
set -e

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "==============================================="
echo "  Mojo vs Warp — GPU Matrix Multiplication"
echo "  Auto-select: Mojo(Metal/CUDA)  Warp(CUDA)"
echo "==============================================="
echo ""

echo "==============================================="
echo "  1. Mojo GPU Matmul"
echo "     Auto: Metal on macOS, CUDA on NVIDIA"
echo "==============================================="
pixi run mojo benchmarks/matmul_gpu.mojo 2>&1 | tee "$TMPDIR/mojo.txt"
echo ""

echo "==============================================="
echo "  2. Warp GPU Matmul"
echo "     Requires NVIDIA CUDA"
echo "==============================================="
pixi run python benchmarks/warp_matmul_gpu.py 2>&1 | tee "$TMPDIR/warp.txt"
echo ""

echo "==============================================="
echo "  SUMMARY (GFLOPS/s)"
echo "==============================================="
echo ""

if grep -q "No CUDA" "$TMPDIR/warp.txt"; then
    printf "  %-7s | %-30s | %s\n" "Size" "Mojo (Metal/CUDA)" "Warp"
    printf "  %-7s | %-30s | %s\n" "-------" "------------------------------" "---------------------------"
    sizes=(256 512 1024)
    i=0
    while read -r gflops; do
        if [ $i -lt ${#sizes[@]} ]; then
            size=${sizes[$i]}
            printf "  %-7s | %-30s | %s\n" "$size" "$gflops GFLOPS/s" "No NVIDIA GPU"
            i=$((i + 1))
        fi
    done < <(grep "GPU Matmul:" "$TMPDIR/mojo.txt" | awk '{print $(NF-1)}')
else
    printf "  %-7s | %-30s | %-25s\n" "Size" "Mojo (Metal/CUDA)" "Warp (CUDA)"
    printf "  %-7s | %-30s | %-25s\n" "-------" "------------------------------" "-------------------------"
    sizes=(256 512 1024)
    i=0
    while IFS=$'\t' read -r mojo warp; do
        if [ $i -lt ${#sizes[@]} ]; then
            size=${sizes[$i]}
            winner=$(echo "$mojo > $warp" | bc -l 2>/dev/null)
            label="Warp wins"
            [ "$winner" = "1" ] && label="Mojo wins"
            printf "  %-7s | %-30s | %-25s\n" "$size" "$mojo GFLOPS/s" "$warp GFLOPS/s ($label)"
            i=$((i + 1))
        fi
    done < <(paste \
        <(grep "GPU Matmul:" "$TMPDIR/mojo.txt" | awk '{print $(NF-1)}') \
        <(grep "GPU Tile (record_cmd):" "$TMPDIR/warp.txt" | awk '{print $(NF-1)}'))
fi
echo ""
