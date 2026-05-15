#!/bin/bash
set -e

# run_gpu_comparison.sh — Run all GPU matmul benchmarks and compare with PyTorch
# Detects GPU type automatically and runs the appropriate specialized kernel.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  GPU Matmul Benchmark Comparison${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# Detect GPU type
gpu_type=""
has_mps=$(pixi run python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "False")
has_cuda=$(pixi run python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [ "$has_cuda" = "True" ]; then
    gpu_type="nvidia"
    echo -e "${GREEN}Detected: NVIDIA GPU (CUDA)${NC}"
elif [ "$has_mps" = "True" ]; then
    gpu_type="apple"
    echo -e "${GREEN}Detected: Apple GPU (Metal)${NC}"
else
    echo -e "${RED}No GPU detected${NC}"
    exit 1
fi
echo ""

# ── 1. PyTorch baseline ──
echo -e "${YELLOW}═══ PyTorch Baseline ═══${NC}"
pixi run python3 torch_matmul_mps.py
echo ""

# ── 2. Mojo general FMA kernel ──
echo -e "${YELLOW}═══ Mojo General (FMA) Kernel ═══${NC}"
pixi run mojo matmul_gpu_general.mojo
echo ""

# ── 3. Mojo specialized kernel ──
if [ "$gpu_type" = "nvidia" ]; then
    # Check if Blackwell GPU (sm_120a)
    sm_version=$(pixi run python3 -c "import torch; v=torch.cuda.get_device_capability(); print(f'{v[0]}{v[1]}')" 2>/dev/null)
    if [ "$sm_version" = "120" ]; then
        echo -e "${YELLOW}═══ Mojo Blackwell tcgen05 MMA Kernel ═══${NC}"
        pixi run mojo matmul_rtxpro6000.mojo
    else
        echo -e "${YELLOW}═══ Mojo T4 Tensor-Core MMA Kernel ═══${NC}"
        pixi run mojo matmul_t4.mojo
    fi
elif [ "$gpu_type" = "apple" ]; then
    echo -e "${YELLOW}═══ Mojo M5 Pro Simdgroup MMA Kernel ═══${NC}"
    pixi run mojo matmul_m5pro.mojo
fi
echo ""

# ── Summary ──
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  All benchmarks complete${NC}"
echo -e "${CYAN}============================================${NC}"

echo ""
echo "See output above for individual benchmark results."
