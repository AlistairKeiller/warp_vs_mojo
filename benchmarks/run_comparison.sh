#!/bin/bash
echo "============================================"
echo "  Mojo vs Warp: Matrix Multiplication"
echo "  Apple M1 Max — CPU only"
echo "============================================"
echo ""

echo "━━━ Mojo ━━━"
pixi run mojo benchmarks/matmul.mojo 2>&1 | grep -E "(Size|Naive|Tiled|Max error|Correctness)"

echo ""
echo "━━━ Warp ━━━"
pixi run python benchmarks/warp_matmul.py 2>&1 | grep -E "(Size|Naive|Tile|Max error|Correctness)"

echo ""
echo "============================================"
echo "  Summary (GFLOPS/s)"
echo "============================================"
echo ""
echo "  Size    | Warp (Tile) | Mojo (Opt)  | Speedup"
echo "  --------+-------------+-------------+--------"

for size in 256 512 1024; do
  warp_gflops=$(pixi run python -c "
import re
import subprocess
result = subprocess.run(['pixi', 'run', 'python', 'benchmarks/warp_matmul.py'], capture_output=True, text=True, cwd='.')
lines = result.stdout.split('\n')
in_size = False
for line in lines:
    if '--- Size $size ---' in line:
        in_size = True
    elif in_size and 'Tile:' in line:
        m = re.search(r'([\d.]+) GFLOPS/s', line)
        if m:
            print(m.group(1))
        break
" 2>/dev/null)

  mojo_gflops=$(pixi run mojo benchmarks/matmul.mojo 2>&1 | grep "Size $size" -A1 | grep "Tiled" | grep -oP '([\d.]+)(?= GFLOPS/s)')

  if [ -n "$warp_gflops" ] && [ -n "$mojo_gflops" ]; then
    speedup=$(echo "scale=1; $mojo_gflops / $warp_gflops" | bc)
    printf "  %-7d | %-11s | %-11s | %.1fx\n" $size "$warp_gflops" "$mojo_gflops" $speedup
  fi
done
