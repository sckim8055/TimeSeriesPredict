#!/bin/bash
# configs/ 의 실험을 순서대로 실행. 로그는 results/logs/
cd "$(dirname "$0")"; mkdir -p results/logs
for f in configs/*.json; do n=$(basename "$f" .json); echo "== $n"; OMP_NUM_THREADS=2 python3 cli.py run "$f" > "results/logs/$n.txt" 2>&1 || echo "   FAIL (results/logs/$n.txt)"; tail -1 "results/logs/$n.txt" | cut -c1-200; done
