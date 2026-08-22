#!/bin/bash
echo "=== 所有 python 进程 ==="
ps aux | grep -E "python" | grep -v grep | head -8
echo "=== 显存占用进程 ==="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | head -5
echo "=== dmesg 最近 kill ==="
dmesg 2>/dev/null | tail -8
