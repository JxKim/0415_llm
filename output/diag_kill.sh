#!/bin/bash
echo "=== dmesg OOM/kill 记录 ==="
dmesg 2>/dev/null | grep -iE "killed|oom" | tail -5 || echo "dmesg 不可用或无记录"
echo "=== tmux 可用性 ==="
which tmux screen 2>/dev/null || echo "无 tmux/screen"
echo "=== 系统内存 ==="
free -g | head -2
