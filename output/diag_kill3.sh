#!/bin/bash
echo "=== screen 会话 ==="
screen -ls 2>/dev/null | head -5
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
echo "=== 系统日志（找 kill/cleanup）==="
journalctl -n 20 --no-pager 2>/dev/null | tail -10 || echo "journalctl 不可用"
echo "=== 最近进程启动时间 ==="
ps -eo pid,lstart,cmd 2>/dev/null | grep -E "SCREEN|train_dpo" | grep -v grep | head -3
