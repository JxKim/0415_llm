#!/bin/bash
echo "=== screen 会话 ==="
screen -ls 2>/dev/null | head -6
echo "=== eval 进程 ==="
ps aux | grep eval_runner | grep -v grep | head -1 | awk '{print "PID:", $2, "CPU:", $3"%"}' || echo "无 eval 进程"
echo "=== eval 日志 ==="
tail -8 /root/autodl-tmp/finetune_proj/output/eval_dpo_8b.log 2>/dev/null || echo "无日志"
