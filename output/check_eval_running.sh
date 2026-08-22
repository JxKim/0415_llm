#!/bin/bash
echo "=== eval 进程 ==="
ps aux | grep eval_runner | grep -v grep | head -1 | awk '{print "PID:", $2, "CPU:", $3"%"}' || echo "无进程"
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
echo "=== eval 日志尾部 ==="
tail -5 /root/autodl-tmp/finetune_proj/output/eval_dpo_8b.log 2>/dev/null
