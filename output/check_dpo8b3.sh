#!/bin/bash
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "=== 训练进程 ==="
ps aux | grep train_dpo | grep -v grep | head -1 || echo "无训练进程"
echo "=== 日志尾部 ==="
tail -12 /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null
