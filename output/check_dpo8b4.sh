#!/bin/bash
echo "=== 显存 ==="
timeout 15 nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi 失败"
echo "=== 进程 ==="
ps aux | grep train_dpo | grep -v grep | head -1 || echo "无 train_dpo 进程"
echo "=== screen 会话 ==="
screen -ls 2>/dev/null | grep dpo || echo "无 screen 会话"
echo "=== 训练进度 ==="
tail -4 /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null
