#!/bin/bash
echo "=== 训练进程 ==="
ps aux | grep "scripts/train_dpo" | grep -v grep | head -1 | awk '{print "PID:", $2, "CPU:", $3"%"}' || echo "无进程（可能已完成）"
echo "=== 最新进度 ==="
grep -oE "[0-9]+/64" /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null | tail -1
echo "=== adapter 检查 ==="
ls /root/autodl-tmp/finetune_proj/output/dpo_model_8b/ 2>/dev/null | head -6
