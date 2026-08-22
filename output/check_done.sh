#!/bin/bash
echo "=== 训练进程 ==="
ps aux | grep "scripts/train_dpo" | grep -v grep | head -1 || echo "无训练进程（可能已完成）"
echo "=== 日志尾部 ==="
tail -10 /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null
echo "=== 输出目录 ==="
ls /root/autodl-tmp/finetune_proj/output/dpo_model_8b/ 2>/dev/null | head -6
