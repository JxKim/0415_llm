#!/bin/bash
echo "=== 用 screen 启动 8B DPO ==="
pkill -f train_dpo 2>/dev/null
sleep 2
screen -dmS dpo bash -c 'cd /root/autodl-tmp/finetune_proj && /root/miniconda3/bin/python -u scripts/train_dpo.py --base /root/autodl-tmp/sft_model_8b_merged --data data/dpo_pairs.jsonl --output output/dpo_model_8b --epochs 2 --lr 1e-5 --batch 2 --accum 16 --qlora > output/dpo_8b.log 2>&1'
echo "screen 会话已创建"
sleep 5
echo "=== screen 会话 ==="
screen -ls | grep dpo
echo "=== 进程 ==="
ps aux | grep train_dpo | grep -v grep | awk '{print $2}' | head -1
