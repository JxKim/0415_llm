#!/bin/bash
echo "=== 启动 8B DPO 训练（screen）==="
pkill -f train_dpo 2>/dev/null
sleep 2
cd /root/autodl-tmp/finetune_proj
screen -dmS dpo8b bash -c 'cd /root/autodl-tmp/finetune_proj && /root/miniconda3/bin/python -u scripts/train_dpo.py --base output/sft_model_8b_merged --data data/dpo_pairs.jsonl --output output/dpo_model_8b --epochs 2 --lr 1e-5 --batch 2 --accum 16 --qlora > output/dpo_8b.log 2>&1'
sleep 6
echo "=== screen 会话 ==="
screen -ls | grep dpo8b
