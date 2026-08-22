#!/bin/bash
echo "=== 启动 8B SFT 训练（screen）==="
pkill -f train_sft 2>/dev/null
sleep 2
cd /root/autodl-tmp/finetune_proj
screen -dmS sft8b bash -c 'cd /root/autodl-tmp/finetune_proj && /root/miniconda3/bin/python -u scripts/train_sft.py --model /root/.cache/modelscope/hub/Qwen3-8B --data data/sft_train.jsonl --eval-data data/sft_test.jsonl --output output/sft_model_8b --epochs 3 --lr 2e-4 --batch 2 --accum 16 --max-len 1024 --qlora > output/train_8b.log 2>&1'
sleep 6
echo "=== screen 会话 ==="
screen -ls | grep sft8b
echo "=== 进程 ==="
ps aux | grep "scripts/train_sft" | grep -v grep | awk '{print $2}' | head -1
