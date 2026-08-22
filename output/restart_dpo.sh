#!/bin/bash
echo "=== 停掉 vLLM 服务 ==="
pkill -f "vllm serve" 2>/dev/null
sleep 8
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "=== 重启 8B DPO ==="
cd /root/autodl-tmp/finetune_proj
pkill -f train_dpo 2>/dev/null
sleep 1
nohup /root/miniconda3/bin/python -u scripts/train_dpo.py --base /root/autodl-tmp/sft_model_8b_merged --data data/dpo_pairs.jsonl --output output/dpo_model_8b --epochs 2 --lr 1e-5 --batch 2 --accum 16 --qlora > output/dpo_8b.log 2>&1 &
echo "8B DPO PID=$!"
