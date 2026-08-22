#!/bin/bash
echo "=== 用 setsid 重启 8B DPO（彻底脱离 ssh 会话）==="
cd /root/autodl-tmp/finetune_proj
pkill -f train_dpo 2>/dev/null
sleep 2
setsid nohup /root/miniconda3/bin/python -u scripts/train_dpo.py --base /root/autodl-tmp/sft_model_8b_merged --data data/dpo_pairs.jsonl --output output/dpo_model_8b --epochs 2 --lr 1e-5 --batch 2 --accum 16 --qlora > output/dpo_8b.log 2>&1 < /dev/null &
echo "启动 PID=$!"
sleep 5
echo "=== 确认进程存活 ==="
ps aux | grep train_dpo | grep -v grep | awk '{print $2, $3"% CPU"}' | head -1
