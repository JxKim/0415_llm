#!/bin/bash
echo "=== 启动 8B DPO 评估（本地直评模式，screen）==="
pkill -f eval_runner 2>/dev/null
sleep 2
cd /root/autodl-tmp/finetune_proj
screen -dmS eval8b bash -c 'cd /root/autodl-tmp/finetune_proj && /root/miniconda3/bin/python -u scripts/eval_runner.py --model output/dpo_model_8b_merged --local-model --tag dpo_model_8b > output/eval_dpo_8b.log 2>&1'
sleep 6
echo "=== screen 会话 ==="
screen -ls | grep eval8b
