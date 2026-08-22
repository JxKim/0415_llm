#!/bin/bash
echo "=== 合并 8B DPO 模型（screen）==="
cd /root/autodl-tmp/finetune_proj
screen -dmS merge_dpo8b bash -c 'cd /root/autodl-tmp/finetune_proj && /root/miniconda3/bin/python -u scripts/merge_model.py --base output/sft_model_8b_merged --adapter output/dpo_model_8b --output output/dpo_model_8b_merged > output/merge_dpo8b.log 2>&1'
echo "合并 screen 已启动"
echo "=== 安装 evalscope（后台）==="
nohup /root/miniconda3/bin/pip install evalscope > /root/evalscope_install.log 2>&1 &
echo "evalscope 安装 PID=$!"
