#!/bin/bash
echo "=== adapter 确认 ==="
ls -la /root/autodl-tmp/finetune_proj/output/sft_model_8b/ 2>/dev/null | head -8
echo "=== 启动合并 ==="
cd /root/autodl-tmp/finetune_proj
screen -dmS merge8b bash -c 'cd /root/autodl-tmp/finetune_proj && /root/miniconda3/bin/python -u scripts/merge_model.py --base /root/.cache/modelscope/hub/Qwen3-8B --adapter output/sft_model_8b --output output/sft_model_8b_merged > output/merge_8b.log 2>&1'
echo "合并 screen 已启动"
