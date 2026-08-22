#!/bin/bash
echo "=== merge 日志 ==="
tail -4 /root/autodl-tmp/finetune_proj/output/merge_8b.log 2>/dev/null
echo "=== 合并输出 ==="
ls -la /root/autodl-tmp/finetune_proj/output/sft_model_8b_merged/ 2>/dev/null | grep -E "safetensors|config" | head -4
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
