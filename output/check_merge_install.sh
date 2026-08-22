#!/bin/bash
echo "=== 合并结果 ==="
tail -2 /root/autodl-tmp/finetune_proj/output/merge_dpo8b.log 2>/dev/null
ls /root/autodl-tmp/finetune_proj/output/dpo_model_8b_merged/model.safetensors 2>/dev/null && echo "DPO 合并 OK" || echo "合并未完成"
echo "=== evalscope 安装 ==="
tail -2 /root/evalscope_install.log 2>/dev/null
/root/miniconda3/bin/python -c "import evalscope; print('evalscope', evalscope.__version__)" 2>&1 | tail -1
