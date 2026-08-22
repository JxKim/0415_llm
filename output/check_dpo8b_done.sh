#!/bin/bash
echo "=== DPO 训练日志尾部 ==="
tail -8 /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null
echo "=== 最终指标 ==="
grep "rewards/accuracies" /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null | tail -1
echo "=== adapter ==="
ls /root/autodl-tmp/finetune_proj/output/dpo_model_8b/ 2>/dev/null | head -5
