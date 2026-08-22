#!/bin/bash
echo "=== 训练日志尾部 ==="
tail -8 /root/autodl-tmp/finetune_proj/output/train_8b.log 2>/dev/null
echo "=== 最终 eval_loss ==="
grep "eval_loss" /root/autodl-tmp/finetune_proj/output/train_8b.log 2>/dev/null | tail -1
echo "=== 输出 ==="
ls /root/autodl-tmp/finetune_proj/output/sft_model_8b/ 2>/dev/null | head -6
