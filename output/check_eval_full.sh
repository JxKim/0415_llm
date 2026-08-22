#!/bin/bash
echo "=== eval 日志（尾部 30 行）==="
tail -30 /root/autodl-tmp/finetune_proj/output/eval_dpo_8b.log 2>/dev/null
