#!/bin/bash
echo "=== 完整日志（找错误）==="
cat /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null | grep -vE "it/s\]|examples/s" | tail -30
echo "=== 日志行数 ==="
wc -l /root/autodl-tmp/finetune_proj/output/dpo_8b.log
