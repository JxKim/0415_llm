#!/bin/bash
echo "=== eval 日志（完整错误）==="
grep -A5 -B2 "Error|error|No such|not found" /root/autodl-tmp/finetune_proj/output/eval_dpo_8b.log 2>/dev/null | head -20
echo "=== evalscope 命令检查 ==="
which evalscope
ls /root/miniconda3/bin/evalscope 2>/dev/null
