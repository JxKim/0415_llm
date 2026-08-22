#!/bin/bash
echo "=== 训练进度（最后进度条）==="
grep -oE "[0-9]+/64" /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null | tail -1
echo "=== 最后 loss ==="
grep -oE "\{'loss': '[0-9.]+'.*rewards/accuracies': '[0-9.]+'" /root/autodl-tmp/finetune_proj/output/dpo_8b.log 2>/dev/null | tail -1
echo "=== 进程 CPU 时间 ==="
ps -o pid,etime,time,%cpu -p 14950 2>/dev/null | tail -1
