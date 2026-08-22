#!/bin/bash
echo "=== eval 进程 ==="
ps aux | grep eval_runner | grep -v grep | head -1 | awk '{print "PID:", $2}' || echo "无进程（可能已完成）"
echo "=== 最终分数 ==="
grep "mean_acc" /root/autodl-tmp/finetune_proj/output/eval_dpo_8b.log 2>/dev/null | tail -6
