#!/bin/bash
echo "=== vllm 日志中关键错误 ==="
grep -nE "Error|error|Traceback|CUDA|assert|Failed" /root/vllm_8b.log | head -25
echo ""
echo "=== 日志总行数 ==="
wc -l /root/vllm_8b.log
