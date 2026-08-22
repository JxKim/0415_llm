#!/bin/bash
echo "=== vllm 进程 ==="
ps aux | grep "vllm serve" | grep -v grep | awk '{print $2, $3"%CPU", $11, $12}' | head -2
echo "=== 服务检查 ==="
timeout 10 curl -s http://127.0.0.1:8000/v1/models | head -c 200
echo ""
echo "=== 日志尾部 ==="
tail -5 /root/vllm_8b.log
