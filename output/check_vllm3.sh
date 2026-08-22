#!/bin/bash
echo "=== vllm 服务 ==="
curl -s http://127.0.0.1:8000/v1/models | head -c 150
echo ""
echo "=== 日志尾部 ==="
tail -6 /root/vllm_8b.log
echo "=== ninja ==="
which ninja 2>/dev/null || echo "无 ninja"
