#!/bin/bash
# 检查 vllm 安装进度
echo "=== 安装日志尾部 ==="
tail -8 /root/vllm_install.log 2>/dev/null || echo "日志不存在"
echo "=== 安装进程 ==="
ps aux | grep -E "pip|vllm" | grep -v grep | head -3 || echo "无安装进程"
echo "=== vllm 可用性 ==="
/root/miniconda3/bin/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
