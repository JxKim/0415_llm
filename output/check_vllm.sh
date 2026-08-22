#!/bin/bash
# 检查 vllm 并启动 8B 推理服务
PY=/root/miniconda3/bin/python
echo "=== vllm 检查 ==="
which vllm 2>/dev/null && echo "vllm CLI 存在" || echo "无 vllm CLI"
$PY -c "import vllm; print('vllm python 包:', vllm.__version__)" 2>&1 | tail -1
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
