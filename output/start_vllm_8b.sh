#!/bin/bash
# 下载 ninja 二进制并重启 vLLM
echo "=== 下载 ninja ==="
cd /tmp
curl -sL -o ninja-linux.zip https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip 2>&1
ls -la ninja-linux.zip 2>/dev/null && echo "下载成功" || echo "下载失败"
unzip -o ninja-linux.zip -d /usr/local/bin/ 2>&1 | tail -2
chmod +x /usr/local/bin/ninja
which ninja && ninja --version
pkill -f "vllm serve" 2>/dev/null
sleep 2
PY=/root/miniconda3/bin
nohup $PY/vllm serve /root/autodl-tmp/sft_model_8b_merged \
    --served-model-name Qwen3-8B-sft \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port 8000 > /root/vllm_8b.log 2>&1 &
echo "vLLM 重启 PID=$!"
