#!/bin/bash
echo "=== 安装 llm_blender ==="
/root/miniconda3/bin/pip install llm_blender 2>&1 | tail -2
echo "=== vllm 的 transformers 约束 ==="
/root/miniconda3/bin/pip show vllm 2>/dev/null | grep -i requires | head -3
