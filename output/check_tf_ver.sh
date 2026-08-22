#!/bin/bash
echo "=== transformers 实际版本 ==="
/root/miniconda3/bin/python -c "import transformers; print(transformers.__version__); print(transformers.__file__)" 2>&1 | tail -2
echo "=== TRANSFORMERS_CACHE 检查 ==="
/root/miniconda3/bin/python -c "from transformers.utils.hub import TRANSFORMERS_CACHE; print('有 TRANSFORMERS_CACHE')" 2>&1 | tail -1
echo "=== 本地(Windows)对比 ==="
echo "本地 transformers 版本见日志"
