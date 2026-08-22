#!/bin/bash
echo "=== 清理 transformers __pycache__ ==="
find /root/miniconda3/lib/python3.12/site-packages/transformers -name "__pycache__" -type d 2>/dev/null | wc -l
find /root/miniconda3/lib/python3.12/site-packages/transformers -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "清理完成"
echo "=== 重新测试 import ==="
/root/miniconda3/bin/python -c "import transformers; print('tf', transformers.__version__)" 2>&1 | tail -1
/root/miniconda3/bin/python -c "from trl import DPOTrainer; print('DPOTrainer OK')" 2>&1 | tail -1
