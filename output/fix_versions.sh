#!/bin/bash
echo "=== 降级 transformers 到 5.2.0 ==="
/root/miniconda3/bin/pip install transformers==5.2.0 2>&1 | tail -3
echo "=== 验证 vllm 可导入 ==="
/root/miniconda3/bin/python -c "import vllm; print('vllm import OK', vllm.__version__)" 2>&1 | tail -1
echo "=== 验证 trl dpo_trainer 可导入 ==="
/root/miniconda3/bin/python -c "from trl import DPOTrainer; print('DPOTrainer OK')" 2>&1 | tail -1
