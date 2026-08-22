#!/bin/bash
echo "=== 卸载 llm_blender（trl 对其 import 是可选跳过，装了反而报错）==="
/root/miniconda3/bin/pip uninstall -y llm_blender 2>&1 | tail -2
echo "=== 验证 DPOTrainer import ==="
/root/miniconda3/bin/python -c "from trl import DPOTrainer; print('DPOTrainer OK')" 2>&1 | tail -1
