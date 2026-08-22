#!/bin/bash
echo "=== 完整错误栈 ==="
/root/miniconda3/bin/python -c "from trl import DPOTrainer" 2>&1 | grep -E "File \"/|import" | head -20
