#!/bin/bash
echo "=== 连接成功 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
PY=/root/miniconda3/bin/python
[ -x $PY ] || PY=$(which python)
echo "Python: $($PY --version 2>&1)"
$PY - <<'EOF'
import transformers, trl
print("transformers", transformers.__version__)
print("trl", trl.__version__)
EOF
echo "=== 8B 模型 ==="
ls /root/.cache/modelscope/hub/ 2>/dev/null
echo "=== autodl-tmp ==="
ls /root/autodl-tmp/ 2>/dev/null
df -h /root/autodl-tmp 2>/dev/null | tail -1
