#!/bin/bash
# 云环境检查脚本
PY=/root/miniconda3/bin/python
echo "=== Python ==="
$PY --version
echo "=== torch ==="
$PY - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "=== 关键包 ==="
$PY - <<'EOF'
import transformers, trl, peft, datasets, modelscope
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("modelscope", modelscope.__version__)
EOF
echo "=== 模型目录 ==="
ls ~/.cache/hub/modelscope/ 2>/dev/null | head -20
echo "=== 磁盘空间 ==="
df -h /root | tail -1
echo "=== 内存 ==="
free -g | head -2
