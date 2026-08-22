#!/bin/bash
echo "=== 云上 trl 版本 ==="
/root/miniconda3/bin/pip show trl 2>/dev/null | grep -E "Version|Location"
/root/miniconda3/bin/python -c "import trl; print('trl', trl.__version__)" 2>&1 | tail -1
echo "=== 云上 trl 是否引用 TRANSFORMERS_CACHE ==="
grep -rn "TRANSFORMERS_CACHE" /root/miniconda3/lib/python3.12/site-packages/trl/ 2>/dev/null | head -3
