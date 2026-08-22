#!/bin/bash
echo "=== mergekit 安装日志尾部 ==="
tail -5 /root/mergekit_install.log
echo "=== mergekit 是否可导入 ==="
/root/miniconda3/bin/python -c "import mergekit; print('mergekit OK')" 2>&1 | tail -1
echo "=== pip list 查 mergekit ==="
/root/miniconda3/bin/pip list 2>/dev/null | grep -i mergekit
