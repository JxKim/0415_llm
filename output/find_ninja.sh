#!/bin/bash
echo "=== ninja 包位置 ==="
/root/miniconda3/bin/python -c "import ninja, os; print(os.path.dirname(ninja.__file__))" 2>&1
echo "=== ninja 包内容 ==="
find /root/miniconda3/lib/python3.12/site-packages/ninja -type f 2>/dev/null | head -20
echo "=== BIN_DIR ==="
/root/miniconda3/bin/python -c "from ninja import BIN_DIR; print(BIN_DIR)" 2>&1
ls -la $(/root/miniconda3/bin/python -c "from ninja import BIN_DIR; print(BIN_DIR)" 2>/dev/null) 2>/dev/null | head -10
