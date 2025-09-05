#!/bin/bash
#
# Quick test script to verify Python compatibility across supported versions
#

set -e

echo "=== Python Compatibility Test ==="
echo "Testing Python version: $(python --version)"
echo

# Determine project root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "1. Testing config module import..."
python - <<PYCODE
import os, sys
sys.path.append('$PROJECT_ROOT')
import config
print('✓ Config module imports successfully')
PYCODE

echo

echo "2. Testing utils module import..."
python - <<PYCODE
import os, sys
sys.path.append('$PROJECT_ROOT')
from utils import run_command
print('✓ Utils module imports successfully')
PYCODE

echo

echo "3. Testing run_command function..."
python - <<'PYCODE'
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import run_command
output = run_command(['echo', 'Hello'], capture_output=True)
if output.stdout.strip() == 'Hello':
    print('✓ Subprocess test passed')
    sys.exit(0)
print('✗ Subprocess test failed')
sys.exit(1)
PYCODE

echo

echo "4. Testing LSTM script syntax..."
python -m py_compile "$PROJECT_ROOT/app-lstm/lstm.py" && echo "✓ LSTM script compiles successfully"

echo

echo "5. Testing profile script syntax..."
python -m py_compile "$PROJECT_ROOT/sample-collection-scripts/profile.py" && echo "✓ Profile script compiles successfully"

echo

echo "=== All Python compatibility tests passed! ==="
