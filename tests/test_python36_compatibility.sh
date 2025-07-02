#!/bin/bash
#
# Quick test script to verify Python 3.6 compatibility fixes
# Run this on the cluster to verify that the subprocess fixes work properly
#

echo "=== Python 3.6 Compatibility Test ==="
echo "Testing Python version: $(python --version)"
echo ""

echo "1. Testing config module import..."
python -c "import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('test_config.py')))); import config; print('✓ Config module imports successfully')" || {
    echo "✗ Config module import failed"
    exit 1
}

echo ""
echo "2. Testing utils module import..."
python -c "import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('test_subprocess_fix.py')))); from utils import run_command; print('✓ Utils module imports successfully')" || {
    echo "✗ Utils module import failed"
    exit 1
}

echo ""
echo "3. Testing subprocess functionality..."
python test_subprocess_fix.py || {
    echo "✗ Subprocess test failed"
    exit 1
}

echo ""
echo "4. Testing LSTM script syntax..."
python -m py_compile ../sample-collection-scripts/lstm.py && echo "✓ LSTM script compiles successfully" || {
    echo "✗ LSTM script compilation failed"
    exit 1
}

echo ""
echo "5. Testing profile script syntax..."
python -m py_compile ../sample-collection-scripts/profile.py && echo "✓ Profile script compiles successfully" || {
    echo "✗ Profile script compilation failed" 
    exit 1
}

echo ""
echo "=== All Python 3.6 compatibility tests passed! ==="
echo "The framework should now work correctly on Python 3.6+ environments."
