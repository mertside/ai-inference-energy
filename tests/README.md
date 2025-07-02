# Tests Directory

This directory contains all test scripts and unit tests for the AI Inference Energy Profiling Framework.

## Overview

The test suite ensures the framework works correctly across different Python versions, GPU configurations, and profiling tools. All tests are designed to be run locally or in CI/CD environments.

## Test Files

### **Configuration Tests**
- **`test_config.py`** - Tests configuration module loading and Python 3.6+ compatibility

### **Compatibility Tests**  
- **`test_subprocess_fix.py`** - Tests subprocess functionality and Python 3.6 compatibility
- **`test_python36_compatibility.sh`** - Comprehensive Python 3.6+ compatibility test suite

## Running Tests

### Individual Tests

#### Test Configuration Module
```bash
cd tests
python test_config.py
```

#### Test Subprocess Compatibility
```bash
cd tests  
python test_subprocess_fix.py
```

#### Comprehensive Compatibility Test
```bash
cd tests
./test_python36_compatibility.sh
```

### Run All Tests
```bash
# From project root
cd tests
python test_config.py && python test_subprocess_fix.py && ./test_python36_compatibility.sh
```

## Test Coverage

### Configuration Testing
- ✅ Config module import validation
- ✅ DCGMI fields configuration
- ✅ GPU frequency settings (A100/V100)
- ✅ Model configuration parameters
- ✅ Python 3.6+ class compatibility

### Subprocess Testing  
- ✅ `run_command()` function with output capture
- ✅ `run_command()` function without output capture
- ✅ Python 3.6+ subprocess.run() compatibility
- ✅ Error handling and timeout functionality

### Compatibility Testing
- ✅ Python version detection
- ✅ Module import validation
- ✅ Script compilation testing
- ✅ Core functionality verification

## Test Output

### Successful Test Run
```
✓ Config module loaded successfully
  - DCGMI fields count: 17
  - A100 frequencies count: 61
  - Default interval: 50ms
  - LLaMA model: huggyllama/llama-7b

✓ Successfully imported run_command from utils
✓ Command executed successfully: Hello World
✓ Command executed without capture
✓ All subprocess tests passed!

=== All Python 3.6 compatibility tests passed! ===
```

### Failed Test Indicators
- ✗ Import errors
- ✗ Subprocess compatibility issues  
- ✗ Configuration loading failures
- ✗ Script compilation errors

## Requirements

- Python 3.6+ (tests verify compatibility across versions)
- Access to project modules (`config.py`, `utils.py`)
- Basic shell environment for bash tests

## Troubleshooting

### Import Errors
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Or run from project root
cd .. && python tests/test_config.py
```

### Permission Issues
```bash
# Make test scripts executable
chmod +x tests/*.sh
```

### Python Version Issues
```bash
# Check Python version
python --version

# Test with specific Python version
python3.6 tests/test_config.py  # If available
```

## Adding New Tests

### Python Test Template
```python
#!/usr/bin/env python3
"""
Test description
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_new_feature():
    """Test new functionality."""
    try:
        # Test implementation
        print("✓ Test passed")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_new_feature()
    sys.exit(0 if success else 1)
```

### Bash Test Template
```bash
#!/bin/bash
echo "=== New Feature Test ==="
echo "Testing: [description]"

# Test implementation
if [condition]; then
    echo "✓ Test passed"
else
    echo "✗ Test failed"
    exit 1
fi

echo "=== Test completed successfully ==="
```

## Integration

These tests integrate with:
- **CI/CD pipelines** for automated validation
- **Development workflow** for local testing
- **Deployment verification** on HPC clusters
- **Python compatibility** validation across versions

## Test Philosophy

- **Comprehensive**: Cover all critical functionality
- **Fast**: Run quickly for rapid development feedback  
- **Portable**: Work across different environments
- **Clear**: Provide clear pass/fail indicators
- **Maintainable**: Easy to update and extend

For framework usage examples, see [`../documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md).
