# Tests Directory

This directory contains all test scripts and unit tests for the AI Inference Energy Profiling Framework.

## Overview

The test suite ensures the framework works correctly across different Python versions, GPU configurations, and profiling tools. All tests are designed to be run locally or in CI/CD environments.

## Test Files

### **Configuration Tests**
- **`test_configuration.py`** - Tests configuration module loading and Python 3.8+ compatibility

### **Compatibility Tests**  
- **`run_tests.py`** - Entry point for running the full test suite
- **`test_python_compatibility.sh`** - Comprehensive Python compatibility test suite

## Running Tests

### Individual Tests

#### Test Configuration Module
```bash
cd tests
python test_configuration.py
```

#### Test Utils Module
```bash
cd tests
python -m pytest test_utils.py -v
```

#### Run Test Suite
```bash
cd tests  
python run_tests.py
```

#### Comprehensive Compatibility Test
```bash
cd tests
./test_python_compatibility.sh
```

### Run All Tests
```bash
# From project root
cd tests
python run_tests.py && ./test_python_compatibility.sh
# Or using pytest
cd ..
pytest -v
```

## Test Coverage

### Configuration Testing
- ✅ Config module import validation
- ✅ DCGMI fields configuration
- ✅ GPU frequency settings (A100/V100)
- ✅ Model configuration parameters
- ✅ Python 3.8+ class compatibility

### Subprocess Testing  
- ✅ `run_command()` function with output capture
- ✅ `run_command()` function without output capture
- ✅ Python 3.8+ subprocess.run() compatibility
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

=== All Python compatibility tests passed! ===
```

### Failed Test Indicators
- ✗ Import errors
- ✗ Subprocess compatibility issues  
- ✗ Configuration loading failures
- ✗ Script compilation errors

## Requirements

- Python 3.8+ (tests verify compatibility across versions)
- Access to project modules (`config.py`, `utils.py`)
- Basic shell environment for bash tests

## Troubleshooting

### Import Errors
```bash
# Ensure proper Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# Or run from project root
cd .. && python tests/test_configuration.py
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
python3.8 tests/test_configuration.py  # If available
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
