# Examples Directory

This directory contains example scripts and usage demonstrations for the AI Inference Energy Profiling Framework, including comprehensive A100 SLURM job submission examples and production-ready power modeling demonstrations.

## Example Files

### **Framework Usage Examples**
- **`example_usage.py`** - Comprehensive demonstration of framework capabilities with robust error handling
- **`power_modeling_example.py`** - Power modeling framework usage, model training, and validation
- **`edp_optimization_example.py`** - EDP optimization, frequency selection, and robustness testing

### **A100 SLURM Job Examples**
- **`submit_lstm_a100_baseline.sh`** - LSTM baseline profiling on A100
- **`submit_lstm_a100_comprehensive.sh`** - LSTM comprehensive profiling with frequency sweeps
- **`submit_lstm_a100_custom.sh`** - LSTM custom frequency profiling
- **`submit_stablediffusion_a100_baseline.sh`** - Stable Diffusion profiling example
- **`submit_custom_app_a100_template.sh`** - Template for custom applications
- **`submit_helper.sh`** - Interactive helper for submitting jobs
- **`A100_EXAMPLES_README.md`** - Detailed guide for A100 examples

## ✨ Latest Improvements (v1.0.0)

### Production-Ready Features
- ✅ **Comprehensive Error Handling**: All examples now include robust error handling for edge cases
- ✅ **Runtime Warning Elimination**: Examples run without mathematical warnings or errors
- ✅ **Input Validation**: Automatic validation of configuration parameters and data inputs
- ✅ **Graceful Fallbacks**: Examples handle invalid inputs and provide meaningful feedback

## Quick Start for A100 Examples

### Using the Submit Helper (Recommended)
```bash
cd examples/
./submit_helper.sh                    # Show help
./submit_helper.sh lstm-baseline      # Submit quick LSTM test
./submit_helper.sh lstm-comprehensive # Submit full analysis
./submit_helper.sh status             # Check job status
```

### Manual Submission
```bash
# Edit email address in script first
nano submit_lstm_a100_baseline.sh
# Submit job
sbatch submit_lstm_a100_baseline.sh
```

## Overview

The examples in this directory demonstrate real-world usage patterns and provide templates for building custom profiling experiments on various GPU types.

## Running Examples

### Framework Usage Example
```bash
cd examples
python example_usage.py
```

### Power Modeling Examples
```bash
# Quick power analysis demonstration
python power_modeling_example.py

# EDP optimization example
python edp_optimization_example.py
```

### Demo Mode (Shorter Experiments)
```bash
cd examples  
python example_usage.py --demo-mode
```

### Robustness Testing
```bash
# Test error handling and edge cases
python example_usage.py --test-robustness

# Validate with intentionally problematic data
python power_modeling_example.py --validate-edge-cases

# Test EDP calculation robustness
python edp_optimization_example.py --test-error-handling
```

## Example Scripts Description

### `example_usage.py`
A comprehensive demonstration script that shows:
- ✅ Configuration loading and validation with error handling
- ✅ GPU profiling setup and execution with robustness checks
- ✅ Application integration examples with edge case handling
- ✅ Production-ready error handling and logging
- ✅ Results processing and analysis with validation
- ✅ **NEW**: Division by zero protection and NaN handling

**Features:**
- Configurable demo mode for quick testing
- Multiple application examples (LSTM, LLaMA, Stable Diffusion)
- Professional logging and error handling
- Automatic cleanup and resource management
- **Enhanced robustness** for production environments

**Usage Scenarios:**
- Learning framework capabilities safely
- Testing new installations with comprehensive validation
- Developing custom profiling workflows with error handling
- Debugging and troubleshooting with detailed logging

## Requirements

- Python 3.6+
- All framework dependencies (see `../requirements.txt`)
- Access to GPU for profiling examples
- Proper framework installation and configuration

## Creating New Examples

### Example Template
```python
#!/usr/bin/env python3
"""
Example: [Description]
"""
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import profiling_config
from utils import setup_logging

def main():
    """Main example function."""
    logger = setup_logging()
    logger.info("Starting example")
    
    # Example implementation here
    
    logger.info("Example completed")

if __name__ == "__main__":
    main()
```

### Best Practices for Examples
1. **Clear Documentation**: Include purpose and usage instructions
2. **Error Handling**: Demonstrate proper error handling patterns
3. **Logging**: Use framework logging utilities
4. **Resource Cleanup**: Properly clean up resources
5. **Configurability**: Allow customization via arguments or configuration

## Integration

These examples integrate with:
- **Main framework** (`../sample-collection-scripts/`)
- **Configuration system** (`../config.py`)
- **Utility functions** (`../utils.py`)
- **Application modules** (`../app-*/`)

## Related Documentation

For more information:
- **[`../documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md)** - CLI usage examples
- **[`../README.md`](../README.md)** - Main project documentation
- **[`../sample-collection-scripts/README.md`](../sample-collection-scripts/README.md)** - Profiling framework guide

## Contributing Examples

When adding new examples:
1. **Follow** the template structure
2. **Include** clear documentation and comments
3. **Test** thoroughly before committing
4. **Update** this README with new example descriptions
5. **Maintain** consistent coding style and patterns
