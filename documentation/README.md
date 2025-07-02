# Documentation Directory

This directory contains comprehensive documentation for the AI Inference Energy Profiling Framework.

## Documentation Files

### **Usage and Examples**
- **`USAGE_EXAMPLES.md`** - Complete CLI usage examples and automation scripts
- **`SUBMIT_JOBS_README.md`** - SLURM job submission guide and HPC cluster usage
- **`A100_USAGE_GUIDE.md`** - Comprehensive A100 GPU support and usage guide (HPCC)
- **`H100_USAGE_GUIDE.md`** - Comprehensive H100 GPU support and usage guide (REPACSS)

### **Technical Implementation**  
- **`CLI_ENHANCEMENT_SUMMARY.md`** - Technical details of CLI enhancements and new features
- **`REFACTORING_SUMMARY.md`** - Complete overview of the codebase refactoring process

### **Compatibility and Fixes**
- **`PYTHON36_COMPATIBILITY_FIX.md`** - Python 3.6 compatibility guide and fixes
- **`SUBPROCESS_FIX_SUMMARY.md`** - Recent subprocess compatibility fix documentation

### **Troubleshooting**
- **`QUICK_FIX_GUIDE.md`** - Common issues, troubleshooting steps, and quick fixes

## Documentation Overview

### For New Users
1. **Start with:** [`USAGE_EXAMPLES.md`](USAGE_EXAMPLES.md) - Learn the CLI interface
2. **A100 Users:** [`A100_USAGE_GUIDE.md`](A100_USAGE_GUIDE.md) - Complete A100 support guide (HPCC)
3. **H100 Users:** [`H100_USAGE_GUIDE.md`](H100_USAGE_GUIDE.md) - Complete H100 support guide (REPACSS)
4. **HPC Users:** [`SUBMIT_JOBS_README.md`](SUBMIT_JOBS_README.md) - SLURM job submission
5. **Troubleshooting:** [`QUICK_FIX_GUIDE.md`](QUICK_FIX_GUIDE.md) - Common issues and solutions

### For Developers
1. **Architecture:** [`REFACTORING_SUMMARY.md`](REFACTORING_SUMMARY.md) - Codebase structure
2. **New Features:** [`CLI_ENHANCEMENT_SUMMARY.md`](CLI_ENHANCEMENT_SUMMARY.md) - Implementation details
3. **Compatibility:** [`PYTHON36_COMPATIBILITY_FIX.md`](PYTHON36_COMPATIBILITY_FIX.md) - Python version support

### For System Administrators
1. **Setup:** [`QUICK_FIX_GUIDE.md`](QUICK_FIX_GUIDE.md) - Installation and configuration
2. **HPC Integration:** [`SUBMIT_JOBS_README.md`](SUBMIT_JOBS_README.md) - Cluster deployment
3. **Recent Fixes:** [`SUBPROCESS_FIX_SUMMARY.md`](SUBPROCESS_FIX_SUMMARY.md) - Latest updates

## Quick Reference

### Command-Line Interface
```bash
# Show all options
./launch.sh --help

# Basic usage examples
./launch.sh                                    # Default A100 DVFS
./launch.sh --gpu-type V100 --profiling-mode baseline  # V100 baseline
./launch.sh --gpu-type H100 --profiling-mode baseline  # H100 baseline
./launch.sh --app-name "CustomApp" --app-executable "my_app"  # Custom app
```

### SLURM Job Submission
```bash
sbatch submit_job.sh                    # A100 main job
sbatch submit_job_v100_baseline.sh     # V100 baseline
sbatch submit_job_custom_app.sh        # Custom applications
sbatch submit_job_comprehensive.sh     # Full DVFS study
```

### Troubleshooting Commands
```bash
# Test configuration
python tests/test_config.py

# Test subprocess compatibility  
python tests/test_subprocess_fix.py

# Comprehensive compatibility test
./tests/test_python36_compatibility.sh

# Check GPU and tools
nvidia-smi
dcgmi discovery --list
```

## Documentation Standards

### File Naming Convention
- `*_EXAMPLES.md` - Usage examples and tutorials
- `*_README.md` - Component-specific documentation  
- `*_SUMMARY.md` - Technical implementation summaries
- `*_FIX.md` - Problem resolution and compatibility fixes
- `*_GUIDE.md` - Step-by-step guides and troubleshooting

### Content Structure
1. **Overview** - Brief description and purpose
2. **Quick Start** - Immediate usage examples
3. **Detailed Examples** - Comprehensive usage scenarios
4. **Configuration** - Parameter and option details
5. **Troubleshooting** - Common issues and solutions
6. **Integration** - How it fits with other components

## Maintenance

### Updating Documentation
- Update examples when CLI changes
- Add new troubleshooting entries for common issues
- Keep version compatibility information current
- Verify all examples work with latest code

### Adding New Documentation
1. Follow naming conventions
2. Include practical examples
3. Cross-reference related documents
4. Update this index file

## Related Files

### Project Root
- **`README.md`** - Main project documentation and overview
- **`requirements.txt`** - Python dependencies
- **`setup.py`** - Package installation instructions

### Application Directories
- **`app-*/README.md`** - Individual application documentation
- **`sample-collection-scripts/README.md`** - Profiling framework documentation
- **`tests/README.md`** - Test suite documentation

## Support

For additional help:
- 📖 **Read** the relevant documentation file
- 🔍 **Search** this directory for specific topics
- 🐛 **Check** [`QUICK_FIX_GUIDE.md`](QUICK_FIX_GUIDE.md) for common issues
- 💬 **Submit** GitHub issues for framework bugs or feature requests

## Contributing

When contributing documentation:
1. **Follow** the established structure and style
2. **Include** practical examples and code snippets
3. **Test** all commands and examples before submitting
4. **Update** cross-references and index files
5. **Maintain** consistent formatting and organization
