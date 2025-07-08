# Documentation Directory

This directory contains concise, production-ready documentation for the AI Inference Energy Profiling Framework.

## 📚 Core Documentation

### **Essential Guides**
- **[`GPU_USAGE_GUIDE.md`](GPU_USAGE_GUIDE.md)** - **Complete GPU support guide** for A100, V100, and H100 across HPCC and REPACSS clusters
- **[`USAGE_EXAMPLES.md`](USAGE_EXAMPLES.md)** - CLI usage examples and automation scripts
- **[`SUBMIT_JOBS_README.md`](SUBMIT_JOBS_README.md)** - SLURM job submission and HPC cluster deployment
- **[`README_POWER_MODELING.md`](README_POWER_MODELING.md)** - **Production-ready** power modeling and EDP optimization framework

## 🚀 Quick Start Paths

### For New Users
1. **[`GPU_USAGE_GUIDE.md`](GPU_USAGE_GUIDE.md)** - Start here for GPU-specific setup and usage
2. **[`USAGE_EXAMPLES.md`](USAGE_EXAMPLES.md)** - Learn the CLI interface with examples
3. **[`SUBMIT_JOBS_README.md`](SUBMIT_JOBS_README.md)** - Submit jobs to HPC clusters

### For Researchers and Advanced Users  
1. **[`README_POWER_MODELING.md`](README_POWER_MODELING.md)** - Advanced power modeling and EDP optimization
2. **[`GPU_USAGE_GUIDE.md`](GPU_USAGE_GUIDE.md)** - Complete GPU specifications and performance characteristics

### For HPC System Administrators
1. **[`SUBMIT_JOBS_README.md`](SUBMIT_JOBS_README.md)** - Cluster integration and job management
2. **[`GPU_USAGE_GUIDE.md`](GPU_USAGE_GUIDE.md)** - GPU resource specifications and requirements

## 🔍 Quick Reference

### GPU-Specific Commands
```bash
# Unified interactive helper (auto-detects GPU type)
cd sample-collection-scripts
./interactive_gpu.sh              # Auto-detection and setup guidance

# A100 (HPCC toreador partition)
./launch.sh --gpu-type A100 --profiling-mode baseline
sbatch submit_job_a100_baseline.sh

# V100 (HPCC matador partition)  
./launch.sh --gpu-type V100 --profiling-mode baseline
sbatch submit_job_v100_baseline.sh

# H100 (REPACSS h100-build partition)
./launch.sh --gpu-type H100 --profiling-mode baseline  
sbatch submit_job_h100_baseline.sh
```

### Power Modeling Framework
```bash
# Quick power analysis
python -c "
from power_modeling import analyze_application
results = analyze_application('profiling_data.csv', app_name='MyApp', gpu_type='A100')
print(f'Optimal frequency: {results[\"summary\"][\"optimal_frequency\"]} MHz')
"

# Advanced framework usage
python -c "
from power_modeling import FGCSPowerModelingFramework
framework = FGCSPowerModelingFramework(gpu_type='A100')
results = framework.analyze_from_file('data/profiling.csv', app_name='MyApp')
"
```

### Troubleshooting Commands
```bash
# Check GPU and tools availability
nvidia-smi
dcgmi discovery --list

# Test framework configuration  
./launch.sh --help
./launch.sh --gpu-type A100 --profiling-mode baseline --num-runs 1

# Interactive sessions for testing (use unified script)
./interactive_gpu.sh a100 test   # A100 testing
./interactive_gpu.sh v100 test   # V100 testing  
./interactive_gpu.sh h100 test   # H100 testing

```

## 📋 Documentation Standards

All documentation follows consistent patterns:
- **Overview** with quick start examples
- **Detailed configuration** options and parameters  
- **Practical examples** with real commands
- **Troubleshooting** with common issues and solutions
- **Cross-references** to related documentation

## 🔗 Related Files

### Project Root Documentation
- **[`../README.md`](../README.md)** - Main project overview and installation guide
- **[`../power_modeling/README.md`](../power_modeling/README.md)** - Power modeling module documentation
- **[`../edp_analysis/README.md`](../edp_analysis/README.md)** - EDP optimization module documentation
- **[`../examples/README.md`](../examples/README.md)** - Usage examples and demonstrations

### Application-Specific Documentation  
- **[`../sample-collection-scripts/README.md`](../sample-collection-scripts/README.md)** - Profiling framework and scripts
- **[`../app-llama/README.md`](../app-llama/README.md)** - LLaMA inference application
- **[`../app-stable-diffusion/README.md`](../app-stable-diffusion/README.md)** - Stable Diffusion application

## 📞 Support and Contribution

### Getting Help
- 📖 Check the relevant documentation file for your use case
- 🔍 Use browser search (Ctrl+F / Cmd+F) within documentation files  
- 🐛 Check troubleshooting sections for common issues
- 💬 Submit GitHub issues for bugs or feature requests

### Contributing Documentation
1. **Follow** established structure: Overview → Examples → Configuration → Troubleshooting
2. **Include** practical, tested examples with real commands
3. **Cross-reference** related documentation appropriately
4. **Test** all commands and examples before submitting
5. **Update** this index file when adding new documentation

---

**Framework Status**: ✅ **Production Ready (v1.0.0)** - Complete documentation for production deployment
