# A100 SLURM Job Submission Guide

## 🎯 Quick Reference for `submit_job_a100.sh`

This guide helps you quickly select the right configuration for your A100 GPU profiling needs.

### 📊 A100 GPU Specifications
- **GPU**: Tesla A100 (40GB HBM2e)
- **Partition**: toreador (HPCC Texas Tech)
- **Frequencies**: 61 available (510-1410 MHz) 
- **Memory**: 1215 MHz (fixed)
- **Architecture**: Ampere (GA100)
- **Advanced Features**: Tensor Cores, MIG support, PCIe Gen4

## 🚀 Configuration Selection

### Quick Tests (1-2 hours)
```bash
# 1. 🚀 QUICK TEST - Fastest validation
# Uncomment: Configuration #1
# Time needed: 3-5 minutes
# Use for: Initial testing, debugging

# 2. 🔬 RESEARCH BASELINE - Statistical significance  
# Uncomment: Configuration #2
# Time needed: 8-12 minutes
# Use for: Publication-quality baseline data
```

### Application Profiling (1-3 hours)
```bash
# 4. 🤖 LSTM PROFILING - Lightweight benchmark
# Uncomment: Configuration #4
# Time needed: 15-25 minutes
# Use for: NLP workload analysis

# 5. 🎨 STABLE DIFFUSION - Image generation
# Uncomment: Configuration #5  
# Time needed: 20-30 minutes
# Use for: Generative AI workload analysis

# 6. 📝 LLAMA - Text generation
# Uncomment: Configuration #6
# Time needed: 25-35 minutes
# Use for: Large language model analysis

# 7. 🔧 CUSTOM APPLICATION - Your own app
# Uncomment: Configuration #7
# Time needed: Varies
# Use for: Custom application profiling
```

### Advanced Studies (4-24 hours)
```bash
# 8. ⚡ COMPREHENSIVE DVFS - All 61 frequencies
# Uncomment: Configuration #8
# Time needed: 4-8 hours ⚠️ LONG JOB
# Use for: Complete energy characterization

# 9. 📈 TENSOR CORE ANALYSIS - A100-specific features
# Uncomment: Configuration #9
# Time needed: 2-4 hours
# Use for: Mixed precision studies

# 14. 🔬 MIG CONFIGURATION - Multi-Instance GPU
# Uncomment: Configuration #14
# Time needed: 1-3 hours
# Use for: Resource partitioning studies
```

## ⏱️ Time Planning

| Configuration | Estimated Time | SLURM --time Setting |
|---------------|----------------|---------------------|
| #1 (Quick Test) | 3-5 min | --time=01:00:00 |
| #2 (Research Baseline) | 8-12 min | --time=01:00:00 |
| #3 (Frequency Sampling) | 15-25 min | --time=02:00:00 |
| #4-7 (Applications) | 15-35 min | --time=02:00:00 |
| #8 (Comprehensive DVFS) | 4-8 hours | --time=10:00:00 |
| #9-16 (Advanced) | 1-4 hours | --time=05:00:00 |

## 🛠️ Usage Instructions

1. **Edit the script**:
   ```bash
   nano submit_job_a100.sh
   ```

2. **Select configuration**:
   - Find your desired configuration (e.g., Configuration #4)
   - Uncomment the `LAUNCH_ARGS` line
   - Comment out any other active `LAUNCH_ARGS` lines

3. **Adjust timing** (if needed):
   - Update `#SBATCH --time=XX:XX:XX` based on the table above

4. **Submit the job**:
   ```bash
   sbatch submit_job_a100.sh
   ```

5. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f A100_PROFILING.*.out
   ```

## 🎯 A100-Specific Features

### Tensor Core Utilization
- **Configuration #9**: Analyzes mixed precision (FP16/FP32) performance
- **Configuration #15**: Compares precision impacts on energy efficiency

### MIG (Multi-Instance GPU) Support
- **Configuration #14**: Tests resource partitioning scenarios
- Useful for multi-tenant environments

### Advanced Memory Hierarchy
- **40GB HBM2e**: Larger memory than V100 (32GB)
- **Higher bandwidth**: Better for memory-intensive workloads
- **Configuration #13**: Memory stress testing with large models

## 🚨 Common Issues & Solutions

### Permission Issues
```bash
# If DCGMI fails, use nvidia-smi fallback
# Uncomment: Configuration #11
```

### Long Runtime Warnings
```bash
# Configuration #8 (Comprehensive DVFS) can take 4-8 hours
# Make sure to:
# 1. Set --time=10:00:00 or higher
# 2. Run during off-peak hours
# 3. Consider Configuration #3 (sampling) for faster results
```

### Memory Limitations
```bash
# A100 has 40GB memory vs H100's 80GB
# For very large models, consider:
# 1. Model sharding
# 2. Gradient checkpointing  
# 3. Mixed precision (Configuration #9)
```

## 📊 Results Analysis

After job completion, analyze results with:

```bash
# Power modeling analysis
python -c "from power_modeling import analyze_application; analyze_application('results/GA100*.csv')"

# EDP optimization
python -c "from edp_analysis import edp_calculator; edp_calculator.find_optimal_configuration('results/GA100*.csv')"

# A100-specific Tensor Core analysis
python -c "from power_modeling import analyze_tensor_cores; analyze_tensor_cores('results/GA100*.csv')"
```

## 🔗 Related Resources

- **Main Documentation**: `../README.md`
- **V100 Guide**: `V100_SCRIPT_GUIDE.md`
- **H100 Guide**: `H100_SCRIPT_GUIDE.md`
- **Advanced Usage**: `../documentation/USAGE_EXAMPLES.md`
- **Troubleshooting**: `../documentation/SUBMIT_JOBS_README.md`

---

💡 **Pro Tip**: Start with Configuration #1 (Quick Test) to validate your setup before running longer experiments.
