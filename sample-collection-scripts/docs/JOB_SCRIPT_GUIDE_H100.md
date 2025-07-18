# H100 SLURM Job Submission Guide

## 🎯 Quick Reference for `submit_job_h100.sh`

This guide helps you quickly select the right configuration for your H100 GPU profiling needs.

### 📊 H100 GPU Specifications
- **GPU**: H100 (80GB HBM3)
- **Partition**: h100-build (REPACSS Texas Tech)
- **Frequencies**: 86 available (510-1785 MHz)
- **Memory**: 2619 MHz (maximum)
- **Architecture**: Hopper (GH100)
- **Advanced Features**: FP8 precision, Transformer Engine, HBM3, PCIe Gen5

## 🚀 Configuration Selection

### Quick Tests (1-2 hours)
```bash
# 1. 🚀 QUICK TEST - Fastest validation
# Uncomment: Configuration #1
# Time needed: 2-4 minutes
# Use for: Initial testing, debugging

# 2. 🔬 RESEARCH BASELINE - Statistical significance  
# Uncomment: Configuration #2
# Time needed: 6-10 minutes
# Use for: Publication-quality baseline data
```

### Application Profiling (1-3 hours)
```bash
# 4. 🎤 WHISPER - Speech recognition benchmark
# Uncomment: Configuration #4
# Time needed: 10-20 minutes
# Use for: Audio processing workload analysis

# 5. 🤖 LSTM PROFILING - Lightweight benchmark
# Uncomment: Configuration #5
# Time needed: 10-20 minutes
# Use for: NLP workload analysis

# 6. 🎨 STABLE DIFFUSION - Image generation
# Uncomment: Configuration #6  
# Time needed: 15-25 minutes
# Use for: Generative AI workload analysis

# 7. 📝 LLAMA - Text generation
# Uncomment: Configuration #7
# Time needed: 20-30 minutes
# Use for: Large language model analysis

# 8. 🔧 CUSTOM APPLICATION - Your own app
# Uncomment: Configuration #8
# Time needed: Varies
# Use for: Custom application profiling
```

### Advanced H100 Features (2-6 hours)
```bash
# 8. ⚡ FP8 PRECISION - Next-gen mixed precision
# Uncomment: Configuration #8
# Time needed: 2-4 hours
# Use for: Ultra-efficient inference studies

# 9. 🤖 TRANSFORMER ENGINE - Optimized transformers
# Uncomment: Configuration #9
# Time needed: 3-5 hours
# Use for: Transformer model optimization

# 10. 💾 HBM3 MEMORY ANALYSIS - Advanced memory study
# Uncomment: Configuration #10
# Time needed: 2-4 hours
# Use for: Memory bandwidth characterization
```

### Comprehensive Studies (6-24 hours)
```bash
# 4. ⚡ FULL DVFS STUDY - All 86 frequencies
# Uncomment: Configuration #4
# Time needed: 6-12 hours ⚠️ LONG JOB
# Use for: Complete energy characterization

# 13. 📊 ENERGY EFFICIENCY STUDY - Power vs performance
# Uncomment: Configuration #13
# Time needed: 4-8 hours
# Use for: Optimal frequency selection

# 16. 🔬 SCALING ANALYSIS - Performance scaling
# Uncomment: Configuration #16
# Time needed: 3-6 hours
# Use for: Workload scaling studies
```

## ⏱️ Time Planning

| Configuration | Estimated Time | SLURM --time Setting |
|---------------|----------------|---------------------|
| #1 (Quick Test) | 2-4 min | --time=01:00:00 |
| #2 (Research Baseline) | 6-10 min | --time=01:00:00 |
| #3 (Frequency Sampling) | 12-20 min | --time=02:00:00 |
| #4 (Full DVFS) | 6-12 hours | --time=14:00:00 |
| #4-8 (Applications) | 10-30 min | --time=02:00:00 |
| #8-10 (H100 Features) | 2-5 hours | --time=06:00:00 |
| #11-16 (Advanced) | 1-8 hours | --time=10:00:00 |

## 🛠️ Usage Instructions

1. **Edit the script**:
   ```bash
   nano submit_job_h100.sh
   ```

2. **Select configuration**:
   - Find your desired configuration (e.g., Configuration #8)
   - Uncomment the `LAUNCH_ARGS` line
   - Comment out any other active `LAUNCH_ARGS` lines

3. **Adjust timing** (if needed):
   - Update `#SBATCH --time=XX:XX:XX` based on the table above

4. **Submit the job**:
   ```bash
   sbatch submit_job_h100.sh
   ```

5. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f H100_PROFILING.*.out
   ```

## 🎯 H100-Specific Features

### FP8 Precision Support
- **Configuration #8**: Tests cutting-edge FP8 mixed precision
- **2x efficiency**: Potential 2x speedup vs FP16 on supported models
- **Configuration #15**: Compares FP8 vs FP16 vs FP32 precision

### Transformer Engine
- **Configuration #9**: Optimized attention mechanism analysis
- **Hardware acceleration**: Native support for transformer operations
- **Auto-mixed precision**: Automatic precision selection

### HBM3 Memory Advantages
- **80GB capacity**: Double the memory of A100 (40GB)
- **Higher bandwidth**: ~3TB/s vs A100's ~1.5TB/s
- **Configuration #10**: Memory bandwidth characterization
- **Configuration #12**: Large model memory stress testing

### Advanced Frequency Range
- **86 frequencies**: More granular than V100 (117) and A100 (61)
- **Higher peak**: 1785 MHz vs A100's 1410 MHz
- **15 MHz steps**: Consistent frequency stepping

## 🚨 Common Issues & Solutions

### Node Availability
```bash
# H100 nodes are limited and in high demand
# Consider:
# 1. Submit jobs during off-peak hours
# 2. Use shorter configurations (#1-3) for testing
# 3. Check node availability: sinfo -p h100-build
```

### Memory Intensive Workloads
```bash
# H100's 80GB memory enables larger models
# For maximum utilization:
# 1. Use Configuration #12 (Memory stress test)
# 2. Test large language models (70B+ parameters)
# 3. Enable gradient checkpointing for training
```

### FP8 Compatibility
```bash
# Not all frameworks support FP8 yet
# Check compatibility:
# 1. PyTorch >= 2.1 with appropriate backends
# 2. TensorFlow >= 2.13 with Hopper support
# 3. Use Configuration #11 (Debug mode) for troubleshooting
```

### Long Runtime Management
```bash
# Configuration #4 (Full DVFS) can take 6-12 hours
# Best practices:
# 1. Set --time=14:00:00 or higher
# 2. Use Configuration #3 (sampling) for faster results
# 3. Submit during weekends or off-peak hours
# 4. Monitor queue times: squeue -p h100-build
```

## 📊 Results Analysis

After job completion, analyze results with:

```bash
# Power modeling analysis
python -c "from power_modeling import analyze_application; analyze_application('results/GH100*.csv')"

# EDP optimization
python -c "from edp_analysis import edp_calculator; edp_calculator.find_optimal_configuration('results/GH100*.csv')"

# H100-specific FP8 analysis
python -c "from power_modeling import analyze_fp8_precision; analyze_fp8_precision('results/GH100*.csv')"

# Transformer Engine analysis
python -c "from power_modeling import analyze_transformer_engine; analyze_transformer_engine('results/GH100*.csv')"

# HBM3 memory analysis
python -c "from power_modeling import analyze_memory_bandwidth; analyze_memory_bandwidth('results/GH100*.csv')"
```

## 🔬 Research Use Cases

### AI/ML Research
- **Large Language Models**: 70B+ parameter models (Configuration #6, #12)
- **Generative AI**: High-resolution image/video generation (Configuration #5)
- **Mixed Precision**: FP8 efficiency studies (Configuration #8, #15)

### Energy Efficiency Research
- **Frequency Optimization**: Configuration #4 (Full DVFS)
- **Power Modeling**: Configuration #13 (Energy efficiency study)
- **Thermal Analysis**: Configuration #14 (Thermal profiling)

### System Architecture Research
- **Memory Hierarchy**: Configuration #10 (HBM3 analysis)
- **Scaling Studies**: Configuration #16 (Batch size scaling)
- **Performance Characterization**: Configuration #2 (Research baseline)

## 🔗 Related Resources

- **Main Documentation**: `../README.md`
- **V100 Guide**: `V100_SCRIPT_GUIDE.md`
- **A100 Guide**: `A100_SCRIPT_GUIDE.md`
- **H100 Architecture**: `../hardware/INFO_H100.txt`
- **Advanced Usage**: `../documentation/USAGE_EXAMPLES.md`
- **Troubleshooting**: `../documentation/SUBMIT_JOBS_README.md`

---

💡 **Pro Tip**: H100 is the most advanced GPU in the cluster. Start with Configuration #1 (Quick Test) to validate your setup, then explore H100-specific features like FP8 precision (Configuration #8) and Transformer Engine (Configuration #9).
