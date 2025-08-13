# AI Inference Energy Optimization: Complete Guide

## Overview

This guide provides a comprehensive walkthrough for reproducing our energy optimization results for AI inference workloads on NVIDIA GPUs. Our methodology uses **100% measured experimental data** to identify optimal GPU frequencies that maximize energy savings while maintaining performance within acceptable bounds.

## 🎯 Project Objectives

- **Primary Goal**: Identify optimal GPU frequencies for AI inference workloads
- **Performance Constraint**: ≤5% performance degradation
- **Optimization Target**: Minimize Energy-Delay Product (EDP)
- **Data Requirement**: Only measured experimental data (no estimates)

## 📊 Experimental Setup

### GPU Architectures Tested
- **NVIDIA V100**: Tesla V100-SXM2-32GB
- **NVIDIA A100**: A100-SXM4-80GB  
- **NVIDIA H100**: H100-SXM5-80GB

### AI Workloads Evaluated
- **LLaMA**: Large Language Model inference
- **Stable Diffusion**: Text-to-image generation
- **Vision Transformer (ViT)**: Image classification
- **Whisper**: Speech-to-text transcription

### Frequency Range
- **Test Range**: 510MHz to 1785MHz (full DVFS range)
- **Step Size**: 15MHz increments
- **Total Frequencies**: ~85 frequency points per configuration

## 🔬 Methodology

### 1. Data Collection Process

**Profiling Infrastructure:**
```bash
# NVIDIA DCGMI profiling with 25 comprehensive metrics
nvidia-smi -lgc <frequency>  # Set GPU frequency
dcgmi profile --start        # Begin profiling
<run_ai_workload>            # Execute inference task
dcgmi profile --stop         # End profiling
```

**Collected Metrics:**
- Power consumption (Watts)
- Energy consumption (Joules)
- Execution time (seconds)
- GPU utilization
- Memory utilization
- Temperature data

**Experimental Design:**
- **3 repetitions** per frequency configuration
- **12 total configurations** (3 GPUs × 4 workloads)
- **~1,000+ measurements** per configuration
- **Total dataset**: >12,000 experimental measurements

### 2. Data Processing Pipeline

**Step 1: Raw Data Extraction**
```bash
# Extract timing from application logs
grep "Total Inference Time" app_output.out

# Extract power/energy from DCGMI CSV
python tools/analysis/measured_data_analysis.py
```

**Step 2: Data Validation**
- Remove outliers (>3 standard deviations)
- Validate measurement consistency
- Check for thermal throttling events
- Ensure minimum 3 valid runs per frequency

**Step 3: Performance Baseline Calculation**
- Identify maximum frequency performance
- Calculate 5% degradation threshold
- Filter valid frequencies within constraint

### 3. Optimization Algorithm

**Energy-Delay Product (EDP) Minimization:**
```
EDP = Energy × Execution_Time
EDP = Power × Time²

Objective: min(EDP) subject to Performance_Degradation ≤ 5%
```

**Constraint Application:**
```python
max_allowed_time = baseline_time * 1.05  # 5% constraint
valid_frequencies = [f for f in frequencies 
                    if execution_time[f] <= max_allowed_time]
optimal_freq = min(valid_frequencies, key=lambda f: edp[f])
```

## 🚀 Reproduction Steps

### Prerequisites

**Hardware Requirements:**
- NVIDIA GPU (V100, A100, or H100)
- NVIDIA DCGMI installed
- Sufficient GPU memory for AI workloads

**Software Requirements:**
```bash
# Install dependencies
pip install torch transformers diffusers
pip install numpy pandas matplotlib

# Clone repository
git clone https://github.com/mertside/ai-inference-energy.git
cd ai-inference-energy
```

### Step 1: Data Collection (Optional)

If you want to collect new data:

```bash
# Submit SLURM jobs for data collection
sbatch sample-collection-scripts/submit_job_v100.sh
sbatch sample-collection-scripts/submit_job_a100.sh  
sbatch sample-collection-scripts/submit_job_h100.sh

# Or run locally for specific workload
python app-llama/LlamaViaHF.py --dvfs_mode --profile_dcgmi
```

### Step 2: Analysis Using Existing Data

Our repository includes pre-collected experimental data:

```bash
# Run main analysis
python tools/analysis/measured_data_analysis.py

# Expected output:
# V100 + whisper: 645MHz (6.0% energy savings, 1.2% performance impact)
# A100 + vit: 525MHz (15.3% energy savings, 0.3% performance impact)
# H100 + llama: 1035MHz (11.6% energy savings, 1.3% performance impact)
# ... (12 total configurations)
```

### Step 3: Generate Deployment Configurations

The analysis automatically generates:
- `MEASURED_DATA_OPTIMAL_FREQUENCIES.md` - Detailed results report
- `measured_data_optimal_frequencies_deployment.json` - Machine-readable config

### Step 4: Deploy Optimal Frequencies

```bash
# Example deployment commands:
# A100 + Vision Transformer (best result: 15.3% savings, 0.3% impact)
sudo nvidia-smi -i 0 -lgc 525

# V100 + Stable Diffusion (highest savings: 18.4% savings, 4.4% impact)  
sudo nvidia-smi -i 0 -lgc 1110

# H100 + LLaMA (solid efficiency: 11.6% savings, 1.3% impact)
sudo nvidia-smi -i 0 -lgc 1035
```

## 🔍 Data Quality Validation

### Measurement Validation
- **Consistency Check**: Standard deviation < 10% across runs
- **Thermal Validation**: No throttling events detected
- **Outlier Removal**: Z-score filtering (|z| < 3)
- **Coverage Validation**: Minimum 3 valid measurements per frequency

### Statistical Significance
- **Sample Size**: 3+ repetitions per configuration
- **Confidence Level**: 95% confidence intervals calculated
- **Validation Method**: Cross-validation with held-out frequencies

## 📈 Expected Results Summary

### Top Performing Configurations

| Rank | Configuration | Optimal Freq | Energy Savings | Performance Impact |
|------|---------------|--------------|----------------|--------------------|
| 1 | V100 + Stable Diffusion | 1110MHz | 18.4% | 4.4% |
| 2 | A100 + Whisper | 930MHz | 16.6% | 1.2% |
| 3 | A100 + Vision Transformer | 525MHz | 15.3% | 0.3% |
| 4 | A100 + LLaMA | 735MHz | 14.0% | 1.1% |

### Performance by GPU Architecture

- **A100**: Best overall efficiency (13.1% average savings)
- **V100**: Excellent for Stable Diffusion (18.4% savings)
- **H100**: Conservative but stable gains (4.2% average savings)

## 🛠 Troubleshooting

### Common Issues

**1. DCGMI Permission Errors**
```bash
# Solution: Run with proper permissions
sudo dcgmi profile --start
```

**2. Memory Insufficient Errors**
```bash
# Solution: Reduce batch size or model precision
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**3. No Valid Measurements**
```bash
# Solution: Check data directory structure
ls sample-collection-scripts/results_*/
```

### Data Validation
```bash
# Verify data extraction
python tools/analysis/data_source_summary.py

# Check measurement counts
grep -r "Found timing" tools/analysis/measured_data_analysis.py
```

## 📊 Understanding the Output

### Analysis Output Format
```
=== Analyzing A100 + vit ===
    Baseline: 1215MHz, 9.98s
    Max allowed: 10.48s
      Valid: 525MHz, 10.01s, EDP: 1876.54
      Valid: 540MHz, 10.12s, EDP: 1923.71
      ...
    ✓ Optimal: 525MHz
    Energy savings: 15.3%
    Performance impact: 0.3%
```

### Key Metrics Explained
- **Baseline**: Maximum frequency performance reference
- **Max allowed**: 5% performance degradation threshold
- **EDP**: Energy-Delay Product (lower is better)
- **Energy savings**: Percentage reduction in energy consumption
- **Performance impact**: Actual performance degradation measured

## 🎯 Research Applications

This methodology enables:
- **Production Deployment**: Immediate energy savings in inference clusters
- **Research Extension**: Framework for new workloads/architectures
- **Policy Development**: Data-driven frequency scaling policies
- **Cost Optimization**: Reduced operational costs in cloud deployments

## 📚 References

- **DVFS Theory**: Dynamic Voltage and Frequency Scaling principles
- **EDP Optimization**: Energy-Delay Product minimization methodology
- **GPU Profiling**: NVIDIA DCGMI comprehensive profiling guide
- **Statistical Analysis**: Confidence interval and outlier detection methods

---

*This guide provides complete reproducibility for our energy optimization research using measured experimental data on modern NVIDIA GPU architectures.*
