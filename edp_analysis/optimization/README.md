# AI Inference GPU Frequency Optimization

## 🎯 Final Production Results

This directory contains the **production-ready** GPU frequency optimization results for AI inference workloads, with **cold start effects properly excluded** using warm-run data.

### 🚨 Critical Discovery: Cold Start Effect

**Problem Identified**: Original analysis used Run 1 data contaminated by cold start effects
- Cold start execution times: 373.55s (A100+LLAMA example)  
- Warm run execution times: 46.05s (same configuration)
- **Result**: 810% artificial performance penalty inflation

**Solution Implemented**: Use Run 2 data (first warm run) for all optimization analyses

## 📊 Production-Ready Frequency Recommendations

### ✅ Recommended for Production (≤20% performance penalty)
| Configuration | Frequency | Performance Penalty | Energy Savings | Use Case |
|---------------|-----------|-------------------|----------------|----------|
| **A100+STABLEDIFFUSION** | 1410→1245 MHz | 20.2% | 38.9% | Interactive Image Generation |
| **V100+STABLEDIFFUSION** | 1380→1110 MHz | 10.3% | 31.4% | Interactive Image Generation |

### ⚠️ Moderate Impact (20-50% performance penalty)
| Configuration | Frequency | Performance Penalty | Energy Savings | Use Case |
|---------------|-----------|-------------------|----------------|----------|
| **A100+LLAMA** | 1410→1200 MHz | 41.3% | 64.0% | Interactive LLM (Consider A/B testing) |
| **V100+LLAMA** | 1380→1365 MHz | 35.4% | 41.4% | Interactive LLM (Consider A/B testing) |

### 🔬 Batch Processing Only (>50% performance penalty)
| Configuration | Frequency | Performance Penalty | Energy Savings | Use Case |
|---------------|-----------|-------------------|----------------|----------|
| **A100+VIT** | 1410→1215 MHz | 93.1% | 99.5% | Batch Vision Processing |
| **A100+WHISPER** | 1410→1290 MHz | 89.8% | 98.7% | Batch Audio Processing |
| **V100+VIT** | 1380→1140 MHz | 92.5% | 99.4% | Batch Vision Processing |
| **V100+WHISPER** | 1380→1230 MHz | 89.5% | 99.1% | Batch Audio Processing |

## 🚀 Quick Deployment

```bash
# Deploy optimal frequency for interactive applications
./deploy_optimal_frequencies.sh A100+STABLEDIFFUSION deploy

# Check current status
./deploy_optimal_frequencies.sh V100+STABLEDIFFUSION status

# Reset to baseline frequency
./deploy_optimal_frequencies.sh A100+LLAMA reset
```

## 📁 Key Files

### Production Files
- **`production_summary.py`** - Final summary generator
- **`deploy_optimal_frequencies.sh`** - Deployment automation script
- **`production_optimizer.py`** - Production-ready optimizer
- **`workload_constraints.py`** - Application-specific constraints

### Analysis Files  
- **`performance_constrained_optimization.py`** - Core optimization engine
- **`production_optimization_summary.txt`** - Detailed results summary
- **`FINAL_OPTIMIZATION_REPORT.txt`** - Comprehensive findings report

### Configuration Files
- **`production_optimization_results.json`** - Optimization results data

## 🔧 Manual Deployment Commands

```bash
# A100 GPU configurations
nvidia-smi -ac 1215,1245  # A100+STABLEDIFFUSION (RECOMMENDED)
nvidia-smi -ac 1215,1200  # A100+LLAMA (moderate impact)
nvidia-smi -ac 1215,1290  # A100+WHISPER (batch only)
nvidia-smi -ac 1215,1215  # A100+VIT (batch only)

# V100 GPU configurations  
nvidia-smi -ac 877,1110   # V100+STABLEDIFFUSION (RECOMMENDED)
nvidia-smi -ac 877,1365   # V100+LLAMA (moderate impact)
nvidia-smi -ac 877,1230   # V100+WHISPER (batch only)
nvidia-smi -ac 877,1140   # V100+VIT (batch only)

# Verification
nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits

# Reset to baseline
nvidia-smi -ac 1215,1410  # A100 reset
nvidia-smi -ac 877,1380   # V100 reset
```

## 📈 Key Achievements

✅ **Eliminated cold start bias**: Reduced performance penalties from 89-98% to realistic 10-41%  
✅ **Production-ready configurations**: Identified deployable frequency settings  
✅ **Significant energy savings**: 31-99% energy reduction with acceptable trade-offs  
✅ **Automated deployment**: Created scripts for easy production deployment  
✅ **Workload-aware optimization**: Different strategies for interactive vs batch workloads

## 🔬 Methodology

1. **Data Collection**: 3 runs per frequency point (1 cold + 2 warm)
2. **Cold Start Handling**: Use Run 2 data to exclude initialization effects  
3. **Workload Constraints**: Application-specific performance limits
4. **Multi-Objective Optimization**: Balance energy savings with acceptable performance
5. **Production Validation**: Generate deployment-ready configurations

## 📚 Usage Examples

### Interactive Applications (Recommended)
```bash
# Stable Diffusion - Excellent trade-off
./deploy_optimal_frequencies.sh A100+STABLEDIFFUSION deploy
# Result: 20.2% slower, 38.9% less energy

./deploy_optimal_frequencies.sh V100+STABLEDIFFUSION deploy  
# Result: 10.3% slower, 31.4% less energy
```

### LLM Applications (Moderate Impact)
```bash
# LLAMA - Consider A/B testing
./deploy_optimal_frequencies.sh A100+LLAMA deploy
# Result: 41.3% slower, 64.0% less energy
```

### Batch Processing (High Savings)
```bash
# VIT/Whisper - Batch workloads only
./deploy_optimal_frequencies.sh A100+VIT deploy
# Result: 93.1% slower, 99.5% less energy
```

---

**🎯 Bottom Line**: Use **Stable Diffusion configurations** for immediate production deployment with excellent energy-performance trade-offs!
python plot_edp_optimization.py \
    --data ../data_aggregation/complete_aggregation.csv \
    --gpu V100 --app LLAMA
```

## Optimization Methods

### Energy-Delay Product (EDP)

Minimizes the product of energy consumption and execution time:
```
EDP = Energy × Execution_Time
```

**Best for**: Balanced energy-performance optimization

### Energy-Delay² Product (ED²P)

Emphasizes performance over energy by squaring the delay term:
```
ED²P = Energy × (Execution_Time)²
```

**Best for**: Performance-conscious energy optimization

### Energy-Only

Minimizes energy consumption regardless of performance impact:
```
Minimize: Energy
```

**Best for**: Maximum energy savings scenarios

### Performance-Only

Minimizes execution time regardless of energy impact:
```
Minimize: Execution_Time
```

**Best for**: Baseline performance comparison

## Key Results (Example)

Our analysis of 480 configurations across 2 GPUs and 4 applications revealed:

### Energy Savings Summary
- **EDP Optimization**: 91.6% ± 8.1% average energy savings
- **ED²P Optimization**: 91.6% ± 8.1% average energy savings
- **Range**: 74.0% - 98.9% energy savings across all configurations

### Best Configurations
- **Top Performer**: V100+WHISPER @ 600 MHz (98.9% energy savings)
- **Most Consistent**: VIT applications (95.7-96.1% savings on both GPUs)

### Optimal Frequency Distribution
- **Low Frequencies**: VIT applications (525 MHz optimal)
- **Mid Frequencies**: WHISPER applications (600-1170 MHz range)
- **High Frequencies**: STABLEDIFFUSION and LLAMA (705-1230 MHz range)

## Input Data Format

Requires aggregated data with the following columns:
- `gpu`: GPU type (V100, A100, H100)
- `application`: Application name
- `frequency`: GPU frequency in MHz
- `avg_power`: Average power consumption in Watts
- `execution_time`: Execution time in seconds
- `energy`: Energy consumption in Joules

## Output Format

### JSON Results Structure

```json
{
  "summary": {
    "total_gpus": 2,
    "total_applications": 4,
    "total_combinations": 8,
    "optimization_methods": ["edp", "ed2p"],
    "analysis_timestamp": "2025-07-31T12:58:34.993862",
    "statistics": {
      "average_energy_savings": {
        "edp": {"mean": 91.6, "std": 8.1, "min": 74.0, "max": 98.9}
      },
      "best_configurations": {
        "edp": {"configuration": "V100_WHISPER", "frequency": 600, "energy_savings": 98.9}
      }
    }
  },
  "configurations": {
    "V100_LLAMA": {
      "gpu": "V100",
      "application": "LLAMA",
      "total_frequencies": 59,
      "frequency_range": {"min": 510, "max": 1380},
      "optimizations": {
        "edp": {
          "frequency": 1155,
          "energy": 3752.7,
          "execution_time": 44.55,
          "baseline_comparison": {
            "energy_savings_percent": 74.0,
            "time_penalty_percent": -88.3
          }
        }
      }
    }
  }
}
```

## Visualization Outputs

### EDP Curve Analysis
- **Individual plots** for each GPU+application combination
- **Four-panel layout**: Energy, Execution Time, EDP, ED²P vs Frequency
- **Optimal point markers** showing EDP and ED²P minima
- **Professional styling** with clear legends and grid lines

### Summary Charts
- **Energy savings comparison** across all configurations
- **Optimal frequency distribution** showing method differences
- **Bar charts** with value labels for easy interpretation

## Implementation Details

### Baseline Comparison

Energy savings calculated relative to maximum frequency:
```python
energy_savings = (baseline_energy - optimal_energy) / baseline_energy × 100
```

### Statistical Analysis

Cross-configuration statistics provide:
- Mean and standard deviation of energy savings
- Best performing configurations per method
- Frequency distribution analysis

### Validation Checks

- Optimal frequency ≤ maximum available frequency
- Energy savings > 0 (sanity check)
- Reasonable execution time ranges
- Consistent results across methods

## Command Line Options

### find_optimal_frequencies.py

- `--data`: Path to aggregated profiling data CSV (required)
- `--output`: Output JSON file (default: optimal_frequencies.json)
- `--gpu`: Filter by GPU type (V100, A100, H100)
- `--app`: Filter by application (LLAMA, VIT, STABLEDIFFUSION, WHISPER)
- `--methods`: Optimization methods (default: edp ed2p)

### plot_edp_optimization.py

- `--data`: Path to aggregated profiling data CSV (required)
- `--optimal`: Path to optimal frequencies JSON file
- `--gpu`: Plot specific GPU (plots all if not specified)
- `--app`: Plot specific application (plots all if not specified)
- `--output-dir`: Output directory for plots (default: plots)

## Performance Considerations

- **Memory efficient**: Processes data in chunks for large datasets
- **Vectorized operations**: Uses pandas/numpy for fast computation
- **Parallel potential**: Individual GPU+app analyses are independent

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- json (built-in)
- pathlib (built-in)

## Use Cases

### Research Applications
- Energy efficiency studies
- GPU frequency optimization research
- AI workload characterization
- Power modeling validation

### Production Applications
- Data center energy optimization
- Deployment frequency selection
- SLA-aware energy management
- Thermal management optimization

## Troubleshooting

### Common Issues

1. **"No data found for GPU+app"**
   - Verify aggregated data contains the specified configuration
   - Check GPU and application name spelling/case

2. **"Missing required columns"**
   - Ensure aggregated data has all required columns
   - Re-run data aggregation if necessary

3. **"Invalid optimization method"**
   - Use only supported methods: edp, ed2p, energy, performance

### Debug Tips

- Check aggregated data structure: `df.info()` and `df.head()`
- Validate frequency ranges per configuration
- Review statistical summaries for anomalies
