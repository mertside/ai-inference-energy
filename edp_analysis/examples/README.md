# 📚 EDP Analysis Examples

This directory contains practical examples showing how to use the EDP analysis framework for GPU frequency optimization on new datasets.

## 🚀 Quick Start Examples

### 1. Analyze New Profiling Data (Current Framework)
```bash
# Basic analysis of new profiling data
python analyze_new_data_current.py --data-dir ../../sample-collection-scripts --output ./results

# Filter by specific GPU and application
python analyze_new_data_current.py \
    --data-dir ../../sample-collection-scripts \
    --gpu V100 \
    --app STABLEDIFFUSION \
    --output ./v100_sd_results \
    --deploy

# Use run 3 data instead of run 2 (different warm run)
python analyze_new_data_current.py \
    --data-dir ../../sample-collection-scripts \
    --run 3 \
    --output ./run3_results
```

### 2. Analyze New Profiling Data (Refactored Framework)
```bash
# Future framework example (after refactoring)
python analyze_new_data.py --data-dir ./new_profiling_data --output ./results --validate --deploy
```

## 📋 Example Scenarios

### Scenario 1: New GPU Model Analysis
You have collected profiling data for a new GPU model and want to find optimal frequencies:

```bash
# 1. Collect profiling data using the main framework
cd ../../sample-collection-scripts
./launch_v2.sh --gpu-type H100 --app-name LLAMA --profiling-mode dvfs --num-runs 3

# 2. Analyze the collected data
cd ../edp_analysis/examples
python analyze_new_data_current.py \
    --data-dir ../../sample-collection-scripts \
    --gpu H100 \
    --app LLAMA \
    --output ./h100_analysis \
    --deploy

# 3. Review results
cat ./h100_analysis/optimization_summary.txt

# 4. Deploy optimal configuration
./h100_analysis/deploy_optimal_frequencies.sh H100+LLAMA deploy
```

### Scenario 2: New AI Application Analysis
You have a new AI application and want to optimize its GPU frequency:

```bash
# 1. Add your application to the framework (modify app-new-model/)
# 2. Collect profiling data
cd ../../sample-collection-scripts
./launch_v2.sh --gpu-type A100 --app-name NEWMODEL --profiling-mode dvfs --num-runs 3

# 3. Analyze with custom constraints
python analyze_new_data_current.py \
    --data-dir ../../sample-collection-scripts \
    --app NEWMODEL \
    --output ./newmodel_analysis \
    --deploy

# 4. A/B test the results before production deployment
```

### Scenario 3: Reproduce Published Results
You want to verify that new data produces consistent results with published analysis:

```bash
# 1. Collect data with identical parameters
cd ../../sample-collection-scripts
for gpu in V100 A100; do
    for app in LLAMA STABLEDIFFUSION VIT WHISPER; do
        ./launch_v2.sh --gpu-type "$gpu" --app-name "$app" --profiling-mode dvfs --num-runs 3
    done
done

# 2. Run analysis with run 2 data (exclude cold start)
cd ../edp_analysis/examples
python analyze_new_data_current.py \
    --data-dir ../../sample-collection-scripts \
    --run 2 \
    --output ./reproduction_analysis \
    --deploy

# 3. Compare with published results
# Expected: V100+STABLEDIFFUSION ~1110MHz, A100+STABLEDIFFUSION ~1245MHz
```

### Scenario 4: Custom Performance Constraints
You need different performance constraints for your specific use case:

```python
# Custom analysis with modified constraints
import sys
sys.path.append('..')

from edp_analysis.optimization.production_optimizer import ProductionOptimizer
import pandas as pd

# Load your aggregated data
data = pd.read_csv('./results/aggregated_data.csv')

# Create optimizer with custom constraints
optimizer = ProductionOptimizer()

# Modify constraints for your use case
custom_constraints = {
    'LLAMA': {'max_penalty': 0.03},      # 3% max for ultra-low latency
    'STABLEDIFFUSION': {'max_penalty': 0.25},  # 25% max for batch processing
}

# Run optimization with custom constraints
results = optimizer.optimize_with_custom_constraints(data, custom_constraints)
```

## 📊 Expected Outputs

### Successful Analysis Output
```
🚀 Starting GPU Frequency Optimization Analysis
Data directory: ../../sample-collection-scripts
Output directory: ./results
Using run 2 data (excludes cold start if run > 1)

📊 Step 1: Aggregating profiling data...
✅ Found 8 result directories
Processing results_V100_LLAMA_job_12345...
  ✅ Processed 60 frequency points
Processing results_A100_STABLEDIFFUSION_job_12346...
  ✅ Processed 61 frequency points
...
✅ Aggregated 480 total configurations
✅ Saved aggregated data to: ./results/aggregated_data.csv

🎯 Step 2: Finding optimal frequencies...
✅ Optimized 8 configurations

📋 Step 3: Analyzing results...
  🟢 Production ready: 2 configurations
    - A100+STABLEDIFFUSION: 1245MHz (20.2% slower, 38.9% energy savings)
    - V100+STABLEDIFFUSION: 1110MHz (10.3% slower, 31.4% energy savings)
  🟡 Needs A/B testing: 2 configurations
    - A100+LLAMA: 1200MHz (41.3% slower, 64.0% energy savings)
    - V100+LLAMA: 1365MHz (35.4% slower, 41.4% energy savings)
  🔵 Batch processing only: 4 configurations
    - A100+VIT: 1215MHz (93.1% slower, 99.5% energy savings)
    - A100+WHISPER: 1290MHz (89.8% slower, 98.7% energy savings)
    - V100+VIT: 1140MHz (92.5% slower, 99.4% energy savings)
    - V100+WHISPER: 1230MHz (89.5% slower, 99.1% energy savings)

🎉 Analysis Complete!
📁 All results saved to: ./results

🚀 Ready for immediate deployment:
Recommended: A100+STABLEDIFFUSION
Command: sudo nvidia-smi -ac 1215,1245
Expected: 20.2% slower, 38.9% energy savings
```

### Generated Files
```
results/
├── aggregated_data.csv              # Raw aggregated profiling data
├── optimization_results.json        # Detailed optimization results
├── optimization_summary.txt         # Human-readable summary
└── deploy_optimal_frequencies.sh    # Deployment automation script
```

## 🔧 Troubleshooting

### Common Issues

#### "No data found to process"
- **Cause**: Incorrect data directory or no matching files
- **Solution**: Check directory structure and file naming pattern
```bash
# Verify data directory structure
ls -la ../../sample-collection-scripts/results_*/
# Expected: run_01_*_freq_*_profile.csv files
```

#### "Error processing directory"
- **Cause**: Corrupted CSV files or parsing errors
- **Solution**: Check file integrity and format
```bash
# Check file format
head -5 ../../sample-collection-scripts/results_V100_LLAMA_job_*/run_01_2_freq_510_profile.csv
# Expected: CSV with Entity,POWER,GPUTL,... headers
```

#### "Low energy savings or high performance penalties"
- **Cause**: Suboptimal optimization or data quality issues
- **Solution**: Check for cold start contamination
```bash
# Compare run 1 vs run 2 results
python analyze_new_data_current.py --data-dir ../../sample-collection-scripts --run 1 --output ./run1_check
python analyze_new_data_current.py --data-dir ../../sample-collection-scripts --run 2 --output ./run2_check
# Run 2 should show better (lower) performance penalties
```

### Performance Tips

#### Speed up analysis for large datasets
```bash
# Filter to specific configurations first
python analyze_new_data_current.py \
    --data-dir ../../sample-collection-scripts \
    --gpu A100 \
    --app STABLEDIFFUSION \
    --output ./fast_analysis
```

#### Parallelize multiple analyses
```bash
# Run analyses in parallel for different GPUs
python analyze_new_data_current.py --gpu V100 --output ./v100_results &
python analyze_new_data_current.py --gpu A100 --output ./a100_results &
wait  # Wait for both to complete
```

## 📈 Next Steps

### After Analysis
1. **Review results**: Check optimization_summary.txt for recommendations
2. **Validate expectations**: Ensure performance penalties are acceptable
3. **Test deployment**: Use deployment script in test environment first
4. **Monitor performance**: Track actual vs predicted performance impact

### For Production
1. **Start with lowest impact**: Deploy configurations with <15% performance penalty
2. **A/B testing**: Test configurations with 15-50% performance penalty
3. **Monitor continuously**: Track GPU temperature, power, and throughput
4. **Create rollback plan**: Keep baseline frequency reset commands ready

### For Research
1. **Compare with baselines**: Validate results against published benchmarks
2. **Cross-validate**: Test optimization robustness with different runs
3. **Document methodology**: Record exact parameters for reproducibility
4. **Share results**: Contribute findings back to the research community

---

These examples provide a complete workflow for applying GPU frequency optimization to new datasets, from data collection through production deployment.
