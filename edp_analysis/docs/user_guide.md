# 📚 EDP Analysis Framework - Complete User Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+ with pandas, numpy, matplotlib
- NVIDIA GPU with nvidia-smi access
- DCGMI profiling data from AI inference experiments

### 1. Install and Setup
```bash
cd ai-inference-energy/
pip install -r requirements.txt
```

### 2. Run Analysis on Sample Data
```python
from edp_analysis import quick_analysis

# Analyze existing production data
results = quick_analysis("edp_analysis/data_aggregation/complete_aggregation_run2.csv")
print(f"✅ Found {len(results['optimizations'])} optimal configurations")
```

### 3. Deploy Recommended Configuration
```bash
# Deploy the best configuration immediately
sudo nvidia-smi -ac 877,1110  # V100+STABLEDIFFUSION: 10.3% slower, 31.4% energy savings
```

---

## 📊 Analyzing New Datasets - Complete Workflow

### Step 1: Data Collection
First, collect profiling data using the main framework:

```bash
# Example: Profile LLAMA on V100 GPU
cd sample-collection-scripts/
./launch_v2.sh --gpu-type V100 --app-name LLAMA --profiling-mode dvfs --num-runs 3
```

**Expected Directory Structure:**
```
sample-collection-scripts/
├── results_V100_LLAMA_job_12345/
│   ├── run_01_1_freq_510_profile.csv    # Cold start run
│   ├── run_01_2_freq_510_profile.csv    # First warm run  
│   ├── run_01_3_freq_510_profile.csv    # Second warm run
│   ├── run_01_1_freq_960_profile.csv
│   └── ... (more frequencies)
```

### Step 2: Data Aggregation Pipeline
Convert raw profiling data to analysis-ready format:

```python
# Method 1: Python API (Recommended)
from edp_analysis.core import ProfilingDataLoader
from edp_analysis.analysis import aggregation

# Load and aggregate data
loader = ProfilingDataLoader()
aggregator = aggregation.DataAggregator()

# Aggregate with warm-run selection (excludes cold start)
aggregated_data = aggregator.aggregate_profiling_data(
    input_dir="sample-collection-scripts/",
    output_file="new_analysis_data.csv",
    run_selection="warm_runs",  # Uses run 2 by default
    gpu_filter=["V100", "A100"],  # Optional: filter specific GPUs
    app_filter=["LLAMA", "STABLEDIFFUSION"]  # Optional: filter apps
)

print(f"✅ Aggregated {aggregated_data['total_configurations']} configurations")
```

```bash
# Method 2: Command Line
cd edp_analysis/data_aggregation/
python aggregate_profiling_data.py \
    --input-dir ../../sample-collection-scripts \
    --output new_analysis_data.csv \
    --run 2 \
    --gpu V100 A100 \
    --app LLAMA STABLEDIFFUSION
```

**Quality Validation:**
```python
# Validate aggregated data quality
from edp_analysis.tools import data_validator

validator = data_validator.DataQualityValidator()
validation_report = validator.validate_aggregated_data("new_analysis_data.csv")

if validation_report.passed:
    print("✅ Data quality validation passed")
else:
    print(f"❌ Validation failed: {validation_report.issues}")
    # Fix data quality issues before proceeding
```

### Step 3: Optimization Pipeline
Find optimal frequencies with performance constraints:

```python
# Method 1: Automated Pipeline (Recommended)
from edp_analysis import analyze_dataset

results = analyze_dataset(
    data_path="new_analysis_data.csv",
    config_path="configs/production.yaml",  # Optional: custom config
    output_dir="analysis_results/"
)

# Review results
print("🎯 Production-Ready Configurations:")
for config in results.production_ready:
    print(f"  {config.configuration}: {config.optimal_frequency}MHz "
          f"({config.performance_penalty:.1f}% slower, "
          f"{config.energy_savings:.1f}% energy savings)")
```

```python
# Method 2: Custom Optimization
from edp_analysis.analysis.optimization import UnifiedOptimizer
from edp_analysis.configs import load_config

# Load configuration
config = load_config("configs/custom_constraints.yaml")

# Create optimizer with custom constraints
optimizer = UnifiedOptimizer(config)

# Optimize all configurations
optimization_results = optimizer.optimize_all_configurations(
    data_file="new_analysis_data.csv"
)

# Generate deployment script
from edp_analysis.analysis.reporting import ReportGenerator
reporter = ReportGenerator()
deployment_script = reporter.generate_deployment_script(optimization_results)

with open("deploy_optimal_frequencies.sh", "w") as f:
    f.write(deployment_script)
```

### Step 4: Results Validation and Deployment

```python
# Validate optimization results
from edp_analysis.analysis.validation import ResultValidator

validator = ResultValidator()
validation_report = validator.validate_optimization_results(
    results=optimization_results,
    data_file="new_analysis_data.csv"
)

# Check for issues
if validation_report.has_warnings:
    print("⚠️ Validation warnings:")
    for warning in validation_report.warnings:
        print(f"  - {warning}")

# Generate comprehensive report
report = reporter.generate_markdown_report(
    results=optimization_results,
    include_deployment_guide=True,
    include_monitoring_recommendations=True
)

with open("analysis_results/optimization_report.md", "w") as f:
    f.write(report)
```

**Production Deployment:**
```bash
# Deploy recommended configuration
chmod +x deploy_optimal_frequencies.sh
./deploy_optimal_frequencies.sh V100+STABLEDIFFUSION deploy

# Monitor performance impact
./deploy_optimal_frequencies.sh V100+STABLEDIFFUSION status
watch -n 5 'nvidia-smi --query-gpu=clocks.gr,temperature.gpu,power.draw --format=csv,noheader'
```

---

## 🎯 Advanced Configuration

### Custom Performance Constraints
Create application-specific optimization constraints:

```yaml
# configs/custom_constraints.yaml
framework:
  version: "2.0.0"
  name: "Custom LLM Optimization"

optimization:
  method: "performance_constrained"
  
  # Application-specific constraints
  constraints:
    LLAMA:
      max_performance_penalty: 0.05  # 5% max for interactive LLM
      min_frequency_ratio: 0.85      # Don't go below 85% of max freq
      priority: "latency"            # Prioritize response time
      
    STABLEDIFFUSION:
      max_performance_penalty: 0.15  # 15% max for image generation
      min_frequency_ratio: 0.70      # More aggressive scaling allowed
      priority: "energy"             # Prioritize energy savings
      
    VIT:
      max_performance_penalty: 0.50  # 50% max for batch processing
      min_frequency_ratio: 0.60      # Very aggressive scaling
      priority: "batch_throughput"   # Optimize for batch efficiency

  # Hardware-specific settings
  hardware:
    V100:
      thermal_limit: 83              # Temperature constraint (°C)
      power_limit: 250               # Power constraint (W)
    A100:
      thermal_limit: 90
      power_limit: 400

  # Output preferences
  output:
    generate_deployment_script: true
    include_monitoring_commands: true
    create_rollback_script: true
```

### Batch Analysis of Multiple Datasets
Process multiple profiling datasets in parallel:

```python
from edp_analysis.workflows import batch_analysis

# Define multiple datasets
datasets = {
    "experiment_1": "profiling_data_v1/",
    "experiment_2": "profiling_data_v2/", 
    "baseline_comparison": "baseline_data/"
}

# Run batch analysis
batch_results = batch_analysis.run_batch_optimization(
    datasets=datasets,
    config_path="configs/batch_analysis.yaml",
    output_dir="batch_results/",
    parallel=True  # Process datasets in parallel
)

# Compare results across experiments
comparison_report = batch_analysis.generate_comparison_report(batch_results)
print(comparison_report.summary_table)
```

---

## 🔄 Reproducibility Validation

### Verify Analysis Consistency
Ensure your new analysis produces consistent results:

```python
from edp_analysis.workflows.reproducibility import ReproducibilityValidator

validator = ReproducibilityValidator()

# Compare new results with published baseline
comparison = validator.validate_analysis(
    original_results="published_results.json",
    new_results=optimization_results,
    tolerance=0.05  # 5% tolerance for frequency differences
)

if comparison.is_consistent:
    print("✅ Results are consistent with published analysis")
else:
    print(f"❌ Inconsistencies found: {comparison.differences}")
    
# Generate reproduction guide
reproduction_guide = validator.generate_reproduction_guide(
    analysis_config=config,
    data_summary=aggregated_data.summary,
    optimization_results=optimization_results
)

with open("REPRODUCTION_GUIDE.md", "w") as f:
    f.write(reproduction_guide)
```

### Cross-Platform Validation
Test analysis across different environments:

```python
# Platform compatibility check
platform_report = validator.test_cross_platform_compatibility(
    data_file="new_analysis_data.csv",
    config_file="configs/production.yaml"
)

print(f"Platform compatibility: {platform_report.status}")
if not platform_report.compatible:
    print("Platform-specific issues:")
    for issue in platform_report.issues:
        print(f"  - {issue}")
```

---

## 📈 Monitoring and Validation

### Performance Monitoring
Monitor GPU frequency changes in production:

```bash
# Continuous monitoring script
cat > monitor_gpu_performance.sh << 'EOF'
#!/bin/bash
echo "Monitoring GPU performance after frequency optimization..."
echo "Timestamp,GPU_Freq,Memory_Freq,Temperature,Power,Utilization" > gpu_monitoring.csv

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    gpu_freq=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)
    mem_freq=$(nvidia-smi --query-gpu=clocks.mem --format=csv,noheader,nounits)
    temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    echo "$timestamp,$gpu_freq,$mem_freq,$temp,$power,$util" >> gpu_monitoring.csv
    sleep 30  # Monitor every 30 seconds
done
EOF

chmod +x monitor_gpu_performance.sh
./monitor_gpu_performance.sh &
```

### Application Performance Validation
Measure inference performance impact:

```python
# Application performance validation
from edp_analysis.tools import performance_validator

validator = performance_validator.ApplicationValidator()

# Test application performance at optimized frequency
performance_test = validator.benchmark_application(
    app_name="STABLEDIFFUSION",
    gpu_frequency=1110,  # Optimized frequency
    baseline_frequency=1380,  # Original frequency
    test_duration=300,  # 5 minutes
    num_iterations=10
)

print(f"Performance impact: {performance_test.performance_change:.1f}%")
print(f"Energy savings: {performance_test.energy_savings:.1f}%")
print(f"Temperature change: {performance_test.temperature_change:.1f}°C")

if performance_test.within_tolerance:
    print("✅ Performance impact within acceptable limits")
else:
    print("⚠️ Performance impact exceeds tolerance, consider reverting")
```

---

## 🐛 Troubleshooting Guide

### Common Issues and Solutions

#### 1. "No profiling data found"
**Cause**: Incorrect directory structure or file naming
**Solution**:
```python
# Check data directory structure
from edp_analysis.tools import data_validator
validator = data_validator.DataDiscovery()
discovered_data = validator.discover_profiling_data("sample-collection-scripts/")
print(f"Found {len(discovered_data.result_directories)} result directories")
print("Expected pattern: results_{GPU}_{APP}_job_{JOBID}/")
```

#### 2. "Cold start contamination detected"
**Cause**: Using run 1 data which includes initialization overhead
**Solution**:
```python
# Always use warm run data (run 2 or 3)
aggregated_data = aggregator.aggregate_profiling_data(
    input_dir="sample-collection-scripts/",
    run_selection="warm_runs",  # This excludes run 1
    cold_start_detection=True   # Enable automatic detection
)
```

#### 3. "Optimization finds no valid configurations"
**Cause**: Performance constraints too strict
**Solution**:
```python
# Relax constraints gradually
relaxed_config = config.copy()
relaxed_config['optimization']['constraints']['max_performance_penalty'] = 0.30  # Increase from 0.20
relaxed_config['optimization']['constraints']['min_frequency_ratio'] = 0.60      # Decrease from 0.70

results = analyzer.optimize_with_config(data, relaxed_config)
```

#### 4. "Inconsistent results across runs"
**Cause**: Non-deterministic aggregation or configuration differences
**Solution**:
```python
# Enable deterministic mode
config['framework']['deterministic'] = True
config['framework']['random_seed'] = 42

# Verify configuration consistency
from edp_analysis.tools import config_validator
config_validator.validate_config_consistency(config_file_1, config_file_2)
```

### Performance Issues

#### Slow aggregation pipeline
```python
# Enable parallel processing
aggregated_data = aggregator.aggregate_profiling_data(
    input_dir="sample-collection-scripts/",
    parallel=True,
    num_workers=4  # Adjust based on available CPU cores
)
```

#### Memory issues with large datasets
```python
# Enable streaming mode for large datasets
aggregated_data = aggregator.aggregate_profiling_data(
    input_dir="sample-collection-scripts/",
    streaming=True,
    chunk_size=1000  # Process 1000 files at a time
)
```

### Deployment Issues

#### nvidia-smi permission denied
```bash
# Add user to appropriate group or use sudo
sudo nvidia-smi -ac 1215,1245

# Or create udev rule for persistent access
echo 'KERNEL=="nvidia*", MODE="0666"' | sudo tee /etc/udev/rules.d/99-nvidia.rules
sudo udevadm control --reload-rules
```

#### Frequency setting not persistent
```bash
# Create systemd service for persistent frequency setting
sudo tee /etc/systemd/system/nvidia-freq-optimization.service << 'EOF'
[Unit]
Description=NVIDIA GPU Frequency Optimization
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -ac 1215,1245
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable nvidia-freq-optimization.service
sudo systemctl start nvidia-freq-optimization.service
```

---

## 📚 Complete API Reference

### High-Level Functions
```python
from edp_analysis import (
    analyze_dataset,           # Complete analysis pipeline
    quick_analysis,           # Fast analysis for existing data
    validate_results,         # Result validation
    deploy_configuration      # Production deployment
)

# Analyze new dataset
results = analyze_dataset(
    data_path="profiling_data/",      # Raw profiling data directory
    config_path="config.yaml",        # Optional configuration file
    output_dir="results/",            # Output directory
    include_validation=True,          # Include result validation
    generate_deployment=True          # Generate deployment scripts
)

# Quick analysis of aggregated data
quick_results = quick_analysis(
    data_file="aggregated_data.csv",  # Pre-aggregated data
    config=config_dict,               # Configuration dictionary
    save_results=True                 # Save results to file
)
```

### Core Analysis Classes
```python
from edp_analysis.core import (
    ProfilingDataLoader,      # Data loading and validation
    EnergyCalculator,         # Energy and EDP calculations
    FrequencyOptimizer,       # Frequency optimization
    PerformanceAnalyzer       # Performance analysis
)

# Example usage
loader = ProfilingDataLoader()
data = loader.load_aggregated_data("data.csv")

calculator = EnergyCalculator()
edp_results = calculator.calculate_edp_metrics(data)

optimizer = FrequencyOptimizer(constraints=constraints)
optimal_frequencies = optimizer.find_optimal_frequencies(data)
```

### Pipeline Classes
```python
from edp_analysis.pipelines import (
    DataAggregationPipeline,   # Raw data → aggregated CSV
    OptimizationPipeline,      # Aggregated CSV → optimal frequencies
    ValidationPipeline,        # Result validation and testing
    DeploymentPipeline         # Production deployment
)

# Full pipeline execution
aggregation = DataAggregationPipeline()
aggregated_data = aggregation.run("profiling_data/", "aggregated.csv", config)

optimization = OptimizationPipeline()
optimal_configs = optimization.run("aggregated.csv", config)

deployment = DeploymentPipeline()
deployment.deploy_configuration("A100+STABLEDIFFUSION", optimal_configs)
```

---

This comprehensive user guide provides everything needed to reproduce and extend the GPU frequency optimization analysis on new datasets. The framework is designed to be both powerful for advanced users and simple for quick analyses.

For additional support, check the troubleshooting section or refer to the API documentation for detailed function parameters and return values.
