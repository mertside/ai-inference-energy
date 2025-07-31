# 🔄 Reproducibility Guide - Regenerating Analysis on New Results

## 📋 Overview

This guide provides step-by-step instructions for reproducing the GPU frequency optimization analysis on new profiling datasets, ensuring consistent and reliable results.

## 🎯 Reproducibility Checklist

### ✅ Environment Setup
- [ ] Python 3.8+ with exact package versions
- [ ] NVIDIA driver compatibility
- [ ] GPU hardware validation
- [ ] Directory structure verification

### ✅ Data Quality
- [ ] Cold start detection and exclusion
- [ ] Sampling rate consistency (50ms DCGMI)
- [ ] Frequency range validation
- [ ] Statistical significance verification

### ✅ Analysis Configuration
- [ ] Identical optimization parameters
- [ ] Performance constraint validation
- [ ] Baseline frequency consistency
- [ ] Result format standardization

---

## 🏗️ Step 1: Environment Preparation

### Python Environment Setup
```bash
# Create isolated environment
conda create -n edp-analysis python=3.8
conda activate edp-analysis

# Install exact package versions for reproducibility
pip install pandas==1.5.3 numpy==1.24.3 matplotlib==3.7.1 seaborn==0.12.2

# Verify installation
python -c "
import pandas as pd
import numpy as np
print(f'✅ pandas: {pd.__version__}')
print(f'✅ numpy: {np.__version__}')
"
```

### GPU Environment Validation
```python
# Validate GPU environment consistency
from edp_analysis.tools import environment_validator

validator = environment_validator.EnvironmentValidator()
env_report = validator.validate_environment()

print("Environment Validation Report:")
print(f"GPU Count: {env_report.gpu_count}")
print(f"Driver Version: {env_report.driver_version}")
print(f"CUDA Version: {env_report.cuda_version}")
print(f"GPU Models: {env_report.gpu_models}")

if not env_report.is_compatible:
    print("❌ Environment compatibility issues:")
    for issue in env_report.issues:
        print(f"  - {issue}")
    exit(1)
else:
    print("✅ Environment is compatible")
```

### Configuration Template
```yaml
# configs/reproducibility_config.yaml
framework:
  version: "2.0.0"
  reproducible_mode: true
  random_seed: 42
  
data_processing:
  cold_start_handling: "exclude_run_1"
  sampling_validation: true
  frequency_validation: true
  statistical_validation: true
  
optimization:
  method: "performance_constrained"
  deterministic: true
  
  # Exact constraints from published analysis
  constraints:
    LLAMA:
      max_performance_penalty: 0.05
      min_frequency_ratio: 0.85
    STABLEDIFFUSION:
      max_performance_penalty: 0.20
      min_frequency_ratio: 0.70
    VIT:
      max_performance_penalty: 0.95
      min_frequency_ratio: 0.60
    WHISPER:
      max_performance_penalty: 0.95
      min_frequency_ratio: 0.60
      
validation:
  enable_cross_validation: true
  statistical_significance: 0.05
  tolerance_percentage: 2.0  # 2% tolerance for frequency differences
```

---

## 📊 Step 2: Data Collection and Validation

### Consistent Data Collection
```bash
# Use identical collection parameters
cd sample-collection-scripts/

# Collect data with exact same parameters as original study
for gpu in V100 A100; do
    for app in LLAMA STABLEDIFFUSION VIT WHISPER; do
        echo "Collecting data for ${gpu}+${app}..."
        ./launch_v2.sh \
            --gpu-type "$gpu" \
            --app-name "$app" \
            --profiling-mode dvfs \
            --num-runs 3 \
            --sampling-interval 50ms \
            --frequency-sweep full
    done
done
```

### Data Quality Validation
```python
from edp_analysis.tools import data_quality_validator

# Validate collected data meets quality standards
validator = data_quality_validator.DataQualityValidator()

quality_report = validator.comprehensive_validation(
    data_directory="sample-collection-scripts/",
    reference_baseline="published_baseline.json"  # Optional: compare with published data
)

print("Data Quality Report:")
print(f"Total Configurations: {quality_report.total_configurations}")
print(f"Valid Configurations: {quality_report.valid_configurations}")
print(f"Quality Score: {quality_report.quality_score:.2f}/10")

# Check for critical issues
if quality_report.has_critical_issues:
    print("❌ Critical data quality issues found:")
    for issue in quality_report.critical_issues:
        print(f"  - {issue}")
    print("Please fix these issues before proceeding.")
    exit(1)

# Check for warnings
if quality_report.has_warnings:
    print("⚠️ Data quality warnings:")
    for warning in quality_report.warnings:
        print(f"  - {warning}")
```

### Cold Start Detection
```python
from edp_analysis.analysis import cold_start_detector

# Detect and handle cold start effects
detector = cold_start_detector.ColdStartDetector()

cold_start_analysis = detector.analyze_cold_start_effects(
    data_directory="sample-collection-scripts/",
    baseline_run=1,  # Check run 1 for cold start contamination
    warm_runs=[2, 3]  # Use runs 2 and 3 for analysis
)

print("Cold Start Analysis:")
print(f"Cold start contamination detected: {cold_start_analysis.contamination_detected}")
if cold_start_analysis.contamination_detected:
    print(f"Average cold start penalty: {cold_start_analysis.avg_penalty:.1f}%")
    print(f"Recommended warm run: {cold_start_analysis.recommended_run}")
```

---

## 🔧 Step 3: Reproducible Aggregation

### Deterministic Aggregation
```python
from edp_analysis.pipelines import DataAggregationPipeline

# Create aggregation pipeline with reproducible settings
aggregator = DataAggregationPipeline(config="configs/reproducibility_config.yaml")

# Run aggregation with identical parameters
aggregated_data = aggregator.run(
    input_dir="sample-collection-scripts/",
    output_file="new_aggregated_data.csv",
    run_selection=2,  # Use run 2 (first warm run)
    deterministic=True,
    validation=True
)

print("Aggregation Results:")
print(f"Total configurations processed: {aggregated_data.total_configurations}")
print(f"Successfully aggregated: {aggregated_data.successful_configurations}")
print(f"Data quality score: {aggregated_data.quality_score:.2f}")
```

### Cross-Validation with Baseline
```python
from edp_analysis.workflows.reproducibility import BaselineComparator

# Compare aggregated data with published baseline
comparator = BaselineComparator()

comparison_result = comparator.compare_aggregated_data(
    new_data="new_aggregated_data.csv",
    baseline_data="published_aggregated_data.csv",  # From original study
    tolerance=0.05  # 5% tolerance
)

print("Baseline Comparison:")
print(f"Data compatibility: {comparison_result.compatible}")
print(f"Statistical similarity: {comparison_result.statistically_similar}")

if not comparison_result.compatible:
    print("❌ Data incompatibility issues:")
    for issue in comparison_result.compatibility_issues:
        print(f"  - {issue}")
        
if comparison_result.has_differences:
    print("⚠️ Notable differences from baseline:")
    for diff in comparison_result.differences:
        print(f"  - {diff}")
```

---

## 🎯 Step 4: Reproducible Optimization

### Exact Parameter Replication
```python
from edp_analysis.pipelines import OptimizationPipeline

# Load exact optimization configuration
optimization_pipeline = OptimizationPipeline(
    config="configs/reproducibility_config.yaml"
)

# Run optimization with identical parameters
optimization_results = optimization_pipeline.run(
    data_file="new_aggregated_data.csv",
    reproducible_mode=True
)

print("Optimization Results:")
print(f"Total configurations optimized: {len(optimization_results.configurations)}")
print(f"Production-ready configurations: {len(optimization_results.production_ready)}")
print(f"Energy savings range: {optimization_results.energy_savings_range}")
```

### Result Validation Against Published Results
```python
from edp_analysis.workflows.reproducibility import ResultValidator

validator = ResultValidator()

# Compare optimization results with published results
validation_result = validator.validate_optimization_consistency(
    new_results=optimization_results,
    published_results="published_optimization_results.json",
    tolerance_frequency=15,  # 15 MHz tolerance
    tolerance_percentage=2.0  # 2% tolerance for energy/performance metrics
)

print("Result Validation:")
print(f"Overall consistency: {validation_result.consistent}")
print(f"Configurations within tolerance: {validation_result.within_tolerance_count}")
print(f"Configurations outside tolerance: {validation_result.outside_tolerance_count}")

if validation_result.has_discrepancies:
    print("⚠️ Discrepancies found:")
    for discrepancy in validation_result.discrepancies:
        print(f"  - {discrepancy.configuration}: "
              f"Expected {discrepancy.expected_frequency}MHz, "
              f"Got {discrepancy.actual_frequency}MHz "
              f"(diff: {discrepancy.difference:.1f}MHz)")
```

---

## 📈 Step 5: Statistical Validation

### Cross-Validation Analysis
```python
from edp_analysis.analysis import statistical_validator

# Perform k-fold cross-validation to ensure robustness
validator = statistical_validator.StatisticalValidator()

cross_validation_results = validator.cross_validate_optimization(
    data_file="new_aggregated_data.csv",
    k_folds=5,
    config="configs/reproducibility_config.yaml"
)

print("Cross-Validation Results:")
print(f"Mean frequency stability: {cross_validation_results.mean_stability:.3f}")
print(f"Standard deviation: {cross_validation_results.std_deviation:.3f}")
print(f"Coefficient of variation: {cross_validation_results.coefficient_variation:.3f}")

# Check if results are statistically stable
if cross_validation_results.is_stable:
    print("✅ Optimization results are statistically stable")
else:
    print("❌ Optimization results show high variability")
    print("Consider collecting more data or adjusting parameters")
```

### Confidence Interval Analysis
```python
# Calculate confidence intervals for optimization results
confidence_analysis = validator.calculate_confidence_intervals(
    optimization_results=optimization_results,
    confidence_level=0.95
)

print("Confidence Interval Analysis:")
for config in optimization_results.configurations:
    ci = confidence_analysis[config.name]
    print(f"{config.name}: "
          f"{ci.optimal_frequency:.0f}MHz "
          f"[{ci.lower_bound:.0f}, {ci.upper_bound:.0f}] "
          f"(95% CI)")
```

---

## 🔬 Step 6: Reproducibility Reporting

### Generate Reproducibility Report
```python
from edp_analysis.workflows.reproducibility import ReproducibilityReporter

reporter = ReproducibilityReporter()

# Generate comprehensive reproducibility report
report = reporter.generate_complete_report(
    environment_info=env_report,
    data_quality=quality_report,
    aggregation_results=aggregated_data,
    optimization_results=optimization_results,
    validation_results=validation_result,
    statistical_analysis=cross_validation_results
)

# Save report
with open("REPRODUCIBILITY_REPORT.md", "w") as f:
    f.write(report.markdown_content)

print("✅ Reproducibility report generated: REPRODUCIBILITY_REPORT.md")
```

### Configuration Archive
```python
# Archive all configuration files for future reproduction
from edp_analysis.tools import config_archiver

archiver = config_archiver.ConfigurationArchiver()

archive_info = archiver.create_reproduction_archive(
    output_file="reproduction_archive.tar.gz",
    include_data_checksums=True,
    include_environment_info=True,
    include_git_info=True
)

print("🗃️ Reproduction archive created:")
print(f"Archive file: {archive_info.archive_path}")
print(f"Data checksum: {archive_info.data_checksum}")
print(f"Git commit: {archive_info.git_commit}")
print(f"Timestamp: {archive_info.timestamp}")
```

---

## 🎯 Step 7: Result Deployment Validation

### Pre-Deployment Validation
```python
from edp_analysis.deployment import deployment_validator

validator = deployment_validator.DeploymentValidator()

# Validate deployment readiness
deployment_check = validator.validate_deployment_readiness(
    optimization_results=optimization_results,
    target_gpu="A100",  # Specify target deployment GPU
    safety_checks=True
)

if deployment_check.ready_for_deployment:
    print("✅ Configurations ready for deployment")
    print(f"Recommended configurations: {len(deployment_check.recommended_configs)}")
else:
    print("❌ Deployment validation failed:")
    for issue in deployment_check.blocking_issues:
        print(f"  - {issue}")
```

### Test Deployment
```python
# Test deployment in safe mode (non-persistent)
test_results = validator.test_deployment(
    configuration="A100+STABLEDIFFUSION",
    optimal_frequency=1245,
    test_duration=60,  # 1 minute test
    rollback_automatic=True
)

print("Test Deployment Results:")
print(f"Deployment successful: {test_results.successful}")
print(f"Performance impact: {test_results.performance_impact:.1f}%")
print(f"Energy savings: {test_results.energy_savings:.1f}%")
print(f"Temperature change: {test_results.temperature_change:.1f}°C")

if test_results.within_expectations:
    print("✅ Test deployment results match expectations")
else:
    print("⚠️ Test deployment results differ from expectations")
```

---

## 📋 Reproducibility Checklist

### Final Validation Checklist
```python
from edp_analysis.workflows.reproducibility import ReproducibilityChecklist

checklist = ReproducibilityChecklist()

# Run comprehensive reproducibility validation
final_check = checklist.validate_full_reproducibility(
    original_config="published_config.yaml",
    new_results=optimization_results,
    tolerance_strict=True
)

print("📋 Final Reproducibility Checklist:")
for check_item in final_check.items:
    status = "✅" if check_item.passed else "❌"
    print(f"{status} {check_item.description}")
    if not check_item.passed:
        print(f"    Issue: {check_item.issue}")
        print(f"    Solution: {check_item.suggested_solution}")

print(f"\nOverall Reproducibility Score: {final_check.score:.1f}/10")

if final_check.reproducible:
    print("🎉 Analysis is fully reproducible!")
else:
    print("⚠️ Reproducibility issues found. Address issues above before proceeding.")
```

---

## 🚀 Quick Reproduction Commands

### One-Command Reproduction
```bash
# Complete reproduction pipeline
python -m edp_analysis reproduce \
    --data-dir "sample-collection-scripts/" \
    --config "configs/reproducibility_config.yaml" \
    --baseline "published_results.json" \
    --output "reproduction_results/" \
    --validate-all
```

### Docker-Based Reproduction
```dockerfile
# Dockerfile for reproducible analysis environment
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.8 python3-pip
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy analysis framework
COPY edp_analysis/ /app/edp_analysis/
COPY configs/ /app/configs/

WORKDIR /app
CMD ["python3", "-m", "edp_analysis", "reproduce", "--config", "configs/reproducibility_config.yaml"]
```

```bash
# Run reproduction in isolated Docker environment
docker build -t edp-analysis-repro .
docker run --gpus all -v $(pwd)/data:/app/data edp-analysis-repro
```

---

## 📊 Expected Reproducible Results

### Key Metrics That Should Match
- **Optimal frequencies** within ±15 MHz
- **Energy savings** within ±2%
- **Performance penalties** within ±2%
- **Configuration categories** (production/testing/batch) identical

### Acceptable Variations
- **Minor frequency differences** due to hardware variations
- **Small metric variations** due to thermal/environmental conditions
- **Timestamp differences** in execution time measurements

### Red Flags (Investigate Immediately)
- **Major frequency differences** (>50 MHz)
- **Category mismatches** (production vs batch classification)
- **Missing configurations** that should be present
- **Cold start contamination** in new results

---

This reproducibility guide ensures that new profiling datasets can be analyzed with the same methodology and produce consistent, comparable results to the original study.
