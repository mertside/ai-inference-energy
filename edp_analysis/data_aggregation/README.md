# EDP Analysis - Data Aggregation

This module provides tools for aggregating time-series GPU profiling data into statistical summaries suitable for Energy-Delay Product (EDP) optimization analysis.

## Overview

The data aggregation pipeline converts raw DCGMI profiling CSV files from job results into per-frequency statistical summaries, enabling downstream EDP analysis and optimal frequency selection.

## Features

- **Robust CSV Parsing**: Handles DCGMI format variations and device name parsing
- **Auto-Discovery**: Automatically detects available frequencies from data files
- **Statistical Aggregation**: Computes mean, std, min, max for key metrics
- **Energy Calculation**: Derives energy consumption from power and execution time
- **Multi-GPU Support**: Handles V100, A100, H100 configurations
- **Flexible Filtering**: Filter by GPU type, application, or run number

## Usage

### Basic Aggregation

```bash
# Aggregate all available data
python aggregate_profiling_data.py \
    --input-dir ../../sample-collection-scripts \
    --output complete_aggregation.csv

# Filter by specific GPU and application
python aggregate_profiling_data.py \
    --input-dir ../../sample-collection-scripts \
    --gpu V100 --app LLAMA \
    --output v100_llama_data.csv

# Specify run number
python aggregate_profiling_data.py \
    --input-dir ../../sample-collection-scripts \
    --run 2 \
    --output run2_data.csv
```

### Command Line Options

- `--input-dir`: Directory containing result folders (default: ../../sample-collection-scripts)
- `--output`: Output CSV file path (default: aggregated_profiling_data.csv)
- `--gpu`: Filter by GPU type (V100, A100, H100)
- `--app`: Filter by application (LLAMA, VIT, STABLEDIFFUSION, WHISPER)
- `--run`: Run number to process (default: 1)

## Input Data Format

The script expects result directories with the pattern:
```
sample-collection-scripts/
├── results_{gpu}_{app}_job_{jobid}/
│   ├── run_{run_id}_{run_number}_freq_{frequency}_profile.csv
│   ├── run_{run_id}_{run_number}_freq_{frequency}_app.out
│   └── ...
```

### DCGMI CSV Format

Input CSV files should follow DCGMI format:
```csv
# Entity  NVIDX  DVNAM       POWER  PMLMT  TMPTR  GPUTL  MCUTL  DRAMA  ...
     GPU     0   Tesla V100   45.7   250.0   52.3   78.2   48.4   0.145  ...
```

## Output Data Format

Aggregated data includes per-frequency statistics:

| Column | Description | Units |
|--------|-------------|-------|
| `frequency` | GPU frequency | MHz |
| `execution_time` | Total execution time | seconds |
| `num_samples` | Number of profiling samples | count |
| `avg_power` | Average power consumption | W |
| `std_power` | Power standard deviation | W |
| `energy` | Total energy consumption | J |
| `avg_gputl` | Average GPU utilization | % |
| `avg_drama` | Average DRAM activity | % |
| `avg_temp` | Average GPU temperature | °C |
| `edp` | Energy-Delay Product | J⋅s |
| `ed2p` | Energy-Delay² Product | J⋅s² |

## Implementation Details

### Execution Time Calculation

Execution time is calculated from the number of profiling samples:
```python
execution_time = num_samples × sampling_interval
# DCGMI sampling interval = 50ms
```

### Energy Calculation

Energy consumption is derived from average power and execution time:
```python
energy = avg_power × execution_time  # Joules
```

### Metric Processing

- **Utilization metrics** (GPUTL, DRAMA, etc.): Converted to percentages if values are 0-1
- **Power metrics**: Validated for reasonable ranges (15-500W)
- **Temperature metrics**: Includes average and maximum values
- **Clock frequencies**: Averaged across the profiling period

## Error Handling

- **Missing files**: Logs warnings and continues processing
- **Parse errors**: Skips problematic files with detailed error messages
- **Invalid data**: Filters out NaN values and validates ranges
- **Directory structure**: Robust pattern matching for various naming conventions

## Example Output

```
============================================================
AGGREGATED DATA SUMMARY
============================================================
Total configurations: 480
GPUs: ['A100', 'V100']
Applications: ['LLAMA', 'STABLEDIFFUSION', 'VIT', 'WHISPER']
Frequencies: [510, 525, ..., 1380, 1395, 1410] MHz
Runs: [1]
Energy range: 217.7 - 30332.9 J
Execution time range: 8.0 - 660.9 s
Power range: 26.1 - 133.8 W
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- pathlib (built-in)
- re (built-in)

## Performance Notes

- Processing ~480 configurations with 60+ frequencies each
- Memory efficient streaming of large CSV files
- Parallel processing potential for large datasets

## Troubleshooting

### Common Issues

1. **"No result directories found"**
   - Verify input directory path
   - Check directory naming convention
   - Ensure read permissions

2. **"No profile files found"**
   - Verify run number exists in data
   - Check file naming pattern matches expected format

3. **"Empty dataframe"**
   - Check CSV file format and headers
   - Verify DCGMI data is properly formatted
   - Look for parse errors in logs

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
