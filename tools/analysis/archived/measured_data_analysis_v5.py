#!/usr/bin/env python3

"""
Measured Data Analysis for AI Inference Energy Project - VERSION 5

This script analyzes measured experimental data to extract optimal GPU frequencies
for AI inference workloads, balancing energy efficiency with performance constraints.

Key Features:
- Hybrid timing extraction from experiment_summary.log (primary) and .out files (fallback)
- Dual baseline validation: energy savings vs max frequency, performance vs fastest execution
- Statistical aggregation with cold-start exclusion and outlier detection
- Comprehensive reporting across GPU architectures (A100, H100, V100)
- Energy-Delay Product (EDP) optimization for finding optimal operating points

Analysis Methodology:
1. Extract timing data from experiment logs using hybrid approach
2. Extract power measurements from DCGMI CSV profiles
3. Aggregate multiple runs per frequency for statistical reliability
4. Apply performance constraints (≤5% degradation from fastest execution)
5. Find optimal frequency minimizing EDP within constraints
6. Report energy savings vs hardware specification maximum frequency for deployment context

IMPROVEMENTS OVER v4:
- Primary timing source switched to experiment_summary.log for universal reliability
- Enhanced baseline frequency validation using hardware specification maximum frequencies
- Improved statistical reporting with timing source transparency
- Better handling of thermal throttling and frequency scaling effects
"""

import re
import csv
import statistics
from pathlib import Path
from collections import defaultdict

# GPU Architecture Constants
# Define hardware specification maximum frequencies for each GPU architecture (MHz)
# These represent the manufacturer-specified maximum frequencies, not necessarily
# the highest measured frequencies in experiments (which may be lower due to thermal throttling)
GPU_HARDWARE_MAX_FREQUENCIES = {
    'A100': 1410,  # NVIDIA A100 specification maximum GPU clock
    'H100': 1785,  # NVIDIA H100 specification maximum GPU clock
    'V100': 1380   # NVIDIA V100 specification maximum GPU clock
}

# Utility Functions
# ==================

def parse_directory_name(dir_name):
    """
    Parse result directory name to extract GPU type, workload, and job ID.
    
    Expected format: results_<gpu>_<workload>_job_<job_id>
    Example: results_a100_llama_job_20892442
    
    Args:
        dir_name (str): Directory name to parse
        
    Returns:
        tuple: (gpu_type, workload, job_id) or (None, None, None) if parsing fails
    """
    match = re.match(r'results_([^_]+)_([^_]+)_job_(\d+)', dir_name)
    if match:
        return match.group(1).upper(), match.group(2), match.group(3)
    return None, None, None

# Timing Extraction Functions
# ===========================

def extract_inference_timing_from_out(out_file):
    """
    Extract inference timing from application .out files using comprehensive patterns.
    
    This function searches for various timing patterns that different AI workloads
    might output. It's designed to be robust across different application formats.
    
    Args:
        out_file (Path): Path to the .out file
        
    Returns:
        tuple: (timing_value, pattern_name) or (None, error_reason)
    """
    if not out_file.exists():
        return None, "file_not_found"
    
    try:
        with open(out_file, 'r') as f:
            content = f.read()
        
        # Comprehensive timing patterns ordered by preference
        # More specific patterns first, then general fallbacks
        timing_patterns = [
            (r'Total Inference Time:\s*([0-9]+\.?[0-9]*)\s*s', 'total_inference_time'),
            (r'Inference Time:\s*([0-9]+\.?[0-9]*)\s*s', 'inference_time'),
            (r'Model Inference Time:\s*([0-9]+\.?[0-9]*)\s*s', 'model_inference_time'),
            (r'Generation Time:\s*([0-9]+\.?[0-9]*)\s*s', 'generation_time'),
            (r'Benchmark completed in\s*([0-9]+\.?[0-9]*)\s*seconds', 'benchmark_time'),
            (r'Total Time:\s*([0-9]+\.?[0-9]*)\s*s', 'total_time'),
            (r'Execution Time:\s*([0-9]+\.?[0-9]*)\s*s', 'execution_time'),
            (r'Processing took\s*([0-9]+\.?[0-9]*)\s*seconds', 'processing_time'),
            (r'Runtime:\s*([0-9]+\.?[0-9]*)\s*s', 'runtime'),
        ]
        
        for pattern, pattern_name in timing_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                try:
                    # Use last match (often the final/summary timing)
                    time_value = float(matches[-1])
                    if time_value > 0:
                        return time_value, pattern_name
                except ValueError:
                    continue
        
        return None, "no_timing_pattern_found"
        
    except Exception as e:
        return None, f"extraction_error_{str(e)}"

def extract_timing_from_experiment_summary(summary_file):
    """
    Extract timing information from experiment_summary.log file.
    
    This file contains a comprehensive "Run Timing Summary" section with all
    experimental run timings, marked with success/failure status. This is the
    most reliable timing source across all workloads.
    
    Args:
        summary_file (Path): Path to experiment_summary.log
        
    Returns:
        dict: Mapping of run_keys to timing data
              Format: {run_key: {'time': float, 'frequency': int, 'status': str}}
    """
    timing_data = {}
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # Look for timing summary section
        timing_section = False
        for line in content.split('\n'):
            line = line.strip()
            
            if 'Run Timing Summary' in line:
                timing_section = True
                continue
            
            if timing_section and line:
                # Parse lines like: "Run 186_01 : 35s (freq: 825MHz, status: success)"
                match = re.match(r'Run\s+(\d+)_(\d+)\s*:\s*(\d+)s\s*\(freq:\s*(\d+)MHz,\s*status:\s*(\w+)\)', line)
                if match:
                    run_id, run_number, time_str, freq_str, status = match.groups()
                    if status == 'success':
                        key = f"{run_id}_{run_number.zfill(2)}"
                        timing_data[key] = {
                            'time': float(time_str),
                            'frequency': int(freq_str),
                            'status': status
                        }
                elif line.startswith('=') or 'Total runs' in line:
                    # End of timing section
                    break
        
        return timing_data
        
    except Exception as e:
        print(f"    ⚠️ Error reading experiment summary: {e}")
        return {}

def extract_timing_hybrid(result_dir, run_id, run_number, frequency):
    """
    Extract timing using hybrid approach with experiment_summary.log as primary source.
    
    This function implements a robust timing extraction strategy:
    1. PRIMARY: Extract session time from experiment_summary.log (most reliable)
    2. SECONDARY: Extract inference time from .out file (for validation)
    3. Use session time as primary timing source for universal reliability
    4. Validate consistency and report any significant discrepancies
    
    Args:
        result_dir (Path): Directory containing experiment files
        run_id (str): Run identifier (e.g., "186")
        run_number (int): Run number within the experiment
        frequency (int): GPU frequency for this run
        
    Returns:
        dict: Comprehensive timing information with validation status
    """
    
    # PRIMARY: Get session time from experiment_summary.log (most reliable)
    summary_file = result_dir / 'experiment_summary.log'
    summary_data = extract_timing_from_experiment_summary(summary_file)
    run_key = f"{run_id}_{run_number:02d}"
    session_time = summary_data.get(run_key, {}).get('time', None)
    
    # SECONDARY: Try to get inference time from .out file for validation
    out_file = result_dir / f"run_{run_id}_{run_number:02d}_freq_{frequency}_app.out"
    inference_time, inference_status = extract_inference_timing_from_out(out_file)
    
    # Build comprehensive timing information
    timing_info = {
        'inference_time': inference_time,
        'inference_status': inference_status,
        'session_time': session_time,
        'selected_time': None,
        'timing_type': None,
        'overhead_ratio': None,
        'validation_status': 'unknown'
    }
    
    # Use session time as primary (most reliable across all workloads)
    if session_time and session_time > 0:
        timing_info['selected_time'] = session_time
        timing_info['timing_type'] = 'session'
        
        # Optional validation with inference time if available
        if inference_time and inference_time > 0:
            if inference_time <= session_time:
                overhead = session_time - inference_time
                timing_info['overhead_ratio'] = overhead / session_time
                timing_info['validation_status'] = 'validated_with_inference'
                
                # Warning for unusually high overhead (>70%)
                if timing_info['overhead_ratio'] > 0.7:
                    timing_info['validation_status'] = 'high_overhead_warning'
            else:
                timing_info['validation_status'] = 'inference_time_inconsistent'
        else:
            timing_info['validation_status'] = 'session_timing_only'
    
    # Fallback to inference time only if session time unavailable
    elif inference_time and inference_time > 0:
        timing_info['selected_time'] = inference_time
        timing_info['timing_type'] = 'inference'
        timing_info['validation_status'] = 'inference_fallback'
    
    else:
        timing_info['validation_status'] = 'no_timing_found'
    
    return timing_info

# Power Data Extraction Functions
# ================================

def extract_power_from_csv(csv_file):
    """
    Extract average power consumption from DCGMI profile CSV files.
    
    DCGMI (Data Center GPU Manager Interface) provides detailed GPU metrics.
    This function parses the power consumption data to calculate average power
    and energy consumption for the experimental run.
    
    Expected CSV format:
    Line 0: #Entity   NVIDX   DVNAM   POWER   PMLMT   ...
    Line 1: ID                        W       W       ...  (units)
    Line 2+: GPU 0    0       NVIDIA  34.535  250.000 ...  (data)
    
    Args:
        csv_file (Path): Path to DCGMI CSV profile file
        
    Returns:
        tuple: (average_power_watts, number_of_samples) or (None, 0) if extraction fails
    """
    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            return None, 0
        
        # Parse header from line 0 (remove # prefix)
        header_line = lines[0].strip()
        if header_line.startswith('#'):
            header_line = header_line[1:]
        
        # Extract power values from data lines
        # The POWER field appears after device identification fields
        power_values = []
        
        for line in lines[2:]:  # Skip header and units lines
            if line.strip() and not line.startswith('#'):
                # Look for patterns like floating point numbers
                # After the GPU device name, power should be the next float
                parts = line.strip().split()
                
                # Typical format: GPU 0 0 NVIDIA A100-PCIE-40GB 34.535 250.000 ...
                # Look for first reasonable power value (should be < 1000W)
                for i, part in enumerate(parts):
                    try:
                        value = float(part)
                        # Power should be reasonable (10-1000W typically for data center GPUs)
                        if 10.0 <= value <= 1000.0:
                            # Check if this could be power by looking at position
                            # Power is typically after device name
                            if i >= 4:  # After "GPU 0 0 NVIDIA A100-PCIE-40GB"
                                power_values.append(value)
                                break
                    except ValueError:
                        continue
        
        if power_values:
            avg_power = statistics.mean(power_values)
            return avg_power, len(power_values)
        else:
            return None, 0
            
    except Exception as e:
        return None, 0

def extract_frequency_from_filename(filename):
    """
    Extract GPU frequency from profile filename.
    
    Profile filenames contain the GPU frequency used during the measurement.
    Example: run_1_01_freq_1410_profile.csv -> 1410 MHz
    
    Args:
        filename (str): Profile CSV filename
        
    Returns:
        int: GPU frequency in MHz, or None if extraction fails
    """
    match = re.search(r'freq_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# Data Processing Functions
# ==========================

def process_result_directory(result_dir):
    """
    Process a single experimental result directory to extract all measurements.
    
    This function processes all profile CSV files in a result directory,
    extracting timing, power, and energy data for each experimental run.
    It uses the hybrid timing approach for maximum reliability.
    
    Args:
        result_dir (Path): Directory containing experimental results
        
    Returns:
        list: List of measurement dictionaries, each containing:
              - Basic info: gpu, workload, frequency, run_id, run_number
              - Measurements: execution_time, avg_power, total_energy, edp
              - Metadata: timing_type, validation status, etc.
    """
    print(f"\nProcessing: {result_dir.name}")
    
    # Parse directory name to get GPU type and workload
    gpu, workload, job_id = parse_directory_name(result_dir.name)
    if not gpu or not workload:
        print(f"  ⚠️ Could not parse directory name: {result_dir.name}")
        return []
    
    print(f"  GPU: {gpu}, Workload: {workload}")
    
    measurements = []
    profile_files = list(result_dir.glob("*_profile.csv"))
    print(f"  Found {len(profile_files)} profile files")
    
    # Track timing extraction statistics
    timing_stats = {
        'session': 0,
        'inference': 0,
        'failed': 0,
        'high_overhead': 0,
        'inconsistent': 0
    }
    
    # Process each profile file
    for profile_file in profile_files:
        # Extract frequency from filename
        frequency = extract_frequency_from_filename(profile_file.name)
        if not frequency:
            continue
        
        # Extract run information from filename
        # Expected format: run_1_01_freq_1410_profile.csv
        match = re.search(r'run_(\d+)_(\d+)_freq_\d+_profile\.csv', profile_file.name)
        if not match:
            continue
        
        run_id, run_number = match.groups()
        run_number = int(run_number)
        
        # Extract timing using hybrid approach
        timing_info = extract_timing_hybrid(result_dir, run_id, run_number, frequency)
        
        # Skip if no valid timing found
        if not timing_info['selected_time'] or timing_info['selected_time'] <= 0:
            timing_stats['failed'] += 1
            continue
        
        # Update timing statistics
        if timing_info['timing_type'] == 'session':
            timing_stats['session'] += 1
        elif timing_info['timing_type'] == 'inference':
            timing_stats['inference'] += 1
        
        if timing_info['validation_status'] == 'high_overhead_warning':
            timing_stats['high_overhead'] += 1
        elif timing_info['validation_status'] == 'inconsistent_timing':
            timing_stats['inconsistent'] += 1
        
        # Extract power data from CSV
        avg_power, power_samples = extract_power_from_csv(profile_file)
        if not avg_power:
            continue
        
        # Calculate energy metrics
        execution_time = timing_info['selected_time']
        total_energy = avg_power * execution_time  # Watts * seconds = Joules
        edp = total_energy * execution_time        # Energy-Delay Product (Joules * seconds)
        
        # Create measurement record
        measurement = {
            # Basic identification
            'gpu': gpu,
            'workload': workload,
            'frequency': frequency,
            'run_id': run_id,
            'run_number': run_number,
            
            # Core measurements
            'execution_time': execution_time,
            'avg_power': avg_power,
            'total_energy': total_energy,
            'edp': edp,
            
            # Metadata
            'power_samples': power_samples,
            'timing_type': timing_info['timing_type'],
            'timing_validation': timing_info['validation_status'],
            'inference_time': timing_info['inference_time'],
            'session_time': timing_info['session_time'],
            'overhead_ratio': timing_info['overhead_ratio']
        }
        
        measurements.append(measurement)
    
    # Print processing summary
    print(f"  Successfully processed {len(measurements)} measurements")
    print(f"  Timing sources: {timing_stats['session']} session, {timing_stats['inference']} inference")
    
    # Report any issues
    if timing_stats['failed'] > 0:
        print(f"  ⚠️ Failed timing extraction: {timing_stats['failed']} runs")
    if timing_stats['high_overhead'] > 0:
        print(f"  ⚠️ High overhead detected: {timing_stats['high_overhead']} runs")
    if timing_stats['inconsistent'] > 0:
        print(f"  ⚠️ Inconsistent timing: {timing_stats['inconsistent']} runs")
    
    return measurements

# Statistical Analysis Functions
# ===============================

def aggregate_measurements_by_frequency(measurements, exclude_first_run=True):
    """
    Group measurements by frequency and calculate statistical averages with outlier detection.
    
    This function aggregates multiple experimental runs at the same frequency to provide
    statistically reliable measurements. It handles cold-start exclusion and outlier detection
    to ensure data quality.
    
    Args:
        measurements (list): List of individual measurement dictionaries
        exclude_first_run (bool): Whether to exclude first run at each frequency (cold-start)
        
    Returns:
        list: List of aggregated measurements by frequency with statistical metadata
    """
    if not measurements:
        return []
    
    print(f"    Aggregating measurements by frequency (exclude_first_run={exclude_first_run})...")
    
    # Group measurements by frequency
    freq_groups = defaultdict(list)
    for m in measurements:
        freq_groups[m['frequency']].append(m)
    
    # Sort runs within each frequency group to identify first runs (cold-start)
    for freq in freq_groups:
        freq_groups[freq] = sorted(freq_groups[freq], key=lambda x: (x['run_id'], x['run_number']))
    
    # Calculate statistical averages for each frequency with outlier detection
    aggregated = []
    total_excluded = 0
    
    for freq, runs in freq_groups.items():
        if len(runs) == 0:
            continue
        
        excluded_cold_runs = 0
        excluded_outlier_runs = 0
        valid_runs = []
        
        # Exclude first run if requested (cold-start elimination)
        if exclude_first_run and len(runs) > 1:
            excluded_cold_runs = 1
            valid_runs = runs[1:]
            print(f"      Excluding first run at {freq}MHz (cold-start, time: {runs[0]['execution_time']:.0f}s)")
        else:
            valid_runs = runs
        
        if len(valid_runs) == 0:
            continue
        
        # Outlier detection: exclude runs >3x median time (likely measurement errors)
        if len(valid_runs) > 2:
            times = [r['execution_time'] for r in valid_runs]
            median_time = statistics.median(times)
            
            outlier_threshold = median_time * 3
            outliers = [r for r in valid_runs if r['execution_time'] > outlier_threshold]
            
            if outliers:
                excluded_outlier_runs = len(outliers)
                outlier_times = [r['execution_time'] for r in outliers]
                print(f"      Excluding {excluded_outlier_runs} outlier runs at {freq}MHz (times: {outlier_times}, >3x median {median_time:.1f}s)")
                valid_runs = [r for r in valid_runs if r['execution_time'] <= outlier_threshold]
        
        if len(valid_runs) == 0:
            continue
        
        # Calculate averages for all metrics
        avg_time = statistics.mean([r['execution_time'] for r in valid_runs])
        avg_power = statistics.mean([r['avg_power'] for r in valid_runs])
        avg_energy = statistics.mean([r['total_energy'] for r in valid_runs])
        avg_edp = statistics.mean([r['edp'] for r in valid_runs])
        
        # Calculate standard deviations for quality assessment
        if len(valid_runs) > 1:
            time_std = statistics.stdev([r['execution_time'] for r in valid_runs])
            # Warn about high variability (>15% standard deviation)
            if time_std > avg_time * 0.15:
                print(f"        Warning: High variation in timing at {freq}MHz (std: {time_std:.2f}s)")
        else:
            time_std = 0.0
        
        print(f"      Averaging {len(valid_runs)} valid runs at {freq}MHz (excluded {excluded_cold_runs + excluded_outlier_runs} total)")
        
        # Count timing source types for transparency
        session_count = sum(1 for r in valid_runs if r['timing_type'] == 'session')
        inference_count = sum(1 for r in valid_runs if r['timing_type'] == 'inference')
        
        # Create aggregated measurement record
        aggregated_measurement = {
            # Basic identification
            'gpu': valid_runs[0]['gpu'],
            'workload': valid_runs[0]['workload'],
            'frequency': freq,
            
            # Aggregated measurements
            'execution_time': avg_time,
            'avg_power': avg_power,
            'total_energy': avg_energy,
            'edp': avg_edp,
            
            # Statistical metadata
            'run_count': len(valid_runs),
            'excluded_cold_runs': excluded_cold_runs,
            'excluded_outlier_runs': excluded_outlier_runs,
            'execution_time_std': time_std,
            
            # Timing source information
            'session_timing_count': session_count,
            'inference_timing_count': inference_count,
            'primary_timing_type': 'session' if session_count > inference_count else 'inference'
        }
        
        aggregated.append(aggregated_measurement)
        total_excluded += excluded_cold_runs + excluded_outlier_runs
    
    print(f"    Aggregated {len(measurements)} individual measurements into {len(aggregated)} frequency points")
    print(f"    Total excluded runs: {total_excluded} (cold-start + outliers)")
    return aggregated

# Optimization Analysis Functions
# ================================

def calculate_optimal_frequency_hybrid(measurements, constraint_pct=5.0):
    """
    Calculate optimal GPU frequency using dual baseline validation approach.
    
    This function implements the core optimization logic:
    1. Aggregate measurements by frequency for statistical reliability
    2. Establish dual baselines: true max frequency (energy) and fastest execution (performance)
    3. Apply performance constraints to ensure acceptable execution time
    4. Find optimal frequency minimizing Energy-Delay Product (EDP) within constraints
    5. Report comprehensive metrics for both deployment and validation contexts
    
    Args:
        measurements (list): Raw measurement data for a GPU-workload combination
        constraint_pct (float): Maximum performance degradation allowed (default: 5%)
        
    Returns:
        dict: Comprehensive optimization results including optimal frequency,
              energy savings, performance impact, and baseline comparisons
    """
    if not measurements:
        return None
    
    # STEP 1: Aggregate multiple runs at same frequency for statistical reliability
    aggregated_measurements = aggregate_measurements_by_frequency(measurements)
    
    if not aggregated_measurements:
        return None
    
    # Sort by frequency for consistent processing
    aggregated_measurements = sorted(aggregated_measurements, key=lambda x: x['frequency'])
    
    # STEP 2: Establish dual baselines
    gpu_type = measurements[0]['gpu']
    hardware_max_freq = GPU_HARDWARE_MAX_FREQUENCIES.get(gpu_type, None)
    
    # Energy baseline: Hardware specification maximum frequency (deployment context)
    if hardware_max_freq:
        # Find measurement closest to the hardware specification maximum frequency
        max_freq_baseline = min(aggregated_measurements, 
                               key=lambda x: abs(x['frequency'] - hardware_max_freq))
        
        # Warn if we're significantly off from the hardware specification max frequency
        freq_diff = abs(max_freq_baseline['frequency'] - hardware_max_freq)
        if freq_diff > 30:  # More than 30MHz difference is concerning
            print(f"    ⚠️ Warning: No measurement at hardware max frequency {hardware_max_freq}MHz")
            print(f"    ⚠️ Using closest available: {max_freq_baseline['frequency']}MHz (diff: {freq_diff}MHz)")
    else:
        # Fallback to highest measured frequency if GPU type unknown
        max_freq_baseline = max(aggregated_measurements, key=lambda x: x['frequency'])
        print(f"    ⚠️ Unknown GPU type '{gpu_type}', using highest measured frequency")
    
    # Performance baseline: Fastest execution (validation context)
    fastest_baseline = min(aggregated_measurements, key=lambda x: x['execution_time'])
    
    # STEP 3: Apply performance constraints
    # Use fastest execution for performance constraint validation
    baseline_time = fastest_baseline['execution_time']
    max_allowed_time = baseline_time * (1 + constraint_pct / 100)
    
    # Print baseline information for transparency
    print(f"    Baselines:")
    print(f"      Energy baseline (max freq): {max_freq_baseline['frequency']}MHz, {max_freq_baseline['execution_time']:.2f}s")
    if hardware_max_freq and max_freq_baseline['frequency'] == hardware_max_freq:
        print(f"        ✓ Using hardware specification maximum frequency ({hardware_max_freq}MHz)")
    elif hardware_max_freq:
        print(f"        ⚠️ Hardware spec max frequency: {hardware_max_freq}MHz, using: {max_freq_baseline['frequency']}MHz")
    
    if max_freq_baseline['run_count'] > 1:
        print(f"        Averaged from {max_freq_baseline['run_count']} runs (std: {max_freq_baseline['execution_time_std']:.3f}s)")
    
    print(f"      Performance baseline (fastest): {fastest_baseline['frequency']}MHz, {fastest_baseline['execution_time']:.2f}s")
    if fastest_baseline['run_count'] > 1:
        print(f"        Averaged from {fastest_baseline['run_count']} runs (std: {fastest_baseline['execution_time_std']:.3f}s)")
    
    # Check for thermal throttling or frequency scaling effects
    if max_freq_baseline['frequency'] != fastest_baseline['frequency']:
        perf_diff = ((max_freq_baseline['execution_time'] - fastest_baseline['execution_time']) / fastest_baseline['execution_time']) * 100
        print(f"      ⚠️ Max frequency baseline is {perf_diff:.1f}% slower than fastest")
        print(f"      ⚠️ This suggests thermal throttling or frequency scaling effects")
        print(f"      ⚠️ Energy savings reported vs max frequency, performance constraint vs fastest")
    
    print(f"    Performance constraint: ≤{constraint_pct}% degradation from fastest ({max_allowed_time:.2f}s)")
    
    # STEP 4: Filter valid frequencies based on performance constraints
    valid_frequencies = []
    for m in aggregated_measurements:
        if m['execution_time'] <= max_allowed_time:
            print(f"      Valid: {m['frequency']}MHz, {m['execution_time']:.2f}s, EDP: {m['edp']:.2f} ({m['run_count']} runs)")
            valid_frequencies.append(m)
        else:
            print(f"      Invalid: {m['frequency']}MHz, {m['execution_time']:.2f}s (too slow) ({m['run_count']} runs)")
    
    if not valid_frequencies:
        print(f"    ❌ No valid frequencies found within {constraint_pct}% performance constraint")
        return None
    
    # STEP 5: Find optimal frequency (minimum EDP among valid frequencies)
    optimal = min(valid_frequencies, key=lambda x: x['edp'])
    
    # STEP 6: Calculate comprehensive metrics relative to BOTH baselines
    # Energy metrics
    energy_vs_maxfreq = ((max_freq_baseline['total_energy'] - optimal['total_energy']) / max_freq_baseline['total_energy']) * 100
    energy_vs_fastest = ((fastest_baseline['total_energy'] - optimal['total_energy']) / fastest_baseline['total_energy']) * 100
    
    # Performance metrics
    perf_vs_maxfreq = ((optimal['execution_time'] - max_freq_baseline['execution_time']) / max_freq_baseline['execution_time']) * 100
    perf_vs_fastest = ((optimal['execution_time'] - fastest_baseline['execution_time']) / fastest_baseline['execution_time']) * 100
    
    # Energy-Delay Product improvements
    edp_improvement_vs_maxfreq = ((max_freq_baseline['edp'] - optimal['edp']) / max_freq_baseline['edp']) * 100
    edp_improvement_vs_fastest = ((fastest_baseline['edp'] - optimal['edp']) / fastest_baseline['edp']) * 100
    
    # Compile comprehensive results
    result = {
        # Basic identification
        'gpu': optimal['gpu'],
        'workload': optimal['workload'],
        
        # Optimal frequency information
        'optimal_frequency': optimal['frequency'],
        'optimal_time': optimal['execution_time'],
        'optimal_power': optimal['avg_power'],
        'optimal_energy': optimal['total_energy'],
        'optimal_edp': optimal['edp'],
        'optimal_run_count': optimal['run_count'],
        'optimal_timing_type': optimal['primary_timing_type'],
        'optimal_excluded_cold_runs': optimal.get('excluded_cold_runs', 0),
        'optimal_excluded_outlier_runs': optimal.get('excluded_outlier_runs', 0),
        'optimal_time_std': optimal.get('execution_time_std', 0.0),
        
        # Baseline frequency information
        'max_frequency': max_freq_baseline['frequency'],
        'hardware_max_frequency': hardware_max_freq,
        'using_hardware_max_freq': hardware_max_freq and max_freq_baseline['frequency'] == hardware_max_freq,
        'max_freq_diff': abs(max_freq_baseline['frequency'] - hardware_max_freq) if hardware_max_freq else 0,
        'fastest_frequency': fastest_baseline['frequency'],
        'baseline_differs': max_freq_baseline['frequency'] != fastest_baseline['frequency'],
        
        # Metrics vs max frequency (deployment context)
        'energy_savings_vs_maxfreq': energy_vs_maxfreq,
        'performance_change_vs_maxfreq': perf_vs_maxfreq,
        'edp_improvement_vs_maxfreq': edp_improvement_vs_maxfreq,
        'max_freq_time': max_freq_baseline['execution_time'],
        'max_freq_energy': max_freq_baseline['total_energy'],
        'max_freq_run_count': max_freq_baseline['run_count'],
        'max_freq_excluded_cold_runs': max_freq_baseline.get('excluded_cold_runs', 0),
        'max_freq_time_std': max_freq_baseline.get('execution_time_std', 0.0),
        
        # Metrics vs fastest (performance validation)
        'energy_savings_vs_fastest': energy_vs_fastest,
        'performance_change_vs_fastest': perf_vs_fastest,
        'edp_improvement_vs_fastest': edp_improvement_vs_fastest,
        'fastest_time': fastest_baseline['execution_time'],
        'fastest_energy': fastest_baseline['total_energy'],
        'fastest_run_count': fastest_baseline['run_count'],
        'fastest_excluded_cold_runs': fastest_baseline.get('excluded_cold_runs', 0),
        'fastest_time_std': fastest_baseline.get('execution_time_std', 0.0),
        
        # Analysis metadata
        'measurements_count': len(measurements),
        'aggregated_frequencies_count': len(aggregated_measurements),
        'valid_frequencies_count': len(valid_frequencies),
        'constraint_pct': constraint_pct,
        'data_source': 'measured_experimental_data_hybrid_timing',
        'timing_source': 'hybrid_inference_and_session'
    }
    
    # Print optimization results
    print(f"    ✓ Optimal: {optimal['frequency']}MHz")
    print(f"    Energy savings vs max frequency: {energy_vs_maxfreq:.1f}%")
    
    # Performance impact interpretation (vs max frequency for consistency)
    if perf_vs_maxfreq < -5:
        print(f"    ⚠️  UNEXPECTED: Lower frequency faster than max by {abs(perf_vs_maxfreq):.1f}%")
        print(f"    ⚠️  This may indicate thermal throttling or measurement artifacts")
    elif perf_vs_maxfreq < 0:
        print(f"    Performance vs max frequency: {abs(perf_vs_maxfreq):.1f}% faster")
    else:
        print(f"    Performance degradation vs max frequency: {perf_vs_maxfreq:.1f}%")
    
    # Additional statistical reporting
    if optimal['run_count'] > 1:
        print(f"    Optimal frequency averaged from {optimal['run_count']} warm runs (time std: {optimal['execution_time_std']:.3f}s)")
    if optimal.get('excluded_cold_runs', 0) > 0:
        print(f"    Excluded {optimal['excluded_cold_runs']} cold-start run(s) from optimal frequency")
    if optimal.get('excluded_outlier_runs', 0) > 0:
        print(f"    Excluded {optimal['excluded_outlier_runs']} outlier run(s) from optimal frequency")
    
    return result

# Main Analysis Functions
# ========================

def analyze_all_measured_data(constraint_pct=5.0):
    """
    Analyze all measured experimental data directories with hybrid timing approach.
    
    This is the main analysis function that:
    1. Discovers all result directories containing experimental data
    2. Processes each directory to extract measurements
    3. Groups measurements by GPU-workload combinations
    4. Applies optimization analysis to find optimal frequencies
    5. Generates comprehensive summary report
    
    Args:
        constraint_pct (float): Maximum performance degradation allowed (default: 5%)
        
    Returns:
        list: List of optimization results for all GPU-workload combinations
    """
    # Locate the sample collection scripts directory containing result folders
    script_dir = Path(__file__).parent
    results_base = script_dir / '../../sample-collection-scripts'
    
    # Find all result directories
    result_dirs = []
    for item in results_base.iterdir():
        if item.is_dir() and item.name.startswith('results_'):
            result_dirs.append(item)
    
    # Sort for consistent processing order
    result_dirs = sorted(result_dirs, key=lambda x: x.name)
    
    print(f"Found {len(result_dirs)} result directories:")
    for result_dir in result_dirs:
        print(f"  {result_dir.name}")
    
    # Process all directories to extract measurements
    all_measurements = []
    for result_dir in result_dirs:
        measurements = process_result_directory(result_dir)
        all_measurements.extend(measurements)
    
    print(f"\nTotal measurements: {len(all_measurements)}")
    
    if len(all_measurements) == 0:
        print("No measurements found!")
        return
    
    # Group measurements by GPU-workload combination
    groups = {}
    for measurement in all_measurements:
        key = f"{measurement['gpu']}_{measurement['workload']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(measurement)
    
    print(f"Found {len(groups)} GPU-workload combinations")
    
    # Analyze each group to find optimal frequency
    results = []
    for key, measurements in groups.items():
        gpu, workload = key.split('_', 1)
        print(f"\n=== Analyzing {gpu} + {workload} ===")
        
        result = calculate_optimal_frequency_hybrid(measurements, constraint_pct)
        if result:
            results.append(result)
    
    # Generate comprehensive summary report
    print_hybrid_summary(results)
    
    return results

# Reporting Functions
# ====================

def print_hybrid_summary(results):
    """
    Print comprehensive summary report with dual baseline reporting.
    
    This function generates a detailed analysis report including:
    - Individual GPU results grouped by architecture
    - Cross-GPU workload analysis
    - Statistical reliability metrics
    - Global insights and recommendations
    
    Args:
        results (list): List of optimization results from calculate_optimal_frequency_hybrid
    """
    if not results:
        return
    
    print("\n" + "="*80)
    print(" OPTIMAL FREQUENCY ANALYSIS - HYBRID TIMING WITH DUAL BASELINES")
    print("="*80)
    
    # Group results by GPU architecture for organized reporting
    gpu_groups = {}
    for result in results:
        gpu = result['gpu']
        if gpu not in gpu_groups:
            gpu_groups[gpu] = []
        gpu_groups[gpu].append(result)
    
    # Print results grouped by GPU architecture
    for gpu in sorted(gpu_groups.keys()):
        gpu_results = gpu_groups[gpu]
        print(f"\n📊 {gpu} GPU RESULTS:")
        print("-" * 50)
        
        # Sort workloads alphabetically for consistent presentation
        for result in sorted(gpu_results, key=lambda x: x['workload']):
            workload = result['workload']
            optimal_freq = result['optimal_frequency']
            energy_savings = result['energy_savings_vs_maxfreq']
            perf_change = result['performance_change_vs_maxfreq']
            edp_improvement = result['edp_improvement_vs_maxfreq']
            time_std = result['optimal_time_std']
            run_count = result['optimal_run_count']
            timing_type = result['optimal_timing_type']
            
            print(f"  🎯 {workload}:")
            print(f"    • Optimal: {optimal_freq}MHz (±{time_std:.3f}s std)")
            print(f"    • Energy savings: {energy_savings:.1f}%")
            
            # Interpret performance change with appropriate messaging
            if perf_change < -5:
                print(f"    • Performance vs max freq: ⚠️ {abs(perf_change):.1f}% FASTER (unexpected)")
            elif perf_change < 0:
                print(f"    • Performance vs max freq: {abs(perf_change):.1f}% faster")
            else:
                print(f"    • Performance vs max freq: {perf_change:.1f}% slower")
            
            print(f"    • Runs averaged: {run_count} ({timing_type} timing)")
            print(f"    • EDP improvement: {edp_improvement:.1f}%")
            
            # Highlight thermal throttling or frequency scaling issues
            if result['baseline_differs']:
                fastest_freq = result['fastest_frequency']
                max_freq = result['max_frequency']
                print(f"    • ⚠️ Max freq ({max_freq}MHz) ≠ fastest freq ({fastest_freq}MHz)")
    
    # Cross-GPU workload analysis
    print(f"\n🔬 CROSS-GPU ANALYSIS:")
    print("-" * 50)
    
    # Group results by workload for cross-GPU comparison
    workload_groups = {}
    for result in results:
        workload = result['workload']
        if workload not in workload_groups:
            workload_groups[workload] = []
        workload_groups[workload].append(result)
    
    # Analyze each workload across all GPU architectures
    for workload in sorted(workload_groups.keys()):
        workload_results = workload_groups[workload]
        avg_energy_savings = sum(r['energy_savings_vs_maxfreq'] for r in workload_results) / len(workload_results)
        avg_perf_change = sum(r['performance_change_vs_maxfreq'] for r in workload_results) / len(workload_results)
        
        print(f"\n  📈 {workload} across GPUs:")
        print(f"    • Average energy savings: {avg_energy_savings:.1f}%")
        if avg_perf_change < 0:
            print(f"    • Average performance change: {abs(avg_perf_change):.1f}% FASTER than max frequency")
        else:
            print(f"    • Average performance degradation: {avg_perf_change:.1f}%")
        
        # List optimal frequencies for each GPU
        for result in sorted(workload_results, key=lambda x: x['gpu']):
            optimal_freq = result['optimal_frequency']
            run_count = result['optimal_run_count']
            timing_type = result['optimal_timing_type']
            print(f"      - {result['gpu']}: {optimal_freq}MHz ({run_count} runs, {timing_type})")
    
    # Statistical reliability assessment
    print(f"\n📊 STATISTICAL RELIABILITY:")
    print("-" * 50)
    
    # Calculate aggregate statistics across all results
    total_measurements = sum(r['measurements_count'] for r in results)
    total_optimal_runs = sum(r['optimal_run_count'] for r in results)
    total_aggregated_freqs = sum(r['aggregated_frequencies_count'] for r in results)
    
    # Count timing source distribution
    session_configs = sum(1 for r in results if r['optimal_timing_type'] == 'session')
    inference_configs = sum(1 for r in results if r['optimal_timing_type'] == 'inference')
    baseline_differs_count = sum(1 for r in results if r['baseline_differs'])
    hardware_max_freq_count = sum(1 for r in results if r.get('using_hardware_max_freq', False))
    
    print(f"  • Total raw measurements processed: {total_measurements}")
    print(f"  • Total runs averaged for optimal frequencies: {total_optimal_runs}")
    print(f"  • Total frequency points after aggregation: {total_aggregated_freqs}")
    print(f"  • Timing sources: {session_configs} session, {inference_configs} inference")
    print(f"  • Configurations using hardware spec max frequency: {hardware_max_freq_count}/{len(results)}")
    print(f"  • Configurations with max freq ≠ fastest: {baseline_differs_count}/{len(results)}")
    
    # Report any baseline frequency issues
    # non_hardware_max = [r for r in results if not r.get('using_hardware_max_freq', False)]
    # if non_hardware_max:
    #     print(f"  • ⚠️ Configurations NOT using hardware specification max frequency:")
    #     for r in non_hardware_max:
    #         hardware_spec = r.get('hardware_max_frequency', 'unknown')
    #         actual = r['max_frequency']
    #         diff = r.get('max_freq_diff', 0)
    #         print(f"    - {r['gpu']} + {r['workload']}: using {actual}MHz (hardware spec: {hardware_spec}MHz, diff: {diff}MHz)")
    
    # Document analysis methodology
    print(f"  • Analysis method: Hybrid timing with dual baseline validation")
    print(f"  • Performance constraint: ≤{results[0]['constraint_pct']}% degradation from fastest")
    print(f"  • Energy baseline: Hardware specification maximum frequency (deployment context)")
    print(f"  • Performance baseline: Fastest execution (validation)")
    
    # Global insights and recommendations
    print(f"\n🎯 GLOBAL INSIGHTS:")
    print("-" * 50)
    
    # Calculate aggregate metrics across all configurations
    all_energy_savings = [r['energy_savings_vs_maxfreq'] for r in results]
    all_perf_changes = [r['performance_change_vs_maxfreq'] for r in results]
    all_edp_improvements = [r['edp_improvement_vs_maxfreq'] for r in results]
    
    print(f"  • Energy savings range: {min(all_energy_savings):.1f}% to {max(all_energy_savings):.1f}%")
    print(f"  • Performance impact range: {min(all_perf_changes):.1f}% to {max(all_perf_changes):.1f}%")
    print(f"  • Average energy savings: {sum(all_energy_savings)/len(all_energy_savings):.1f}%")
    
    # Overall performance impact assessment
    avg_perf = sum(all_perf_changes)/len(all_perf_changes)
    if avg_perf < 0:
        print(f"  • Average performance change: {abs(avg_perf):.1f}% FASTER than max frequency")
    else:
        print(f"  • Average performance degradation: {avg_perf:.1f}%")
    
    print(f"  • Average EDP improvement: {sum(all_edp_improvements)/len(all_edp_improvements):.1f}%")
    
    # Identify unexpected results (thermal throttling indicators)
    unexpected_count = sum(1 for perf in all_perf_changes if perf < -5)
    if unexpected_count > 0:
        print(f"  • ⚠️ Configurations with unexpected speedup: {unexpected_count}/{len(results)}")
        print(f"      (Lower frequencies faster than max by >5% - likely thermal throttling)")
    
    print(f"\n✅ V5 hybrid analysis complete!")
    print("="*80)

# Entry Point
# ============

def main():
    """
    Main entry point for the measured data analysis script.
    
    This function initiates the complete analysis pipeline for all experimental data,
    applying the hybrid timing extraction methodology and dual baseline validation
    to find optimal GPU frequencies for AI inference workloads.
    """
    analyze_all_measured_data()


if __name__ == "__main__":
    main()
