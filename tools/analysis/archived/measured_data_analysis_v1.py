#!/usr/bin/env python3

"""
Measured Data Analysis for AI Inference Energy Project
Extracts optimal GPU frequencies using measured experimental data
No estimates - only processes actual measurements from profiling runs
"""

import os
import sys
import re
import json
from pathlib import Path

def parse_directory_name(dir_name):
    """Parse directory name like results_a100_llama_job_20892442"""
    match = re.match(r'results_([^_]+)_([^_]+)_job_(\d+)', dir_name)
    if match:
        gpu = match.group(1).upper()
        workload = match.group(2)
        job_id = match.group(3)
        return gpu, workload, job_id
    return None, None, None

def extract_timing_from_out_file(file_path):
    """Extract timing information from .out file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Look for timing patterns - Updated for actual log format
            patterns = [
                r'Total Inference Time:\s*(\d+\.?\d*)\s*s',  # Primary pattern from logs
                r'Command completed in\s*(\d+\.?\d*)\s*s',   # Alternative from profiler
                r'Application duration:\s*(\d+\.?\d*)\s*s',  # Another alternative
                r'Inference Time:\s*(\d+\.?\d*)\s*s',
                r'Execution Time:\s*(\d+\.?\d*)\s*s',
                r'Duration:\s*(\d+\.?\d*)\s*s',
                r'Total Time:\s*(\d+\.?\d*)\s*s'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    timing = float(match.group(1))
                    print(f"    Found timing: {timing}s")
                    return timing
                    
        print(f"    No timing pattern found in {file_path}")
        return None
    except Exception as e:
        print(f"Error reading timing from {file_path}: {e}")
        return None

def extract_power_from_csv(csv_file):
    """Extract average power and total energy from DCGMI profile CSV"""
    try:
        power_values = []
        
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            
        # Find header line with POWER column - handle different formats
        power_col_idx = None
        data_start_line = None
        
        for i, line in enumerate(lines):
            # Look for header lines (both A100 and V100 formats)
            if ('Entity' in line and 'POWER' in line) or (line.startswith('#Entity') and 'POWER' in line):
                header_parts = line.strip().replace('#', '').split()
                for j, part in enumerate(header_parts):
                    if part == 'POWER':
                        power_col_idx = j
                        # Data starts after header line and units line
                        data_start_line = i + 2
                        break
                if power_col_idx is not None:
                    break
        
        if power_col_idx is None:
            print(f"    No POWER column found")
            return None, None
        
        print(f"    Found POWER column at header index {power_col_idx}")
        
        # Extract power values from data lines
        # For data lines, we need to find the power value by position in the split parts
        # Look at the first data line to determine actual power column index
        first_data_line = None
        for line in lines[data_start_line:]:
            if not line.startswith('#') and line.strip():
                first_data_line = line
                break
        
        if not first_data_line:
            print(f"    No data lines found")
            return None, None
        
        # Parse first data line to find power value position
        parts = first_data_line.strip().split()
        actual_power_idx = None
        
        # Look for a reasonable power value (20-500W typically for GPUs)
        for idx, part in enumerate(parts):
            try:
                val = float(part)
                if 20.0 <= val <= 500.0:  # Reasonable power range
                    actual_power_idx = idx
                    print(f"    Found power value {val}W at data index {idx}")
                    break
            except ValueError:
                continue
        
        if actual_power_idx is None:
            print(f"    Could not locate power values in data")
            return None, None
        
        # Extract all power values using the found index
        for line in lines[data_start_line:]:
            # Skip comment lines and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split()
            if len(parts) > actual_power_idx:
                try:
                    power = float(parts[actual_power_idx])
                    if power > 0:  # Only valid positive power readings
                        power_values.append(power)
                except (ValueError, IndexError):
                    continue
        
        if len(power_values) == 0:
            print(f"    No valid power readings found")
            return None, None
            
        avg_power = sum(power_values) / len(power_values)
        
        # Calculate energy: power * time * sampling_interval
        sampling_interval_s = 0.05  # 50ms from logs
        total_energy_j = avg_power * len(power_values) * sampling_interval_s
        
        print(f"    Power: {avg_power:.2f}W, Energy: {total_energy_j:.2f}J ({len(power_values)} samples)")
        return avg_power, total_energy_j
        
    except Exception as e:
        print(f"Error reading power: {e}")
        return None, None

def extract_frequency_from_filename(filename):
    """Extract frequency from filename like run_1_01_freq_1410_profile.csv"""
    match = re.search(r'freq_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def process_result_directory(result_dir):
    """Process a single result directory and extract all measurements"""
    print(f"\nProcessing: {result_dir.name}")
    
    gpu, workload, job_id = parse_directory_name(result_dir.name)
    if not gpu or not workload:
        print(f"  Skipping - cannot parse directory name")
        return []
    
    print(f"  GPU: {gpu}, Workload: {workload}")
    
    measurements = []
    profile_files = list(result_dir.glob("*_profile.csv"))
    print(f"  Found {len(profile_files)} profile files")
    
    for profile_file in profile_files:
        print(f"    Processing: {profile_file.name}")
        
        # Extract frequency
        frequency = extract_frequency_from_filename(profile_file.name)
        if not frequency:
            print(f"      Skipping - no frequency found")
            continue
        
        print(f"      Frequency: {frequency} MHz")
        
        # Find corresponding .out file
        base_name = profile_file.stem.replace('_profile', '_app')
        out_file = profile_file.parent / f"{base_name}.out"
        
        if not out_file.exists():
            print(f"      Skipping - no .out file")
            continue
        
        # Extract timing
        execution_time = extract_timing_from_out_file(out_file)
        if not execution_time:
            print(f"      Skipping - no timing data")
            continue
        
        # Extract power
        avg_power, total_energy = extract_power_from_csv(profile_file)
        if not avg_power or not total_energy:
            print(f"      Skipping - no power data")
            continue
        
        # Store measurement
        measurement = {
            'gpu': gpu,
            'workload': workload,
            'frequency': frequency,
            'execution_time': execution_time,
            'avg_power': avg_power,
            'total_energy': total_energy,
            'edp': total_energy * execution_time,
            'job_id': job_id
        }
        
        measurements.append(measurement)
        print(f"      ✓ Added: {frequency}MHz, {execution_time:.2f}s, {avg_power:.2f}W")
    
    print(f"  Successfully processed {len(measurements)} measurements")
    return measurements

def calculate_optimal_frequency(measurements, constraint_pct=5.0):
    """Calculate optimal frequency for a given GPU-workload combination"""
    if not measurements:
        return None
    
    # Sort by frequency
    measurements = sorted(measurements, key=lambda x: x['frequency'])
    
    # Find baseline (best performance - minimum execution time)
    baseline = min(measurements, key=lambda x: x['execution_time'])
    baseline_time = baseline['execution_time']
    max_allowed_time = baseline_time * (1 + constraint_pct / 100)
    
    print(f"    Baseline: {baseline['frequency']}MHz, {baseline_time:.2f}s")
    print(f"    Max allowed: {max_allowed_time:.2f}s")
    
    # Filter valid frequencies
    valid_frequencies = []
    for m in measurements:
        if m['execution_time'] <= max_allowed_time:
            valid_frequencies.append(m)
            print(f"      Valid: {m['frequency']}MHz, {m['execution_time']:.2f}s, EDP: {m['edp']:.2f}")
        else:
            print(f"      Invalid: {m['frequency']}MHz, {m['execution_time']:.2f}s (too slow)")
    
    if not valid_frequencies:
        print(f"    No frequencies meet constraint")
        return None
    
    # Find optimal (minimum EDP)
    optimal = min(valid_frequencies, key=lambda x: x['edp'])
    
    # Calculate savings
    energy_savings = ((baseline['total_energy'] - optimal['total_energy']) / baseline['total_energy']) * 100
    
    result = {
        'gpu': optimal['gpu'],
        'workload': optimal['workload'],
        'optimal_frequency': optimal['frequency'],
        'baseline_frequency': baseline['frequency'],
        'execution_time': optimal['execution_time'],
        'baseline_time': baseline_time,
        'performance_degradation': ((optimal['execution_time'] - baseline_time) / baseline_time) * 100,
        'avg_power': optimal['avg_power'],
        'baseline_power': baseline['avg_power'],
        'total_energy': optimal['total_energy'],
        'baseline_energy': baseline['total_energy'],
        'energy_savings_pct': energy_savings,
        'edp': optimal['edp'],
        'baseline_edp': baseline['edp'],
        'measurements_count': len(measurements),
        'valid_frequencies_count': len(valid_frequencies),
        'data_source': 'measured_experimental_data'
    }
    
    print(f"    ✓ Optimal: {optimal['frequency']}MHz")
    print(f"    Energy savings: {energy_savings:.1f}%")
    print(f"    Performance impact: {result['performance_degradation']:.1f}%")
    
    return result

def analyze_all_measured_data(constraint_pct=5.0):
    """Analyze all measured data directories"""
    script_dir = Path(__file__).parent
    results_base = script_dir / '../../sample-collection-scripts'
    
    # Find result directories
    result_dirs = [p for p in results_base.iterdir() if p.is_dir() and p.name.startswith('results_')]
    print(f"Found {len(result_dirs)} result directories")
    
    # Process all directories
    all_measurements = []
    for result_dir in result_dirs:
        measurements = process_result_directory(result_dir)
        all_measurements.extend(measurements)
    
    print(f"\nTotal measurements: {len(all_measurements)}")
    
    if len(all_measurements) == 0:
        print("ERROR: No measurements found")
        return None
    
    # Group by GPU and workload
    groups = {}
    for measurement in all_measurements:
        key = (measurement['gpu'], measurement['workload'])
        if key not in groups:
            groups[key] = []
        groups[key].append(measurement)
    
    print(f"GPU-workload combinations: {len(groups)}")
    for key, measurements in groups.items():
        print(f"  {key[0]} + {key[1]}: {len(measurements)} measurements")
    
    # Calculate optimal frequencies
    results = {}
    for (gpu, workload), measurements in groups.items():
        print(f"\n=== Analyzing {gpu} + {workload} ===")
        optimal = calculate_optimal_frequency(measurements, constraint_pct)
        if optimal:
            results[f"{gpu}_{workload}"] = optimal
    
    return results

def main():
    """Main analysis function"""
    print("=== Measured Data Analysis ===")
    print("Using measured experimental data\n")
    
    results = analyze_all_measured_data(constraint_pct=10.0)
    
    if not results:
        print("\nERROR: No results generated")
        return
    
    print(f"\n=== Analysis Complete ===")
    print(f"Analyzed {len(results)} configurations")
    
    # Print summary
    print("\n=== Summary ===")
    for config_name, result in results.items():
        print(f"{result['gpu']} + {result['workload']}: {result['optimal_frequency']}MHz "
              f"({result['energy_savings_pct']:.1f}% energy savings, "
              f"{result['performance_degradation']:.1f}% performance impact)")

if __name__ == "__main__":
    main()
