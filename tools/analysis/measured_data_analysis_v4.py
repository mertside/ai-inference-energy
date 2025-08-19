#!/usr/bin/env python3

"""
Measured Data Analysis for AI Inference Energy Project - VERSION 4
Extracts optimal GPU frequencies using measured experimental data
Uses experiment_summary.log for consistent timing data

IMPROVEMENTS OVER v3:
- Uses experiment_summary.log for consistent timing data across all runs
- Eliminates timing extraction failures from individual .out files  
- Provides more reliable data source with systematic timing recording
- Maintains statistical aggregation, cold-start exclusion, and outlier detection
"""

import os
import sys
import re
import json
import statistics
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

def extract_timing_from_experiment_summary(summary_file):
    """Extract timing information from experiment_summary.log file."""
    timing_data = {}
    
    try:
        with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Find the Run Timing Summary section
        timing_section_match = re.search(r'Run Timing Summary:\s*=+\s*\n(.*?)\n\nTiming Statistics:', content, re.DOTALL)
        if not timing_section_match:
            print(f"    No timing summary section found in {summary_file}")
            return timing_data
            
        timing_lines = timing_section_match.group(1).strip().split('\n')
        
        for line in timing_lines:
            # Parse lines like: "  Run 186_01         :  35s (freq:  825MHz, status: success)"
            match = re.match(r'\s*Run\s+(\d+)_(\d+)\s*:\s*(\d+)s\s*\(freq:\s*(\d+)MHz,\s*status:\s*(\w+)\)', line)
            if match:
                run_id = int(match.group(1))
                run_number = int(match.group(2))
                duration = int(match.group(3))
                frequency = int(match.group(4))
                status = match.group(5)
                
                if status == 'success':
                    key = f"run_{run_id}_{run_number:02d}_freq_{frequency}"
                    timing_data[key] = {
                        'run_id': run_id,
                        'run_number': run_number,
                        'duration': duration,
                        'frequency': frequency,
                        'status': status
                    }
                    print(f"    Extracted timing: {key} -> {duration}s")
                else:
                    print(f"    Skipping failed run: Run {run_id}_{run_number:02d} (status: {status})")
        
        print(f"    Successfully extracted {len(timing_data)} timing entries from experiment summary")
        return timing_data
        
    except Exception as e:
        print(f"Error reading experiment summary from {summary_file}: {e}")
        return timing_data

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
    """Process a single result directory and extract all measurements using experiment_summary.log"""
    print(f"\nProcessing: {result_dir.name}")
    
    gpu, workload, job_id = parse_directory_name(result_dir.name)
    if not gpu or not workload:
        print(f"  Skipping - cannot parse directory name")
        return []
    
    print(f"  GPU: {gpu}, Workload: {workload}")
    
    # NEW v4: Load timing data from experiment_summary.log
    experiment_summary = result_dir / 'experiment_summary.log'
    if not experiment_summary.exists():
        print(f"  Skipping - no experiment_summary.log found")
        return []
    
    timing_data = extract_timing_from_experiment_summary(experiment_summary)
    if not timing_data:
        print(f"  Skipping - no timing data extracted from experiment summary")
        return []
    
    measurements = []
    profile_files = list(result_dir.glob("*_profile.csv"))
    print(f"  Found {len(profile_files)} profile files")
    
    for profile_file in profile_files:
        print(f"    Processing: {profile_file.name}")
        
        # Extract frequency and run info from filename
        frequency = extract_frequency_from_filename(profile_file.name)
        if not frequency:
            print(f"      Skipping - no frequency found")
            continue
        
        # Extract run number from filename like run_1_01_freq_1410_profile.csv
        run_match = re.search(r'run_(\d+)_(\d+)_freq_', profile_file.name)
        if not run_match:
            print(f"      Skipping - no run number found")
            continue
        
        run_id = int(run_match.group(1))
        run_number = int(run_match.group(2))
        
        # NEW v4: Get timing from experiment summary instead of .out file
        timing_key = f"run_{run_id}_{run_number:02d}_freq_{frequency}"
        if timing_key not in timing_data:
            print(f"      Skipping - no timing data for {timing_key}")
            continue
        
        execution_time = timing_data[timing_key]['duration']
        print(f"      Frequency: {frequency} MHz, Run: {run_number}, Timing: {execution_time}s (from experiment summary)")
        
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
            'run_number': run_number,
            'execution_time': execution_time,
            'avg_power': avg_power,
            'total_energy': total_energy,
            'edp': total_energy * execution_time,
            'job_id': job_id,
            'timing_source': 'experiment_summary_log'
        }
        
        measurements.append(measurement)
        print(f"      ✓ Added: {frequency}MHz run {run_number}, {execution_time}s, {avg_power:.2f}W")
    
    print(f"  Successfully processed {len(measurements)} measurements")
    return measurements

def aggregate_measurements_by_frequency(measurements, exclude_first_run=True):
    """Group measurements by frequency and calculate statistical averages with outlier detection
    
    Args:
        measurements: List of measurement dictionaries
        exclude_first_run: If True, exclude first run at each frequency to avoid cold-start effects
    """
    if not measurements:
        return []
    
    print(f"    Aggregating measurements by frequency (exclude_first_run={exclude_first_run})...")
    
    # Group measurements by frequency
    freq_groups = {}
    for m in measurements:
        freq = m['frequency']
        if freq not in freq_groups:
            freq_groups[freq] = []
        freq_groups[freq].append(m)
    
    # Sort runs within each frequency group to identify first runs
    for freq in freq_groups:
        # Sort by run_number to get consistent ordering
        freq_groups[freq] = sorted(freq_groups[freq], key=lambda x: x.get('run_number', 0))
    
    # Calculate statistical averages for each frequency with outlier detection
    aggregated = []
    total_excluded = 0
    
    for freq, runs in freq_groups.items():
        original_count = len(runs)
        
        # First, exclude first run if requested and we have multiple runs
        if exclude_first_run and len(runs) > 1:
            excluded_run = runs[0]
            runs = runs[1:]  # Skip first run
            total_excluded += 1
            print(f"      Excluding first run at {freq}MHz (cold-start, time: {excluded_run['execution_time']}s)")
        
        if len(runs) == 0:
            print(f"      Skipping {freq}MHz - no runs left after cold-start exclusion")
            continue
        
        # ENHANCED: Outlier detection - remove runs that are >3x the median time
        if len(runs) > 1:
            times = [r['execution_time'] for r in runs]
            median_time = sorted(times)[len(times)//2]
            
            # Filter out extreme outliers (>3x median time)
            outlier_threshold = median_time * 3.0
            valid_runs = []
            outlier_runs = []
            
            for run in runs:
                if run['execution_time'] <= outlier_threshold:
                    valid_runs.append(run)
                else:
                    outlier_runs.append(run)
            
            if outlier_runs:
                outlier_times = [r['execution_time'] for r in outlier_runs]
                print(f"      Excluding {len(outlier_runs)} outlier runs at {freq}MHz (times: {outlier_times}, >3x median {median_time:.1f}s)")
                total_excluded += len(outlier_runs)
                runs = valid_runs
        
        if len(runs) == 0:
            print(f"      Skipping {freq}MHz - no runs left after outlier exclusion")
            continue
            
        if len(runs) > 1:
            print(f"      Averaging {len(runs)} valid runs at {freq}MHz (excluded {original_count - len(runs)} total)")
            
            # Calculate statistics for validation
            times = [r['execution_time'] for r in runs]
            powers = [r['avg_power'] for r in runs]
            energies = [r['total_energy'] for r in runs]
            
            # Check for outliers (basic validation)
            time_std = statistics.stdev(times) if len(times) > 1 else 0
            time_mean = statistics.mean(times)
            
            if time_std > 0.1 * time_mean:  # More than 10% variation
                print(f"        Warning: High variation in timing at {freq}MHz (std: {time_std:.2f}s)")
        else:
            print(f"      Using single valid run at {freq}MHz")
        
        # Calculate averages
        avg_measurement = {
            'gpu': runs[0]['gpu'],
            'workload': runs[0]['workload'],
            'frequency': freq,
            'execution_time': sum(r['execution_time'] for r in runs) / len(runs),
            'avg_power': sum(r['avg_power'] for r in runs) / len(runs),
            'total_energy': sum(r['total_energy'] for r in runs) / len(runs),
            'run_count': len(runs),
            'original_run_count': original_count,
            'excluded_cold_runs': 1 if exclude_first_run and original_count > 1 else 0,
            'excluded_outlier_runs': original_count - len(runs) - (1 if exclude_first_run and original_count > 1 else 0),
            'job_id': runs[0]['job_id'],  # Keep first job_id for reference
            'data_source': 'measured_experimental_data_summary_timing',
            'timing_source': 'experiment_summary_log'
        }
        
        # Add statistical measures if multiple runs
        if len(runs) > 1:
            avg_measurement['execution_time_std'] = statistics.stdev([r['execution_time'] for r in runs])
            avg_measurement['avg_power_std'] = statistics.stdev([r['avg_power'] for r in runs])
            avg_measurement['total_energy_std'] = statistics.stdev([r['total_energy'] for r in runs])
        else:
            avg_measurement['execution_time_std'] = 0.0
            avg_measurement['avg_power_std'] = 0.0
            avg_measurement['total_energy_std'] = 0.0
        
        # Recalculate EDP with averaged values
        avg_measurement['edp'] = avg_measurement['total_energy'] * avg_measurement['execution_time']
        
        aggregated.append(avg_measurement)
    
    print(f"    Aggregated {len(measurements)} individual measurements into {len(aggregated)} frequency points")
    print(f"    Total excluded runs: {total_excluded} (cold-start + outliers)")
    return aggregated

def calculate_optimal_frequency(measurements, constraint_pct=5.0):
    """Calculate optimal frequency for a given GPU-workload combination"""
    if not measurements:
        return None
    
    # FIRST: Aggregate multiple runs at same frequency for statistical reliability
    aggregated_measurements = aggregate_measurements_by_frequency(measurements)
    
    if not aggregated_measurements:
        print(f"    No aggregated measurements available")
        return None
    
    # Sort by frequency
    aggregated_measurements = sorted(aggregated_measurements, key=lambda x: x['frequency'])
    
    # Use maximum frequency as baseline (proper energy optimization reference)
    baseline = max(aggregated_measurements, key=lambda x: x['frequency'])
    baseline_time = baseline['execution_time']
    max_allowed_time = baseline_time * (1 + constraint_pct / 100)
    
    print(f"    Baseline: {baseline['frequency']}MHz (max freq), {baseline_time:.2f}s")
    if baseline['run_count'] > 1:
        print(f"    Baseline averaged from {baseline['run_count']} warm runs (std: {baseline['execution_time_std']:.3f}s)")
        if baseline.get('excluded_cold_runs', 0) > 0:
            print(f"    Baseline excluded {baseline['excluded_cold_runs']} cold-start run(s)")
    print(f"    Max allowed: {max_allowed_time:.2f}s (≤{constraint_pct}% degradation)")
    
    # Filter valid frequencies
    valid_frequencies = []
    for m in aggregated_measurements:
        if m['execution_time'] <= max_allowed_time:
            valid_frequencies.append(m)
            run_info = f" ({m['run_count']} runs)" if m['run_count'] > 1 else ""
            print(f"      Valid: {m['frequency']}MHz, {m['execution_time']:.2f}s, EDP: {m['edp']:.2f}{run_info}")
        else:
            run_info = f" ({m['run_count']} runs)" if m['run_count'] > 1 else ""
            print(f"      Invalid: {m['frequency']}MHz, {m['execution_time']:.2f}s (too slow){run_info}")
    
    if not valid_frequencies:
        print(f"    No frequencies meet constraint - all frequencies slower than allowed")
        return None
    
    # Find optimal (minimum EDP among valid frequencies)
    optimal = min(valid_frequencies, key=lambda x: x['edp'])
    
    # Calculate savings relative to MAXIMUM FREQUENCY baseline
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
        'aggregated_frequencies_count': len(aggregated_measurements),
        'valid_frequencies_count': len(valid_frequencies),
        'optimal_run_count': optimal['run_count'],
        'baseline_run_count': baseline['run_count'],
        'optimal_excluded_cold_runs': optimal.get('excluded_cold_runs', 0),
        'baseline_excluded_cold_runs': baseline.get('excluded_cold_runs', 0),
        'optimal_excluded_outlier_runs': optimal.get('excluded_outlier_runs', 0),
        'baseline_excluded_outlier_runs': baseline.get('excluded_outlier_runs', 0),
        'execution_time_std': optimal.get('execution_time_std', 0.0),
        'baseline_time_std': baseline.get('execution_time_std', 0.0),
        'data_source': 'measured_experimental_data_summary_timing',
        'timing_source': 'experiment_summary_log'
    }
    
    print(f"    ✓ Optimal: {optimal['frequency']}MHz")
    print(f"    Energy savings vs max frequency: {energy_savings:.1f}%")
    
    # Proper performance impact interpretation
    if result['performance_degradation'] < -5:
        print(f"    ⚠️  UNEXPECTED: Lower frequency faster than max by {abs(result['performance_degradation']):.1f}%")
        print(f"    ⚠️  This may indicate thermal throttling or measurement artifacts")
    elif result['performance_degradation'] < 0:
        print(f"    Performance improvement vs max frequency: {abs(result['performance_degradation']):.1f}% faster")
    else:
        print(f"    Performance degradation vs max frequency: {result['performance_degradation']:.1f}%")
    
    if optimal['run_count'] > 1:
        print(f"    Optimal frequency averaged from {optimal['run_count']} warm runs (time std: {optimal['execution_time_std']:.3f}s)")
    if optimal.get('excluded_cold_runs', 0) > 0:
        print(f"    Excluded {optimal['excluded_cold_runs']} cold-start run(s) from optimal frequency")
    if optimal.get('excluded_outlier_runs', 0) > 0:
        print(f"    Excluded {optimal['excluded_outlier_runs']} outlier run(s) from optimal frequency")
    if baseline.get('excluded_outlier_runs', 0) > 0:
        print(f"    Baseline excluded {baseline['excluded_outlier_runs']} outlier run(s) for reliability")
    
    return result

def analyze_all_measured_data(constraint_pct=5.0):
    """Analyze all measured data directories"""
    script_dir = Path(__file__).parent
    results_base = script_dir / '../../sample-collection-scripts'
    
    # Find result directories - use a more comprehensive search
    result_dirs = []
    
    # Look for all directories matching results_* pattern
    for item in results_base.iterdir():
        if item.is_dir() and item.name.startswith('results_'):
            result_dirs.append(item)
    
    # Sort for consistent processing order
    result_dirs = sorted(result_dirs, key=lambda x: x.name)
    
    print(f"Found {len(result_dirs)} result directories:")
    for result_dir in result_dirs:
        print(f"  {result_dir.name}")
    
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
    """Main analysis function using experiment_summary.log timing with max frequency baseline"""
    print("=== MEASURED DATA ANALYSIS V4 ===")
    print("Using systematic timing from experiment_summary.log files with max frequency baseline")
    print("Includes statistical aggregation, cold-start exclusion, and outlier detection\n")
    
    results = analyze_all_measured_data(constraint_pct=5.0)
    
    if not results:
        print("\nERROR: No results generated")
        return
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print(" OPTIMAL FREQUENCY ANALYSIS - SYSTEMATIC EXPERIMENT TIMING")
    print("="*80)
    
    # Group by GPU for organized display
    by_gpu = {}
    for config_name, result in results.items():
        gpu = result['gpu']
        if gpu not in by_gpu:
            by_gpu[gpu] = {}
        by_gpu[gpu][result['workload']] = result
    
    # Print per-GPU summaries
    for gpu in sorted(by_gpu.keys()):
        print(f"\n📊 {gpu} GPU RESULTS:")
        print("-" * 50)
        
        workloads = by_gpu[gpu]
        for workload in sorted(workloads.keys()):
            result = workloads[workload]
            conf_info = ""
            if result.get('optimal_run_count', 1) > 1:
                conf_info = f" (±{result.get('execution_time_std', 0):.3f}s std)"
            
            # Proper performance impact display
            perf_impact = result['performance_degradation']
            if perf_impact < -5:
                perf_display = f"⚠️ {abs(perf_impact):.1f}% FASTER (unexpected)"
            elif perf_impact < 0:
                perf_display = f"{abs(perf_impact):.1f}% faster"
            else:
                perf_display = f"{perf_impact:.1f}% slower"
            
            print(f"  🎯 {workload}:")
            print(f"    • Optimal: {result['optimal_frequency']}MHz{conf_info}")
            print(f"    • Energy savings: {result['energy_savings_pct']:.1f}%")
            print(f"    • Performance vs max freq: {perf_display}")
            print(f"    • Runs averaged: {result.get('optimal_run_count', 1)}")
            print(f"    • EDP improvement: {((result['baseline_edp'] - result['edp']) / result['baseline_edp'] * 100):.1f}%")
    
    # Cross-GPU analysis
    print(f"\n🔬 CROSS-GPU ANALYSIS:")
    print("-" * 50)
    
    workload_comparison = {}
    for result in results.values():
        workload = result['workload']
        if workload not in workload_comparison:
            workload_comparison[workload] = []
        workload_comparison[workload].append(result)
    
    for workload in sorted(workload_comparison.keys()):
        workload_results = workload_comparison[workload]
        print(f"\n  📈 {workload} across GPUs:")
        
        avg_energy_savings = sum(r['energy_savings_pct'] for r in workload_results) / len(workload_results)
        avg_perf_impact = sum(r['performance_degradation'] for r in workload_results) / len(workload_results)
        
        print(f"    • Average energy savings: {avg_energy_savings:.1f}%")
        if avg_perf_impact < 0:
            print(f"    • Average performance change: {abs(avg_perf_impact):.1f}% FASTER than max frequency")
        else:
            print(f"    • Average performance degradation: {avg_perf_impact:.1f}%")
        
        for result in sorted(workload_results, key=lambda x: x['gpu']):
            runs_info = f" ({result.get('optimal_run_count', 1)} runs)" if result.get('optimal_run_count', 1) > 1 else ""
            print(f"      - {result['gpu']}: {result['optimal_frequency']}MHz{runs_info}")
    
    # Statistical reliability summary
    print(f"\n📊 STATISTICAL RELIABILITY:")
    print("-" * 50)
    total_measurements = sum(r.get('measurements_count', 0) for r in results.values())
    total_runs = sum(r.get('optimal_run_count', 1) for r in results.values())
    total_aggregated = sum(r.get('aggregated_frequencies_count', 0) for r in results.values())
    total_outliers_excluded = sum(r.get('optimal_excluded_outlier_runs', 0) + r.get('baseline_excluded_outlier_runs', 0) for r in results.values())
    
    print(f"  • Total raw measurements processed: {total_measurements}")
    print(f"  • Total runs averaged for optimal frequencies: {total_runs}")
    print(f"  • Total frequency points after aggregation: {total_aggregated}")
    print(f"  • Total outlier runs excluded: {total_outliers_excluded}")
    print(f"  • Analysis method: Statistical aggregation with outlier detection")
    print(f"  • Cold-start exclusion: First run at each frequency excluded")
    print(f"  • Outlier detection: Runs >3x median time excluded")
    print(f"  • Performance constraint: ≤5% degradation")
    print(f"  • Timing source: Systematic experiment_summary.log files")
    print(f"  • Baseline: Maximum frequency (proper energy optimization reference)")
    print(f"  • Data source: 100% measured experimental data")
    
    # Global statistics
    print(f"\n🎯 GLOBAL INSIGHTS:")
    print("-" * 50)
    
    all_energy_savings = [r['energy_savings_pct'] for r in results.values()]
    all_perf_impacts = [r['performance_degradation'] for r in results.values()]
    
    print(f"  • Energy savings range: {min(all_energy_savings):.1f}% to {max(all_energy_savings):.1f}%")
    print(f"  • Performance impact range: {min(all_perf_impacts):.1f}% to {max(all_perf_impacts):.1f}%")
    print(f"  • Average energy savings: {sum(all_energy_savings)/len(all_energy_savings):.1f}%")
    
    avg_perf = sum(all_perf_impacts)/len(all_perf_impacts)
    if avg_perf < 0:
        print(f"  • Average performance change: {abs(avg_perf):.1f}% FASTER than max frequency")
        print(f"  • ⚠️  Note: Faster performance at lower frequencies may indicate: ")
        print(f"              1. Thermal throttling at maximum frequency")
        print(f"              2. Memory bandwidth limitations")
        print(f"              3. Power delivery constraints")
        print(f"              4. Measurement artifacts in the workload")
    else:
        print(f"  • Average performance degradation: {avg_perf:.1f}%")
    
    print(f"\n✅ V4 analysis complete with systematic experiment timing!")
    print("="*80)

if __name__ == "__main__":
    main()
