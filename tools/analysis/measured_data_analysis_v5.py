#!/usr/bin/env python3

"""
Measured Data Analysis for AI Inference Energy Project - VERSION 5
Extracts optimal GPU frequencies using measured experimental data
Hybrid timing extraction with dual baseline validation

IMPROVEMENTS OVER v4:
- Hybrid timing extraction: Prioritizes inference time from .out files, falls back to experiment_summary.log
- Dual baseline reporting: Performance validation vs fastest, energy savings vs max frequency
- Enhanced timing validation and consistency checking
- Improved data quality reporting and transparency
- Better handling of thermal throttling and measurement artifacts
"""

import re
import csv
import statistics
from pathlib import Path
from collections import defaultdict

def parse_directory_name(dir_name):
    """Parse directory name like results_a100_llama_job_20892442"""
    match = re.match(r'results_([^_]+)_([^_]+)_job_(\d+)', dir_name)
    if match:
        return match.group(1).upper(), match.group(2), match.group(3)
    return None, None, None

def extract_inference_timing_from_out(out_file):
    """Extract inference timing from .out file using comprehensive patterns"""
    if not out_file.exists():
        return None, "file_not_found"
    
    try:
        with open(out_file, 'r') as f:
            content = f.read()
        
        # Comprehensive timing patterns (ordered by preference)
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
                    time_value = float(matches[-1])  # Use last match (often the final/summary timing)
                    if time_value > 0:
                        return time_value, pattern_name
                except ValueError:
                    continue
        
        return None, "no_timing_pattern_found"
        
    except Exception as e:
        return None, f"extraction_error_{str(e)}"

def extract_timing_from_experiment_summary(summary_file):
    """Extract timing information from experiment_summary.log file."""
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
    """Extract timing with hybrid approach: inference time preferred, session time fallback"""
    
    # 1. PRIMARY: Try to get inference time from .out file
    out_file = result_dir / f"run_{run_id}_{run_number:02d}_freq_{frequency}_app.out"
    inference_time, inference_status = extract_inference_timing_from_out(out_file)
    
    # 2. FALLBACK: Get session time from experiment_summary.log
    summary_file = result_dir / 'experiment_summary.log'
    summary_data = extract_timing_from_experiment_summary(summary_file)
    run_key = f"{run_id}_{run_number:02d}"
    session_time = summary_data.get(run_key, {}).get('time', None)
    
    # 3. Validation and selection
    timing_info = {
        'inference_time': inference_time,
        'inference_status': inference_status,
        'session_time': session_time,
        'selected_time': None,
        'timing_type': None,
        'overhead_ratio': None,
        'validation_status': 'unknown'
    }
    
    # Use inference time if available and reasonable
    if inference_time and inference_time > 0:
        timing_info['selected_time'] = inference_time
        timing_info['timing_type'] = 'inference'
        
        if session_time and session_time > 0:
            if inference_time <= session_time:
                overhead = session_time - inference_time
                timing_info['overhead_ratio'] = overhead / session_time
                timing_info['validation_status'] = 'validated'
                
                if timing_info['overhead_ratio'] > 0.7:  # >70% overhead seems excessive
                    timing_info['validation_status'] = 'high_overhead_warning'
            else:
                timing_info['validation_status'] = 'inconsistent_timing'
        else:
            timing_info['validation_status'] = 'inference_only'
    
    # Fallback to session time
    elif session_time and session_time > 0:
        timing_info['selected_time'] = session_time
        timing_info['timing_type'] = 'session'
        timing_info['validation_status'] = 'session_fallback'
    
    else:
        timing_info['validation_status'] = 'no_timing_found'
    
    return timing_info

def extract_power_from_csv(csv_file):
    """Extract average power and total energy from DCGMI profile CSV"""
    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            return None, 0
        
        # DCGMI format:
        # Line 0: #Entity   NVIDX   DVNAM   POWER   PMLMT   ...
        # Line 1: ID                        W       W       ...  (units)
        # Line 2+: GPU 0    0       NVIDIA  34.535  250.000 ...  (data)
        
        # Parse header from line 0 (remove # prefix)
        header_line = lines[0].strip()
        if header_line.startswith('#'):
            header_line = header_line[1:]
        
        # The fields are separated by multiple spaces, use fixed-width parsing
        # Looking at the format, POWER appears to be around position ~50-60
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
                        # Power should be reasonable (10-500W typically)
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
    """Extract frequency from filename like run_1_01_freq_1410_profile.csv"""
    match = re.search(r'freq_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def process_result_directory(result_dir):
    """Process a single result directory and extract all measurements using hybrid timing"""
    print(f"\nProcessing: {result_dir.name}")
    
    gpu, workload, job_id = parse_directory_name(result_dir.name)
    if not gpu or not workload:
        print(f"  ⚠️ Could not parse directory name: {result_dir.name}")
        return []
    
    print(f"  GPU: {gpu}, Workload: {workload}")
    
    measurements = []
    profile_files = list(result_dir.glob("*_profile.csv"))
    print(f"  Found {len(profile_files)} profile files")
    
    timing_stats = {
        'inference': 0,
        'session': 0,
        'failed': 0,
        'high_overhead': 0,
        'inconsistent': 0
    }
    
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
        
        if not timing_info['selected_time'] or timing_info['selected_time'] <= 0:
            timing_stats['failed'] += 1
            continue
        
        # Update timing statistics
        if timing_info['timing_type'] == 'inference':
            timing_stats['inference'] += 1
        elif timing_info['timing_type'] == 'session':
            timing_stats['session'] += 1
        
        if timing_info['validation_status'] == 'high_overhead_warning':
            timing_stats['high_overhead'] += 1
        elif timing_info['validation_status'] == 'inconsistent_timing':
            timing_stats['inconsistent'] += 1
        
        # Extract power data
        avg_power, power_samples = extract_power_from_csv(profile_file)
        if not avg_power:
            continue
        
        # Calculate energy
        execution_time = timing_info['selected_time']
        total_energy = avg_power * execution_time  # Watts * seconds = Joules
        edp = total_energy * execution_time  # Energy-Delay Product
        
        measurement = {
            'gpu': gpu,
            'workload': workload,
            'frequency': frequency,
            'run_id': run_id,
            'run_number': run_number,
            'execution_time': execution_time,
            'avg_power': avg_power,
            'total_energy': total_energy,
            'edp': edp,
            'power_samples': power_samples,
            'timing_type': timing_info['timing_type'],
            'timing_validation': timing_info['validation_status'],
            'inference_time': timing_info['inference_time'],
            'session_time': timing_info['session_time'],
            'overhead_ratio': timing_info['overhead_ratio']
        }
        
        measurements.append(measurement)
    
    print(f"  Successfully processed {len(measurements)} measurements")
    print(f"  Timing sources: {timing_stats['inference']} inference, {timing_stats['session']} session")
    if timing_stats['failed'] > 0:
        print(f"  ⚠️ Failed timing extraction: {timing_stats['failed']} runs")
    if timing_stats['high_overhead'] > 0:
        print(f"  ⚠️ High overhead detected: {timing_stats['high_overhead']} runs")
    if timing_stats['inconsistent'] > 0:
        print(f"  ⚠️ Inconsistent timing: {timing_stats['inconsistent']} runs")
    
    return measurements

def aggregate_measurements_by_frequency(measurements, exclude_first_run=True):
    """Group measurements by frequency and calculate statistical averages with outlier detection"""
    if not measurements:
        return []
    
    print(f"    Aggregating measurements by frequency (exclude_first_run={exclude_first_run})...")
    
    # Group measurements by frequency
    freq_groups = defaultdict(list)
    for m in measurements:
        freq_groups[m['frequency']].append(m)
    
    # Sort runs within each frequency group to identify first runs
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
        
        # Exclude first run if requested (cold-start)
        if exclude_first_run and len(runs) > 1:
            excluded_cold_runs = 1
            valid_runs = runs[1:]
            print(f"      Excluding first run at {freq}MHz (cold-start, time: {runs[0]['execution_time']:.0f}s)")
        else:
            valid_runs = runs
        
        if len(valid_runs) == 0:
            continue
        
        # Outlier detection: exclude runs >3x median time
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
        
        # Calculate averages
        avg_time = statistics.mean([r['execution_time'] for r in valid_runs])
        avg_power = statistics.mean([r['avg_power'] for r in valid_runs])
        avg_energy = statistics.mean([r['total_energy'] for r in valid_runs])
        avg_edp = statistics.mean([r['edp'] for r in valid_runs])
        
        # Calculate standard deviations
        if len(valid_runs) > 1:
            time_std = statistics.stdev([r['execution_time'] for r in valid_runs])
            if time_std > avg_time * 0.15:  # >15% variation
                print(f"        Warning: High variation in timing at {freq}MHz (std: {time_std:.2f}s)")
        else:
            time_std = 0.0
        
        print(f"      Averaging {len(valid_runs)} valid runs at {freq}MHz (excluded {excluded_cold_runs + excluded_outlier_runs} total)")
        
        # Count timing types
        inference_count = sum(1 for r in valid_runs if r['timing_type'] == 'inference')
        session_count = sum(1 for r in valid_runs if r['timing_type'] == 'session')
        
        aggregated_measurement = {
            'gpu': valid_runs[0]['gpu'],
            'workload': valid_runs[0]['workload'],
            'frequency': freq,
            'execution_time': avg_time,
            'avg_power': avg_power,
            'total_energy': avg_energy,
            'edp': avg_edp,
            'run_count': len(valid_runs),
            'excluded_cold_runs': excluded_cold_runs,
            'excluded_outlier_runs': excluded_outlier_runs,
            'execution_time_std': time_std,
            'inference_timing_count': inference_count,
            'session_timing_count': session_count,
            'primary_timing_type': 'inference' if inference_count > session_count else 'session'
        }
        
        aggregated.append(aggregated_measurement)
        total_excluded += excluded_cold_runs + excluded_outlier_runs
    
    print(f"    Aggregated {len(measurements)} individual measurements into {len(aggregated)} frequency points")
    print(f"    Total excluded runs: {total_excluded} (cold-start + outliers)")
    return aggregated

def calculate_optimal_frequency_hybrid(measurements, constraint_pct=5.0):
    """Calculate optimal frequency with dual baseline validation"""
    if not measurements:
        return None
    
    # FIRST: Aggregate multiple runs at same frequency for statistical reliability
    aggregated_measurements = aggregate_measurements_by_frequency(measurements)
    
    if not aggregated_measurements:
        return None
    
    # Sort by frequency
    aggregated_measurements = sorted(aggregated_measurements, key=lambda x: x['frequency'])
    
    # Identify both baselines
    max_freq_baseline = max(aggregated_measurements, key=lambda x: x['frequency'])
    fastest_baseline = min(aggregated_measurements, key=lambda x: x['execution_time'])
    
    # Use fastest execution for performance constraint validation
    baseline_time = fastest_baseline['execution_time']
    max_allowed_time = baseline_time * (1 + constraint_pct / 100)
    
    print(f"    Baselines:")
    print(f"      Max frequency: {max_freq_baseline['frequency']}MHz, {max_freq_baseline['execution_time']:.2f}s")
    if max_freq_baseline['run_count'] > 1:
        print(f"        Averaged from {max_freq_baseline['run_count']} runs (std: {max_freq_baseline['execution_time_std']:.3f}s)")
    
    print(f"      Fastest execution: {fastest_baseline['frequency']}MHz, {fastest_baseline['execution_time']:.2f}s")
    if fastest_baseline['run_count'] > 1:
        print(f"        Averaged from {fastest_baseline['run_count']} runs (std: {fastest_baseline['execution_time_std']:.3f}s)")
    
    if max_freq_baseline['frequency'] != fastest_baseline['frequency']:
        perf_diff = ((max_freq_baseline['execution_time'] - fastest_baseline['execution_time']) / fastest_baseline['execution_time']) * 100
        print(f"      ⚠️ Max frequency is {perf_diff:.1f}% slower than fastest (possible throttling)")
    
    print(f"    Performance constraint: ≤{constraint_pct}% degradation from fastest ({max_allowed_time:.2f}s)")
    
    # Filter valid frequencies based on fastest execution baseline
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
    
    # Find optimal (minimum EDP among valid frequencies)
    optimal = min(valid_frequencies, key=lambda x: x['edp'])
    
    # Calculate metrics relative to BOTH baselines
    energy_vs_maxfreq = ((max_freq_baseline['total_energy'] - optimal['total_energy']) / max_freq_baseline['total_energy']) * 100
    energy_vs_fastest = ((fastest_baseline['total_energy'] - optimal['total_energy']) / fastest_baseline['total_energy']) * 100
    
    perf_vs_maxfreq = ((optimal['execution_time'] - max_freq_baseline['execution_time']) / max_freq_baseline['execution_time']) * 100
    perf_vs_fastest = ((optimal['execution_time'] - fastest_baseline['execution_time']) / fastest_baseline['execution_time']) * 100
    
    edp_improvement_vs_maxfreq = ((max_freq_baseline['edp'] - optimal['edp']) / max_freq_baseline['edp']) * 100
    edp_improvement_vs_fastest = ((fastest_baseline['edp'] - optimal['edp']) / fastest_baseline['edp']) * 100
    
    result = {
        'gpu': optimal['gpu'],
        'workload': optimal['workload'],
        
        # Optimal frequency info
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
        
        # Baseline frequencies
        'max_frequency': max_freq_baseline['frequency'],
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
    
    # Additional reporting
    if optimal['run_count'] > 1:
        print(f"    Optimal frequency averaged from {optimal['run_count']} warm runs (time std: {optimal['execution_time_std']:.3f}s)")
    if optimal.get('excluded_cold_runs', 0) > 0:
        print(f"    Excluded {optimal['excluded_cold_runs']} cold-start run(s) from optimal frequency")
    if optimal.get('excluded_outlier_runs', 0) > 0:
        print(f"    Excluded {optimal['excluded_outlier_runs']} outlier run(s) from optimal frequency")
    
    return result

def analyze_all_measured_data(constraint_pct=5.0):
    """Analyze all measured data directories with hybrid timing approach"""
    script_dir = Path(__file__).parent
    results_base = script_dir / '../../sample-collection-scripts'
    
    # Find result directories
    result_dirs = []
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
        print("No measurements found!")
        return
    
    # Group by GPU and workload
    groups = {}
    for measurement in all_measurements:
        key = f"{measurement['gpu']}_{measurement['workload']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(measurement)
    
    print(f"Found {len(groups)} GPU-workload combinations")
    
    # Analyze each group
    results = []
    for key, measurements in groups.items():
        gpu, workload = key.split('_', 1)
        print(f"\n=== Analyzing {gpu} + {workload} ===")
        
        result = calculate_optimal_frequency_hybrid(measurements, constraint_pct)
        if result:
            results.append(result)
    
    # Print comprehensive summary
    print_hybrid_summary(results)
    
    return results

def print_hybrid_summary(results):
    """Print comprehensive summary with dual baseline reporting"""
    if not results:
        return
    
    print("\n" + "="*80)
    print(" OPTIMAL FREQUENCY ANALYSIS - HYBRID TIMING WITH DUAL BASELINES")
    print("="*80)
    
    # Group by GPU
    gpu_groups = {}
    for result in results:
        gpu = result['gpu']
        if gpu not in gpu_groups:
            gpu_groups[gpu] = []
        gpu_groups[gpu].append(result)
    
    for gpu in sorted(gpu_groups.keys()):
        gpu_results = gpu_groups[gpu]
        print(f"\n📊 {gpu} GPU RESULTS:")
        print("-" * 50)
        
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
            
            if perf_change < -5:
                print(f"    • Performance vs max freq: ⚠️ {abs(perf_change):.1f}% FASTER (unexpected)")
            elif perf_change < 0:
                print(f"    • Performance vs max freq: {abs(perf_change):.1f}% faster")
            else:
                print(f"    • Performance vs max freq: {perf_change:.1f}% slower")
            
            print(f"    • Runs averaged: {run_count} ({timing_type} timing)")
            print(f"    • EDP improvement: {edp_improvement:.1f}%")
            
            if result['baseline_differs']:
                fastest_freq = result['fastest_frequency']
                max_freq = result['max_frequency']
                print(f"    • ⚠️ Max freq ({max_freq}MHz) ≠ fastest freq ({fastest_freq}MHz)")
    
    # Cross-GPU analysis
    print(f"\n🔬 CROSS-GPU ANALYSIS:")
    print("-" * 50)
    
    workload_groups = {}
    for result in results:
        workload = result['workload']
        if workload not in workload_groups:
            workload_groups[workload] = []
        workload_groups[workload].append(result)
    
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
        
        for result in sorted(workload_results, key=lambda x: x['gpu']):
            optimal_freq = result['optimal_frequency']
            run_count = result['optimal_run_count']
            timing_type = result['optimal_timing_type']
            print(f"      - {result['gpu']}: {optimal_freq}MHz ({run_count} runs, {timing_type})")
    
    # Statistical reliability
    print(f"\n📊 STATISTICAL RELIABILITY:")
    print("-" * 50)
    total_measurements = sum(r['measurements_count'] for r in results)
    total_optimal_runs = sum(r['optimal_run_count'] for r in results)
    total_aggregated_freqs = sum(r['aggregated_frequencies_count'] for r in results)
    
    inference_configs = sum(1 for r in results if r['optimal_timing_type'] == 'inference')
    session_configs = sum(1 for r in results if r['optimal_timing_type'] == 'session')
    baseline_differs_count = sum(1 for r in results if r['baseline_differs'])
    
    print(f"  • Total raw measurements processed: {total_measurements}")
    print(f"  • Total runs averaged for optimal frequencies: {total_optimal_runs}")
    print(f"  • Total frequency points after aggregation: {total_aggregated_freqs}")
    print(f"  • Timing sources: {inference_configs} inference, {session_configs} session")
    print(f"  • Configurations with max freq ≠ fastest: {baseline_differs_count}/{len(results)}")
    print(f"  • Analysis method: Hybrid timing with dual baseline validation")
    print(f"  • Performance constraint: ≤{results[0]['constraint_pct']}% degradation from fastest")
    print(f"  • Energy baseline: Maximum frequency (deployment context)")
    print(f"  • Performance baseline: Fastest execution (validation)")
    
    # Global insights
    print(f"\n🎯 GLOBAL INSIGHTS:")
    print("-" * 50)
    all_energy_savings = [r['energy_savings_vs_maxfreq'] for r in results]
    all_perf_changes = [r['performance_change_vs_maxfreq'] for r in results]
    all_edp_improvements = [r['edp_improvement_vs_maxfreq'] for r in results]
    
    print(f"  • Energy savings range: {min(all_energy_savings):.1f}% to {max(all_energy_savings):.1f}%")
    print(f"  • Performance impact range: {min(all_perf_changes):.1f}% to {max(all_perf_changes):.1f}%")
    print(f"  • Average energy savings: {sum(all_energy_savings)/len(all_energy_savings):.1f}%")
    
    avg_perf = sum(all_perf_changes)/len(all_perf_changes)
    if avg_perf < 0:
        print(f"  • Average performance change: {abs(avg_perf):.1f}% FASTER than max frequency")
    else:
        print(f"  • Average performance degradation: {avg_perf:.1f}%")
    
    print(f"  • Average EDP improvement: {sum(all_edp_improvements)/len(all_edp_improvements):.1f}%")
    
    unexpected_count = sum(1 for perf in all_perf_changes if perf < -5)
    if unexpected_count > 0:
        print(f"  • ⚠️ Configurations with unexpected speedup: {unexpected_count}/{len(results)}")
        print(f"      (Lower frequencies faster than max by >5% - likely thermal throttling)")
    
    print(f"\n✅ V5 hybrid analysis complete!")
    print("="*80)

def main():
    analyze_all_measured_data()

if __name__ == "__main__":
    main()
