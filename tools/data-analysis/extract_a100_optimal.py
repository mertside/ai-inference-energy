#!/usr/bin/env python
"""
Extract real optimal frequencies for A100 using our experimental data
Following the same methodology as H100 analysis
"""

import csv
import os
from collections import defaultdict

def load_and_clean_csv_data(filename):
    """Load CSV data and remove duplicates"""
    data = []
    seen = set()
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create unique key
            key = (row['gpu'], row['workload'], row['frequency_mhz'], row['run_id'])
            
            if key in seen:
                continue  # Skip duplicate
            seen.add(key)
            
            # Convert numeric fields
            row['frequency_mhz'] = float(row['frequency_mhz'])
            row['duration_seconds'] = float(row['duration_seconds'])
            row['avg_power_watts'] = float(row['avg_power_watts'])
            row['total_energy_joules'] = float(row['total_energy_joules'])
            
            # Calculate EDP
            row['edp'] = row['total_energy_joules'] * row['duration_seconds']
            data.append(row)
    
    return data

def find_optimal_frequencies_for_gpu(data, target_gpu='A100'):
    """Find optimal frequencies for specific GPU"""
    
    # Filter data for target GPU only
    gpu_data = [row for row in data if row['gpu'] == target_gpu]
    
    if not gpu_data:
        print(f"No data found for {target_gpu}")
        return {}
    
    # Group by workload
    workload_groups = defaultdict(list)
    for row in gpu_data:
        workload_groups[row['workload']].append(row)
    
    optimal_results = {}
    
    print(f"REAL DATA OPTIMAL FREQUENCY ANALYSIS FOR {target_gpu}")
    print("=" * 60)
    
    for workload, workload_data in workload_groups.items():
        print(f"\nWorkload: {workload}")
        print("-" * 30)
        
        # Group by frequency and average metrics
        freq_groups = defaultdict(list)
        for row in workload_data:
            freq_groups[row['frequency_mhz']].append(row)
        
        freq_averages = {}
        for freq, freq_data in freq_groups.items():
            avg_duration = sum(row['duration_seconds'] for row in freq_data) / len(freq_data)
            avg_energy = sum(row['total_energy_joules'] for row in freq_data) / len(freq_data)
            avg_edp = sum(row['edp'] for row in freq_data) / len(freq_data)
            
            freq_averages[freq] = {
                'duration': avg_duration,
                'energy': avg_energy,
                'edp': avg_edp,
                'samples': len(freq_data)
            }
        
        # Find baseline (highest frequency)
        max_freq = max(freq_averages.keys())
        baseline = freq_averages[max_freq]
        
        print(f"Baseline (max freq {max_freq}MHz): {baseline['duration']:.1f}s, {baseline['energy']:.1f}J")
        print(f"Frequencies tested: {sorted(freq_averages.keys())}")
        
        # Find optimal frequency (minimum EDP with ≤5% performance degradation)
        best_freq = None
        best_metrics = None
        
        for freq in sorted(freq_averages.keys()):
            metrics = freq_averages[freq]
            
            # Calculate performance impact (duration increase)
            perf_impact = ((metrics['duration'] - baseline['duration']) / baseline['duration']) * 100
            
            # Calculate energy savings
            energy_savings = ((baseline['energy'] - metrics['energy']) / baseline['energy']) * 100
            
            print(f"  {freq:>4.0f}MHz: {metrics['duration']:>5.1f}s ({perf_impact:+4.1f}%) "
                  f"{metrics['energy']:>6.1f}J ({energy_savings:+4.1f}%) "
                  f"EDP: {metrics['edp']:>8.0f} [{metrics['samples']} samples]")
            
            # Check if this frequency meets performance constraint
            if perf_impact <= 5.0:  # ≤5% performance degradation
                if best_freq is None or metrics['edp'] < best_metrics['edp']:
                    best_freq = freq
                    best_metrics = {
                        'frequency': freq,
                        'energy_savings': energy_savings,
                        'performance_impact': perf_impact,
                        'edp': metrics['edp']
                    }
        
        if best_freq is not None:
            optimal_results[workload] = best_metrics
            print(f"  → OPTIMAL: {best_freq}MHz ({best_metrics['energy_savings']:+.1f}% energy, "
                  f"{best_metrics['performance_impact']:+.1f}% perf)")
        else:
            print(f"  → No valid optimal frequency found (all exceed 5% performance impact)")
    
    return optimal_results

def main():
    # Load our aggregated data
    data_file = "aggregated_results/ai_inference_aggregated_data_20250807_134913.csv"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    print("Loading aggregated experimental data...")
    data = load_and_clean_csv_data(data_file)
    print(f"Loaded {len(data)} unique data points")
    
    # Analyze A100 data
    a100_optimal = find_optimal_frequencies_for_gpu(data, 'A100')
    
    print(f"\n{'=' * 60}")
    print("A100 OPTIMAL FREQUENCY SUMMARY")
    print(f"{'=' * 60}")
    
    if a100_optimal:
        for workload, metrics in a100_optimal.items():
            freq = metrics['frequency']
            energy = metrics['energy_savings']
            perf = metrics['performance_impact']
            print(f"{workload:>15}: {freq:>4.0f}MHz ({energy:+4.1f}% energy, {perf:+4.1f}% perf)")
    else:
        print("No optimal frequencies found for A100")
    
    # Also get H100 data for comparison
    print(f"\n{'=' * 60}")
    print("H100 COMPARISON (from previous analysis)")
    print(f"{'=' * 60}")
    
    h100_optimal = find_optimal_frequencies_for_gpu(data, 'H100')
    
    if h100_optimal:
        for workload, metrics in h100_optimal.items():
            freq = metrics['frequency']
            energy = metrics['energy_savings']
            perf = metrics['performance_impact']
            print(f"{workload:>15}: {freq:>4.0f}MHz ({energy:+4.1f}% energy, {perf:+4.1f}% perf)")
    
    return a100_optimal, h100_optimal

if __name__ == "__main__":
    main()
