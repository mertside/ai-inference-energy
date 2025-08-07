#!/usr/bin/env python
"""
Fixed real data-driven optimal frequency selection with proper baseline analysis
"""

import csv
import os
from collections import defaultdict

def load_csv_data(filename):
    """Load CSV data using built-in csv module"""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['frequency_mhz'] = float(row['frequency_mhz'])
            row['duration_seconds'] = float(row['duration_seconds'])
            row['avg_power_watts'] = float(row['avg_power_watts'])
            row['total_energy_joules'] = float(row['total_energy_joules'])
            
            # Calculate EDP if not present
            if 'edp' not in row or not row['edp']:
                row['edp'] = row['total_energy_joules'] * row['duration_seconds']
            else:
                row['edp'] = float(row['edp'])
                
            data.append(row)
    return data

def analyze_data_overview(data):
    """Analyze the data to understand the frequency-performance relationship"""
    
    # Group data by GPU and workload
    groups = defaultdict(list)
    for row in data:
        key = (row['gpu'], row['workload'])
        groups[key].append(row)
    
    print("Data Overview:")
    print("=" * 50)
    
    for (gpu, workload), subset in groups.items():
        freqs = sorted(set(row['frequency_mhz'] for row in subset))
        times = []
        energies = []
        
        for freq in freqs:
            freq_data = [row for row in subset if row['frequency_mhz'] == freq]
            avg_time = sum(row['duration_seconds'] for row in freq_data) / len(freq_data)
            avg_energy = sum(row['total_energy_joules'] for row in freq_data) / len(freq_data)
            times.append(avg_time)
            energies.append(avg_energy)
        
        print(f"\n{gpu} {workload}:")
        print(f"  Frequency range: {min(freqs):.0f} - {max(freqs):.0f} MHz")
        print(f"  Time range: {min(times):.1f} - {max(times):.1f} seconds")
        print(f"  Energy range: {min(energies):.0f} - {max(energies):.0f} Joules")
        
        # Check frequency-time relationship
        if len(freqs) >= 3:
            # Simple correlation check
            high_freq_time = times[freqs.index(max(freqs))]
            low_freq_time = times[freqs.index(min(freqs))]
            print(f"  Time ratio (low/high freq): {low_freq_time/high_freq_time:.2f}")

def find_optimal_frequencies_fixed(data, constraint_pct=5.0):
    """Find optimal frequencies with proper baseline establishment"""
    
    # Group data by GPU and workload
    groups = defaultdict(list)
    for row in data:
        key = (row['gpu'], row['workload'])
        groups[key].append(row)
    
    results = []
    
    print(f"\nReal Data-Driven Optimal Frequency Selection")
    print("=" * 50)
    print(f"Performance constraint: ≤{constraint_pct}% degradation")
    print()
    
    for (gpu, workload), subset in groups.items():
        if not subset:
            continue
        
        # Create frequency-performance mapping
        freq_data = defaultdict(list)
        for row in subset:
            freq_data[row['frequency_mhz']].append(row)
        
        # Calculate averages for each frequency
        freq_averages = {}
        for freq, rows in freq_data.items():
            avg_time = sum(row['duration_seconds'] for row in rows) / len(rows)
            avg_energy = sum(row['total_energy_joules'] for row in rows) / len(rows)
            avg_power = sum(row['avg_power_watts'] for row in rows) / len(rows)
            avg_edp = sum(row['edp'] for row in rows) / len(rows)
            
            freq_averages[freq] = {
                'frequency_mhz': freq,
                'duration_seconds': avg_time,
                'total_energy_joules': avg_energy,
                'avg_power_watts': avg_power,
                'edp': avg_edp,
                'count': len(rows)
            }
        
        if len(freq_averages) < 2:
            print(f"{gpu} {workload:>15}: Insufficient frequency points ({len(freq_averages)})")
            continue
            
        # Find baseline (maximum frequency = best performance = shortest time)
        frequencies = list(freq_averages.keys())
        max_freq = max(frequencies)
        baseline = freq_averages[max_freq]
        
        baseline_time = baseline['duration_seconds']
        baseline_energy = baseline['total_energy_joules']
        baseline_power = baseline['avg_power_watts']
        
        # Performance constraint (max acceptable time)
        max_acceptable_time = baseline_time * (1 + constraint_pct / 100.0)
        
        print(f"{gpu} {workload}:")
        print(f"  Baseline ({max_freq:.0f}MHz): {baseline_time:.1f}s, {baseline_energy:.0f}J")
        print(f"  Max acceptable time: {max_acceptable_time:.1f}s")
        
        # Find feasible frequencies
        feasible_freqs = []
        for freq, data in freq_averages.items():
            if data['duration_seconds'] <= max_acceptable_time:
                feasible_freqs.append(freq)
        
        if not feasible_freqs:
            print(f"  No feasible solutions within {constraint_pct}% constraint")
            continue
        
        print(f"  Feasible frequencies: {len(feasible_freqs)}/{len(frequencies)}")
        
        # Find optimal EDP among feasible frequencies
        feasible_data = {freq: freq_averages[freq] for freq in feasible_freqs}
        optimal_freq = min(feasible_data.keys(), key=lambda f: feasible_data[f]['edp'])
        optimal = feasible_data[optimal_freq]
        
        # Calculate metrics
        energy_savings_pct = (1 - optimal['total_energy_joules'] / baseline_energy) * 100
        power_savings_pct = (1 - optimal['avg_power_watts'] / baseline_power) * 100
        perf_impact_pct = (optimal['duration_seconds'] / baseline_time - 1) * 100
        edp_reduction_pct = (1 - optimal['edp'] / baseline['edp']) * 100
        
        result = {
            'gpu': gpu,
            'workload': workload,
            'optimal_frequency_mhz': int(optimal_freq),
            'baseline_frequency_mhz': int(max_freq),
            'optimal_energy_joules': optimal['total_energy_joules'],
            'baseline_energy_joules': baseline_energy,
            'optimal_time_seconds': optimal['duration_seconds'],
            'baseline_time_seconds': baseline_time,
            'optimal_power_watts': optimal['avg_power_watts'],
            'baseline_power_watts': baseline_power,
            'energy_savings_pct': energy_savings_pct,
            'power_savings_pct': power_savings_pct,
            'performance_impact_pct': perf_impact_pct,
            'edp_reduction_pct': edp_reduction_pct,
            'optimal_edp': optimal['edp'],
            'baseline_edp': baseline['edp'],
            'within_constraint': optimal['duration_seconds'] <= max_acceptable_time,
            'feasible_count': len(feasible_freqs),
            'total_frequency_points': len(frequencies)
        }
        
        results.append(result)
        
        print(f"  Optimal: {int(optimal_freq):>4}MHz "
              f"({energy_savings_pct:+5.1f}% energy, {perf_impact_pct:+4.1f}% perf, "
              f"{edp_reduction_pct:+4.1f}% EDP)")
        print()
    
    return results

def main():
    """Main function"""
    data_file = "aggregated_results/ai_inference_aggregated_data_20250807_134913.csv"
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return 1
    
    try:
        # Load data
        print(f"Loading data from: {data_file}")
        data = load_csv_data(data_file)
        print(f"Loaded {len(data)} data points\n")
        
        # Analyze data overview
        analyze_data_overview(data)
        
        # Find optimal frequencies
        results = find_optimal_frequencies_fixed(data)
        
        if not results:
            print("No optimal frequencies found!")
            return 1
        
        # Summary
        energy_savings = [r['energy_savings_pct'] for r in results]
        perf_impacts = [r['performance_impact_pct'] for r in results]
        edp_reductions = [r['edp_reduction_pct'] for r in results]
        
        print(f"📊 SUMMARY RESULTS")
        print(f"=" * 30)
        print(f"Combinations analyzed: {len(results)}")
        print(f"Average energy savings: {sum(energy_savings)/len(energy_savings):.1f}%")
        print(f"Average performance impact: {sum(perf_impacts)/len(perf_impacts):.1f}%")
        print(f"Average EDP reduction: {sum(edp_reductions)/len(edp_reductions):.1f}%")
        print(f"Constraint satisfaction: {sum(1 for r in results if r['within_constraint'])/len(results)*100:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
