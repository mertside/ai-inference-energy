#!/usr/bin/env python
"""
Corrected real data-driven optimal frequency selection
Handles duplicates and analyzes true frequency-performance relationships
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

def analyze_frequency_performance_relationship(data):
    """Analyze the actual frequency-performance relationship"""
    
    # Group data by GPU and workload
    groups = defaultdict(list)
    for row in data:
        key = (row['gpu'], row['workload'])
        groups[key].append(row)
    
    print("Frequency-Performance Relationship Analysis:")
    print("=" * 60)
    
    relationship_summary = {}
    
    for (gpu, workload), subset in groups.items():
        # Group by frequency
        freq_groups = defaultdict(list)
        for row in subset:
            freq_groups[row['frequency_mhz']].append(row)
        
        # Calculate statistics for each frequency
        freq_stats = {}
        for freq, rows in freq_groups.items():
            times = [row['duration_seconds'] for row in rows]
            energies = [row['total_energy_joules'] for row in rows]
            powers = [row['avg_power_watts'] for row in rows]
            
            freq_stats[freq] = {
                'mean_time': sum(times) / len(times),
                'mean_energy': sum(energies) / len(energies),
                'mean_power': sum(powers) / len(powers),
                'count': len(times),
                'time_std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
            }
        
        # Analyze relationship
        frequencies = sorted(freq_stats.keys())
        if len(frequencies) < 3:
            continue
            
        times = [freq_stats[f]['mean_time'] for f in frequencies]
        energies = [freq_stats[f]['mean_energy'] for f in frequencies]
        
        # Check if higher frequency = faster execution (normal behavior)
        high_freq_time = freq_stats[max(frequencies)]['mean_time']
        low_freq_time = freq_stats[min(frequencies)]['mean_time']
        normal_behavior = high_freq_time < low_freq_time
        
        relationship_summary[(gpu, workload)] = {
            'normal_behavior': normal_behavior,
            'freq_stats': freq_stats,
            'frequencies': frequencies
        }
        
        print(f"\n{gpu} {workload}:")
        print(f"  Frequencies tested: {len(frequencies)} ({min(frequencies):.0f}-{max(frequencies):.0f} MHz)")
        print(f"  Normal freq-perf relationship: {normal_behavior}")
        print(f"  Time at max freq ({max(frequencies):.0f}MHz): {high_freq_time:.1f}s")
        print(f"  Time at min freq ({min(frequencies):.0f}MHz): {low_freq_time:.1f}s")
        
        # Show some frequency points
        sample_freqs = frequencies[::len(frequencies)//3] if len(frequencies) > 3 else frequencies
        for freq in sample_freqs:
            stats = freq_stats[freq]
            print(f"    {freq:>4.0f}MHz: {stats['mean_time']:5.1f}s, {stats['mean_energy']:5.0f}J, {stats['count']:2d} runs")
    
    return relationship_summary

def find_optimal_frequencies_corrected(data, constraint_pct=5.0):
    """Find optimal frequencies using corrected analysis"""
    
    # First analyze relationships
    relationships = analyze_frequency_performance_relationship(data)
    
    # Group data by GPU and workload
    groups = defaultdict(list)
    for row in data:
        key = (row['gpu'], row['workload'])
        groups[key].append(row)
    
    results = []
    
    print(f"\n\nOptimal Frequency Selection (≤{constraint_pct}% degradation):")
    print("=" * 60)
    
    for (gpu, workload), subset in groups.items():
        if (gpu, workload) not in relationships:
            continue
            
        rel_info = relationships[(gpu, workload)]
        freq_stats = rel_info['freq_stats']
        frequencies = rel_info['frequencies']
        
        if len(frequencies) < 2:
            continue
        
        # Determine baseline based on normal vs abnormal behavior
        if rel_info['normal_behavior']:
            # Normal: highest frequency = best performance
            baseline_freq = max(frequencies)
        else:
            # Abnormal: lowest frequency = best performance (this shouldn't happen but handle it)
            baseline_freq = min(frequencies, key=lambda f: freq_stats[f]['mean_time'])
        
        baseline_stats = freq_stats[baseline_freq]
        baseline_time = baseline_stats['mean_time']
        baseline_energy = baseline_stats['mean_energy']
        baseline_power = baseline_stats['mean_power']
        
        # Performance constraint
        max_acceptable_time = baseline_time * (1 + constraint_pct / 100.0)
        
        # Find feasible frequencies
        feasible_freqs = []
        for freq in frequencies:
            if freq_stats[freq]['mean_time'] <= max_acceptable_time:
                feasible_freqs.append(freq)
        
        if not feasible_freqs:
            print(f"{gpu} {workload:>15}: No feasible solutions")
            continue
        
        # Find optimal EDP among feasible frequencies
        optimal_freq = min(feasible_freqs, key=lambda f: freq_stats[f]['mean_time'] * freq_stats[f]['mean_energy'])
        optimal_stats = freq_stats[optimal_freq]
        
        # Calculate metrics
        energy_savings_pct = (1 - optimal_stats['mean_energy'] / baseline_energy) * 100
        power_savings_pct = (1 - optimal_stats['mean_power'] / baseline_power) * 100
        perf_impact_pct = (optimal_stats['mean_time'] / baseline_time - 1) * 100
        
        # EDP metrics
        baseline_edp = baseline_time * baseline_energy
        optimal_edp = optimal_stats['mean_time'] * optimal_stats['mean_energy']
        edp_reduction_pct = (1 - optimal_edp / baseline_edp) * 100
        
        result = {
            'gpu': gpu,
            'workload': workload,
            'optimal_frequency_mhz': int(optimal_freq),
            'baseline_frequency_mhz': int(baseline_freq),
            'optimal_energy_joules': optimal_stats['mean_energy'],
            'baseline_energy_joules': baseline_energy,
            'optimal_time_seconds': optimal_stats['mean_time'],
            'baseline_time_seconds': baseline_time,
            'optimal_power_watts': optimal_stats['mean_power'],
            'baseline_power_watts': baseline_power,
            'energy_savings_pct': energy_savings_pct,
            'power_savings_pct': power_savings_pct,
            'performance_impact_pct': perf_impact_pct,
            'edp_reduction_pct': edp_reduction_pct,
            'normal_behavior': rel_info['normal_behavior'],
            'within_constraint': optimal_stats['mean_time'] <= max_acceptable_time,
            'feasible_count': len(feasible_freqs),
            'total_frequency_points': len(frequencies)
        }
        
        results.append(result)
        
        print(f"{gpu} {workload:>15}: {int(optimal_freq):>4}MHz "
              f"(baseline: {int(baseline_freq)}MHz)")
        print(f"                     Energy: {energy_savings_pct:+5.1f}%, "
              f"Perf: {perf_impact_pct:+4.1f}%, EDP: {edp_reduction_pct:+4.1f}%")
        print(f"                     Normal behavior: {rel_info['normal_behavior']}")
        print()
    
    return results

def save_corrected_results(results, output_dir="corrected_optimal_results"):
    """Save corrected results"""
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    json_file = os.path.join(output_dir, f"corrected_optimal_frequencies_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary report
    report_file = os.path.join(output_dir, f"corrected_optimal_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write("Corrected Real Data-Driven Optimal Frequency Selection Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Performance constraint: ≤5.0% degradation\n\n")
        
        normal_count = sum(1 for r in results if r['normal_behavior'])
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"  Total combinations: {len(results)}\n")
        f.write(f"  Normal freq-perf behavior: {normal_count}/{len(results)}\n")
        
        if results:
            energy_savings = [r['energy_savings_pct'] for r in results if r['normal_behavior']]
            perf_impacts = [r['performance_impact_pct'] for r in results if r['normal_behavior']]
            
            if energy_savings:
                f.write(f"  Avg energy savings (normal): {sum(energy_savings)/len(energy_savings):.1f}%\n")
                f.write(f"  Avg performance impact (normal): {sum(perf_impacts)/len(perf_impacts):.1f}%\n")
        
        f.write("\nOPTIMAL FREQUENCIES BY GPU-WORKLOAD:\n")
        for result in sorted(results, key=lambda x: (x['gpu'], x['workload'])):
            behavior = "✓" if result['normal_behavior'] else "⚠"
            f.write(f"  {behavior} {result['gpu']} {result['workload']:>15}: "
                   f"{result['optimal_frequency_mhz']:>4}MHz "
                   f"({result['energy_savings_pct']:+5.1f}% energy, "
                   f"{result['performance_impact_pct']:+4.1f}% perf)\n")
    
    print(f"\nCorrected results saved to:")
    print(f"  Report: {report_file}")
    print(f"  JSON: {json_file}")

def main():
    """Main function"""
    data_file = "aggregated_results/ai_inference_aggregated_data_20250807_134913.csv"
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return 1
    
    try:
        # Load and clean data
        print(f"Loading and cleaning data from: {data_file}")
        data = load_and_clean_csv_data(data_file)
        print(f"Loaded {len(data)} unique data points (duplicates removed)\n")
        
        # Find optimal frequencies with corrected analysis
        results = find_optimal_frequencies_corrected(data)
        
        if not results:
            print("No optimal frequencies found!")
            return 1
        
        # Summary
        normal_results = [r for r in results if r['normal_behavior']]
        abnormal_results = [r for r in results if not r['normal_behavior']]
        
        print(f"📊 CORRECTED ANALYSIS SUMMARY")
        print(f"=" * 35)
        print(f"Total combinations: {len(results)}")
        print(f"Normal behavior: {len(normal_results)}")
        print(f"Abnormal behavior: {len(abnormal_results)}")
        
        if normal_results:
            energy_savings = [r['energy_savings_pct'] for r in normal_results]
            perf_impacts = [r['performance_impact_pct'] for r in normal_results]
            
            print(f"\nNormal behavior results:")
            print(f"  Average energy savings: {sum(energy_savings)/len(energy_savings):.1f}%")
            print(f"  Average performance impact: {sum(perf_impacts)/len(perf_impacts):.1f}%")
        
        if abnormal_results:
            print(f"\n⚠  {len(abnormal_results)} combinations show abnormal freq-perf relationship")
            print("   (Lower frequencies showing better performance - needs investigation)")
        
        # Save results
        save_corrected_results(results)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
