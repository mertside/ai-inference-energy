#!/usr/bin/env python
"""
Simple CSV-based optimal frequency selection using built-in modules only
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

def find_optimal_frequencies(data, constraint_pct=5.0):
    """Find optimal frequencies for each GPU-workload combination"""
    
    # Group data by GPU and workload
    groups = defaultdict(list)
    for row in data:
        key = (row['gpu'], row['workload'])
        groups[key].append(row)
    
    results = []
    
    print("Real Data-Driven Optimal Frequency Selection")
    print("=" * 50)
    print(f"Performance constraint: ≤{constraint_pct}% degradation")
    print()
    
    for (gpu, workload), subset in groups.items():
        if not subset:
            continue
            
        # Find baseline (maximum frequency)
        max_freq = max(row['frequency_mhz'] for row in subset)
        baseline_rows = [row for row in subset if row['frequency_mhz'] == max_freq]
        
        if not baseline_rows:
            continue
            
        # Calculate baseline averages
        baseline_time = sum(row['duration_seconds'] for row in baseline_rows) / len(baseline_rows)
        baseline_energy = sum(row['total_energy_joules'] for row in baseline_rows) / len(baseline_rows)
        
        # Performance constraint
        max_acceptable_time = baseline_time * (1 + constraint_pct / 100.0)
        
        # Apply constraint
        feasible = [row for row in subset if row['duration_seconds'] <= max_acceptable_time]
        
        if not feasible:
            print(f"{gpu} {workload:>15}: No feasible solutions within {constraint_pct}% constraint")
            continue
        
        # Find optimal EDP
        optimal = min(feasible, key=lambda x: x['edp'])
        
        # Calculate metrics
        energy_savings_pct = (1 - optimal['total_energy_joules'] / baseline_energy) * 100
        perf_impact_pct = (optimal['duration_seconds'] / baseline_time - 1) * 100
        
        result = {
            'gpu': gpu,
            'workload': workload,
            'optimal_frequency_mhz': int(optimal['frequency_mhz']),
            'baseline_frequency_mhz': int(max_freq),
            'optimal_energy_joules': optimal['total_energy_joules'],
            'baseline_energy_joules': baseline_energy,
            'optimal_time_seconds': optimal['duration_seconds'],
            'baseline_time_seconds': baseline_time,
            'energy_savings_pct': energy_savings_pct,
            'performance_impact_pct': perf_impact_pct,
            'optimal_edp': optimal['edp'],
            'within_constraint': optimal['duration_seconds'] <= max_acceptable_time,
            'feasible_count': len(feasible),
            'total_count': len(subset)
        }
        
        results.append(result)
        
        print(f"{gpu} {workload:>15}: {int(optimal['frequency_mhz']):>4}MHz "
              f"({energy_savings_pct:+5.1f}% energy, {perf_impact_pct:+4.1f}% perf)")
    
    return results

def generate_summary(results):
    """Generate summary statistics"""
    if not results:
        return {}
    
    energy_savings = [r['energy_savings_pct'] for r in results]
    perf_impacts = [r['performance_impact_pct'] for r in results]
    
    summary = {
        'total_combinations': len(results),
        'avg_energy_savings_pct': sum(energy_savings) / len(energy_savings),
        'max_energy_savings_pct': max(energy_savings),
        'min_energy_savings_pct': min(energy_savings),
        'avg_performance_impact_pct': sum(perf_impacts) / len(perf_impacts),
        'max_performance_impact_pct': max(perf_impacts),
        'min_performance_impact_pct': min(perf_impacts),
        'within_constraint_count': sum(1 for r in results if r['within_constraint']),
        'constraint_satisfaction_pct': sum(1 for r in results if r['within_constraint']) / len(results) * 100
    }
    
    return summary

def save_results(results, output_dir="real_optimal_results"):
    """Save results to files"""
    import json
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = os.path.join(output_dir, f"optimal_frequencies_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV results
    csv_file = os.path.join(output_dir, f"optimal_frequencies_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Save summary report
    summary = generate_summary(results)
    report_file = os.path.join(output_dir, f"optimal_frequency_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write("Real Data-Driven Optimal Frequency Selection Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Method: EDP optimization\n")
        f.write(f"Performance constraint: ≤5.0% degradation\n\n")
        
        f.write("SUMMARY STATISTICS:\n")
        f.write(f"  Total combinations: {summary['total_combinations']}\n")
        f.write(f"  Average energy savings: {summary['avg_energy_savings_pct']:.1f}%\n")
        f.write(f"  Average performance impact: {summary['avg_performance_impact_pct']:.1f}%\n")
        f.write(f"  Constraint satisfaction: {summary['constraint_satisfaction_pct']:.1f}%\n\n")
        
        f.write("OPTIMAL FREQUENCIES BY GPU-WORKLOAD:\n")
        for result in sorted(results, key=lambda x: (x['gpu'], x['workload'])):
            f.write(f"  {result['gpu']} {result['workload']:>15}: "
                   f"{result['optimal_frequency_mhz']:>4}MHz "
                   f"({result['energy_savings_pct']:+5.1f}% energy, "
                   f"{result['performance_impact_pct']:+4.1f}% perf)\n")
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")
    print(f"  Report: {report_file}")
    
    return json_file, csv_file, report_file

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
        print(f"Loaded {len(data)} data points")
        
        # Find optimal frequencies
        results = find_optimal_frequencies(data)
        
        if not results:
            print("No optimal frequencies found!")
            return 1
        
        # Generate summary
        summary = generate_summary(results)
        
        print(f"\n📊 OPTIMAL FREQUENCY SELECTION COMPLETED")
        print(f"=" * 50)
        print(f"Method: EDP optimization")
        print(f"Constraint: ≤5.0% performance degradation") 
        print(f"Combinations analyzed: {summary['total_combinations']}")
        print(f"Average energy savings: {summary['avg_energy_savings_pct']:.1f}%")
        print(f"Average performance impact: {summary['avg_performance_impact_pct']:.1f}%")
        print(f"Constraint satisfaction: {summary['constraint_satisfaction_pct']:.1f}%")
        
        # Save results
        save_results(results)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
