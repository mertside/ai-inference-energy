#!/usr/bin/env python3

"""
EDP Results Summary Table Generator

This script reads the JSON output from edp_optimizer.py and generates
nicely formatted summary tables for easy interpretation and CSV export.
"""

import json
import csv
import argparse
from pathlib import Path

def create_summary_table(results_file: str, export_csv: bool = False):
    """Create formatted summary tables from EDP optimization results"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Group by GPU
    gpu_groups = {}
    for result in results:
        gpu = result['gpu']
        if gpu not in gpu_groups:
            gpu_groups[gpu] = []
        gpu_groups[gpu].append(result)
    
    print("📊 EDP OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    
    # Overall summary
    total_results = len(results)
    avg_energy_savings = sum(r['energy_savings_percent'] for r in results) / total_results
    avg_edp_improvement = sum(r['edp_improvement_percent'] for r in results) / total_results
    faster_than_max = sum(1 for r in results if r['performance_vs_max_percent'] < 0)
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   • Total configurations: {total_results}")
    print(f"   • Average energy savings: {avg_energy_savings:.1f}%")
    print(f"   • Average EDP improvement: {avg_edp_improvement:.1f}%")
    print(f"   • Faster than max frequency: {faster_than_max}/{total_results} ({100*faster_than_max/total_results:.1f}%)")
    
    # Per-GPU detailed tables
    for gpu in sorted(gpu_groups.keys()):
        gpu_results = gpu_groups[gpu]
        
        print(f"\n🔧 {gpu} GPU DETAILED RESULTS:")
        print("-" * 80)
        print(f"{'Workload':<15} {'Optimal':<8} {'Max':<6} {'Energy':<8} {'Perf vs':<9} {'EDP':<6} {'Faster':<7}")
        print(f"{'':15} {'Freq':<8} {'Freq':<6} {'Savings':<8} {'Max %':<9} {'Impr%':<6} {'than Max':<7}")
        print("-" * 80)
        
        total_energy = 0
        faster_count = 0
        
        for result in sorted(gpu_results, key=lambda x: x['workload']):
            workload = result['workload']
            optimal_freq = result['optimal_frequency_mhz']
            max_freq = result['max_frequency_mhz']
            energy_savings = result['energy_savings_percent']
            perf_vs_max = result['performance_vs_max_percent']
            edp_improvement = result['edp_improvement_percent']
            faster = "Yes" if perf_vs_max < 0 else "No"
            
            # Format performance change
            perf_str = f"{abs(perf_vs_max):.1f}{'↑' if perf_vs_max < 0 else '↓'}"
            
            print(f"{workload:<15} {optimal_freq:<8} {max_freq:<6} {energy_savings:<7.1f}% {perf_str:<9} {edp_improvement:<5.1f}% {faster:<7}")
            
            total_energy += energy_savings
            if perf_vs_max < 0:
                faster_count += 1
        
        avg_energy = total_energy / len(gpu_results)
        print("-" * 80)
        print(f"{'GPU Average:':<15} {'':<8} {'':<6} {avg_energy:<7.1f}% {'':<9} {'':<6} {faster_count}/{len(gpu_results)}")
    
    # Cross-workload analysis
    print(f"\n🔬 CROSS-GPU WORKLOAD ANALYSIS:")
    print("-" * 80)
    print(f"{'Workload':<15} {'A100':<12} {'H100':<12} {'V100':<12} {'Best GPU':<10}")
    print(f"{'':15} {'Savings':<12} {'Savings':<12} {'Savings':<12} {'(Savings)':<10}")
    print("-" * 80)
    
    # Group by workload
    workload_groups = {}
    for result in results:
        workload = result['workload']
        if workload not in workload_groups:
            workload_groups[workload] = {}
        workload_groups[workload][result['gpu']] = result
    
    for workload in sorted(workload_groups.keys()):
        workload_data = workload_groups[workload]
        
        # Get savings for each GPU
        a100_savings = workload_data.get('A100', {}).get('energy_savings_percent', 0)
        h100_savings = workload_data.get('H100', {}).get('energy_savings_percent', 0)
        v100_savings = workload_data.get('V100', {}).get('energy_savings_percent', 0)
        
        # Find best GPU
        best_gpu = max([
            ('A100', a100_savings),
            ('H100', h100_savings), 
            ('V100', v100_savings)
        ], key=lambda x: x[1])
        
        print(f"{workload:<15} {a100_savings:<11.1f}% {h100_savings:<11.1f}% {v100_savings:<11.1f}% {best_gpu[0]:<10}")
    
    # Frequency reduction analysis
    print(f"\n⚡ FREQUENCY REDUCTION ANALYSIS:")
    print("-" * 80)
    print(f"{'GPU-Workload':<20} {'Max Freq':<10} {'Optimal':<10} {'Reduction':<10} {'Reduction %':<12}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: (x['gpu'], x['workload'])):
        config = f"{result['gpu']} {result['workload']}"
        max_freq = result['max_frequency_mhz']
        optimal_freq = result['optimal_frequency_mhz']
        reduction = max_freq - optimal_freq
        reduction_pct = (reduction / max_freq) * 100
        
        print(f"{config:<20} {max_freq:<10} {optimal_freq:<10} {reduction:<10} {reduction_pct:<11.1f}%")

    # Export to CSV if requested
    if export_csv:
        export_to_csv(results, results_file)

def export_to_csv(results: list, input_file: str):
    """Export results to multiple CSV files for detailed analysis"""
    
    # Generate base filename from input file
    base_name = Path(input_file).stem
    
    # 1. Main results CSV
    main_csv = f"{base_name}_summary.csv"
    with open(main_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'GPU', 'Workload', 'Optimal_Freq_MHz', 'Max_Freq_MHz', 'Fastest_Freq_MHz',
            'Energy_Savings_Percent', 'Performance_vs_Max_Percent', 'Performance_vs_Fastest_Percent',
            'EDP_Improvement_Percent', 'Is_Max_Freq_Fastest', 'Optimal_Timing_Seconds',
            'Optimal_Energy_Joules', 'Optimal_EDP', 'Runs_Averaged'
        ])
        
        # Data rows
        for result in sorted(results, key=lambda x: (x['gpu'], x['workload'])):
            writer.writerow([
                result['gpu'],
                result['workload'],
                result['optimal_frequency_mhz'],
                result['max_frequency_mhz'],
                result['fastest_frequency_mhz'],
                round(result['energy_savings_percent'], 2),
                round(result['performance_vs_max_percent'], 2),
                round(result['performance_vs_fastest_percent'], 2),
                round(result['edp_improvement_percent'], 2),
                result['is_max_frequency_fastest'],
                result['optimal_timing_seconds'],
                result['optimal_energy_joules'],
                result['optimal_edp'],
                result['runs_averaged']
            ])
    
    # 2. Cross-workload comparison CSV
    workload_csv = f"{base_name}_workload_comparison.csv"
    with open(workload_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Workload', 'A100_Energy_Savings', 'H100_Energy_Savings', 'V100_Energy_Savings', 'Best_GPU', 'Best_Savings'])
        
        # Group by workload
        workload_groups = {}
        for result in results:
            workload = result['workload']
            if workload not in workload_groups:
                workload_groups[workload] = {}
            workload_groups[workload][result['gpu']] = result
        
        # Data rows
        for workload in sorted(workload_groups.keys()):
            workload_data = workload_groups[workload]
            
            a100_savings = workload_data.get('A100', {}).get('energy_savings_percent', 0)
            h100_savings = workload_data.get('H100', {}).get('energy_savings_percent', 0)
            v100_savings = workload_data.get('V100', {}).get('energy_savings_percent', 0)
            
            # Find best GPU
            best_gpu_data = max([
                ('A100', a100_savings),
                ('H100', h100_savings),
                ('V100', v100_savings)
            ], key=lambda x: x[1])
            
            writer.writerow([
                workload,
                round(a100_savings, 2),
                round(h100_savings, 2),
                round(v100_savings, 2),
                best_gpu_data[0],
                round(best_gpu_data[1], 2)
            ])
    
    # 3. Frequency reduction analysis CSV
    freq_csv = f"{base_name}_frequency_analysis.csv"
    with open(freq_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['GPU', 'Workload', 'Max_Freq_MHz', 'Optimal_Freq_MHz', 'Reduction_MHz', 'Reduction_Percent'])
        
        # Data rows
        for result in sorted(results, key=lambda x: (x['gpu'], x['workload'])):
            max_freq = result['max_frequency_mhz']
            optimal_freq = result['optimal_frequency_mhz']
            reduction = max_freq - optimal_freq
            reduction_pct = (reduction / max_freq) * 100
            
            writer.writerow([
                result['gpu'],
                result['workload'],
                max_freq,
                optimal_freq,
                reduction,
                round(reduction_pct, 2)
            ])
    
    # 4. GPU summary CSV
    gpu_csv = f"{base_name}_gpu_summary.csv"
    with open(gpu_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['GPU', 'Avg_Energy_Savings', 'Configurations_Faster_Than_Max', 'Total_Configurations', 'Faster_Percentage'])
        
        # Group by GPU and calculate averages
        gpu_groups = {}
        for result in results:
            gpu = result['gpu']
            if gpu not in gpu_groups:
                gpu_groups[gpu] = []
            gpu_groups[gpu].append(result)
        
        # Data rows
        for gpu in sorted(gpu_groups.keys()):
            gpu_results = gpu_groups[gpu]
            avg_energy = sum(r['energy_savings_percent'] for r in gpu_results) / len(gpu_results)
            faster_count = sum(1 for r in gpu_results if r['performance_vs_max_percent'] < 0)
            total_configs = len(gpu_results)
            faster_pct = (faster_count / total_configs) * 100
            
            writer.writerow([
                gpu,
                round(avg_energy, 2),
                faster_count,
                total_configs,
                round(faster_pct, 1)
            ])
    
    print(f"\n📁 CSV FILES EXPORTED:")
    print(f"   • Main results: {main_csv}")
    print(f"   • Workload comparison: {workload_csv}")
    print(f"   • Frequency analysis: {freq_csv}")
    print(f"   • GPU summary: {gpu_csv}")

def main():
    parser = argparse.ArgumentParser(description='Generate summary tables from EDP optimization results')
    parser.add_argument('--input', '-i',
                       default='edp_optimization_results.json',
                       help='Input JSON file with EDP results')
    parser.add_argument('--csv', '-c',
                       action='store_true',
                       help='Export results to CSV files')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return 1
    
    create_summary_table(args.input, export_csv=args.csv)
    return 0

if __name__ == '__main__':
    exit(main())
