#!/usr/bin/env python3
"""
Real Data-Driven Optimal Frequency Analysis

This script analyzes real experimental data to find optimal frequencies for each
GPU-application combination with detailed energy savings and performance impact analysis.

Features:
- Uses actual DCGMI profiling data collected from experiments
- Calculates real energy savings and performance degradation
- Applies performance constraints (≤5% degradation by default)
- Generates comprehensive reports with statistical analysis
- Exports results for deployment and research publication

Author: Mert Side
Date: August 7, 2025
"""

import csv
import json
import os
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Data structures for analysis
FrequencyResult = namedtuple('FrequencyResult', [
    'frequency', 'duration', 'energy', 'power', 'edp', 
    'samples', 'duration_std', 'energy_std', 'power_std'
])

OptimalResult = namedtuple('OptimalResult', [
    'gpu', 'workload', 'optimal_frequency', 'baseline_frequency',
    'energy_savings_pct', 'performance_impact_pct', 'power_savings_pct',
    'optimal_energy', 'baseline_energy', 'optimal_duration', 'baseline_duration',
    'optimal_power', 'baseline_power', 'optimal_edp', 'baseline_edp',
    'constraint_satisfied', 'data_quality'
])

class RealDataOptimalFrequencyAnalyzer:
    """Analyzes real experimental data to find optimal frequencies"""
    
    def __init__(self, performance_constraint_pct: float = 5.0, min_samples: int = 2):
        self.performance_constraint_pct = performance_constraint_pct
        self.min_samples = min_samples
        self.raw_data = []
        self.processed_data = defaultdict(lambda: defaultdict(list))
        self.frequency_averages = defaultdict(lambda: defaultdict(dict))
        self.optimal_results = []
        
    def load_aggregated_data(self, csv_file: str) -> bool:
        """Load aggregated experimental data from CSV file"""
        try:
            print(f"Loading experimental data from {csv_file}...")
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    try:
                        processed_row = {
                            'gpu': row['gpu'],
                            'workload': row['workload'],
                            'frequency_mhz': float(row['frequency_mhz']),
                            'duration_seconds': float(row['duration_seconds']),
                            'avg_power_watts': float(row['avg_power_watts']),
                            'total_energy_joules': float(row['total_energy_joules']),
                            'run_id': row.get('run_id', 'unknown')
                        }
                        
                        # Calculate EDP
                        processed_row['edp'] = processed_row['total_energy_joules'] * processed_row['duration_seconds']
                        
                        self.raw_data.append(processed_row)
                        
                    except (ValueError, KeyError) as e:
                        print(f"Warning: Skipping invalid row: {e}")
                        continue
            
            print(f"Loaded {len(self.raw_data)} valid data points")
            
            if len(self.raw_data) == 0:
                print("Error: No valid data points loaded")
                return False
                
            # Remove duplicates (same GPU, workload, frequency, run_id)
            seen = set()
            unique_data = []
            for row in self.raw_data:
                key = (row['gpu'], row['workload'], row['frequency_mhz'], row['run_id'])
                if key not in seen:
                    seen.add(key)
                    unique_data.append(row)
            
            self.raw_data = unique_data
            print(f"After removing duplicates: {len(self.raw_data)} unique data points")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def process_data(self):
        """Process raw data and compute frequency averages"""
        print("Processing data and computing frequency averages...")
        
        # Group data by GPU, workload, and frequency
        for row in self.raw_data:
            gpu = row['gpu']
            workload = row['workload'] 
            freq = row['frequency_mhz']
            
            self.processed_data[gpu][workload].append(row)
        
        # Compute averages for each frequency
        for gpu in self.processed_data:
            for workload in self.processed_data[gpu]:
                workload_data = self.processed_data[gpu][workload]
                
                # Group by frequency
                freq_groups = defaultdict(list)
                for row in workload_data:
                    freq_groups[row['frequency_mhz']].append(row)
                
                # Compute averages for each frequency
                for freq, freq_data in freq_groups.items():
                    if len(freq_data) >= self.min_samples:
                        durations = [row['duration_seconds'] for row in freq_data]
                        energies = [row['total_energy_joules'] for row in freq_data]
                        powers = [row['avg_power_watts'] for row in freq_data]
                        edps = [row['edp'] for row in freq_data]
                        
                        avg_duration = sum(durations) / len(durations)
                        avg_energy = sum(energies) / len(energies)
                        avg_power = sum(powers) / len(powers)
                        avg_edp = sum(edps) / len(edps)
                        
                        # Calculate standard deviations
                        duration_std = (sum((d - avg_duration)**2 for d in durations) / len(durations))**0.5 if len(durations) > 1 else 0
                        energy_std = (sum((e - avg_energy)**2 for e in energies) / len(energies))**0.5 if len(energies) > 1 else 0
                        power_std = (sum((p - avg_power)**2 for p in powers) / len(powers))**0.5 if len(powers) > 1 else 0
                        
                        self.frequency_averages[gpu][workload][freq] = FrequencyResult(
                            frequency=freq,
                            duration=avg_duration,
                            energy=avg_energy,
                            power=avg_power,
                            edp=avg_edp,
                            samples=len(freq_data),
                            duration_std=duration_std,
                            energy_std=energy_std,
                            power_std=power_std
                        )
        
        print(f"Processed data for {len(self.frequency_averages)} GPUs")
        for gpu in self.frequency_averages:
            print(f"  {gpu}: {len(self.frequency_averages[gpu])} workloads")
    
    def find_optimal_frequency(self, gpu: str, workload: str) -> Optional[OptimalResult]:
        """Find optimal frequency for specific GPU-workload combination"""
        
        if gpu not in self.frequency_averages or workload not in self.frequency_averages[gpu]:
            print(f"Warning: No data for {gpu} {workload}")
            return None
        
        freq_data = self.frequency_averages[gpu][workload]
        
        if len(freq_data) < 2:
            print(f"Warning: Insufficient data for {gpu} {workload} ({len(freq_data)} frequencies)")
            return None
        
        # Find baseline (maximum frequency)
        max_freq = max(freq_data.keys())
        baseline = freq_data[max_freq]
        
        # Calculate performance constraint threshold
        max_acceptable_duration = baseline.duration * (1 + self.performance_constraint_pct / 100)
        
        # Find frequencies that meet performance constraint
        valid_frequencies = []
        for freq, result in freq_data.items():
            if result.duration <= max_acceptable_duration:
                valid_frequencies.append((freq, result))
        
        if not valid_frequencies:
            print(f"Warning: No frequencies meet performance constraint for {gpu} {workload}")
            return None
        
        # Find optimal frequency (minimum EDP among valid frequencies)
        optimal_freq, optimal_result = min(valid_frequencies, key=lambda x: x[1].edp)
        
        # Calculate metrics
        energy_savings_pct = ((baseline.energy - optimal_result.energy) / baseline.energy) * 100
        performance_impact_pct = ((optimal_result.duration - baseline.duration) / baseline.duration) * 100
        power_savings_pct = ((baseline.power - optimal_result.power) / baseline.power) * 100
        
        # Assess data quality
        data_quality = "HIGH" if optimal_result.samples >= 3 else "MEDIUM" if optimal_result.samples >= 2 else "LOW"
        
        return OptimalResult(
            gpu=gpu,
            workload=workload,
            optimal_frequency=int(optimal_freq),
            baseline_frequency=int(max_freq),
            energy_savings_pct=energy_savings_pct,
            performance_impact_pct=performance_impact_pct,
            power_savings_pct=power_savings_pct,
            optimal_energy=optimal_result.energy,
            baseline_energy=baseline.energy,
            optimal_duration=optimal_result.duration,
            baseline_duration=baseline.duration,
            optimal_power=optimal_result.power,
            baseline_power=baseline.power,
            optimal_edp=optimal_result.edp,
            baseline_edp=baseline.edp,
            constraint_satisfied=optimal_result.duration <= max_acceptable_duration,
            data_quality=data_quality
        )
    
    def analyze_all_combinations(self) -> List[OptimalResult]:
        """Analyze all GPU-workload combinations"""
        print(f"Analyzing optimal frequencies with ≤{self.performance_constraint_pct}% performance constraint...")
        
        self.optimal_results = []
        
        for gpu in self.frequency_averages:
            for workload in self.frequency_averages[gpu]:
                result = self.find_optimal_frequency(gpu, workload)
                if result:
                    self.optimal_results.append(result)
                    print(f"  {gpu:>4} {workload:>15}: {result.optimal_frequency:>4d}MHz "
                          f"({result.energy_savings_pct:+5.1f}% energy, {result.performance_impact_pct:+5.1f}% perf)")
        
        print(f"\nSuccessfully analyzed {len(self.optimal_results)} combinations")
        return self.optimal_results
    
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary statistics"""
        if not self.optimal_results:
            return {}
        
        # Overall statistics
        energy_savings = [r.energy_savings_pct for r in self.optimal_results]
        performance_impacts = [r.performance_impact_pct for r in self.optimal_results]
        power_savings = [r.power_savings_pct for r in self.optimal_results]
        
        # GPU-specific statistics
        gpu_stats = defaultdict(lambda: {'energy_savings': [], 'performance_impacts': [], 'count': 0})
        for result in self.optimal_results:
            gpu_stats[result.gpu]['energy_savings'].append(result.energy_savings_pct)
            gpu_stats[result.gpu]['performance_impacts'].append(result.performance_impact_pct)
            gpu_stats[result.gpu]['count'] += 1
        
        # Workload-specific statistics
        workload_stats = defaultdict(lambda: {'energy_savings': [], 'performance_impacts': [], 'count': 0})
        for result in self.optimal_results:
            workload_stats[result.workload]['energy_savings'].append(result.energy_savings_pct)
            workload_stats[result.workload]['performance_impacts'].append(result.performance_impact_pct)
            workload_stats[result.workload]['count'] += 1
        
        # Best and worst cases
        best_energy = max(self.optimal_results, key=lambda r: r.energy_savings_pct)
        worst_energy = min(self.optimal_results, key=lambda r: r.energy_savings_pct)
        best_performance = min(self.optimal_results, key=lambda r: r.performance_impact_pct)
        worst_performance = max(self.optimal_results, key=lambda r: r.performance_impact_pct)
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'performance_constraint_pct': self.performance_constraint_pct,
            'total_combinations': len(self.optimal_results),
            'total_data_points': len(self.raw_data),
            
            # Overall statistics
            'overall': {
                'avg_energy_savings_pct': sum(energy_savings) / len(energy_savings),
                'max_energy_savings_pct': max(energy_savings),
                'min_energy_savings_pct': min(energy_savings),
                'avg_performance_impact_pct': sum(performance_impacts) / len(performance_impacts),
                'max_performance_impact_pct': max(performance_impacts),
                'min_performance_impact_pct': min(performance_impacts),
                'avg_power_savings_pct': sum(power_savings) / len(power_savings)
            },
            
            # GPU-specific statistics
            'by_gpu': {
                gpu: {
                    'count': stats['count'],
                    'avg_energy_savings_pct': sum(stats['energy_savings']) / len(stats['energy_savings']),
                    'avg_performance_impact_pct': sum(stats['performance_impacts']) / len(stats['performance_impacts'])
                }
                for gpu, stats in gpu_stats.items()
            },
            
            # Workload-specific statistics
            'by_workload': {
                workload: {
                    'count': stats['count'],
                    'avg_energy_savings_pct': sum(stats['energy_savings']) / len(stats['energy_savings']),
                    'avg_performance_impact_pct': sum(stats['performance_impacts']) / len(stats['performance_impacts'])
                }
                for workload, stats in workload_stats.items()
            },
            
            # Best/worst cases
            'best_cases': {
                'best_energy_savings': {
                    'gpu': best_energy.gpu,
                    'workload': best_energy.workload,
                    'frequency': best_energy.optimal_frequency,
                    'energy_savings_pct': best_energy.energy_savings_pct,
                    'performance_impact_pct': best_energy.performance_impact_pct
                },
                'worst_energy_savings': {
                    'gpu': worst_energy.gpu,
                    'workload': worst_energy.workload,
                    'frequency': worst_energy.optimal_frequency,
                    'energy_savings_pct': worst_energy.energy_savings_pct,
                    'performance_impact_pct': worst_energy.performance_impact_pct
                },
                'best_performance': {
                    'gpu': best_performance.gpu,
                    'workload': best_performance.workload,
                    'frequency': best_performance.optimal_frequency,
                    'energy_savings_pct': best_performance.energy_savings_pct,
                    'performance_impact_pct': best_performance.performance_impact_pct
                },
                'worst_performance': {
                    'gpu': worst_performance.gpu,
                    'workload': worst_performance.workload,
                    'frequency': worst_performance.optimal_frequency,
                    'energy_savings_pct': worst_performance.energy_savings_pct,
                    'performance_impact_pct': worst_performance.performance_impact_pct
                }
            }
        }
        
        return summary
    
    def print_detailed_report(self):
        """Print comprehensive analysis report"""
        if not self.optimal_results:
            print("No results to report")
            return
        
        print("\n" + "="*80)
        print("REAL DATA-DRIVEN OPTIMAL FREQUENCY ANALYSIS REPORT")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Performance Constraint: ≤{self.performance_constraint_pct}% degradation")
        print(f"Total Data Points: {len(self.raw_data)}")
        print(f"GPU-Workload Combinations: {len(self.optimal_results)}")
        
        # Summary statistics
        summary = self.generate_summary_statistics()
        overall = summary['overall']
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Average Energy Savings: {overall['avg_energy_savings_pct']:6.1f}%")
        print(f"  Energy Savings Range:   {overall['min_energy_savings_pct']:6.1f}% to {overall['max_energy_savings_pct']:6.1f}%")
        print(f"  Average Performance Impact: {overall['avg_performance_impact_pct']:6.1f}%")
        print(f"  Performance Impact Range:   {overall['min_performance_impact_pct']:6.1f}% to {overall['max_performance_impact_pct']:6.1f}%")
        print(f"  Average Power Savings:  {overall['avg_power_savings_pct']:6.1f}%")
        
        # Detailed results table
        print(f"\nDETAILED RESULTS:")
        print(f"{'GPU':<6} {'Workload':<15} {'Optimal':<8} {'Baseline':<8} {'Energy':<8} {'Perf':<8} {'Power':<8} {'Quality'}")
        print(f"{'':^6} {'':^15} {'Freq(MHz)':<8} {'Freq(MHz)':<8} {'Savings':<8} {'Impact':<8} {'Savings':<8} {''}")
        print("-"*80)
        
        for result in sorted(self.optimal_results, key=lambda r: (r.gpu, r.workload)):
            print(f"{result.gpu:<6} {result.workload:<15} {result.optimal_frequency:<8d} {result.baseline_frequency:<8d} "
                  f"{result.energy_savings_pct:>6.1f}% {result.performance_impact_pct:>6.1f}% "
                  f"{result.power_savings_pct:>6.1f}% {result.data_quality}")
        
        # GPU-specific analysis
        print(f"\nBY GPU:")
        for gpu, stats in summary['by_gpu'].items():
            print(f"  {gpu}: {stats['count']} workloads, "
                  f"{stats['avg_energy_savings_pct']:5.1f}% avg energy savings, "
                  f"{stats['avg_performance_impact_pct']:5.1f}% avg performance impact")
        
        # Workload-specific analysis
        print(f"\nBY WORKLOAD:")
        for workload, stats in summary['by_workload'].items():
            print(f"  {workload:>15}: {stats['count']} GPUs, "
                  f"{stats['avg_energy_savings_pct']:5.1f}% avg energy savings, "
                  f"{stats['avg_performance_impact_pct']:5.1f}% avg performance impact")
        
        # Best cases
        best = summary['best_cases']
        print(f"\nBEST CASES:")
        print(f"  Highest Energy Savings: {best['best_energy_savings']['gpu']} {best['best_energy_savings']['workload']} "
              f"({best['best_energy_savings']['energy_savings_pct']:.1f}% at {best['best_energy_savings']['frequency']}MHz)")
        print(f"  Lowest Performance Impact: {best['best_performance']['gpu']} {best['best_performance']['workload']} "
              f"({best['best_performance']['performance_impact_pct']:.1f}% at {best['best_performance']['frequency']}MHz)")
    
    def export_results(self, output_dir: str = "optimal_frequency_analysis"):
        """Export results to various formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed results as JSON
        results_data = []
        for result in self.optimal_results:
            results_data.append({
                'gpu': result.gpu,
                'workload': result.workload,
                'optimal_frequency_mhz': result.optimal_frequency,
                'baseline_frequency_mhz': result.baseline_frequency,
                'energy_savings_percent': round(result.energy_savings_pct, 2),
                'performance_impact_percent': round(result.performance_impact_pct, 2),
                'power_savings_percent': round(result.power_savings_pct, 2),
                'optimal_energy_joules': round(result.optimal_energy, 2),
                'baseline_energy_joules': round(result.baseline_energy, 2),
                'optimal_duration_seconds': round(result.optimal_duration, 2),
                'baseline_duration_seconds': round(result.baseline_duration, 2),
                'optimal_power_watts': round(result.optimal_power, 2),
                'baseline_power_watts': round(result.baseline_power, 2),
                'constraint_satisfied': result.constraint_satisfied,
                'data_quality': result.data_quality
            })
        
        # Save JSON results
        json_file = output_path / f"optimal_frequencies_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'performance_constraint_percent': self.performance_constraint_pct,
                    'total_combinations': len(self.optimal_results),
                    'data_source': 'real_experimental_data'
                },
                'results': results_data,
                'summary': self.generate_summary_statistics()
            }, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = output_path / f"optimal_frequencies_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'GPU', 'Workload', 'Optimal_Freq_MHz', 'Baseline_Freq_MHz',
                'Energy_Savings_Pct', 'Performance_Impact_Pct', 'Power_Savings_Pct',
                'Optimal_Energy_J', 'Baseline_Energy_J', 'Optimal_Duration_s', 'Baseline_Duration_s',
                'Optimal_Power_W', 'Baseline_Power_W', 'Data_Quality'
            ])
            
            for result in self.optimal_results:
                writer.writerow([
                    result.gpu, result.workload, result.optimal_frequency, result.baseline_frequency,
                    round(result.energy_savings_pct, 2), round(result.performance_impact_pct, 2), 
                    round(result.power_savings_pct, 2), round(result.optimal_energy, 2), 
                    round(result.baseline_energy, 2), round(result.optimal_duration, 2), 
                    round(result.baseline_duration, 2), round(result.optimal_power, 2), 
                    round(result.baseline_power, 2), result.data_quality
                ])
        
        # Save deployment-ready frequencies
        deployment_file = output_path / f"deployment_frequencies_{timestamp}.json"
        deployment_data = {}
        for result in self.optimal_results:
            if result.gpu not in deployment_data:
                deployment_data[result.gpu] = {}
            deployment_data[result.gpu][result.workload] = {
                'frequency_mhz': result.optimal_frequency,
                'energy_savings_percent': round(result.energy_savings_pct, 1),
                'performance_impact_percent': round(result.performance_impact_pct, 1),
                'nvidia_smi_command': f"sudo nvidia-smi -lgc {result.optimal_frequency}"
            }
        
        with open(deployment_file, 'w') as f:
            json.dump(deployment_data, f, indent=2)
        
        print(f"\nResults exported to {output_dir}/:")
        print(f"  Detailed results: {json_file.name}")
        print(f"  CSV analysis:     {csv_file.name}")
        print(f"  Deployment:       {deployment_file.name}")
        
        return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real data-driven optimal frequency analysis")
    parser.add_argument("--data-file", 
                       default="aggregated_results/ai_inference_aggregated_data_20250807_134913.csv",
                       help="Aggregated experimental data CSV file")
    parser.add_argument("--constraint-pct", type=float, default=5.0,
                       help="Performance degradation constraint percentage (default: 5.0)")
    parser.add_argument("--min-samples", type=int, default=2,
                       help="Minimum samples required per frequency (default: 2)")
    parser.add_argument("--output-dir", default="optimal_frequency_analysis",
                       help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RealDataOptimalFrequencyAnalyzer(
        performance_constraint_pct=args.constraint_pct,
        min_samples=args.min_samples
    )
    
    # Load and process data
    if not analyzer.load_aggregated_data(args.data_file):
        print("Failed to load data")
        return 1
    
    analyzer.process_data()
    
    # Analyze all combinations
    results = analyzer.analyze_all_combinations()
    
    if not results:
        print("No optimal frequencies found")
        return 1
    
    # Print detailed report
    if not args.quiet:
        analyzer.print_detailed_report()
    
    # Export results
    output_path = analyzer.export_results(args.output_dir)
    
    print(f"\n🎯 ANALYSIS COMPLETE")
    print(f"Found optimal frequencies for {len(results)} GPU-workload combinations")
    print(f"Average energy savings: {sum(r.energy_savings_pct for r in results)/len(results):.1f}%")
    print(f"Results saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
