#!/usr/bin/env python3

"""
Energy Delay Product (EDP) Optimizer for AI Inference Energy Project

This script calculates the Energy Delay Product (EDP) for each run of each application
on each GPU to determine optimal frequency settings that balance energy efficiency
with performance constraints.

Key Features:
- Extracts power metrics from DCGMI CSV files
- Extracts timing information from experiment_summary.log files
- Ignores cold runs (first run at each frequency)
- Aggregates remaining runs by averaging
- Calculates EDP for each frequency point
- Finds optimal frequency based on configurable performance degradation threshold
- Provides comprehensive analysis output

Author: Mert Side
Version: 1.0
"""

import re
import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse

# GPU Configuration Constants
GPU_MAX_FREQUENCIES = {
    'A100': 1410,  # MHz
    'H100': 1785,  # MHz  
    'V100': 1380   # MHz
}

@dataclass
class RunData:
    """Data structure for individual run measurements"""
    run_id: str
    frequency: int
    timing: float  # seconds
    energy: float  # Joules
    avg_power: float  # Watts
    run_number: int  # 1-5, used to identify cold runs
    
@dataclass
class FrequencyData:
    """Aggregated data for a specific frequency"""
    frequency: int
    avg_timing: float
    avg_energy: float
    avg_power: float
    edp: float  # Energy Delay Product
    run_count: int
    timing_std: float
    
@dataclass
class OptimalResult:
    """Optimal frequency analysis result"""
    gpu: str
    workload: str
    optimal_frequency: int
    max_frequency: int
    fastest_frequency: int
    energy_savings_vs_max: float  # % savings vs max frequency
    performance_vs_max: float     # % change vs max frequency (negative = faster)
    performance_vs_fastest: float # % change vs fastest frequency
    edp_improvement: float        # % improvement vs max frequency
    is_max_fastest: bool          # Whether max frequency was fastest
    optimal_timing: float
    optimal_energy: float
    optimal_edp: float
    run_count: int

class EDPOptimizer:
    """Main class for EDP optimization analysis"""
    
    def __init__(self, results_dir: str, performance_threshold: float = 5.0):
        """
        Initialize EDP optimizer
        
        Args:
            results_dir: Path to sample-collection-scripts directory
            performance_threshold: Maximum allowed performance degradation (%)
        """
        self.results_dir = Path(results_dir)
        self.performance_threshold = performance_threshold
        self.all_results = []
        
    def parse_directory_name(self, dir_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse results directory name to extract GPU, workload, and job ID"""
        match = re.match(r'results_([^_]+)_([^_]+)_job_(\d+)', dir_name)
        if match:
            return match.group(1).upper(), match.group(2), match.group(3)
        return None, None, None
    
    def extract_timing_from_summary(self, summary_file: Path) -> Dict[str, Tuple[float, int]]:
        """
        Extract timing data from experiment_summary.log
        
        Returns:
            Dict mapping run_id to (timing_seconds, frequency_mhz)
        """
        timing_data = {}
        
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            
            # Find timing summary section
            timing_section = False
            for line in content.split('\n'):
                line = line.strip()
                
                if 'Run Timing Summary' in line:
                    timing_section = True
                    continue
                
                if timing_section and line:
                    # Parse timing entries: "Run 1_01 : 114s (freq: 1410MHz, status: success)"
                    match = re.match(r'Run\s+(\S+)\s*:\s*(\d+)s\s*\(freq:\s*(\d+)MHz', line)
                    if match:
                        run_id = match.group(1)
                        timing = float(match.group(2))
                        frequency = int(match.group(3))
                        timing_data[run_id] = (timing, frequency)
                        
        except Exception as e:
            print(f"Warning: Error reading timing summary: {e}")
            
        return timing_data
    
    def extract_energy_from_csv(self, csv_file: Path) -> float:
        """
        Extract energy consumption from DCGMI CSV profile
        
        Returns:
            Total energy in Joules
        """
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and find data
            power_values = []
            for line in lines:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line or 'GPU' not in line:
                    continue
                
                # Split by whitespace and find power value
                parts = line.split()
                if len(parts) >= 6:  # Ensure we have enough columns
                    # Look for the power value - it should be a float after the GPU name
                    for i in range(3, min(len(parts), 10)):  # Search in reasonable range
                        try:
                            power = float(parts[i])
                            if power > 0 and power < 1000:  # Reasonable power range for GPUs
                                power_values.append(power)
                                break
                        except ValueError:
                            continue
                
            if power_values:
                avg_power = statistics.mean(power_values)
                # Assuming 50ms sampling interval (20 Hz)
                duration = len(power_values) * 0.05  # seconds
                energy = avg_power * duration  # Joules
                return energy
                    
        except Exception as e:
            print(f"Warning: Error reading CSV file {csv_file}: {e}")
            
        return 0.0
    
    def extract_avg_power_from_csv(self, csv_file: Path) -> float:
        """Extract average power consumption from DCGMI CSV profile"""
        try:
            with open(csv_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and find data
            power_values = []
            for line in lines:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line or 'GPU' not in line:
                    continue
                
                # Split by whitespace and find power value
                parts = line.split()
                if len(parts) >= 6:  # Ensure we have enough columns
                    # Look for the power value - it should be a float after the GPU name
                    for i in range(3, min(len(parts), 10)):  # Search in reasonable range
                        try:
                            power = float(parts[i])
                            if power > 0 and power < 1000:  # Reasonable power range for GPUs
                                power_values.append(power)
                                break
                        except ValueError:
                            continue
                
            if power_values:
                return statistics.mean(power_values)
                    
        except Exception as e:
            print(f"Warning: Error reading CSV file {csv_file}: {e}")
            
        return 0.0
    
    def process_result_directory(self, result_dir: Path) -> List[RunData]:
        """Process a single results directory and extract all run data"""
        gpu, workload, job_id = self.parse_directory_name(result_dir.name)
        if not gpu or not workload:
            return []
        
        print(f"Processing {gpu} {workload} (Job {job_id})...")
        
        # Extract timing data from experiment summary
        summary_file = result_dir / 'experiment_summary.log'
        timing_data = self.extract_timing_from_summary(summary_file)
        print(f"  Found {len(timing_data)} timing entries")
        
        runs = []
        
        # Process each run file
        csv_files = list(result_dir.glob('run_*_profile.csv'))
        print(f"  Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            # Extract run information from filename
            match = re.match(r'run_(\d+)_(\d+)_freq_(\d+)_profile\.csv', csv_file.name)
            if not match:
                print(f"    Skipping {csv_file.name} - filename doesn't match pattern")
                continue
                
            run_num = int(match.group(1))
            freq_index = int(match.group(2))
            frequency = int(match.group(3))
            
            # Create run ID for timing lookup
            run_id = f"{run_num}_{freq_index:02d}"
            
            # Get timing from summary (prefer this source)
            if run_id in timing_data:
                timing, freq_check = timing_data[run_id]
                if freq_check != frequency:
                    print(f"Warning: Frequency mismatch for {run_id}: {frequency} vs {freq_check}")
            else:
                print(f"Warning: No timing data found for {run_id}")
                continue
            
            # Extract energy from CSV
            energy = self.extract_energy_from_csv(csv_file)
            avg_power = self.extract_avg_power_from_csv(csv_file)
            
            if timing > 0 and energy > 0:
                run_data = RunData(
                    run_id=run_id,
                    frequency=frequency,
                    timing=timing,
                    energy=energy,
                    avg_power=avg_power,
                    run_number=freq_index
                )
                runs.append(run_data)
            else:
                print(f"    Skipping {run_id} - invalid data: timing={timing}, energy={energy}")
        
        print(f"  Extracted {len(runs)} valid runs")
        return runs
    
    def aggregate_by_frequency(self, runs: List[RunData]) -> List[FrequencyData]:
        """
        Aggregate runs by frequency, excluding cold runs (first run at each frequency)
        """
        freq_groups = defaultdict(list)
        
        # Group runs by frequency
        for run in runs:
            freq_groups[run.frequency].append(run)
        
        aggregated = []
        
        for frequency, freq_runs in freq_groups.items():
            # Sort by run number and exclude first run (cold run)
            freq_runs.sort(key=lambda x: x.run_number)
            warm_runs = [r for r in freq_runs if r.run_number > 1]
            
            if len(warm_runs) < 1:
                print(f"Warning: No warm runs for frequency {frequency} MHz")
                continue
            
            # Calculate averages
            timings = [r.timing for r in warm_runs]
            energies = [r.energy for r in warm_runs]
            powers = [r.avg_power for r in warm_runs]
            
            avg_timing = statistics.mean(timings)
            avg_energy = statistics.mean(energies)
            avg_power = statistics.mean(powers)
            timing_std = statistics.stdev(timings) if len(timings) > 1 else 0.0
            
            # Calculate EDP (Energy Delay Product)
            edp = avg_energy * avg_timing
            
            freq_data = FrequencyData(
                frequency=frequency,
                avg_timing=avg_timing,
                avg_energy=avg_energy,
                avg_power=avg_power,
                edp=edp,
                run_count=len(warm_runs),
                timing_std=timing_std
            )
            aggregated.append(freq_data)
        
        return sorted(aggregated, key=lambda x: x.frequency)
    
    def find_optimal_frequency(self, freq_data: List[FrequencyData], gpu: str, workload: str) -> OptimalResult:
        """
        Find optimal frequency based on EDP optimization with performance constraints
        """
        if not freq_data:
            raise ValueError("No frequency data available")
        
        max_frequency = GPU_MAX_FREQUENCIES.get(gpu, max(f.frequency for f in freq_data))
        
        # Find max frequency data point
        max_freq_data = next((f for f in freq_data if f.frequency == max_frequency), None)
        if not max_freq_data:
            # Use highest available frequency as baseline
            max_freq_data = max(freq_data, key=lambda x: x.frequency)
            max_frequency = max_freq_data.frequency
        
        # Find fastest execution (minimum timing)
        fastest_data = min(freq_data, key=lambda x: x.avg_timing)
        fastest_frequency = fastest_data.frequency
        
        # Apply performance constraint relative to fastest execution
        max_allowed_timing = fastest_data.avg_timing * (1 + self.performance_threshold / 100)
        
        # Filter frequencies that meet performance constraint
        valid_frequencies = [f for f in freq_data if f.avg_timing <= max_allowed_timing]
        
        if not valid_frequencies:
            print(f"Warning: No frequencies meet {self.performance_threshold}% performance constraint")
            valid_frequencies = freq_data
        
        # Find optimal frequency (minimum EDP among valid frequencies)
        optimal_data = min(valid_frequencies, key=lambda x: x.edp)
        
        # Calculate metrics
        energy_savings = ((max_freq_data.avg_energy - optimal_data.avg_energy) / max_freq_data.avg_energy) * 100
        performance_vs_max = ((optimal_data.avg_timing - max_freq_data.avg_timing) / max_freq_data.avg_timing) * 100
        performance_vs_fastest = ((optimal_data.avg_timing - fastest_data.avg_timing) / fastest_data.avg_timing) * 100
        edp_improvement = ((max_freq_data.edp - optimal_data.edp) / max_freq_data.edp) * 100
        
        return OptimalResult(
            gpu=gpu,
            workload=workload,
            optimal_frequency=optimal_data.frequency,
            max_frequency=max_frequency,
            fastest_frequency=fastest_frequency,
            energy_savings_vs_max=energy_savings,
            performance_vs_max=performance_vs_max,
            performance_vs_fastest=performance_vs_fastest,
            edp_improvement=edp_improvement,
            is_max_fastest=(fastest_frequency == max_frequency),
            optimal_timing=optimal_data.avg_timing,
            optimal_energy=optimal_data.avg_energy,
            optimal_edp=optimal_data.edp,
            run_count=optimal_data.run_count
        )
    
    def analyze_all_results(self) -> List[OptimalResult]:
        """Process all result directories and find optimal frequencies"""
        results = []
        
        # Find all results directories
        result_dirs = [d for d in self.results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('results_')]
        
        print(f"Found {len(result_dirs)} result directories")
        
        for result_dir in sorted(result_dirs):
            gpu, workload, job_id = self.parse_directory_name(result_dir.name)
            if not gpu or not workload:
                continue
                
            try:
                # Extract run data
                runs = self.process_result_directory(result_dir)
                if not runs:
                    print(f"No valid runs found for {gpu} {workload}")
                    continue
                
                # Aggregate by frequency
                freq_data = self.aggregate_by_frequency(runs)
                if not freq_data:
                    print(f"No frequency data after aggregation for {gpu} {workload}")
                    continue
                
                # Find optimal frequency
                optimal = self.find_optimal_frequency(freq_data, gpu, workload)
                results.append(optimal)
                
            except Exception as e:
                print(f"Error processing {result_dir.name}: {e}")
                continue
        
        self.all_results = results
        return results
    
    def print_summary(self, results: List[OptimalResult]):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("EDP OPTIMIZATION ANALYSIS SUMMARY")
        print("="*80)
        
        # Group by GPU
        gpu_groups = defaultdict(list)
        for result in results:
            gpu_groups[result.gpu].append(result)
        
        for gpu in sorted(gpu_groups.keys()):
            gpu_results = gpu_groups[gpu]
            print(f"\n🔧 {gpu} GPU ANALYSIS:")
            print("-" * 50)
            
            total_energy_savings = 0
            max_faster_count = 0
            
            for result in sorted(gpu_results, key=lambda x: x.workload):
                print(f"\n  🎯 {result.workload}:")
                print(f"    • Optimal frequency: {result.optimal_frequency} MHz")
                print(f"    • Max frequency: {result.max_frequency} MHz")
                print(f"    • Fastest frequency: {result.fastest_frequency} MHz")
                print(f"    • Energy savings vs max: {result.energy_savings_vs_max:.1f}%")
                
                if result.performance_vs_max < 0:
                    print(f"    • Performance vs max: {abs(result.performance_vs_max):.1f}% FASTER")
                    max_faster_count += 1
                else:
                    print(f"    • Performance vs max: {result.performance_vs_max:.1f}% slower")
                
                print(f"    • Performance vs fastest: {result.performance_vs_fastest:.1f}% slower")
                print(f"    • EDP improvement: {result.edp_improvement:.1f}%")
                print(f"    • Max freq is fastest: {'Yes' if result.is_max_fastest else 'No'}")
                print(f"    • Runs averaged: {result.run_count}")
                
                total_energy_savings += result.energy_savings_vs_max
            
            avg_savings = total_energy_savings / len(gpu_results)
            print(f"\n  📊 {gpu} Summary:")
            print(f"    • Average energy savings: {avg_savings:.1f}%")
            print(f"    • Configurations faster than max freq: {max_faster_count}/{len(gpu_results)}")
        
        # Overall statistics
        print(f"\n📈 OVERALL STATISTICS:")
        print("-" * 50)
        avg_energy_savings = statistics.mean([r.energy_savings_vs_max for r in results])
        avg_edp_improvement = statistics.mean([r.edp_improvement for r in results])
        faster_than_max = sum(1 for r in results if r.performance_vs_max < 0)
        
        print(f"  • Total configurations analyzed: {len(results)}")
        print(f"  • Average energy savings: {avg_energy_savings:.1f}%")
        print(f"  • Average EDP improvement: {avg_edp_improvement:.1f}%")
        print(f"  • Configurations faster than max freq: {faster_than_max}/{len(results)} ({100*faster_than_max/len(results):.1f}%)")
        print(f"  • Performance threshold used: {self.performance_threshold}%")
    
    def export_results(self, results: List[OptimalResult], output_file: str):
        """Export results to JSON file"""
        export_data = []
        
        for result in results:
            export_data.append({
                'gpu': result.gpu,
                'workload': result.workload,
                'optimal_frequency_mhz': result.optimal_frequency,
                'max_frequency_mhz': result.max_frequency,
                'fastest_frequency_mhz': result.fastest_frequency,
                'energy_savings_percent': round(result.energy_savings_vs_max, 2),
                'performance_vs_max_percent': round(result.performance_vs_max, 2),
                'performance_vs_fastest_percent': round(result.performance_vs_fastest, 2),
                'edp_improvement_percent': round(result.edp_improvement, 2),
                'is_max_frequency_fastest': result.is_max_fastest,
                'optimal_timing_seconds': round(result.optimal_timing, 2),
                'optimal_energy_joules': round(result.optimal_energy, 2),
                'optimal_edp': round(result.optimal_edp, 2),
                'runs_averaged': result.run_count
            })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nResults exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='EDP Optimizer for AI Inference Energy Project')
    parser.add_argument('--results-dir', '-r', 
                       default='../sample-collection-scripts',
                       help='Path to sample-collection-scripts directory')
    parser.add_argument('--performance-threshold', '-t', 
                       type=float, default=5.0,
                       help='Maximum allowed performance degradation (%) [default: 5.0]')
    parser.add_argument('--output', '-o',
                       default='edp_optimization_results.json',
                       help='Output file for results [default: edp_optimization_results.json]')
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    print(f"EDP Optimizer v1.0")
    print(f"Results directory: {args.results_dir}")
    print(f"Performance threshold: {args.performance_threshold}%")
    print("-" * 60)
    
    # Initialize optimizer
    optimizer = EDPOptimizer(args.results_dir, args.performance_threshold)
    
    # Analyze all results
    results = optimizer.analyze_all_results()
    
    if not results:
        print("No valid results found!")
        return 1
    
    # Print summary
    if not args.quiet:
        optimizer.print_summary(results)
    
    # Export results
    optimizer.export_results(results, args.output)
    
    return 0

if __name__ == '__main__':
    exit(main())
