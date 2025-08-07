#!/usr/bin/env python
"""
Real Data-Driven Optimal Frequency Selection

This script implements optimal frequency selection using your actual collected DVFS data
from the comprehensive experiments. It processes real data to find optimal frequencies
with performance constraints.

Author: Mert Side
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. Using direct optimization only.")
    SKLEARN_AVAILABLE = False

class RealDataOptimalFrequencySelector:
    """
    Real data-driven optimal frequency selection using your collected DVFS data
    """
    
    def __init__(self, constraint_pct: float = 5.0):
        self.constraint_pct = constraint_pct
        self.df = None
        self.baselines = {}
        
    def load_aggregated_data(self, csv_file: str) -> bool:
        """Load aggregated data from CSV file"""
        try:
            self.df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(self.df)} data points from {csv_file}")
            
            # Validate required columns
            required_columns = ['gpu', 'workload', 'frequency_mhz', 'duration_seconds', 
                              'avg_power_watts', 'total_energy_joules']
            missing = [col for col in required_columns if col not in self.df.columns]
            
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return False
                
            # Calculate EDP and ED²P
            self.df['edp'] = self.df['total_energy_joules'] * self.df['duration_seconds']
            self.df['ed2p'] = self.df['total_energy_joules'] * (self.df['duration_seconds'] ** 2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
            
    def establish_baselines(self):
        """Establish performance baselines for each GPU-workload combination"""
        self.baselines = {}
        
        for gpu in self.df['gpu'].unique():
            for workload in self.df['workload'].unique():
                subset = self.df[(self.df['gpu'] == gpu) & (self.df['workload'] == workload)]
                
                if subset.empty:
                    continue
                    
                # Find maximum frequency (baseline performance)
                max_freq = subset['frequency_mhz'].max()
                baseline_data = subset[subset['frequency_mhz'] == max_freq]
                
                if baseline_data.empty:
                    continue
                    
                # Use mean values across runs at max frequency
                baseline_time = baseline_data['duration_seconds'].mean()
                baseline_energy = baseline_data['total_energy_joules'].mean()
                baseline_power = baseline_data['avg_power_watts'].mean()
                
                # Performance constraint (5% degradation)
                max_acceptable_time = baseline_time * (1 + self.constraint_pct / 100.0)
                
                self.baselines[f"{gpu}_{workload}"] = {
                    'gpu': gpu,
                    'workload': workload,
                    'baseline_frequency_mhz': max_freq,
                    'baseline_time_seconds': baseline_time,
                    'baseline_energy_joules': baseline_energy,
                    'baseline_power_watts': baseline_power,
                    'max_acceptable_time_seconds': max_acceptable_time,
                    'constraint_pct': self.constraint_pct
                }
                
        logger.info(f"Established baselines for {len(self.baselines)} GPU-workload combinations")
        
    def find_optimal_frequency(self, gpu: str, workload: str, method: str = 'edp') -> Dict:
        """Find optimal frequency for specific GPU-workload combination"""
        
        # Get data subset
        subset = self.df[(self.df['gpu'] == gpu) & (self.df['workload'] == workload)]
        
        if subset.empty:
            logger.error(f"No data found for {gpu} {workload}")
            return None
            
        # Get baseline for this combination
        baseline_key = f"{gpu}_{workload}"
        if baseline_key not in self.baselines:
            logger.error(f"No baseline established for {gpu} {workload}")
            return None
            
        baseline = self.baselines[baseline_key]
        max_acceptable_time = baseline['max_acceptable_time_seconds']
        
        # Apply performance constraint
        feasible_subset = subset[subset['duration_seconds'] <= max_acceptable_time]
        
        if feasible_subset.empty:
            logger.warning(f"No feasible solutions for {gpu} {workload} within {self.constraint_pct}% constraint")
            logger.warning(f"Using all data points (relaxing constraint)")
            feasible_subset = subset
            
        # Find optimal based on method
        if method == 'edp':
            optimal_idx = feasible_subset['edp'].idxmin()
        elif method == 'ed2p':
            optimal_idx = feasible_subset['ed2p'].idxmin()
        elif method == 'energy':
            optimal_idx = feasible_subset['total_energy_joules'].idxmin()
        elif method == 'power':
            optimal_idx = feasible_subset['avg_power_watts'].idxmin()
        else:
            logger.error(f"Unknown optimization method: {method}")
            return None
            
        optimal_run = feasible_subset.loc[optimal_idx]
        
        # Calculate performance metrics
        time_ratio = optimal_run['duration_seconds'] / baseline['baseline_time_seconds']
        energy_ratio = optimal_run['total_energy_joules'] / baseline['baseline_energy_joules']
        power_ratio = optimal_run['avg_power_watts'] / baseline['baseline_power_watts']
        
        energy_savings_pct = (1 - energy_ratio) * 100
        power_savings_pct = (1 - power_ratio) * 100
        performance_impact_pct = (time_ratio - 1) * 100
        
        result = {
            'gpu': gpu,
            'workload': workload,
            'method': method,
            
            # Optimal configuration
            'optimal_frequency_mhz': int(optimal_run['frequency_mhz']),
            'optimal_power_watts': optimal_run['avg_power_watts'],
            'optimal_time_seconds': optimal_run['duration_seconds'],
            'optimal_energy_joules': optimal_run['total_energy_joules'],
            'optimal_edp': optimal_run['edp'],
            'optimal_ed2p': optimal_run['ed2p'],
            
            # Baseline comparison
            'baseline_frequency_mhz': int(baseline['baseline_frequency_mhz']),
            'baseline_power_watts': baseline['baseline_power_watts'],
            'baseline_time_seconds': baseline['baseline_time_seconds'],
            'baseline_energy_joules': baseline['baseline_energy_joules'],
            
            # Performance metrics
            'energy_savings_pct': energy_savings_pct,
            'power_savings_pct': power_savings_pct,
            'performance_impact_pct': performance_impact_pct,
            
            # Constraint satisfaction
            'within_constraint': optimal_run['duration_seconds'] <= max_acceptable_time,
            'constraint_pct': self.constraint_pct,
            'feasible_configurations': len(feasible_subset),
            'total_configurations': len(subset)
        }
        
        logger.info(f"Optimal for {gpu} {workload}: {optimal_run['frequency_mhz']:.0f}MHz, "
                   f"{energy_savings_pct:.1f}% energy savings, {performance_impact_pct:.1f}% perf impact")
        
        return result
        
    def find_all_optimal_frequencies(self, method: str = 'edp') -> List[Dict]:
        """Find optimal frequencies for all GPU-workload combinations"""
        results = []
        
        combinations = self.df.groupby(['gpu', 'workload']).size().index
        
        for gpu, workload in combinations:
            result = self.find_optimal_frequency(gpu, workload, method)
            if result:
                results.append(result)
                
        return results
        
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not results:
            return {}
            
        summary = {
            'total_combinations': len(results),
            'method': results[0]['method'] if results else 'unknown',
            'constraint_pct': self.constraint_pct,
            'timestamp': datetime.now().isoformat(),
            
            # Energy savings statistics
            'avg_energy_savings_pct': np.mean([r['energy_savings_pct'] for r in results]),
            'max_energy_savings_pct': np.max([r['energy_savings_pct'] for r in results]),
            'min_energy_savings_pct': np.min([r['energy_savings_pct'] for r in results]),
            
            # Performance impact statistics
            'avg_performance_impact_pct': np.mean([r['performance_impact_pct'] for r in results]),
            'max_performance_impact_pct': np.max([r['performance_impact_pct'] for r in results]),
            'min_performance_impact_pct': np.min([r['performance_impact_pct'] for r in results]),
            
            # Constraint satisfaction
            'within_constraint_count': sum(1 for r in results if r['within_constraint']),
            'constraint_satisfaction_pct': sum(1 for r in results if r['within_constraint']) / len(results) * 100,
            
            # Frequency statistics
            'optimal_frequencies': {
                'V100': [r['optimal_frequency_mhz'] for r in results if r['gpu'] == 'V100'],
                'A100': [r['optimal_frequency_mhz'] for r in results if r['gpu'] == 'A100'],
                'H100': [r['optimal_frequency_mhz'] for r in results if r['gpu'] == 'H100']
            }
        }
        
        return summary
        
    def save_results(self, results: List[Dict], output_dir: str = "optimal_frequency_results"):
        """Save optimal frequency results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"optimal_frequencies_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed results: {results_file}")
        
        # Save summary
        summary = self.generate_summary(results)
        summary_file = output_path / f"optimal_frequency_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_file}")
        
        # Create CSV for easy analysis
        results_df = pd.DataFrame(results)
        csv_file = output_path / f"optimal_frequencies_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV: {csv_file}")
        
        # Generate human-readable report
        report_file = output_path / f"optimal_frequency_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("Real Data-Driven Optimal Frequency Selection Report\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Method: {summary['method'].upper()}\n")
            f.write(f"Performance constraint: ≤{summary['constraint_pct']}% degradation\n\n")
            
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
                
        logger.info(f"Saved report: {report_file}")
        
        return results_file, summary_file, csv_file, report_file

def main():
    parser = argparse.ArgumentParser(description="Real data-driven optimal frequency selection")
    parser.add_argument("--data-file", required=True,
                       help="Aggregated data CSV file")
    parser.add_argument("--method", default="edp", choices=['edp', 'ed2p', 'energy', 'power'],
                       help="Optimization method")
    parser.add_argument("--constraint-pct", type=float, default=5.0,
                       help="Performance degradation constraint percentage")
    parser.add_argument("--output-dir", default="optimal_frequency_results",
                       help="Output directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize selector
    selector = RealDataOptimalFrequencySelector(args.constraint_pct)
    
    # Load data
    if not selector.load_aggregated_data(args.data_file):
        logger.error("Failed to load data")
        return 1
        
    # Establish baselines
    selector.establish_baselines()
    
    # Find optimal frequencies
    results = selector.find_all_optimal_frequencies(args.method)
    
    if not results:
        logger.error("No optimal frequencies found")
        return 1
        
    # Save results
    files = selector.save_results(results, args.output_dir)
    
    # Print summary
    summary = selector.generate_summary(results)
    print(f"\n📊 OPTIMAL FREQUENCY SELECTION COMPLETED")
    print(f"=" * 50)
    print(f"Method: {args.method.upper()}")
    print(f"Constraint: ≤{args.constraint_pct}% performance degradation")
    print(f"Combinations analyzed: {summary['total_combinations']}")
    print(f"Average energy savings: {summary['avg_energy_savings_pct']:.1f}%")
    print(f"Average performance impact: {summary['avg_performance_impact_pct']:.1f}%")
    print(f"Constraint satisfaction: {summary['constraint_satisfaction_pct']:.1f}%")
    print(f"\nResults saved to: {args.output_dir}/")
    
    return 0

if __name__ == "__main__":
    exit(main())
