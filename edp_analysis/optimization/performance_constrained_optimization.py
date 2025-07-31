#!/usr/bin/env python3
"""
Performance-Constrained Frequency Optimization

This module performs frequency optimization with strict performance constraints
to ensure practical deployment of energy-efficient AI inference settings.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workload_constraints import (
    get_workload_constraints, 
    get_gpu_specifications,
    get_practical_frequency_range,
    classify_performance_penalty,
    validate_constraints
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceConstrainedOptimizer:
    """
    Optimizer that finds energy-efficient frequencies while respecting 
    strict performance constraints for each workload.
    """
    
    def __init__(self, data_path: str):
        """Initialize the optimizer with aggregated profiling data."""
        self.data_path = data_path
        self.data = None
        self.results = {}
        
        # Load the aggregated data
        self.load_data()
    
    def load_data(self):
        """Load aggregated profiling data."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.data)} configuration records")
            
            # Validate required columns
            required_cols = ['gpu', 'application', 'frequency', 'avg_power', 
                           'execution_time', 'edp', 'ed2p']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Map execution_time to total_time for consistency
            if 'execution_time' in self.data.columns and 'total_time' not in self.data.columns:
                self.data['total_time'] = self.data['execution_time']
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_baseline_performance(self, gpu: str, app: str) -> Dict[str, float]:
        """Get baseline performance metrics at maximum frequency."""
        gpu_data = self.data[(self.data['gpu'] == gpu) & 
                           (self.data['application'] == app)]
        
        if gpu_data.empty:
            raise ValueError(f"No data found for {gpu}+{app}")
        
        # Find maximum frequency (baseline)
        max_freq_row = gpu_data.loc[gpu_data['frequency'].idxmax()]
        
        baseline = {
            'frequency': max_freq_row['frequency'],
            'avg_power': max_freq_row['avg_power'],
            'total_time': max_freq_row['total_time'],
            'edp': max_freq_row['edp'],
            'ed2p': max_freq_row['ed2p']
        }
        
        logger.info(f"Baseline for {gpu}+{app}: {baseline['frequency']}MHz, "
                   f"Time: {baseline['total_time']:.2f}s, Power: {baseline['avg_power']:.2f}W")
        
        return baseline
    
    def calculate_performance_penalty(self, baseline_time: float, current_time: float) -> float:
        """Calculate performance penalty as relative slowdown."""
        return (current_time - baseline_time) / baseline_time
    
    def find_constrained_optimum(self, gpu: str, app: str) -> Dict[str, any]:
        """Find optimal frequency respecting performance constraints."""
        try:
            # Get constraints and baseline
            constraints = get_workload_constraints(app)
            baseline = self.get_baseline_performance(gpu, app)
            min_freq, max_freq = get_practical_frequency_range(gpu, app)
            
            # Filter data for this GPU+app combination
            gpu_data = self.data[(self.data['gpu'] == gpu) & 
                               (self.data['application'] == app)].copy()
            
            # Calculate performance penalties
            gpu_data['performance_penalty'] = gpu_data['total_time'].apply(
                lambda t: self.calculate_performance_penalty(baseline['total_time'], t)
            )
            
            # Apply performance constraint
            max_penalty = constraints['max_penalty']
            feasible_data = gpu_data[gpu_data['performance_penalty'] <= max_penalty]
            
            # Apply frequency constraints
            feasible_data = feasible_data[
                (feasible_data['frequency'] >= min_freq) & 
                (feasible_data['frequency'] <= max_freq)
            ]
            
            if feasible_data.empty:
                logger.warning(f"No feasible solutions for {gpu}+{app} with {max_penalty*100:.1f}% penalty limit")
                return self._create_fallback_solution(gpu, app, baseline, constraints)
            
            # Find optimal solution based on optimization metric
            optimal_solution = self._select_optimal_solution(feasible_data, constraints, baseline)
            
            return optimal_solution
            
        except Exception as e:
            logger.error(f"Error optimizing {gpu}+{app}: {e}")
            return self._create_error_solution(gpu, app, baseline)
    
    def _select_optimal_solution(self, feasible_data: pd.DataFrame, 
                               constraints: Dict, baseline: Dict) -> Dict:
        """Select the optimal solution based on optimization metric."""
        metric = constraints['optimization_metric']
        priority = constraints['priority']
        
        if priority == "performance":
            # Minimize performance penalty, then energy
            optimal_row = feasible_data.loc[feasible_data['performance_penalty'].idxmin()]
        elif priority == "energy":
            # Minimize energy (EDP), respecting performance constraint
            optimal_row = feasible_data.loc[feasible_data['edp'].idxmin()]
        else:  # balanced
            # Balance energy and performance using ED²P
            optimal_row = feasible_data.loc[feasible_data['ed2p'].idxmin()]
        
        # Calculate savings
        energy_savings = (baseline['edp'] - optimal_row['edp']) / baseline['edp']
        power_savings = (baseline['avg_power'] - optimal_row['avg_power']) / baseline['avg_power']
        
        solution = {
            'gpu': optimal_row['gpu'],
            'application': optimal_row['application'],
            'optimal_frequency': optimal_row['frequency'],
            'baseline_frequency': baseline['frequency'],
            'frequency_reduction': (baseline['frequency'] - optimal_row['frequency']) / baseline['frequency'],
            'performance_penalty': optimal_row['performance_penalty'],
            'penalty_category': classify_performance_penalty(optimal_row['performance_penalty']),
            'energy_savings': energy_savings,
            'power_savings': power_savings,
            'baseline_time': baseline['total_time'],
            'optimal_time': optimal_row['total_time'],
            'baseline_power': baseline['avg_power'],
            'optimal_power': optimal_row['avg_power'],
            'baseline_edp': baseline['edp'],
            'optimal_edp': optimal_row['edp'],
            'optimization_metric': constraints['optimization_metric'],
            'priority': constraints['priority'],
            'constraints_met': True,
            'feasible_count': len(feasible_data)
        }
        
        return solution
    
    def _create_fallback_solution(self, gpu: str, app: str, baseline: Dict, 
                                constraints: Dict) -> Dict:
        """Create a fallback solution when no feasible solutions exist."""
        # Use the minimum allowed frequency
        min_freq, max_freq = get_practical_frequency_range(gpu, app)
        
        # Find the highest frequency that might work
        gpu_data = self.data[(self.data['gpu'] == gpu) & 
                           (self.data['application'] == app)]
        
        # Try progressively higher frequencies
        fallback_freq = max_freq
        fallback_row = gpu_data[gpu_data['frequency'] == fallback_freq].iloc[0]
        
        solution = {
            'gpu': gpu,
            'application': app,
            'optimal_frequency': fallback_freq,
            'baseline_frequency': baseline['frequency'],
            'frequency_reduction': 0.0,  # No reduction possible
            'performance_penalty': 0.0,  # Using max frequency
            'penalty_category': 'acceptable',
            'energy_savings': 0.0,  # No savings
            'power_savings': 0.0,
            'baseline_time': baseline['total_time'],
            'optimal_time': baseline['total_time'],
            'baseline_power': baseline['avg_power'],
            'optimal_power': baseline['avg_power'],
            'baseline_edp': baseline['edp'],
            'optimal_edp': baseline['edp'],
            'optimization_metric': constraints['optimization_metric'],
            'priority': constraints['priority'],
            'constraints_met': False,
            'feasible_count': 0,
            'fallback_reason': f"No solutions within {constraints['max_penalty']*100:.1f}% penalty limit"
        }
        
        logger.warning(f"Using fallback solution for {gpu}+{app}")
        return solution
    
    def _create_error_solution(self, gpu: str, app: str, baseline: Dict) -> Dict:
        """Create an error solution when optimization fails."""
        return {
            'gpu': gpu,
            'application': app,
            'optimal_frequency': baseline['frequency'],
            'baseline_frequency': baseline['frequency'],
            'error': True,
            'constraints_met': False
        }
    
    def optimize_all_workloads(self) -> Dict[str, Dict]:
        """Optimize all GPU+application combinations."""
        logger.info("Starting performance-constrained optimization")
        
        # Get unique GPU+application combinations
        combinations = self.data[['gpu', 'application']].drop_duplicates()
        
        results = {}
        total_combinations = len(combinations)
        
        for idx, (_, row) in enumerate(combinations.iterrows(), 1):
            gpu, app = row['gpu'], row['application']
            logger.info(f"Optimizing {idx}/{total_combinations}: {gpu}+{app}")
            
            # Validate constraints first
            if not validate_constraints(app, gpu):
                logger.error(f"Invalid constraints for {gpu}+{app}, skipping")
                continue
            
            # Perform optimization
            result = self.find_constrained_optimum(gpu, app)
            results[f"{gpu}+{app}"] = result
            
            # Log result summary
            if result.get('constraints_met', False):
                logger.info(f"  ✅ Optimal: {result['optimal_frequency']}MHz "
                          f"({result['performance_penalty']*100:.1f}% penalty, "
                          f"{result['energy_savings']*100:.1f}% energy savings)")
            else:
                logger.warning(f"  ❌ Fallback: {result.get('fallback_reason', 'Error')}")
        
        self.results = results
        return results
    
    def save_results(self, output_path: str):
        """Save optimization results to JSON file."""
        try:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = {}
            for key, result in self.results.items():
                serializable_result = {}
                for k, v in result.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serializable_result[k] = v.item()
                    else:
                        serializable_result[k] = v
                serializable_results[key] = serializable_result
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of optimization results."""
        if not self.results:
            return "No optimization results available."
        
        report = []
        report.append("🎯 Performance-Constrained Optimization Summary")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        total_configs = len(self.results)
        successful_configs = sum(1 for r in self.results.values() if r.get('constraints_met', False))
        
        report.append(f"📊 Overall Results:")
        report.append(f"  Total Configurations: {total_configs}")
        report.append(f"  Successful Optimizations: {successful_configs} ({successful_configs/total_configs*100:.1f}%)")
        report.append(f"  Fallback Solutions: {total_configs - successful_configs}")
        report.append("")
        
        # Performance penalty distribution
        penalties = [r['performance_penalty'] for r in self.results.values() 
                    if r.get('constraints_met', False)]
        if penalties:
            report.append(f"⚡ Performance Impact:")
            report.append(f"  Average Penalty: {np.mean(penalties)*100:.1f}%")
            report.append(f"  Max Penalty: {np.max(penalties)*100:.1f}%")
            report.append(f"  Penalty Range: {np.min(penalties)*100:.1f}% - {np.max(penalties)*100:.1f}%")
            report.append("")
        
        # Energy savings distribution
        savings = [r['energy_savings'] for r in self.results.values() 
                  if r.get('constraints_met', False) and r['energy_savings'] > 0]
        if savings:
            report.append(f"🔋 Energy Savings:")
            report.append(f"  Average Energy Savings: {np.mean(savings)*100:.1f}%")
            report.append(f"  Max Energy Savings: {np.max(savings)*100:.1f}%")
            report.append(f"  Savings Range: {np.min(savings)*100:.1f}% - {np.max(savings)*100:.1f}%")
            report.append("")
        
        # Individual results
        report.append("🔍 Individual Results:")
        report.append("-" * 40)
        
        for config_name, result in self.results.items():
            if result.get('constraints_met', False):
                report.append(f"✅ {config_name}:")
                report.append(f"   {result['baseline_frequency']} → {result['optimal_frequency']} MHz")
                report.append(f"   Performance: -{result['performance_penalty']*100:.1f}% ({result['penalty_category']})")
                report.append(f"   Energy: -{result['energy_savings']*100:.1f}%")
                report.append(f"   Power: -{result['power_savings']*100:.1f}%")
            else:
                report.append(f"❌ {config_name}: {result.get('fallback_reason', 'Failed')}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function."""
    # Configuration - using run 2 data to avoid first-run outliers
    data_path = "../data_aggregation/complete_aggregation_run2.csv"
    output_path = "performance_constrained_optimization_results_run2.json"
    report_path = "optimization_summary_report_run2.txt"
    
    try:
        # Initialize optimizer
        optimizer = PerformanceConstrainedOptimizer(data_path)
        
        # Perform optimization
        results = optimizer.optimize_all_workloads()
        
        # Save results
        optimizer.save_results(output_path)
        
        # Generate and save summary report
        summary = optimizer.generate_summary_report()
        with open(report_path, 'w') as f:
            f.write(summary)
        
        # Print summary
        print(summary)
        
        logger.info(f"Optimization complete. Results saved to {output_path}")
        logger.info(f"Summary report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
