#!/usr/bin/env python3
"""
EDP/ED2P Analysis and Optimal Frequency Selection for AI Inference.

This script performs Energy-Delay Product (EDP) and Energy-Delay^2 Product (ED2P)
analysis on aggregated AI inference results to identify optimal GPU frequencies
for energy-efficient operation with performance constraints.

Based on proven methodologies from FGCS and ICPP papers, this script:
- Calculates EDP and ED2P for all frequency configurations
- Applies performance constraints (e.g., ≤5% degradation)
- Identifies optimal frequencies for each GPU-workload combination
- Validates energy savings vs performance trade-offs

Requirements:
    - Aggregated results CSV from aggregate_results.py
    - Performance baselines JSON from aggregate_results.py
    - Python 3.8+ with pandas, numpy, matplotlib

Author: Mert Side
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from utils import setup_logging
except ImportError:
    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)


class OptimalFrequencySelector:
    """
    Optimal frequency selector using EDP/ED2P analysis with performance constraints.
    
    Implements proven methodologies from FGCS/ICPP papers for AI inference workloads,
    incorporating performance degradation constraints for practical deployment.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the optimal frequency selector.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logging()
        self.results_df = None
        self.baselines = None
        self.optimal_frequencies = {}

    def load_aggregated_data(self, results_file: str, baselines_file: str) -> None:
        """
        Load aggregated experimental results and performance baselines.
        
        Args:
            results_file: Path to aggregated results CSV
            baselines_file: Path to performance baselines JSON
        """
        self.logger.info(f"Loading aggregated data from {results_file}")
        
        # Load results
        self.results_df = pd.read_csv(results_file)
        self.logger.info(f"Loaded {len(self.results_df)} experimental records")
        
        # Load baselines
        with open(baselines_file, 'r') as f:
            self.baselines = json.load(f)
        
        self.logger.info(f"Loaded baselines for {len(self.baselines)} GPU-workload combinations")

    def calculate_edp_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EDP and ED2P metrics for all experiments.
        
        Args:
            df: DataFrame with experimental results
            
        Returns:
            DataFrame with added EDP metrics
        """
        df = df.copy()
        
        # Ensure required columns exist
        if 'total_energy' not in df.columns or 'execution_time' not in df.columns:
            self.logger.error("Missing required columns: total_energy, execution_time")
            return df
        
        # Calculate EDP and ED2P (your proven approach)
        df['edp'] = df['total_energy'] * df['execution_time']
        df['ed2p'] = df['total_energy'] * (df['execution_time'] ** 2)
        
        # Calculate energy efficiency metrics
        df['energy_efficiency'] = 1.0 / df['total_energy']
        df['performance_efficiency'] = 1.0 / df['execution_time']
        df['combined_efficiency'] = 1.0 / df['edp']
        
        self.logger.info("Calculated EDP/ED2P metrics for all experiments")
        return df

    def apply_performance_constraints(self, df: pd.DataFrame, gpu: str, workload: str) -> pd.DataFrame:
        """
        Filter experiments that meet performance constraints.
        
        Args:
            df: DataFrame with experimental results
            gpu: GPU architecture
            workload: AI workload name
            
        Returns:
            Filtered DataFrame with only constraint-satisfying experiments
        """
        baseline_key = f"{gpu}_{workload}"
        
        if baseline_key not in self.baselines:
            self.logger.warning(f"No baseline found for {baseline_key}")
            return df
        
        baseline = self.baselines[baseline_key]
        max_acceptable_time = baseline['max_acceptable_time']
        
        # Filter experiments meeting performance constraint
        constrained_df = df[df['execution_time'] <= max_acceptable_time].copy()
        
        # Calculate performance degradation
        constrained_df['performance_degradation_pct'] = (
            (constrained_df['execution_time'] / baseline['baseline_time'] - 1) * 100
        )
        
        # Calculate energy savings relative to baseline
        constrained_df['energy_savings_pct'] = (
            (1 - constrained_df['total_energy'] / baseline['baseline_energy']) * 100
        )
        
        self.logger.debug(
            f"{baseline_key}: {len(constrained_df)}/{len(df)} experiments meet constraint "
            f"(≤{baseline['constraint_pct']}% degradation)"
        )
        
        return constrained_df

    def find_optimal_frequency_edp(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Find optimal frequency using EDP minimization.
        
        Args:
            df: DataFrame with EDP metrics and constraints applied
            
        Returns:
            Tuple of (optimal_frequency, optimization_details)
        """
        if df.empty:
            return None, {'error': 'No experiments meet performance constraints'}
        
        # Find minimum EDP
        optimal_idx = df['edp'].idxmin()
        optimal_freq = df.loc[optimal_idx, 'frequency']
        
        # Gather optimization details
        details = {
            'optimal_frequency': optimal_freq,
            'edp': df.loc[optimal_idx, 'edp'],
            'energy': df.loc[optimal_idx, 'total_energy'],
            'execution_time': df.loc[optimal_idx, 'execution_time'],
            'performance_degradation_pct': df.loc[optimal_idx, 'performance_degradation_pct'],
            'energy_savings_pct': df.loc[optimal_idx, 'energy_savings_pct'],
            'valid_frequencies': len(df['frequency'].unique()),
            'optimization_method': 'EDP_minimization'
        }
        
        return optimal_freq, details

    def find_optimal_frequency_ed2p(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Find optimal frequency using ED2P minimization (performance-centric).
        
        Args:
            df: DataFrame with ED2P metrics and constraints applied
            
        Returns:
            Tuple of (optimal_frequency, optimization_details)
        """
        if df.empty:
            return None, {'error': 'No experiments meet performance constraints'}
        
        # Find minimum ED2P
        optimal_idx = df['ed2p'].idxmin()
        optimal_freq = df.loc[optimal_idx, 'frequency']
        
        # Gather optimization details
        details = {
            'optimal_frequency': optimal_freq,
            'ed2p': df.loc[optimal_idx, 'ed2p'],
            'energy': df.loc[optimal_idx, 'total_energy'],
            'execution_time': df.loc[optimal_idx, 'execution_time'],
            'performance_degradation_pct': df.loc[optimal_idx, 'performance_degradation_pct'],
            'energy_savings_pct': df.loc[optimal_idx, 'energy_savings_pct'],
            'valid_frequencies': len(df['frequency'].unique()),
            'optimization_method': 'ED2P_minimization'
        }
        
        return optimal_freq, details

    def find_optimal_frequency_energy_constrained(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Find optimal frequency using pure energy minimization with constraints.
        
        Args:
            df: DataFrame with energy metrics and constraints applied
            
        Returns:
            Tuple of (optimal_frequency, optimization_details)
        """
        if df.empty:
            return None, {'error': 'No experiments meet performance constraints'}
        
        # Find minimum energy consumption
        optimal_idx = df['total_energy'].idxmin()
        optimal_freq = df.loc[optimal_idx, 'frequency']
        
        # Gather optimization details
        details = {
            'optimal_frequency': optimal_freq,
            'energy': df.loc[optimal_idx, 'total_energy'],
            'execution_time': df.loc[optimal_idx, 'execution_time'],
            'edp': df.loc[optimal_idx, 'edp'],
            'performance_degradation_pct': df.loc[optimal_idx, 'performance_degradation_pct'],
            'energy_savings_pct': df.loc[optimal_idx, 'energy_savings_pct'],
            'valid_frequencies': len(df['frequency'].unique()),
            'optimization_method': 'energy_minimization'
        }
        
        return optimal_freq, details

    def analyze_all_combinations(self, optimization_method: str = 'edp') -> Dict[str, Dict]:
        """
        Analyze optimal frequencies for all GPU-workload combinations.
        
        Args:
            optimization_method: 'edp', 'ed2p', or 'energy'
            
        Returns:
            Dictionary of optimal frequency results
        """
        self.logger.info(f"Analyzing optimal frequencies using {optimization_method} optimization")
        
        if self.results_df is None:
            raise ValueError("No data loaded. Call load_aggregated_data() first.")
        
        # Calculate EDP metrics
        df_with_metrics = self.calculate_edp_metrics(self.results_df)
        
        # Select optimization method
        if optimization_method == 'edp':
            optimize_func = self.find_optimal_frequency_edp
        elif optimization_method == 'ed2p':
            optimize_func = self.find_optimal_frequency_ed2p
        elif optimization_method == 'energy':
            optimize_func = self.find_optimal_frequency_energy_constrained
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
        
        results = {}
        
        # Process each GPU-workload combination
        for gpu in df_with_metrics['gpu'].unique():
            for workload in df_with_metrics['workload'].unique():
                combination_key = f"{gpu}_{workload}"
                
                # Filter data for this combination
                subset = df_with_metrics[
                    (df_with_metrics['gpu'] == gpu) & 
                    (df_with_metrics['workload'] == workload)
                ]
                
                if subset.empty:
                    continue
                
                # Apply performance constraints
                constrained_subset = self.apply_performance_constraints(subset, gpu, workload)
                
                # Find optimal frequency
                optimal_freq, details = optimize_func(constrained_subset)
                
                # Store results
                results[combination_key] = {
                    'gpu': gpu,
                    'workload': workload,
                    'total_experiments': len(subset),
                    'valid_experiments': len(constrained_subset),
                    'frequency_range': {
                        'min': subset['frequency'].min(),
                        'max': subset['frequency'].max(),
                        'count': len(subset['frequency'].unique())
                    },
                    **details
                }
                
                self.logger.info(
                    f"{combination_key}: optimal frequency = {optimal_freq} MHz, "
                    f"energy savings = {details.get('energy_savings_pct', 0):.1f}%, "
                    f"performance impact = {details.get('performance_degradation_pct', 0):.1f}%"
                )
        
        self.optimal_frequencies = results
        return results

    def generate_optimization_report(self, results: Dict[str, Dict]) -> str:
        """
        Generate a detailed optimization report.
        
        Args:
            results: Optimal frequency analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("AI INFERENCE OPTIMAL FREQUENCY SELECTION REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        valid_results = {k: v for k, v in results.items() if 'optimal_frequency' in v}
        
        if not valid_results:
            report.append("\nNo valid optimal frequencies found!")
            return "\n".join(report)
        
        energy_savings = [v['energy_savings_pct'] for v in valid_results.values() if 'energy_savings_pct' in v]
        performance_impacts = [v['performance_degradation_pct'] for v in valid_results.values() if 'performance_degradation_pct' in v]
        
        report.append(f"\nSummary Statistics:")
        report.append(f"  Valid GPU-workload combinations: {len(valid_results)}")
        report.append(f"  Mean energy savings: {np.mean(energy_savings):.1f}% ± {np.std(energy_savings):.1f}%")
        report.append(f"  Mean performance impact: {np.mean(performance_impacts):.1f}% ± {np.std(performance_impacts):.1f}%")
        report.append(f"  Energy savings range: {np.min(energy_savings):.1f}% to {np.max(energy_savings):.1f}%")
        
        # Detailed results by GPU
        for gpu in ['v100', 'a100', 'h100']:
            gpu_results = {k: v for k, v in valid_results.items() if v['gpu'] == gpu}
            if not gpu_results:
                continue
                
            report.append(f"\n{gpu.upper()} Results:")
            report.append("-" * 40)
            
            for workload in ['llama', 'stablediffusion', 'vit', 'whisper']:
                key = f"{gpu}_{workload}"
                if key not in gpu_results:
                    continue
                
                result = gpu_results[key]
                report.append(
                    f"  {workload.upper()}:"
                    f"  {result['optimal_frequency']} MHz"
                    f"  ({result['energy_savings_pct']:+.1f}% energy, {result['performance_degradation_pct']:+.1f}% time)"
                )
        
        # Frequency distribution analysis
        report.append(f"\nOptimal Frequency Distribution:")
        freq_counts = {}
        for result in valid_results.values():
            freq = result['optimal_frequency']
            freq_counts[freq] = freq_counts.get(freq, 0) + 1
        
        for freq in sorted(freq_counts.keys()):
            count = freq_counts[freq]
            report.append(f"  {freq} MHz: {count} workload(s)")
        
        # Constraint satisfaction analysis
        report.append(f"\nPerformance Constraint Analysis:")
        constraint_violations = [p for p in performance_impacts if p > 5.0]
        report.append(f"  Constraint violations (>5%): {len(constraint_violations)}/{len(performance_impacts)}")
        
        if constraint_violations:
            report.append(f"  Max performance impact: {max(constraint_violations):.1f}%")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)

    def save_optimal_frequencies(self, results: Dict[str, Dict], output_file: str) -> None:
        """
        Save optimal frequency results to JSON file.
        
        Args:
            results: Optimal frequency analysis results  
            output_file: Output JSON file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved optimal frequency results to {output_path}")

    def create_optimization_plots(self, results: Dict[str, Dict], output_dir: str) -> None:
        """
        Create visualization plots for optimization results.
        
        Args:
            results: Optimal frequency analysis results
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        valid_results = {k: v for k, v in results.items() if 'optimal_frequency' in v}
        
        if not valid_results:
            self.logger.warning("No valid results to plot")
            return
        
        # Energy savings vs performance impact scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Energy savings vs performance impact
        energy_savings = [v['energy_savings_pct'] for v in valid_results.values()]
        performance_impacts = [v['performance_degradation_pct'] for v in valid_results.values()]
        colors = [{'v100': 'red', 'a100': 'green', 'h100': 'blue'}[v['gpu']] for v in valid_results.values()]
        
        ax1.scatter(performance_impacts, energy_savings, c=colors, alpha=0.7, s=100)
        ax1.axvline(x=5.0, color='red', linestyle='--', alpha=0.5, label='5% Performance Constraint')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Performance Impact (%)')
        ax1.set_ylabel('Energy Savings (%)')
        ax1.set_title('Energy Savings vs Performance Impact')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add GPU legend
        from matplotlib.lines import Line2D
        gpu_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=gpu.upper()) 
                     for gpu, c in [('v100', 'red'), ('a100', 'green'), ('h100', 'blue')]]
        ax1.legend(handles=gpu_legend, loc='upper right')
        
        # Plot 2: Optimal frequency distribution
        frequencies = [v['optimal_frequency'] for v in valid_results.values()]
        ax2.hist(frequencies, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Optimal Frequency (MHz)')
        ax2.set_ylabel('Number of Workloads')
        ax2.set_title('Distribution of Optimal Frequencies')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Workload-specific analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        workloads = ['llama', 'stablediffusion', 'vit', 'whisper']
        
        for i, workload in enumerate(workloads):
            workload_results = {k: v for k, v in valid_results.items() if v['workload'] == workload}
            
            if not workload_results:
                axes[i].text(0.5, 0.5, f'No data for {workload}', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{workload.upper()}')
                continue
            
            gpus = [v['gpu'] for v in workload_results.values()]
            frequencies = [v['optimal_frequency'] for v in workload_results.values()]
            energy_savings = [v['energy_savings_pct'] for v in workload_results.values()]
            
            # Bar plot of optimal frequencies by GPU
            gpu_colors = {'v100': 'red', 'a100': 'green', 'h100': 'blue'}
            colors = [gpu_colors[gpu] for gpu in gpus]
            
            bars = axes[i].bar(range(len(gpus)), frequencies, color=colors, alpha=0.7)
            axes[i].set_xlabel('GPU Architecture')
            axes[i].set_ylabel('Optimal Frequency (MHz)')
            axes[i].set_title(f'{workload.upper()} - Optimal Frequencies')
            axes[i].set_xticks(range(len(gpus)))
            axes[i].set_xticklabels([gpu.upper() for gpu in gpus])
            
            # Add energy savings as text on bars
            for j, (bar, savings) in enumerate(zip(bars, energy_savings)):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 20,
                           f'{savings:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'workload_specific_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved optimization plots to {output_path}")

    def validate_optimal_frequencies(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Validate optimal frequency results against theoretical expectations.
        
        Args:
            results: Optimal frequency analysis results
            
        Returns:
            Validation summary
        """
        valid_results = {k: v for k, v in results.items() if 'optimal_frequency' in v}
        
        validation = {
            'total_combinations': len(results),
            'valid_optimizations': len(valid_results),
            'constraint_violations': 0,
            'energy_degradations': 0,
            'mean_energy_savings': 0,
            'mean_performance_impact': 0,
            'frequency_distribution': {},
            'gpu_performance': {}
        }
        
        if not valid_results:
            return validation
        
        energy_savings = []
        performance_impacts = []
        
        for key, result in valid_results.items():
            energy_savings.append(result.get('energy_savings_pct', 0))
            performance_impacts.append(result.get('performance_degradation_pct', 0))
            
            # Check constraint violations
            if result.get('performance_degradation_pct', 0) > 5.0:
                validation['constraint_violations'] += 1
            
            # Check energy degradations (should be rare)
            if result.get('energy_savings_pct', 0) < 0:
                validation['energy_degradations'] += 1
            
            # Frequency distribution
            freq = result['optimal_frequency']
            validation['frequency_distribution'][freq] = validation['frequency_distribution'].get(freq, 0) + 1
        
        validation['mean_energy_savings'] = np.mean(energy_savings)
        validation['mean_performance_impact'] = np.mean(performance_impacts)
        
        # GPU-specific performance
        for gpu in ['v100', 'a100', 'h100']:
            gpu_results = [v for v in valid_results.values() if v['gpu'] == gpu]
            if gpu_results:
                validation['gpu_performance'][gpu] = {
                    'count': len(gpu_results),
                    'mean_energy_savings': np.mean([r['energy_savings_pct'] for r in gpu_results]),
                    'mean_performance_impact': np.mean([r['performance_degradation_pct'] for r in gpu_results])
                }
        
        return validation


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="EDP/ED2P analysis and optimal frequency selection for AI inference"
    )
    parser.add_argument(
        "-r", "--results",
        required=True,
        help="Path to aggregated results CSV file"
    )
    parser.add_argument(
        "-b", "--baselines",
        required=True,
        help="Path to performance baselines JSON file"
    )
    parser.add_argument(
        "-m", "--method",
        choices=['edp', 'ed2p', 'energy'],
        default='edp',
        help="Optimization method (default: edp)"
    )
    parser.add_argument(
        "-o", "--output",
        default="optimal_frequencies.json",
        help="Output JSON file for optimal frequencies (default: optimal_frequencies.json)"
    )
    parser.add_argument(
        "--report",
        default="optimization_report.txt",
        help="Output file for optimization report (default: optimization_report.txt)"
    )
    parser.add_argument(
        "--plots",
        default="plots",
        help="Output directory for visualization plots (default: plots)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        # Initialize selector
        selector = OptimalFrequencySelector(logger=logger)
        
        # Load data
        selector.load_aggregated_data(args.results, args.baselines)
        
        # Analyze optimal frequencies
        results = selector.analyze_all_combinations(args.method)
        
        # Generate report
        report = selector.generate_optimization_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        
        # Save results
        selector.save_optimal_frequencies(results, args.output)
        
        # Create plots
        selector.create_optimization_plots(results, args.plots)
        
        # Validate results
        validation = selector.validate_optimal_frequencies(results)
        
        # Display summary
        print(report)
        print(f"\nValidation Summary:")
        print(f"  Valid optimizations: {validation['valid_optimizations']}/{validation['total_combinations']}")
        print(f"  Mean energy savings: {validation['mean_energy_savings']:.1f}%")
        print(f"  Mean performance impact: {validation['mean_performance_impact']:.1f}%")
        print(f"  Constraint violations: {validation['constraint_violations']}")
        
        logger.info("Optimal frequency analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Optimal frequency analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
