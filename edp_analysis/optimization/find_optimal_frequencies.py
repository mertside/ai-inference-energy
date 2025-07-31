#!/usr/bin/env python3
"""
Optimal Frequency Selection using EDP/ED²P Analysis

This script analyzes aggregated profiling data to find optimal frequencies
for energy-delay optimization across different GPU applications.

Usage:
    python find_optimal_frequencies.py --data ../data_aggregation/complete_aggregation.csv --output optimal_frequencies.json
    python find_optimal_frequencies.py --data ../data_aggregation/complete_aggregation.csv --gpu V100 --method edp
    python find_optimal_frequencies.py --help

Author: AI Inference Energy Profiling Framework
"""

import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class EDPOptimizer:
    """Find optimal frequencies using Energy-Delay Product analysis."""
    
    def __init__(self):
        self.supported_methods = ["edp", "ed2p", "energy", "performance"]
        self.results = {}
    
    def load_aggregated_data(self, data_path: str) -> pd.DataFrame:
        """Load aggregated profiling data."""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} configurations from {data_path}")
            
            # Validate required columns
            required_cols = ['gpu', 'application', 'frequency', 'avg_power', 'execution_time', 'energy']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Calculate EDP and ED²P if not present
            if 'edp' not in df.columns:
                df['edp'] = df['energy'] * df['execution_time']
            if 'ed2p' not in df.columns:
                df['ed2p'] = df['energy'] * (df['execution_time'] ** 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def find_optimal_frequency(self, df: pd.DataFrame, method: str = "edp") -> Tuple[int, Dict]:
        """Find optimal frequency for a given optimization method."""
        
        if method == "edp":
            # Find frequency with minimum EDP
            optimal_idx = df['edp'].idxmin()
            metric_name = "EDP"
            metric_value = df.loc[optimal_idx, 'edp']
        elif method == "ed2p":
            # Find frequency with minimum ED²P
            optimal_idx = df['ed2p'].idxmin()
            metric_name = "ED²P"
            metric_value = df.loc[optimal_idx, 'ed2p']
        elif method == "energy":
            # Find frequency with minimum energy
            optimal_idx = df['energy'].idxmin()
            metric_name = "Energy"
            metric_value = df.loc[optimal_idx, 'energy']
        elif method == "performance":
            # Find frequency with minimum execution time
            optimal_idx = df['execution_time'].idxmin()
            metric_name = "Performance"
            metric_value = df.loc[optimal_idx, 'execution_time']
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimal_freq = df.loc[optimal_idx, 'frequency']
        optimal_config = df.loc[optimal_idx].to_dict()
        
        return optimal_freq, {
            'frequency': int(optimal_freq),
            'method': method,
            'metric_name': metric_name,
            'metric_value': float(metric_value),
            'energy': float(optimal_config['energy']),
            'execution_time': float(optimal_config['execution_time']),
            'avg_power': float(optimal_config['avg_power']),
            'edp': float(optimal_config['edp']),
            'ed2p': float(optimal_config['ed2p'])
        }
    
    def calculate_baseline_comparison(self, df: pd.DataFrame, optimal_freq: int, baseline_freq: int = None) -> Dict:
        """Calculate energy savings compared to baseline (typically max frequency)."""
        
        if baseline_freq is None:
            baseline_freq = df['frequency'].max()
        
        # Get optimal configuration
        optimal_config = df[df['frequency'] == optimal_freq].iloc[0]
        
        # Get baseline configuration
        baseline_config = df[df['frequency'] == baseline_freq].iloc[0]
        
        # Calculate savings
        energy_savings = (baseline_config['energy'] - optimal_config['energy']) / baseline_config['energy'] * 100
        time_penalty = (optimal_config['execution_time'] - baseline_config['execution_time']) / baseline_config['execution_time'] * 100
        power_savings = (baseline_config['avg_power'] - optimal_config['avg_power']) / baseline_config['avg_power'] * 100
        
        return {
            'baseline_frequency': int(baseline_freq),
            'optimal_frequency': int(optimal_freq),
            'energy_savings_percent': float(energy_savings),
            'time_penalty_percent': float(time_penalty),
            'power_savings_percent': float(power_savings),
            'baseline_energy': float(baseline_config['energy']),
            'optimal_energy': float(optimal_config['energy']),
            'baseline_time': float(baseline_config['execution_time']),
            'optimal_time': float(optimal_config['execution_time'])
        }
    
    def analyze_application(self, df: pd.DataFrame, gpu: str, app: str, methods: List[str] = None) -> Dict:
        """Analyze optimal frequencies for a specific GPU+application combination."""
        
        if methods is None:
            methods = ["edp", "ed2p", "energy", "performance"]
        
        # Filter data for specific GPU and application
        app_data = df[(df['gpu'] == gpu) & (df['application'] == app)].copy()
        
        if app_data.empty:
            logger.warning(f"No data found for {gpu} + {app}")
            return {}
        
        logger.info(f"Analyzing {gpu} + {app}: {len(app_data)} frequency points")
        
        results = {
            'gpu': gpu,
            'application': app,
            'total_frequencies': len(app_data),
            'frequency_range': {
                'min': int(app_data['frequency'].min()),
                'max': int(app_data['frequency'].max())
            },
            'optimizations': {}
        }
        
        # Find optimal frequencies for each method
        for method in methods:
            try:
                optimal_freq, config = self.find_optimal_frequency(app_data, method)
                
                # Calculate baseline comparison
                comparison = self.calculate_baseline_comparison(app_data, optimal_freq)
                
                results['optimizations'][method] = {
                    **config,
                    'baseline_comparison': comparison
                }
                
                logger.info(f"  {method.upper()} optimal: {optimal_freq} MHz "
                          f"(Energy savings: {comparison['energy_savings_percent']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error optimizing {method} for {gpu}+{app}: {e}")
        
        return results
    
    def analyze_all_configurations(self, df: pd.DataFrame, gpu_filter: str = None, 
                                 app_filter: str = None, methods: List[str] = None) -> Dict:
        """Analyze optimal frequencies for all GPU+application combinations."""
        
        if methods is None:
            methods = ["edp", "ed2p"]  # Focus on most important methods
        
        # Get unique combinations
        if gpu_filter:
            gpus = [gpu_filter]
        else:
            gpus = df['gpu'].unique()
        
        if app_filter:
            apps = [app_filter]
        else:
            apps = df['application'].unique()
        
        results = {
            'summary': {
                'total_gpus': len(gpus),
                'total_applications': len(apps),
                'total_combinations': len(gpus) * len(apps),
                'optimization_methods': methods,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'configurations': {}
        }
        
        logger.info(f"Analyzing {len(gpus)} GPUs × {len(apps)} applications = {len(gpus) * len(apps)} combinations")
        
        # Analyze each combination
        for gpu in gpus:
            for app in apps:
                config_key = f"{gpu}_{app}"
                logger.info(f"\n📊 Analyzing {gpu} + {app}")
                
                config_results = self.analyze_application(df, gpu, app, methods)
                if config_results:
                    results['configurations'][config_key] = config_results
        
        # Calculate summary statistics
        self.add_summary_statistics(results)
        
        return results
    
    def add_summary_statistics(self, results: Dict):
        """Add summary statistics across all configurations."""
        
        if not results['configurations']:
            return
        
        summary_stats = {
            'average_energy_savings': {},
            'best_configurations': {},
            'frequency_recommendations': {}
        }
        
        # Calculate average savings by method
        for method in results['summary']['optimization_methods']:
            energy_savings = []
            best_config = None
            best_savings = -float('inf')
            
            for config_key, config in results['configurations'].items():
                if method in config['optimizations']:
                    savings = config['optimizations'][method]['baseline_comparison']['energy_savings_percent']
                    energy_savings.append(savings)
                    
                    if savings > best_savings:
                        best_savings = savings
                        best_config = {
                            'configuration': config_key,
                            'frequency': config['optimizations'][method]['frequency'],
                            'energy_savings': savings
                        }
            
            if energy_savings:
                summary_stats['average_energy_savings'][method] = {
                    'mean': np.mean(energy_savings),
                    'std': np.std(energy_savings),
                    'min': np.min(energy_savings),
                    'max': np.max(energy_savings)
                }
                summary_stats['best_configurations'][method] = best_config
        
        results['summary']['statistics'] = summary_stats
    
    def save_results(self, results: Dict, output_path: str):
        """Save optimization results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary_report(self, results: Dict):
        """Print a comprehensive summary report."""
        
        print("\n" + "="*80)
        print("OPTIMAL FREQUENCY ANALYSIS SUMMARY")
        print("="*80)
        
        summary = results['summary']
        print(f"Analysis Date: {summary['analysis_timestamp']}")
        print(f"Total Configurations: {summary['total_combinations']} "
              f"({summary['total_gpus']} GPUs × {summary['total_applications']} apps)")
        print(f"Optimization Methods: {', '.join(summary['optimization_methods'])}")
        
        # Print statistics if available
        if 'statistics' in summary:
            stats = summary['statistics']
            
            print(f"\n📈 ENERGY SAVINGS SUMMARY:")
            for method, savings_stats in stats['average_energy_savings'].items():
                print(f"  {method.upper()}:")
                print(f"    Average: {savings_stats['mean']:.1f}% ± {savings_stats['std']:.1f}%")
                print(f"    Range: {savings_stats['min']:.1f}% to {savings_stats['max']:.1f}%")
            
            print(f"\n🏆 BEST CONFIGURATIONS:")
            for method, best_config in stats['best_configurations'].items():
                print(f"  {method.upper()}: {best_config['configuration']} @ {best_config['frequency']} MHz")
                print(f"    Energy savings: {best_config['energy_savings']:.1f}%")
        
        # Print detailed results for each configuration
        print(f"\n📊 DETAILED RESULTS:")
        for config_key, config in results['configurations'].items():
            print(f"\n  {config['gpu']} + {config['application']}:")
            print(f"    Frequency range: {config['frequency_range']['min']}-{config['frequency_range']['max']} MHz")
            
            for method, opt_result in config['optimizations'].items():
                freq = opt_result['frequency']
                savings = opt_result['baseline_comparison']['energy_savings_percent']
                penalty = opt_result['baseline_comparison']['time_penalty_percent']
                print(f"    {method.upper()}: {freq} MHz (Energy: {savings:+.1f}%, Time: {penalty:+.1f}%)")
        
        print("\n" + "="*80)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Find optimal frequencies using EDP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze all configurations with EDP and ED²P
    python find_optimal_frequencies.py --data complete_aggregation.csv --output optimal_frequencies.json
    
    # Analyze specific GPU with all methods
    python find_optimal_frequencies.py --data complete_aggregation.csv --gpu V100 --methods edp ed2p energy performance
    
    # Analyze specific application
    python find_optimal_frequencies.py --data complete_aggregation.csv --app LLAMA --methods edp
        """
    )
    
    parser.add_argument("--data", type=str, required=True,
                       help="Path to aggregated profiling data CSV file")
    parser.add_argument("--output", type=str, default="optimal_frequencies.json",
                       help="Output JSON file for results (default: optimal_frequencies.json)")
    parser.add_argument("--gpu", type=str, choices=["V100", "A100", "H100"],
                       help="Filter by specific GPU type")
    parser.add_argument("--app", type=str, choices=["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER"],
                       help="Filter by specific application")
    parser.add_argument("--methods", nargs="+", choices=["edp", "ed2p", "energy", "performance"],
                       default=["edp", "ed2p"],
                       help="Optimization methods to use (default: edp ed2p)")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = EDPOptimizer()
    
    logger.info(f"🚀 Starting EDP optimization analysis")
    logger.info(f"Data: {args.data}")
    logger.info(f"Methods: {args.methods}")
    if args.gpu:
        logger.info(f"GPU filter: {args.gpu}")
    if args.app:
        logger.info(f"Application filter: {args.app}")
    
    try:
        # Load data
        df = optimizer.load_aggregated_data(args.data)
        
        # Perform analysis
        results = optimizer.analyze_all_configurations(
            df, 
            gpu_filter=args.gpu,
            app_filter=args.app,
            methods=args.methods
        )
        
        # Save results
        optimizer.save_results(results, args.output)
        
        # Print summary
        optimizer.print_summary_report(results)
        
        logger.info("✅ EDP optimization analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
