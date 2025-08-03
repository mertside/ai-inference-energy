"""
Comparative Analysis Workflow

This module provides comprehensive comparative analysis capabilities for 
GPU frequency optimization results across different configurations, methods,
and time periods.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import sys

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    ProfilingDataLoader, EnergyCalculator, FrequencyOptimizer, PerformanceConstraintManager,
    load_config, save_results, create_deployment_summary,
    get_framework_version, setup_logging, ensure_directory
)

logger = logging.getLogger(__name__)


class ComparativeAnalysisWorkflow:
    """
    Comprehensive comparative analysis workflow for optimization results.
    """
    
    def __init__(self, 
                 output_dir: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize comparative analysis workflow.
        
        Args:
            output_dir: Optional output directory for results
            config_path: Optional path to configuration file
        """
        self.output_dir = ensure_directory(output_dir or "comparative_analysis_results")
        self.config_path = config_path
        self.config = load_config(config_path) if config_path else self._get_default_config()
        
        # Data storage
        self.datasets = {}
        self.optimization_results = {}
        self.comparison_results = {}
        
        logger.info(f"Comparative analysis initialized: {self.output_dir}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for comparative analysis."""
        return {
            'comparison': {
                'methods': ['edp', 'energy', 'performance'],
                'metrics': ['energy_savings', 'performance_penalty', 'edp_improvement'],
                'statistical_tests': True,
                'confidence_level': 0.95
            },
            'visualization': {
                'include_plots': True,
                'plot_formats': ['png', 'svg'],
                'figure_size': [12, 8],
                'dpi': 300
            }
        }
    
    def add_dataset(self, 
                   name: str, 
                   data_path: str, 
                   description: Optional[str] = None) -> None:
        """
        Add a dataset for comparative analysis.
        
        Args:
            name: Unique name for the dataset
            data_path: Path to the dataset
            description: Optional description of the dataset
        """
        logger.info(f"Adding dataset: {name} from {data_path}")
        
        data_loader = ProfilingDataLoader(self.config_path)
        data = data_loader.load_aggregated_data(data_path)
        
        # Validate data
        validation = data_loader.validate_data_consistency()
        
        self.datasets[name] = {
            'data': data,
            'path': data_path,
            'description': description or f"Dataset {name}",
            'validation': validation,
            'summary': data_loader.get_summary_statistics()
        }
        
        logger.info(f"Dataset {name} added: {len(data)} records")
    
    def add_optimization_results(self, 
                               name: str, 
                               results_path: str,
                               description: Optional[str] = None) -> None:
        """
        Add optimization results for comparative analysis.
        
        Args:
            name: Unique name for the results
            results_path: Path to optimization results JSON file
            description: Optional description
        """
        logger.info(f"Adding optimization results: {name} from {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        self.optimization_results[name] = {
            'results': results,
            'path': results_path,
            'description': description or f"Optimization results {name}",
            'timestamp': results.get('timestamp', 'unknown')
        }
        
        logger.info(f"Optimization results {name} added")
    
    def compare_optimization_methods(self, 
                                   dataset_name: str,
                                   methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare different optimization methods on the same dataset.
        
        Args:
            dataset_name: Name of dataset to analyze
            methods: List of methods to compare
            
        Returns:
            Method comparison results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        methods = methods or ['edp', 'energy', 'performance']
        logger.info(f"Comparing optimization methods {methods} on dataset {dataset_name}")
        
        data = self.datasets[dataset_name]['data']
        
        # Run optimization with different methods
        optimizer = FrequencyOptimizer(self.config_path)
        method_results = {}
        
        for method in methods:
            logger.info(f"Running optimization with method: {method}")
            result = optimizer.optimize_all_configurations(data, methods=[method])
            method_results[method] = result
        
        # Compare results
        comparison = self._analyze_method_comparison(method_results, methods)
        
        # Save comparison results
        comparison_data = {
            'comparison_type': 'optimization_methods',
            'dataset': dataset_name,
            'methods': methods,
            'results': comparison,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results_path = self.output_dir / f"method_comparison_{dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(comparison_data, results_path)
        
        logger.info(f"Method comparison completed and saved to {results_path}")
        return comparison_data
    
    def _analyze_method_comparison(self, 
                                 method_results: Dict[str, Any],
                                 methods: List[str]) -> Dict[str, Any]:
        """Analyze comparison between optimization methods."""
        comparison = {
            'summary': {},
            'detailed_comparison': {},
            'statistical_analysis': {},
            'recommendations': {}
        }
        
        # Extract metrics for comparison
        method_metrics = {}
        
        for method in methods:
            if method in method_results and 'summary' in method_results[method]:
                summary = method_results[method]['summary']
                method_metrics[method] = {
                    'energy_savings_mean': summary.get('energy_savings', {}).get('mean', 0),
                    'energy_savings_std': summary.get('energy_savings', {}).get('std', 0),
                    'performance_penalty_mean': abs(summary.get('performance_penalty', {}).get('mean', 0)),
                    'performance_penalty_std': summary.get('performance_penalty', {}).get('std', 0),
                    'successful_optimizations': summary.get('successful_optimizations', 0),
                    'total_configurations': summary.get('total_configurations', 0)
                }
        
        # Calculate summary statistics
        comparison['summary'] = {
            'best_energy_savings': max(method_metrics.items(), 
                                     key=lambda x: x[1]['energy_savings_mean']) if method_metrics else None,
            'lowest_performance_penalty': min(method_metrics.items(), 
                                            key=lambda x: x[1]['performance_penalty_mean']) if method_metrics else None,
            'highest_success_rate': max(method_metrics.items(), 
                                      key=lambda x: x[1]['successful_optimizations'] / max(x[1]['total_configurations'], 1)) if method_metrics else None
        }
        
        # Detailed comparison
        comparison['detailed_comparison'] = method_metrics
        
        # Generate recommendations
        recommendations = []
        
        if method_metrics:
            best_energy = comparison['summary']['best_energy_savings']
            lowest_penalty = comparison['summary']['lowest_performance_penalty']
            
            if best_energy and lowest_penalty:
                if best_energy[0] == lowest_penalty[0]:
                    recommendations.append(f"Method '{best_energy[0]}' provides the best balance of energy savings and performance")
                else:
                    recommendations.append(f"Method '{best_energy[0]}' for maximum energy savings ({best_energy[1]['energy_savings_mean']:.1f}%)")
                    recommendations.append(f"Method '{lowest_penalty[0]}' for minimum performance impact ({lowest_penalty[1]['performance_penalty_mean']:.1f}%)")
            
            # Add method-specific recommendations
            for method, metrics in method_metrics.items():
                success_rate = metrics['successful_optimizations'] / max(metrics['total_configurations'], 1)
                if success_rate < 0.8:
                    recommendations.append(f"Method '{method}' has low success rate ({success_rate:.1%}) - consider parameter tuning")
        
        comparison['recommendations'] = recommendations
        
        return comparison
    
    def compare_datasets(self, 
                        dataset_names: List[str],
                        optimization_method: str = 'edp') -> Dict[str, Any]:
        """
        Compare optimization results across different datasets.
        
        Args:
            dataset_names: List of dataset names to compare
            optimization_method: Method to use for optimization
            
        Returns:
            Dataset comparison results
        """
        logger.info(f"Comparing datasets {dataset_names} with method {optimization_method}")
        
        # Validate all datasets exist
        for name in dataset_names:
            if name not in self.datasets:
                raise ValueError(f"Dataset {name} not found")
        
        # Run optimization on all datasets
        dataset_results = {}
        optimizer = FrequencyOptimizer(self.config_path)
        
        for dataset_name in dataset_names:
            logger.info(f"Optimizing dataset: {dataset_name}")
            data = self.datasets[dataset_name]['data']
            result = optimizer.optimize_all_configurations(data, methods=[optimization_method])
            dataset_results[dataset_name] = result
        
        # Compare results
        comparison = self._analyze_dataset_comparison(dataset_results, dataset_names)
        
        # Save comparison results
        comparison_data = {
            'comparison_type': 'datasets',
            'datasets': dataset_names,
            'optimization_method': optimization_method,
            'results': comparison,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results_path = self.output_dir / f"dataset_comparison_{optimization_method}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(comparison_data, results_path)
        
        logger.info(f"Dataset comparison completed and saved to {results_path}")
        return comparison_data
    
    def _analyze_dataset_comparison(self, 
                                  dataset_results: Dict[str, Any],
                                  dataset_names: List[str]) -> Dict[str, Any]:
        """Analyze comparison between datasets."""
        comparison = {
            'summary': {},
            'configuration_comparison': {},
            'trends_analysis': {},
            'recommendations': {}
        }
        
        # Extract metrics for each dataset
        dataset_metrics = {}
        
        for dataset_name in dataset_names:
            if dataset_name in dataset_results and 'summary' in dataset_results[dataset_name]:
                summary = dataset_results[dataset_name]['summary']
                dataset_metrics[dataset_name] = {
                    'energy_savings': summary.get('energy_savings', {}),
                    'performance_penalty': summary.get('performance_penalty', {}),
                    'successful_optimizations': summary.get('successful_optimizations', 0),
                    'total_configurations': summary.get('total_configurations', 0),
                    'dataset_info': self.datasets[dataset_name]['summary']
                }
        
        # Compare configurations across datasets
        config_comparison = {}
        
        # Get all unique configurations
        all_configs = set()
        for dataset_name in dataset_names:
            if dataset_name in dataset_results and 'configurations' in dataset_results[dataset_name]:
                all_configs.update(dataset_results[dataset_name]['configurations'].keys())
        
        # Compare each configuration across datasets
        for config in all_configs:
            config_comparison[config] = {}
            
            for dataset_name in dataset_names:
                if (dataset_name in dataset_results and 
                    'configurations' in dataset_results[dataset_name] and
                    config in dataset_results[dataset_name]['configurations']):
                    
                    config_data = dataset_results[dataset_name]['configurations'][config]
                    if 'error' not in config_data:
                        config_comparison[config][dataset_name] = {
                            'energy_savings': config_data.get('configuration_summary', {}).get('energy_savings_percent', 0),
                            'performance_penalty': config_data.get('configuration_summary', {}).get('performance_penalty_percent', 0),
                            'optimal_frequency': config_data.get('optimal_frequency', 0)
                        }
        
        comparison['configuration_comparison'] = config_comparison
        
        # Generate summary statistics
        if dataset_metrics:
            comparison['summary'] = {
                'best_overall_savings': max(dataset_metrics.items(), 
                                          key=lambda x: x[1]['energy_savings'].get('mean', 0)),
                'most_consistent': min(dataset_metrics.items(), 
                                     key=lambda x: x[1]['energy_savings'].get('std', float('inf'))),
                'highest_success_rate': max(dataset_metrics.items(), 
                                          key=lambda x: x[1]['successful_optimizations'] / max(x[1]['total_configurations'], 1))
            }
        
        # Generate recommendations
        recommendations = []
        
        if dataset_metrics:
            # Find datasets with significantly different performance
            energy_means = [metrics['energy_savings'].get('mean', 0) for metrics in dataset_metrics.values()]
            if max(energy_means) - min(energy_means) > 10:  # More than 10% difference
                recommendations.append("Significant variation in energy savings across datasets - investigate environmental factors")
            
            # Check for consistency issues
            for dataset_name, metrics in dataset_metrics.items():
                std_dev = metrics['energy_savings'].get('std', 0)
                mean_savings = metrics['energy_savings'].get('mean', 0)
                if std_dev > mean_savings * 0.5:  # High variability
                    recommendations.append(f"Dataset '{dataset_name}' shows high variability - consider additional data collection")
        
        comparison['recommendations'] = recommendations
        
        return comparison
    
    def compare_optimization_results(self, 
                                   result_names: List[str]) -> Dict[str, Any]:
        """
        Compare pre-computed optimization results.
        
        Args:
            result_names: List of result names to compare
            
        Returns:
            Results comparison analysis
        """
        logger.info(f"Comparing optimization results: {result_names}")
        
        # Validate all results exist
        for name in result_names:
            if name not in self.optimization_results:
                raise ValueError(f"Optimization results {name} not found")
        
        # Extract data for comparison
        results_data = {}
        for name in result_names:
            results_data[name] = self.optimization_results[name]['results']
        
        # Perform comparison analysis
        comparison = self._analyze_results_comparison(results_data, result_names)
        
        # Save comparison results
        comparison_data = {
            'comparison_type': 'optimization_results',
            'result_names': result_names,
            'results': comparison,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        results_path = self.output_dir / f"results_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(comparison_data, results_path)
        
        logger.info(f"Results comparison completed and saved to {results_path}")
        return comparison_data
    
    def _analyze_results_comparison(self, 
                                  results_data: Dict[str, Any],
                                  result_names: List[str]) -> Dict[str, Any]:
        """Analyze comparison between optimization results."""
        comparison = {
            'temporal_analysis': {},
            'performance_comparison': {},
            'configuration_trends': {},
            'recommendations': {}
        }
        
        # Extract timestamps and sort chronologically
        result_timestamps = []
        for name in result_names:
            timestamp = results_data[name].get('timestamp', 'unknown')
            result_timestamps.append((name, timestamp))
        
        # Sort by timestamp if available
        try:
            result_timestamps.sort(key=lambda x: pd.Timestamp(x[1]) if x[1] != 'unknown' else pd.Timestamp.min)
            comparison['temporal_analysis']['chronological_order'] = [name for name, _ in result_timestamps]
        except:
            comparison['temporal_analysis']['chronological_order'] = result_names
        
        # Compare performance metrics
        performance_metrics = {}
        
        for name in result_names:
            if 'summary' in results_data[name]:
                summary = results_data[name]['summary']
                performance_metrics[name] = {
                    'energy_savings': summary.get('energy_savings', {}),
                    'performance_penalty': summary.get('performance_penalty', {}),
                    'total_configurations': summary.get('total_configurations', 0),
                    'successful_optimizations': summary.get('successful_optimizations', 0)
                }
        
        comparison['performance_comparison'] = performance_metrics
        
        # Analyze configuration trends across results
        config_trends = {}
        
        # Get all configurations that appear in multiple results
        all_configs = set()
        for name in result_names:
            if 'configurations' in results_data[name]:
                all_configs.update(results_data[name]['configurations'].keys())
        
        for config in all_configs:
            config_data = {}
            
            for name in result_names:
                if (config in results_data[name].get('configurations', {}) and
                    'error' not in results_data[name]['configurations'][config]):
                    
                    config_result = results_data[name]['configurations'][config]
                    config_data[name] = {
                        'optimal_frequency': config_result.get('optimal_frequency', 0),
                        'energy_savings': config_result.get('configuration_summary', {}).get('energy_savings_percent', 0),
                        'performance_penalty': config_result.get('configuration_summary', {}).get('performance_penalty_percent', 0)
                    }
            
            if len(config_data) > 1:  # Only include configs that appear in multiple results
                config_trends[config] = config_data
        
        comparison['configuration_trends'] = config_trends
        
        # Generate recommendations
        recommendations = []
        
        if performance_metrics:
            # Check for improving or degrading trends
            if len(result_names) >= 2 and comparison['temporal_analysis']['chronological_order']:
                ordered_names = comparison['temporal_analysis']['chronological_order']
                
                # Compare first and last results
                first_result = performance_metrics.get(ordered_names[0], {})
                last_result = performance_metrics.get(ordered_names[-1], {})
                
                if first_result and last_result:
                    first_energy = first_result.get('energy_savings', {}).get('mean', 0)
                    last_energy = last_result.get('energy_savings', {}).get('mean', 0)
                    
                    if last_energy > first_energy + 5:  # 5% improvement
                        recommendations.append(f"Energy savings improved from {first_energy:.1f}% to {last_energy:.1f}% over time")
                    elif first_energy > last_energy + 5:  # 5% degradation
                        recommendations.append(f"Energy savings decreased from {first_energy:.1f}% to {last_energy:.1f}% - investigate causes")
            
            # Check for consistency across results
            energy_means = [metrics.get('energy_savings', {}).get('mean', 0) for metrics in performance_metrics.values()]
            if len(energy_means) > 1 and max(energy_means) - min(energy_means) > 15:  # High variation
                recommendations.append("High variation in results across different optimization runs - consider environmental factors")
        
        comparison['recommendations'] = recommendations
        
        return comparison
    
    def generate_comparative_report(self, 
                                  report_name: Optional[str] = None) -> Path:
        """
        Generate comprehensive comparative analysis report.
        
        Args:
            report_name: Optional name for the report
            
        Returns:
            Path to generated report
        """
        if not report_name:
            report_name = f"comparative_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Generating comparative analysis report: {report_name}")
        
        report_lines = []
        report_lines.append("# Comparative Analysis Report")
        report_lines.append(f"Generated: {pd.Timestamp.now().isoformat()}")
        report_lines.append(f"Framework Version: {get_framework_version()}")
        report_lines.append("")
        
        # Summary of available data
        report_lines.append("## Available Data")
        report_lines.append("")
        
        if self.datasets:
            report_lines.append("### Datasets")
            for name, info in self.datasets.items():
                report_lines.append(f"- **{name}**: {info['description']}")
                report_lines.append(f"  - Records: {len(info['data'])}")
                report_lines.append(f"  - Configurations: {info['summary']['metadata']['configurations']}")
                report_lines.append("")
        
        if self.optimization_results:
            report_lines.append("### Optimization Results")
            for name, info in self.optimization_results.items():
                report_lines.append(f"- **{name}**: {info['description']}")
                report_lines.append(f"  - Timestamp: {info['timestamp']}")
                report_lines.append("")
        
        # Add comparison results if available
        if self.comparison_results:
            report_lines.append("## Comparison Results")
            report_lines.append("")
            
            for comparison_type, results in self.comparison_results.items():
                report_lines.append(f"### {comparison_type.replace('_', ' ').title()}")
                
                if 'recommendations' in results:
                    report_lines.append("**Recommendations:**")
                    for rec in results['recommendations']:
                        report_lines.append(f"- {rec}")
                report_lines.append("")
        
        # Add analysis guidelines
        report_lines.append("## Analysis Guidelines")
        report_lines.append("")
        report_lines.append("### Method Comparison")
        report_lines.append("- Compare EDP, energy, and performance optimization methods")
        report_lines.append("- Evaluate trade-offs between energy savings and performance impact")
        report_lines.append("- Consider success rates and consistency across configurations")
        report_lines.append("")
        
        report_lines.append("### Dataset Comparison")
        report_lines.append("- Analyze optimization effectiveness across different hardware/software configurations")
        report_lines.append("- Identify environmental factors affecting optimization results")
        report_lines.append("- Validate reproducibility of optimization strategies")
        report_lines.append("")
        
        report_lines.append("### Temporal Analysis")
        report_lines.append("- Track optimization performance over time")
        report_lines.append("- Identify trends in energy savings and performance penalties")
        report_lines.append("- Monitor for optimization degradation or improvement")
        report_lines.append("")
        
        # Save report
        report_path = self.output_dir / f"{report_name}.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comparative analysis report generated: {report_path}")
        return report_path
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """
        Run comprehensive comparative analysis on all available data.
        
        Returns:
            Complete comparison results
        """
        logger.info("🔍 Running comprehensive comparative analysis...")
        
        all_results = {
            'analysis_type': 'comprehensive_comparison',
            'timestamp': pd.Timestamp.now().isoformat(),
            'framework_version': get_framework_version(),
            'datasets': list(self.datasets.keys()),
            'optimization_results': list(self.optimization_results.keys()),
            'comparisons_performed': []
        }
        
        try:
            # Method comparison for each dataset
            if self.datasets:
                logger.info("Performing method comparisons...")
                for dataset_name in self.datasets.keys():
                    try:
                        method_comparison = self.compare_optimization_methods(dataset_name)
                        all_results['comparisons_performed'].append({
                            'type': 'method_comparison',
                            'dataset': dataset_name,
                            'status': 'completed'
                        })
                        self.comparison_results[f'method_comparison_{dataset_name}'] = method_comparison
                    except Exception as e:
                        logger.error(f"Method comparison failed for {dataset_name}: {e}")
                        all_results['comparisons_performed'].append({
                            'type': 'method_comparison',
                            'dataset': dataset_name,
                            'status': 'failed',
                            'error': str(e)
                        })
            
            # Dataset comparison
            if len(self.datasets) > 1:
                logger.info("Performing dataset comparison...")
                try:
                    dataset_comparison = self.compare_datasets(list(self.datasets.keys()))
                    all_results['comparisons_performed'].append({
                        'type': 'dataset_comparison',
                        'datasets': list(self.datasets.keys()),
                        'status': 'completed'
                    })
                    self.comparison_results['dataset_comparison'] = dataset_comparison
                except Exception as e:
                    logger.error(f"Dataset comparison failed: {e}")
                    all_results['comparisons_performed'].append({
                        'type': 'dataset_comparison',
                        'datasets': list(self.datasets.keys()),
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Optimization results comparison
            if len(self.optimization_results) > 1:
                logger.info("Performing optimization results comparison...")
                try:
                    results_comparison = self.compare_optimization_results(list(self.optimization_results.keys()))
                    all_results['comparisons_performed'].append({
                        'type': 'results_comparison',
                        'results': list(self.optimization_results.keys()),
                        'status': 'completed'
                    })
                    self.comparison_results['results_comparison'] = results_comparison
                except Exception as e:
                    logger.error(f"Results comparison failed: {e}")
                    all_results['comparisons_performed'].append({
                        'type': 'results_comparison',
                        'results': list(self.optimization_results.keys()),
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Generate comprehensive report
            report_path = self.generate_comparative_report()
            all_results['report_path'] = str(report_path)
            
            # Save complete results
            results_path = self.output_dir / f"comprehensive_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_results(all_results, results_path)
            
            logger.info("✅ Comprehensive comparative analysis completed!")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Report generated: {report_path}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Comprehensive comparison failed: {e}")
            raise


def comparative_analysis(datasets: Optional[Dict[str, str]] = None,
                       results: Optional[Dict[str, str]] = None,
                       config_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for comparative analysis.
    
    Args:
        datasets: Dict mapping names to dataset paths
        results: Dict mapping names to optimization result paths
        config_path: Optional path to configuration file
        output_dir: Optional output directory
        
    Returns:
        Complete comparative analysis results
    """
    # Set up logging
    setup_logging("INFO")
    
    # Initialize workflow
    workflow = ComparativeAnalysisWorkflow(output_dir, config_path)
    
    # Add datasets
    if datasets:
        for name, path in datasets.items():
            workflow.add_dataset(name, path)
    
    # Add optimization results
    if results:
        for name, path in results.items():
            workflow.add_optimization_results(name, path)
    
    # Run comprehensive comparison
    return workflow.run_comprehensive_comparison()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparative Analysis for GPU Optimization")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--datasets", nargs='+', help="Dataset paths (name:path format)")
    parser.add_argument("--results", nargs='+', help="Optimization result paths (name:path format)")
    
    args = parser.parse_args()
    
    # Parse datasets and results
    datasets = {}
    results = {}
    
    if args.datasets:
        for item in args.datasets:
            if ':' in item:
                name, path = item.split(':', 1)
                datasets[name] = path
            else:
                datasets[f"dataset_{len(datasets)}"] = item
    
    if args.results:
        for item in args.results:
            if ':' in item:
                name, path = item.split(':', 1)
                results[name] = path
            else:
                results[f"results_{len(results)}"] = item
    
    # Run comparative analysis
    comparative_analysis(
        datasets=datasets if datasets else None,
        results=results if results else None,
        config_path=args.config,
        output_dir=args.output_dir
    )
