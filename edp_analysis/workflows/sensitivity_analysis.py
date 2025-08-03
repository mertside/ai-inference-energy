"""
Sensitivity Analysis Workflow

This module provides comprehensive sensitivity analysis for GPU frequency
optimization parameters, exploring how changes in constraints, methods,
and environmental factors affect optimization results.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import sys
import itertools

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    ProfilingDataLoader, FrequencyOptimizer, PerformanceConstraintManager,
    load_config, save_results, create_deployment_summary,
    get_framework_version, setup_logging, ensure_directory
)

logger = logging.getLogger(__name__)


class SensitivityAnalysisWorkflow:
    """
    Comprehensive sensitivity analysis workflow for optimization parameters.
    """
    
    def __init__(self, 
                 data_path: str,
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize sensitivity analysis workflow.
        
        Args:
            data_path: Path to aggregated profiling data
            config_path: Optional path to configuration file
            output_dir: Optional output directory for results
        """
        self.data_path = data_path
        self.config_path = config_path
        self.output_dir = ensure_directory(output_dir or "sensitivity_analysis_results")
        
        # Load configuration and data
        self.config = load_config(config_path) if config_path else self._get_default_config()
        self.data_loader = None
        self.data = None
        self.optimizer = None
        
        # Results storage
        self.sensitivity_results = {}
        self.parameter_ranges = {}
        
        logger.info(f"Sensitivity analysis initialized: {data_path}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for sensitivity analysis."""
        return {
            'sensitivity': {
                'performance_constraint_range': [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
                'energy_weight_range': [0.1, 0.3, 0.5, 0.7, 0.9],
                'frequency_ranges': {
                    'min_freq_multipliers': [0.5, 0.6, 0.7, 0.8, 0.9],
                    'max_freq_multipliers': [1.0, 1.1, 1.2, 1.3, 1.4]
                },
                'optimization_methods': ['edp', 'energy', 'performance']
            },
            'analysis': {
                'statistical_tests': True,
                'confidence_level': 0.95,
                'correlation_analysis': True,
                'robustness_testing': True
            }
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for sensitivity analysis."""
        logger.info("Loading data for sensitivity analysis...")
        
        self.data_loader = ProfilingDataLoader(self.config_path)
        self.data = self.data_loader.load_aggregated_data(self.data_path)
        
        # Initialize optimizer
        self.optimizer = FrequencyOptimizer(self.config_path)
        
        logger.info(f"Data loaded: {len(self.data)} records")
        return self.data
    
    def analyze_performance_constraint_sensitivity(self, 
                                                 workloads: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze sensitivity to performance constraint variations.
        
        Args:
            workloads: Optional list of specific workloads to analyze
            
        Returns:
            Performance constraint sensitivity results
        """
        logger.info("Analyzing performance constraint sensitivity...")
        
        if self.data is None:
            self.load_data()
        
        # Get constraint ranges from config
        constraint_range = self.config['sensitivity']['performance_constraint_range']
        
        if workloads is None:
            workloads = self.data['workload'].unique().tolist()
        
        results = {
            'analysis_type': 'performance_constraint_sensitivity',
            'constraint_range': constraint_range,
            'workloads': workloads,
            'detailed_results': {},
            'summary': {},
            'recommendations': []
        }
        
        # Test each constraint value
        for constraint in constraint_range:
            logger.info(f"Testing constraint: {constraint}")
            
            # Create modified config
            test_config = self.config.copy()
            for workload in workloads:
                if 'optimization' not in test_config:
                    test_config['optimization'] = {}
                if 'performance_constraints' not in test_config['optimization']:
                    test_config['optimization']['performance_constraints'] = {}
                test_config['optimization']['performance_constraints'][workload] = constraint
            
            # Create temporary optimizer with modified config
            temp_optimizer = FrequencyOptimizer(config_dict=test_config)
            
            # Run optimization
            try:
                optimization_results = temp_optimizer.optimize_all_configurations(
                    self.data, 
                    methods=['edp']  # Use consistent method for comparison
                )
                
                # Extract key metrics
                constraint_results = {
                    'constraint_value': constraint,
                    'energy_savings': optimization_results['summary']['energy_savings'],
                    'performance_penalty': optimization_results['summary']['performance_penalty'],
                    'successful_optimizations': optimization_results['summary']['successful_optimizations'],
                    'total_configurations': optimization_results['summary']['total_configurations']
                }
                
                results['detailed_results'][str(constraint)] = constraint_results
                
            except Exception as e:
                logger.error(f"Optimization failed for constraint {constraint}: {e}")
                results['detailed_results'][str(constraint)] = {
                    'constraint_value': constraint,
                    'error': str(e)
                }
        
        # Analyze trends
        results['summary'] = self._analyze_constraint_trends(results['detailed_results'])
        results['recommendations'] = self._generate_constraint_recommendations(results)
        
        # Save results
        results_path = self.output_dir / f"constraint_sensitivity_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, results_path)
        
        self.sensitivity_results['performance_constraints'] = results
        logger.info(f"Performance constraint sensitivity analysis completed: {results_path}")
        
        return results
    
    def _analyze_constraint_trends(self, detailed_results: Dict) -> Dict[str, Any]:
        """Analyze trends in constraint sensitivity results."""
        summary = {
            'trends': {},
            'optimal_ranges': {},
            'trade_offs': {}
        }
        
        # Extract data for trend analysis
        constraints = []
        energy_savings = []
        performance_penalties = []
        success_rates = []
        
        for constraint_str, result in detailed_results.items():
            if 'error' not in result:
                constraints.append(float(constraint_str))
                energy_savings.append(result['energy_savings']['mean'])
                performance_penalties.append(abs(result['performance_penalty']['mean']))
                success_rates.append(
                    result['successful_optimizations'] / max(result['total_configurations'], 1)
                )
        
        if len(constraints) > 1:
            # Sort by constraint value
            sorted_data = sorted(zip(constraints, energy_savings, performance_penalties, success_rates))
            constraints, energy_savings, performance_penalties, success_rates = zip(*sorted_data)
            
            # Analyze trends
            summary['trends'] = {
                'constraint_range': f"{min(constraints):.2f} - {max(constraints):.2f}",
                'energy_savings_trend': self._calculate_trend(constraints, energy_savings),
                'performance_penalty_trend': self._calculate_trend(constraints, performance_penalties),
                'success_rate_trend': self._calculate_trend(constraints, success_rates)
            }
            
            # Find optimal ranges
            # Best energy savings
            max_energy_idx = energy_savings.index(max(energy_savings))
            # Best trade-off (high energy savings, low penalty)
            trade_off_scores = [e / (p + 0.01) for e, p in zip(energy_savings, performance_penalties)]
            best_trade_off_idx = trade_off_scores.index(max(trade_off_scores))
            
            summary['optimal_ranges'] = {
                'best_energy_savings': {
                    'constraint': constraints[max_energy_idx],
                    'energy_savings': energy_savings[max_energy_idx],
                    'performance_penalty': performance_penalties[max_energy_idx]
                },
                'best_trade_off': {
                    'constraint': constraints[best_trade_off_idx],
                    'energy_savings': energy_savings[best_trade_off_idx],
                    'performance_penalty': performance_penalties[best_trade_off_idx],
                    'trade_off_score': trade_off_scores[best_trade_off_idx]
                }
            }
        
        return summary
    
    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for two series."""
        if len(x_values) < 2:
            return {'slope': 0, 'correlation': 0}
        
        # Simple linear regression
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Correlation coefficient
        x_std = (sum((x - x_mean) ** 2 for x in x_values) / n) ** 0.5
        y_std = (sum((y - y_mean) ** 2 for y in y_values) / n) ** 0.5
        
        correlation = numerator / (n * x_std * y_std) if (x_std * y_std) != 0 else 0
        
        return {'slope': slope, 'correlation': correlation}
    
    def _generate_constraint_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on constraint sensitivity analysis."""
        recommendations = []
        
        if 'summary' in results and 'optimal_ranges' in results['summary']:
            optimal = results['summary']['optimal_ranges']
            
            if 'best_trade_off' in optimal:
                trade_off = optimal['best_trade_off']
                recommendations.append(
                    f"Optimal performance constraint: {trade_off['constraint']:.2f} "
                    f"(Energy savings: {trade_off['energy_savings']:.1f}%, "
                    f"Performance penalty: {trade_off['performance_penalty']:.1f}%)"
                )
            
            if 'best_energy_savings' in optimal:
                best_energy = optimal['best_energy_savings']
                if best_energy['performance_penalty'] <= 20:  # Reasonable penalty
                    recommendations.append(
                        f"For maximum energy savings: constraint {best_energy['constraint']:.2f} "
                        f"({best_energy['energy_savings']:.1f}% savings)"
                    )
                else:
                    recommendations.append(
                        f"Maximum energy savings constraint ({best_energy['constraint']:.2f}) "
                        f"has high performance penalty ({best_energy['performance_penalty']:.1f}%)"
                    )
        
        # Check trends
        if 'summary' in results and 'trends' in results['summary']:
            trends = results['summary']['trends']
            
            if 'energy_savings_trend' in trends:
                energy_trend = trends['energy_savings_trend']
                if energy_trend['correlation'] > 0.7:
                    recommendations.append("Energy savings increase with looser constraints")
                elif energy_trend['correlation'] < -0.7:
                    recommendations.append("Energy savings decrease with looser constraints")
        
        return recommendations
    
    def analyze_method_sensitivity(self) -> Dict[str, Any]:
        """
        Analyze sensitivity to different optimization methods.
        
        Returns:
            Method sensitivity results
        """
        logger.info("Analyzing optimization method sensitivity...")
        
        if self.data is None:
            self.load_data()
        
        methods = self.config['sensitivity']['optimization_methods']
        
        results = {
            'analysis_type': 'method_sensitivity',
            'methods': methods,
            'detailed_results': {},
            'comparative_analysis': {},
            'recommendations': []
        }
        
        # Test each method
        method_results = {}
        
        for method in methods:
            logger.info(f"Testing method: {method}")
            
            try:
                optimization_result = self.optimizer.optimize_all_configurations(
                    self.data, 
                    methods=[method]
                )
                
                method_results[method] = {
                    'energy_savings': optimization_result['summary']['energy_savings'],
                    'performance_penalty': optimization_result['summary']['performance_penalty'],
                    'successful_optimizations': optimization_result['summary']['successful_optimizations'],
                    'total_configurations': optimization_result['summary']['total_configurations'],
                    'configuration_results': optimization_result['configurations']
                }
                
            except Exception as e:
                logger.error(f"Optimization failed for method {method}: {e}")
                method_results[method] = {'error': str(e)}
        
        results['detailed_results'] = method_results
        
        # Comparative analysis
        results['comparative_analysis'] = self._analyze_method_comparison(method_results)
        results['recommendations'] = self._generate_method_recommendations(results)
        
        # Save results
        results_path = self.output_dir / f"method_sensitivity_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, results_path)
        
        self.sensitivity_results['methods'] = results
        logger.info(f"Method sensitivity analysis completed: {results_path}")
        
        return results
    
    def _analyze_method_comparison(self, method_results: Dict) -> Dict[str, Any]:
        """Compare results across different optimization methods."""
        comparison = {
            'performance_metrics': {},
            'consistency_analysis': {},
            'configuration_agreement': {}
        }
        
        # Performance metrics comparison
        valid_methods = {method: result for method, result in method_results.items() 
                        if 'error' not in result}
        
        if valid_methods:
            # Extract metrics
            for method, result in valid_methods.items():
                comparison['performance_metrics'][method] = {
                    'energy_savings_mean': result['energy_savings']['mean'],
                    'energy_savings_std': result['energy_savings']['std'],
                    'performance_penalty_mean': abs(result['performance_penalty']['mean']),
                    'performance_penalty_std': result['performance_penalty']['std'],
                    'success_rate': result['successful_optimizations'] / max(result['total_configurations'], 1)
                }
            
            # Configuration agreement analysis
            if len(valid_methods) > 1:
                comparison['configuration_agreement'] = self._analyze_configuration_agreement(valid_methods)
        
        return comparison
    
    def _analyze_configuration_agreement(self, method_results: Dict) -> Dict[str, Any]:
        """Analyze agreement between methods on optimal configurations."""
        agreement = {
            'frequency_agreement': {},
            'ranking_agreement': {},
            'consensus_configurations': []
        }
        
        # Get all configurations that appear in all methods
        method_names = list(method_results.keys())
        common_configs = set()
        
        if method_names:
            # Start with configurations from first method
            first_method = method_names[0]
            if 'configuration_results' in method_results[first_method]:
                common_configs = set(method_results[first_method]['configuration_results'].keys())
                
                # Intersect with other methods
                for method in method_names[1:]:
                    if 'configuration_results' in method_results[method]:
                        common_configs &= set(method_results[method]['configuration_results'].keys())
        
        # Analyze agreement for common configurations
        for config in common_configs:
            config_analysis = {}
            
            for method in method_names:
                if (config in method_results[method].get('configuration_results', {}) and
                    'error' not in method_results[method]['configuration_results'][config]):
                    
                    config_result = method_results[method]['configuration_results'][config]
                    config_analysis[method] = {
                        'optimal_frequency': config_result.get('optimal_frequency', 0),
                        'energy_savings': config_result.get('configuration_summary', {}).get('energy_savings_percent', 0)
                    }
            
            if len(config_analysis) > 1:
                # Calculate frequency agreement (how close are the optimal frequencies)
                frequencies = [data['optimal_frequency'] for data in config_analysis.values()]
                freq_std = np.std(frequencies) if len(frequencies) > 1 else 0
                freq_mean = np.mean(frequencies) if len(frequencies) > 0 else 0
                
                agreement['frequency_agreement'][config] = {
                    'mean_frequency': freq_mean,
                    'std_frequency': freq_std,
                    'coefficient_of_variation': freq_std / freq_mean if freq_mean > 0 else 0,
                    'method_frequencies': config_analysis
                }
        
        # Identify consensus configurations (low variability across methods)
        consensus_threshold = 0.1  # 10% coefficient of variation
        
        for config, analysis in agreement['frequency_agreement'].items():
            if analysis['coefficient_of_variation'] <= consensus_threshold:
                agreement['consensus_configurations'].append({
                    'configuration': config,
                    'mean_frequency': analysis['mean_frequency'],
                    'agreement_score': 1 - analysis['coefficient_of_variation']
                })
        
        return agreement
    
    def _generate_method_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on method sensitivity analysis."""
        recommendations = []
        
        if 'comparative_analysis' in results:
            comparison = results['comparative_analysis']
            
            # Performance-based recommendations
            if 'performance_metrics' in comparison:
                metrics = comparison['performance_metrics']
                
                # Find best method for energy savings
                best_energy_method = max(metrics.items(), 
                                       key=lambda x: x[1]['energy_savings_mean'])
                
                # Find most consistent method
                most_consistent_method = min(metrics.items(), 
                                           key=lambda x: x[1]['energy_savings_std'])
                
                recommendations.append(
                    f"Best energy savings: {best_energy_method[0]} "
                    f"({best_energy_method[1]['energy_savings_mean']:.1f}% average)"
                )
                
                recommendations.append(
                    f"Most consistent: {most_consistent_method[0]} "
                    f"(±{most_consistent_method[1]['energy_savings_std']:.1f}% std dev)"
                )
            
            # Agreement-based recommendations
            if 'configuration_agreement' in comparison:
                agreement = comparison['configuration_agreement']
                
                if 'consensus_configurations' in agreement and agreement['consensus_configurations']:
                    consensus_count = len(agreement['consensus_configurations'])
                    recommendations.append(
                        f"Strong method agreement on {consensus_count} configurations - "
                        f"these are robust optimization targets"
                    )
                else:
                    recommendations.append(
                        "Limited agreement between methods - consider ensemble approach"
                    )
        
        return recommendations
    
    def analyze_parameter_robustness(self, 
                                   parameter_variations: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze robustness to parameter variations.
        
        Args:
            parameter_variations: Custom parameter variations to test
            
        Returns:
            Robustness analysis results
        """
        logger.info("Analyzing parameter robustness...")
        
        if self.data is None:
            self.load_data()
        
        # Default parameter variations
        if parameter_variations is None:
            parameter_variations = {
                'energy_weight': [0.3, 0.5, 0.7, 0.9],
                'temperature_factor': [0.8, 1.0, 1.2],
                'convergence_threshold': [0.001, 0.01, 0.1]
            }
        
        results = {
            'analysis_type': 'parameter_robustness',
            'parameter_variations': parameter_variations,
            'detailed_results': {},
            'robustness_metrics': {},
            'recommendations': []
        }
        
        # Baseline optimization (with default parameters)
        logger.info("Running baseline optimization...")
        baseline_result = self.optimizer.optimize_all_configurations(
            self.data, 
            methods=['edp']
        )
        
        results['baseline'] = {
            'energy_savings': baseline_result['summary']['energy_savings'],
            'performance_penalty': baseline_result['summary']['performance_penalty'],
            'successful_optimizations': baseline_result['summary']['successful_optimizations']
        }
        
        # Test parameter variations
        for param_name, param_values in parameter_variations.items():
            logger.info(f"Testing parameter: {param_name}")
            param_results = {}
            
            for param_value in param_values:
                try:
                    # Create modified config
                    test_config = self.config.copy()
                    
                    # Apply parameter variation (simplified - would need more sophisticated config modification)
                    if param_name == 'energy_weight':
                        if 'optimization' not in test_config:
                            test_config['optimization'] = {}
                        test_config['optimization']['energy_weight'] = param_value
                    
                    # Create temporary optimizer
                    temp_optimizer = FrequencyOptimizer(config_dict=test_config)
                    
                    # Run optimization
                    optimization_result = temp_optimizer.optimize_all_configurations(
                        self.data, 
                        methods=['edp']
                    )
                    
                    param_results[str(param_value)] = {
                        'energy_savings': optimization_result['summary']['energy_savings'],
                        'performance_penalty': optimization_result['summary']['performance_penalty'],
                        'successful_optimizations': optimization_result['summary']['successful_optimizations']
                    }
                    
                except Exception as e:
                    logger.error(f"Parameter test failed for {param_name}={param_value}: {e}")
                    param_results[str(param_value)] = {'error': str(e)}
            
            results['detailed_results'][param_name] = param_results
        
        # Calculate robustness metrics
        results['robustness_metrics'] = self._calculate_robustness_metrics(
            results['detailed_results'], 
            results['baseline']
        )
        
        results['recommendations'] = self._generate_robustness_recommendations(results)
        
        # Save results
        results_path = self.output_dir / f"robustness_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, results_path)
        
        self.sensitivity_results['robustness'] = results
        logger.info(f"Robustness analysis completed: {results_path}")
        
        return results
    
    def _calculate_robustness_metrics(self, 
                                    detailed_results: Dict, 
                                    baseline: Dict) -> Dict[str, Any]:
        """Calculate robustness metrics for parameter variations."""
        metrics = {}
        
        baseline_energy = baseline['energy_savings']['mean']
        
        for param_name, param_results in detailed_results.items():
            param_metrics = {
                'stability_score': 0,
                'max_deviation': 0,
                'coefficient_of_variation': 0,
                'robust_range': None
            }
            
            # Extract energy savings for valid results
            energy_values = []
            param_values = []
            
            for param_val, result in param_results.items():
                if 'error' not in result:
                    energy_values.append(result['energy_savings']['mean'])
                    param_values.append(float(param_val))
            
            if len(energy_values) > 1:
                # Calculate stability metrics
                energy_std = np.std(energy_values)
                energy_mean = np.mean(energy_values)
                
                param_metrics['coefficient_of_variation'] = energy_std / energy_mean if energy_mean > 0 else 0
                param_metrics['max_deviation'] = max(abs(e - baseline_energy) for e in energy_values)
                
                # Stability score (lower variation = higher score)
                param_metrics['stability_score'] = 1 / (1 + param_metrics['coefficient_of_variation'])
                
                # Find robust range (parameter values with <10% deviation from baseline)
                robust_values = []
                for param_val, energy_val in zip(param_values, energy_values):
                    deviation = abs(energy_val - baseline_energy) / baseline_energy
                    if deviation < 0.1:  # 10% threshold
                        robust_values.append(param_val)
                
                if robust_values:
                    param_metrics['robust_range'] = [min(robust_values), max(robust_values)]
            
            metrics[param_name] = param_metrics
        
        return metrics
    
    def _generate_robustness_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on robustness analysis."""
        recommendations = []
        
        if 'robustness_metrics' in results:
            metrics = results['robustness_metrics']
            
            # Identify most stable parameters
            stable_params = [(name, metric['stability_score']) 
                           for name, metric in metrics.items() 
                           if metric['stability_score'] > 0]
            
            if stable_params:
                stable_params.sort(key=lambda x: x[1], reverse=True)
                most_stable = stable_params[0]
                
                recommendations.append(
                    f"Most robust parameter: {most_stable[0]} "
                    f"(stability score: {most_stable[1]:.3f})"
                )
                
                # Check for parameters with robust ranges
                for param_name, metric in metrics.items():
                    if metric['robust_range']:
                        recommendations.append(
                            f"Robust range for {param_name}: "
                            f"{metric['robust_range'][0]:.3f} - {metric['robust_range'][1]:.3f}"
                        )
        
        # General robustness recommendations
        recommendations.append(
            "For production deployment, use parameter values within identified robust ranges"
        )
        recommendations.append(
            "Monitor optimization performance when changing parameters outside robust ranges"
        )
        
        return recommendations
    
    def run_comprehensive_sensitivity_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive sensitivity analysis across all parameters.
        
        Returns:
            Complete sensitivity analysis results
        """
        logger.info("🔍 Running comprehensive sensitivity analysis...")
        
        try:
            # Load data
            self.load_data()
            
            # Performance constraint sensitivity
            logger.info("1/3 Analyzing performance constraint sensitivity...")
            constraint_results = self.analyze_performance_constraint_sensitivity()
            
            # Method sensitivity
            logger.info("2/3 Analyzing optimization method sensitivity...")
            method_results = self.analyze_method_sensitivity()
            
            # Parameter robustness
            logger.info("3/3 Analyzing parameter robustness...")
            robustness_results = self.analyze_parameter_robustness()
            
            # Compile comprehensive results
            comprehensive_results = {
                'analysis_type': 'comprehensive_sensitivity',
                'framework_version': get_framework_version(),
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_source': str(self.data_path),
                'individual_analyses': {
                    'performance_constraints': constraint_results,
                    'optimization_methods': method_results,
                    'parameter_robustness': robustness_results
                },
                'overall_recommendations': self._generate_overall_recommendations()
            }
            
            # Save comprehensive results
            results_path = self.output_dir / f"comprehensive_sensitivity_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_results(comprehensive_results, results_path)
            
            # Generate summary report
            report_path = self._generate_sensitivity_report(comprehensive_results)
            
            logger.info("✅ Comprehensive sensitivity analysis completed!")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Report generated: {report_path}")
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Comprehensive sensitivity analysis failed: {e}")
            raise
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations from all sensitivity analyses."""
        recommendations = []
        
        # Collect recommendations from all analyses
        all_recs = []
        for analysis_name, results in self.sensitivity_results.items():
            if 'recommendations' in results:
                all_recs.extend(results['recommendations'])
        
        # Generate consolidated recommendations
        recommendations.append("=== SENSITIVITY ANALYSIS SUMMARY ===")
        
        if 'performance_constraints' in self.sensitivity_results:
            recommendations.append("PERFORMANCE CONSTRAINTS:")
            recommendations.extend(self.sensitivity_results['performance_constraints']['recommendations'])
        
        if 'methods' in self.sensitivity_results:
            recommendations.append("\nOPTIMIZATION METHODS:")
            recommendations.extend(self.sensitivity_results['methods']['recommendations'])
        
        if 'robustness' in self.sensitivity_results:
            recommendations.append("\nPARAMETER ROBUSTNESS:")
            recommendations.extend(self.sensitivity_results['robustness']['recommendations'])
        
        recommendations.append("\n=== OVERALL GUIDANCE ===")
        recommendations.append("1. Use identified optimal performance constraint ranges for target workloads")
        recommendations.append("2. Consider method agreement when selecting optimization approach")
        recommendations.append("3. Stay within robust parameter ranges for production deployment")
        recommendations.append("4. Monitor sensitivity to environmental changes in production")
        
        return recommendations
    
    def _generate_sensitivity_report(self, results: Dict) -> Path:
        """Generate comprehensive sensitivity analysis report."""
        report_lines = []
        report_lines.append("# Sensitivity Analysis Report")
        report_lines.append(f"Generated: {pd.Timestamp.now().isoformat()}")
        report_lines.append(f"Framework Version: {get_framework_version()}")
        report_lines.append(f"Data Source: {self.data_path}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append("This sensitivity analysis evaluates the robustness of GPU frequency optimization")
        report_lines.append("across different parameters, methods, and constraints.")
        report_lines.append("")
        
        # Individual Analysis Summaries
        if 'individual_analyses' in results:
            analyses = results['individual_analyses']
            
            # Performance Constraint Analysis
            if 'performance_constraints' in analyses:
                report_lines.append("### Performance Constraint Sensitivity")
                constraint_analysis = analyses['performance_constraints']
                
                if 'summary' in constraint_analysis and 'optimal_ranges' in constraint_analysis['summary']:
                    optimal = constraint_analysis['summary']['optimal_ranges']
                    if 'best_trade_off' in optimal:
                        trade_off = optimal['best_trade_off']
                        report_lines.append(f"- **Optimal Constraint**: {trade_off['constraint']:.2f}")
                        report_lines.append(f"- **Energy Savings**: {trade_off['energy_savings']:.1f}%")
                        report_lines.append(f"- **Performance Penalty**: {trade_off['performance_penalty']:.1f}%")
                report_lines.append("")
            
            # Method Sensitivity
            if 'optimization_methods' in analyses:
                report_lines.append("### Optimization Method Sensitivity")
                method_analysis = analyses['optimization_methods']
                
                if 'comparative_analysis' in method_analysis and 'performance_metrics' in method_analysis['comparative_analysis']:
                    metrics = method_analysis['comparative_analysis']['performance_metrics']
                    
                    report_lines.append("**Method Performance:**")
                    for method, perf in metrics.items():
                        report_lines.append(f"- {method}: {perf['energy_savings_mean']:.1f}% energy savings")
                report_lines.append("")
            
            # Robustness Analysis
            if 'parameter_robustness' in analyses:
                report_lines.append("### Parameter Robustness")
                robustness_analysis = analyses['parameter_robustness']
                
                if 'robustness_metrics' in robustness_analysis:
                    metrics = robustness_analysis['robustness_metrics']
                    
                    report_lines.append("**Parameter Stability:**")
                    for param, metric in metrics.items():
                        stability = metric.get('stability_score', 0)
                        report_lines.append(f"- {param}: {stability:.3f} stability score")
                report_lines.append("")
        
        # Recommendations
        if 'overall_recommendations' in results:
            report_lines.append("## Recommendations")
            report_lines.append("")
            for rec in results['overall_recommendations']:
                if rec.startswith("==="):
                    report_lines.append(f"### {rec.replace('=', '').strip()}")
                elif rec.startswith(("PERFORMANCE", "OPTIMIZATION", "PARAMETER", "OVERALL")):
                    report_lines.append(f"#### {rec}")
                else:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
        
        # Save report
        report_path = self.output_dir / f"sensitivity_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path


def sensitivity_analysis(data_path: str,
                       config_path: Optional[str] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for sensitivity analysis.
    
    Args:
        data_path: Path to aggregated profiling data
        config_path: Optional path to configuration file
        output_dir: Optional output directory
        
    Returns:
        Complete sensitivity analysis results
    """
    # Set up logging
    setup_logging("INFO")
    
    # Initialize and run workflow
    workflow = SensitivityAnalysisWorkflow(data_path, config_path, output_dir)
    return workflow.run_comprehensive_sensitivity_analysis()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sensitivity Analysis for GPU Optimization")
    parser.add_argument("data_path", help="Path to aggregated profiling data")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run sensitivity analysis
    results = sensitivity_analysis(
        data_path=args.data_path,
        config_path=args.config,
        output_dir=args.output_dir
    )
