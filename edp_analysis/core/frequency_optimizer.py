"""
Frequency Optimizer Module

This module provides advanced frequency optimization algorithms for GPU inference workloads,
consolidating and enhancing functionality from the original optimization modules.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml
from pathlib import Path

from .energy_calculator import EnergyCalculator, EDPResult

logger = logging.getLogger(__name__)


class FrequencyOptimizer:
    """
    Advanced frequency optimizer with multiple optimization strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize frequency optimizer with configuration."""
        self.config = self._load_config(config_path)
        self.energy_calculator = EnergyCalculator()
        self.optimization_results = {}
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'optimization': {
                'performance_constraints': {
                    'llama': 0.05,
                    'stable_diffusion': 0.20,
                    'vit': 0.20,
                    'whisper': 0.15
                },
                'energy_weight': 0.7,
                'performance_weight': 0.3
            },
            'gpus': {
                'a100': {'max_frequency': 1410, 'practical_range': [70, 100]},
                'v100': {'max_frequency': 1380, 'practical_range': [70, 100]}
            }
        }
    
    def get_performance_constraint(self, application: str) -> float:
        """Get performance constraint for an application."""
        app_lower = application.lower().replace('+', '')
        constraints = self.config['optimization']['performance_constraints']
        
        # Try exact match first
        if app_lower in constraints:
            return constraints[app_lower]
        
        # Try partial matches
        for app_key in constraints:
            if app_key in app_lower or app_lower in app_key:
                return constraints[app_key]
        
        # Default constraint
        logger.warning(f"No performance constraint found for {application}, using 0.15 (15%)")
        return 0.15
    
    def get_practical_frequency_range(self, gpu: str, frequencies: List[float]) -> Tuple[float, float]:
        """Get practical frequency range for a GPU."""
        gpu_lower = gpu.lower()
        
        if gpu_lower in self.config['gpus']:
            gpu_config = self.config['gpus'][gpu_lower]
            max_freq = gpu_config['max_frequency']
            range_percent = gpu_config['practical_range']
            
            min_freq = max_freq * range_percent[0] / 100
            max_freq = max_freq * range_percent[1] / 100
            
            # Constrain to available frequencies
            available_freqs = np.array(frequencies)
            min_freq = available_freqs[available_freqs >= min_freq].min()
            max_freq = available_freqs[available_freqs <= max_freq].max()
            
            return min_freq, max_freq
        else:
            # Use full available range if GPU not in config
            return min(frequencies), max(frequencies)
    
    def optimize_single_configuration(self, 
                                    data: pd.DataFrame,
                                    gpu: str,
                                    application: str,
                                    methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize frequency for a single GPU-application configuration.
        
        Args:
            data: Complete profiling dataset
            gpu: GPU type
            application: Application name
            methods: Optimization methods to use
            
        Returns:
            Dictionary with optimization results
        """
        if methods is None:
            methods = ["edp", "energy"]
        
        logger.info(f"Optimizing {gpu}+{application}")
        
        # Filter data for this configuration
        config_data = data[
            (data['gpu'] == gpu) & 
            (data['application'] == application)
        ].copy()
        
        if len(config_data) == 0:
            raise ValueError(f"No data found for {gpu}+{application}")
        
        # Get performance constraint
        performance_constraint = self.get_performance_constraint(application)
        logger.info(f"Performance constraint: {performance_constraint*100:.1f}%")
        
        # Get practical frequency range
        frequencies = config_data['frequency'].tolist()
        min_freq, max_freq = self.get_practical_frequency_range(gpu, frequencies)
        logger.info(f"Practical frequency range: {min_freq}-{max_freq} MHz")
        
        # Filter to practical range
        config_data = config_data[
            (config_data['frequency'] >= min_freq) & 
            (config_data['frequency'] <= max_freq)
        ]
        
        # Set baseline (maximum frequency in practical range)
        baseline_data = config_data[config_data['frequency'] == config_data['frequency'].max()]
        if len(baseline_data) == 0:
            raise ValueError(f"No baseline data found for {gpu}+{application}")
        
        baseline = baseline_data.iloc[0]
        self.energy_calculator.set_baseline(
            baseline['frequency'],
            baseline['execution_time'],
            baseline['power']
        )
        
        # Optimize with each method
        results = {}
        for method in methods:
            try:
                result = self.energy_calculator.find_optimal_frequency(
                    config_data,
                    method=method,
                    performance_constraint=performance_constraint
                )
                results[method] = result
                logger.info(f"{method}: {result.frequency} MHz, {result.energy_savings_percent:.1f}% energy savings")
            except Exception as e:
                logger.error(f"Optimization failed for {method}: {e}")
                results[method] = None
        
        # Select best result (prefer EDP if available)
        best_method = "edp" if "edp" in results and results["edp"] is not None else None
        if best_method is None:
            best_method = next((m for m in methods if results.get(m) is not None), None)
        
        if best_method is None:
            raise ValueError(f"All optimization methods failed for {gpu}+{application}")
        
        best_result = results[best_method]
        
        # Classify performance penalty
        penalty_category = self._classify_performance_penalty(best_result.performance_penalty_percent)
        
        return {
            'gpu': gpu,
            'application': application,
            'baseline_frequency': baseline['frequency'],
            'optimal_frequency': best_result.frequency,
            'optimization_method': best_method,
            'results': results,
            'best_result': best_result,
            'performance_category': penalty_category,
            'deployment_recommendation': self._get_deployment_recommendation(penalty_category),
            'configuration_summary': {
                'frequency_reduction_mhz': baseline['frequency'] - best_result.frequency,
                'frequency_reduction_percent': (baseline['frequency'] - best_result.frequency) / baseline['frequency'] * 100,
                'energy_savings_percent': best_result.energy_savings_percent,
                'performance_penalty_percent': best_result.performance_penalty_percent,
                'edp_improvement_percent': best_result.edp_improvement_percent
            }
        }
    
    def _classify_performance_penalty(self, penalty_percent: float) -> str:
        """Classify performance penalty into categories."""
        abs_penalty = abs(penalty_percent)
        
        if abs_penalty <= 5:
            return "excellent"
        elif abs_penalty <= 10:
            return "good"
        elif abs_penalty <= 20:
            return "acceptable"
        elif abs_penalty <= 50:
            return "moderate"
        else:
            return "high"
    
    def _get_deployment_recommendation(self, category: str) -> str:
        """Get deployment recommendation based on performance category."""
        recommendations = {
            "excellent": "Recommended for immediate production deployment",
            "good": "Recommended for production deployment",
            "acceptable": "Suitable for production with monitoring",
            "moderate": "Consider A/B testing before deployment",
            "high": "Suitable only for batch processing"
        }
        return recommendations.get(category, "Requires careful evaluation")
    
    def optimize_all_configurations(self, 
                                  data: pd.DataFrame,
                                  methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize frequencies for all available configurations.
        
        Args:
            data: Complete profiling dataset
            methods: Optimization methods to use
            
        Returns:
            Dictionary with all optimization results
        """
        if methods is None:
            methods = ["edp", "energy"]
        
        # Get all available configurations
        configurations = data.groupby(['gpu', 'application']).size().index.tolist()
        logger.info(f"Found {len(configurations)} configurations to optimize")
        
        all_results = {}
        successful_optimizations = 0
        
        for gpu, application in configurations:
            config_key = f"{gpu}+{application}"
            
            try:
                result = self.optimize_single_configuration(data, gpu, application, methods)
                all_results[config_key] = result
                successful_optimizations += 1
                logger.info(f"✅ Successfully optimized {config_key}")
            except Exception as e:
                logger.error(f"❌ Failed to optimize {config_key}: {e}")
                all_results[config_key] = {
                    'gpu': gpu,
                    'application': application,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generate summary statistics
        summary = self._generate_optimization_summary(all_results, successful_optimizations)
        
        return {
            'summary': summary,
            'configurations': all_results,
            'metadata': {
                'total_configurations': len(configurations),
                'successful_optimizations': successful_optimizations,
                'optimization_methods': methods,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
    
    def _generate_optimization_summary(self, results: Dict, successful_count: int) -> Dict:
        """Generate summary statistics for optimization results."""
        successful_results = [r for r in results.values() if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful optimizations'}
        
        # Extract metrics
        energy_savings = [r['configuration_summary']['energy_savings_percent'] for r in successful_results]
        performance_penalties = [abs(r['configuration_summary']['performance_penalty_percent']) for r in successful_results]
        frequency_reductions = [r['configuration_summary']['frequency_reduction_percent'] for r in successful_results]
        
        # Count by category
        categories = [r['performance_category'] for r in successful_results]
        category_counts = pd.Series(categories).value_counts().to_dict()
        
        return {
            'successful_optimizations': successful_count,
            'energy_savings': {
                'mean': np.mean(energy_savings),
                'median': np.median(energy_savings),
                'std': np.std(energy_savings),
                'min': np.min(energy_savings),
                'max': np.max(energy_savings)
            },
            'performance_penalties': {
                'mean': np.mean(performance_penalties),
                'median': np.median(performance_penalties),
                'std': np.std(performance_penalties),
                'min': np.min(performance_penalties),
                'max': np.max(performance_penalties)
            },
            'frequency_reductions': {
                'mean': np.mean(frequency_reductions),
                'median': np.median(frequency_reductions),
                'std': np.std(frequency_reductions),
                'min': np.min(frequency_reductions),
                'max': np.max(frequency_reductions)
            },
            'performance_categories': category_counts,
            'deployment_ready_count': sum(1 for r in successful_results if r['performance_category'] in ['excellent', 'good', 'acceptable'])
        }
    
    def generate_deployment_configs(self, 
                                  optimization_results: Dict,
                                  output_path: Optional[str] = None) -> Dict:
        """
        Generate deployment configurations from optimization results.
        
        Args:
            optimization_results: Results from optimize_all_configurations
            output_path: Optional path to save deployment configs
            
        Returns:
            Dictionary with deployment configurations
        """
        deployment_configs = {
            'version': '1.0.0',
            'generated_timestamp': pd.Timestamp.now().isoformat(),
            'configurations': {},
            'deployment_commands': {},
            'categories': {
                'production_ready': [],
                'a_b_testing': [],
                'batch_only': []
            }
        }
        
        for config_name, result in optimization_results['configurations'].items():
            if 'error' in result:
                continue
            
            gpu = result['gpu']
            optimal_freq = result['optimal_frequency']
            baseline_freq = result['baseline_frequency']
            penalty = abs(result['configuration_summary']['performance_penalty_percent'])
            
            # Get GPU memory frequency
            gpu_lower = gpu.lower()
            if gpu_lower in self.config['gpus']:
                memory_freq = self.config['gpus'][gpu_lower].get('memory_frequency', 1215)
            else:
                memory_freq = 1215 if 'a100' in gpu_lower else 877
            
            # Create deployment configuration
            config = {
                'gpu': gpu,
                'application': result['application'],
                'optimal_frequency': optimal_freq,
                'baseline_frequency': baseline_freq,
                'memory_frequency': memory_freq,
                'performance_penalty_percent': result['configuration_summary']['performance_penalty_percent'],
                'energy_savings_percent': result['configuration_summary']['energy_savings_percent'],
                'deployment_category': result['performance_category'],
                'deployment_recommendation': result['deployment_recommendation'],
                'nvidia_smi_command': f"nvidia-smi -ac {memory_freq},{int(optimal_freq)}",
                'reset_command': f"nvidia-smi -ac {memory_freq},{int(baseline_freq)}"
            }
            
            deployment_configs['configurations'][config_name] = config
            deployment_configs['deployment_commands'][config_name] = {
                'deploy': config['nvidia_smi_command'],
                'reset': config['reset_command']
            }
            
            # Categorize for deployment
            if penalty <= 20:
                deployment_configs['categories']['production_ready'].append(config_name)
            elif penalty <= 50:
                deployment_configs['categories']['a_b_testing'].append(config_name)
            else:
                deployment_configs['categories']['batch_only'].append(config_name)
        
        # Save to file if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(deployment_configs, f, indent=2)
            logger.info(f"Deployment configurations saved to {output_path}")
        
        return deployment_configs
    
    def validate_optimization_results(self, results: Dict) -> Dict[str, bool]:
        """
        Validate optimization results for consistency and reasonableness.
        
        Args:
            results: Results from optimize_all_configurations
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {
            'has_successful_optimizations': False,
            'energy_savings_reasonable': False,
            'performance_penalties_acceptable': False,
            'frequency_reductions_reasonable': False,
            'all_validations_passed': False
        }
        
        if results['metadata']['successful_optimizations'] == 0:
            return validation_results
        
        validation_results['has_successful_optimizations'] = True
        
        # Check energy savings (should be positive and reasonable)
        energy_stats = results['summary']['energy_savings']
        validation_results['energy_savings_reasonable'] = (
            energy_stats['min'] >= 0 and
            energy_stats['max'] <= 100 and
            energy_stats['mean'] > 5  # At least 5% average savings
        )
        
        # Check performance penalties (should be within configured constraints)
        penalty_stats = results['summary']['performance_penalties']
        validation_results['performance_penalties_acceptable'] = (
            penalty_stats['max'] <= 100  # No more than 100% penalty (2x slower)
        )
        
        # Check frequency reductions (should be reasonable)
        freq_stats = results['summary']['frequency_reductions']
        validation_results['frequency_reductions_reasonable'] = (
            freq_stats['min'] >= 0 and
            freq_stats['max'] <= 60 and  # No more than 60% reduction
            freq_stats['mean'] > 5  # At least 5% average reduction
        )
        
        validation_results['all_validations_passed'] = all([
            validation_results['has_successful_optimizations'],
            validation_results['energy_savings_reasonable'],
            validation_results['performance_penalties_acceptable'],
            validation_results['frequency_reductions_reasonable']
        ])
        
        return validation_results


def quick_frequency_optimization(data: pd.DataFrame, 
                               gpu: str, 
                               application: str,
                               config_path: Optional[str] = None) -> Dict:
    """
    Quick frequency optimization for a single configuration.
    
    Args:
        data: Complete profiling dataset
        gpu: GPU type
        application: Application name
        config_path: Optional path to configuration file
        
    Returns:
        Optimization results dictionary
    """
    optimizer = FrequencyOptimizer(config_path)
    
    try:
        result = optimizer.optimize_single_configuration(data, gpu, application)
        return result
    except Exception as e:
        logger.error(f"Quick optimization failed: {e}")
        return {'error': str(e)}


def optimize_all_frequencies(data: pd.DataFrame, 
                           config_path: Optional[str] = None,
                           output_path: Optional[str] = None) -> Dict:
    """
    Optimize frequencies for all configurations in dataset.
    
    Args:
        data: Complete profiling dataset
        config_path: Optional path to configuration file
        output_path: Optional path to save results
        
    Returns:
        Complete optimization results
    """
    optimizer = FrequencyOptimizer(config_path)
    
    # Run optimization
    results = optimizer.optimize_all_configurations(data)
    
    # Generate deployment configurations
    deployment_configs = optimizer.generate_deployment_configs(results, output_path)
    results['deployment_configs'] = deployment_configs
    
    # Validate results
    validation = optimizer.validate_optimization_results(results)
    results['validation'] = validation
    
    if validation['all_validations_passed']:
        logger.info("✅ All optimizations completed successfully with validation passed")
    else:
        logger.warning("⚠️ Some validation checks failed")
    
    return results
