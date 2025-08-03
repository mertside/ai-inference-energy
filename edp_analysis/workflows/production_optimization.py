"""
Production Optimization Workflow

This module provides a comprehensive workflow for production-ready GPU frequency
optimization with deployment automation and monitoring.

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
    ProfilingDataLoader, FrequencyOptimizer, PerformanceConstraintManager,
    load_config, save_results, create_deployment_summary,
    get_framework_version, setup_logging, ensure_directory
)

logger = logging.getLogger(__name__)


class ProductionOptimizationWorkflow:
    """
    Production-ready workflow for GPU frequency optimization with deployment automation.
    """
    
    def __init__(self, 
                 data_path: str,
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize production optimization workflow.
        
        Args:
            data_path: Path to aggregated profiling data
            config_path: Optional path to configuration file
            output_dir: Optional output directory for results
        """
        self.data_path = data_path
        self.config_path = config_path
        self.output_dir = ensure_directory(output_dir or "production_optimization_results")
        
        # Load configuration
        self.config = load_config(config_path) if config_path else self._get_default_config()
        
        # Initialize components
        self.data_loader = None
        self.optimizer = None
        self.constraint_manager = PerformanceConstraintManager(
            self.config.get('optimization', {}).get('performance_constraints', {})
        )
        
        self.data = None
        self.optimization_results = {}
        self.deployment_configs = {}
        self.validation_results = {}
        
        logger.info(f"Production optimization initialized: {data_path}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for production optimization."""
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
            'deployment': {
                'categories': {
                    'production_ready': 0.20,
                    'a_b_testing': 0.50,
                    'batch_only': 1.0
                },
                'validation_timeout': 30,
                'rollback_on_failure': True
            }
        }
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load and validate profiling data for production use."""
        logger.info("Loading and validating production data...")
        
        self.data_loader = ProfilingDataLoader(self.config_path)
        self.data = self.data_loader.load_aggregated_data(self.data_path)
        
        # Perform comprehensive validation
        validation = self.data_loader.validate_data_consistency()
        
        if not all(validation.values()):
            logger.warning(f"Data validation issues detected: {validation}")
            # Could add stricter validation for production
        
        # Log data quality metrics
        summary = self.data_loader.get_summary_statistics()
        logger.info(f"Production data loaded: {summary['metadata']['total_records']} records")
        logger.info(f"Configurations: {summary['metadata']['configurations']}")
        
        return self.data
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run comprehensive frequency optimization analysis."""
        logger.info("Running production frequency optimization...")
        
        if self.data is None:
            self.load_and_validate_data()
        
        if self.optimizer is None:
            self.optimizer = FrequencyOptimizer(self.config_path)
        
        # Run optimization with production settings
        results = self.optimizer.optimize_all_configurations(
            self.data,
            methods=["edp", "energy"]  # Focus on proven methods for production
        )
        
        self.optimization_results = results
        logger.info(f"Optimization completed: {results['metadata']['successful_optimizations']} configurations")
        
        return results
    
    def validate_optimization_results(self) -> Dict[str, bool]:
        """Validate optimization results for production deployment."""
        logger.info("Validating optimization results for production...")
        
        if not self.optimization_results:
            raise ValueError("No optimization results to validate. Run run_optimization() first.")
        
        validation = self.optimizer.validate_optimization_results(self.optimization_results)
        
        # Additional production-specific validations
        production_validations = {}
        
        # Check if we have any production-ready configurations
        prod_ready_count = 0
        total_configs = 0
        
        for config_name, result in self.optimization_results['configurations'].items():
            if 'error' not in result:
                total_configs += 1
                penalty = abs(result['configuration_summary']['performance_penalty_percent'])
                if penalty <= 20:  # Production ready threshold
                    prod_ready_count += 1
        
        production_validations['has_production_ready_configs'] = prod_ready_count > 0
        production_validations['production_ready_ratio'] = prod_ready_count / max(total_configs, 1)
        
        # Check energy savings are significant
        if 'summary' in self.optimization_results:
            energy_stats = self.optimization_results['summary']['energy_savings']
            production_validations['significant_energy_savings'] = energy_stats['mean'] >= 10  # At least 10% average
        
        # Combine all validations
        validation.update(production_validations)
        self.validation_results = validation
        
        if validation['all_validations_passed'] and validation['has_production_ready_configs']:
            logger.info("✅ All production validations passed")
        else:
            logger.warning("⚠️ Some production validations failed")
        
        return validation
    
    def generate_deployment_configurations(self) -> Dict[str, Any]:
        """Generate comprehensive deployment configurations."""
        logger.info("Generating deployment configurations...")
        
        if not self.optimization_results:
            raise ValueError("No optimization results available. Run run_optimization() first.")
        
        # Generate deployment configs
        deployment_configs = self.optimizer.generate_deployment_configs(
            self.optimization_results,
            output_path=str(self.output_dir / "deployment_configurations.json")
        )
        
        # Enhance with production-specific information
        deployment_configs['production_metadata'] = {
            'validation_passed': self.validation_results.get('all_validations_passed', False),
            'recommended_order': self._get_deployment_order(deployment_configs),
            'monitoring_recommendations': self._get_monitoring_recommendations(),
            'rollback_procedures': self._get_rollback_procedures(deployment_configs)
        }
        
        self.deployment_configs = deployment_configs
        return deployment_configs
    
    def _get_deployment_order(self, configs: Dict) -> List[str]:
        """Get recommended deployment order based on risk assessment."""
        # Sort configurations by deployment safety (lowest penalty first)
        config_penalties = []
        
        for config_name, config in configs['configurations'].items():
            penalty = abs(config['performance_penalty_percent'])
            config_penalties.append((penalty, config_name))
        
        # Sort by penalty (ascending)
        config_penalties.sort()
        
        return [config_name for _, config_name in config_penalties]
    
    def _get_monitoring_recommendations(self) -> Dict[str, List[str]]:
        """Get monitoring recommendations for production deployment."""
        return {
            'gpu_metrics': [
                'GPU temperature (keep below 85°C)',
                'GPU utilization',
                'Power consumption',
                'Memory utilization',
                'Clock frequencies'
            ],
            'application_metrics': [
                'Inference latency',
                'Throughput (requests/second)',
                'Error rates',
                'Queue lengths',
                'Response times'
            ],
            'system_metrics': [
                'CPU utilization',
                'Memory usage',
                'Network latency',
                'Disk I/O',
                'System stability'
            ],
            'monitoring_tools': [
                'nvidia-smi for GPU monitoring',
                'DCGMI for detailed GPU metrics',
                'Application-specific monitoring',
                'Custom dashboards'
            ]
        }
    
    def _get_rollback_procedures(self, configs: Dict) -> Dict[str, str]:
        """Get rollback procedures for each configuration."""
        rollback_procedures = {}
        
        for config_name, config in configs['configurations'].items():
            rollback_procedures[config_name] = f"""
Rollback procedure for {config_name}:
1. Execute: {config['reset_command']}
2. Verify frequency: nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits
3. Check application performance returns to baseline
4. Monitor for 5 minutes to ensure stability
5. Document rollback reason and time
"""
        
        return rollback_procedures
    
    def create_production_deployment_script(self, 
                                          categories: Optional[List[str]] = None) -> Path:
        """Create production-grade deployment script with safety features."""
        if categories is None:
            categories = ['production_ready']
        
        script_lines = []
        script_lines.append("#!/bin/bash")
        script_lines.append("# Production GPU Frequency Deployment Script")
        script_lines.append("# Generated by EDP Analysis Framework")
        script_lines.append(f"# Generated: {pd.Timestamp.now().isoformat()}")
        script_lines.append("")
        script_lines.append("set -e")
        script_lines.append("")
        
        # Add safety configuration
        script_lines.append("# Safety configuration")
        script_lines.append("VALIDATION_TIMEOUT=30")
        script_lines.append("TEMP_THRESHOLD=85")
        script_lines.append("LOG_FILE=\"/var/log/gpu_frequency_deployment.log\"")
        script_lines.append("")
        
        # Add logging function
        script_lines.append("log_message() {")
        script_lines.append('    echo "$(date): $1" | tee -a "$LOG_FILE"')
        script_lines.append("}")
        script_lines.append("")
        
        # Add validation functions
        script_lines.append("validate_deployment() {")
        script_lines.append('    log_message "Validating deployment..."')
        script_lines.append('    ')
        script_lines.append('    # Check if frequency was applied')
        script_lines.append('    CURRENT_FREQ=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)')
        script_lines.append('    if [ "$CURRENT_FREQ" != "$OPTIMAL_FREQ" ]; then')
        script_lines.append('        log_message "ERROR: Frequency not applied correctly. Expected: $OPTIMAL_FREQ, Got: $CURRENT_FREQ"')
        script_lines.append('        return 1')
        script_lines.append('    fi')
        script_lines.append('    ')
        script_lines.append('    # Check temperature')
        script_lines.append('    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)')
        script_lines.append('    if [ "$TEMP" -gt "$TEMP_THRESHOLD" ]; then')
        script_lines.append('        log_message "WARNING: GPU temperature high: ${TEMP}°C"')
        script_lines.append('    fi')
        script_lines.append('    ')
        script_lines.append('    log_message "Deployment validation passed"')
        script_lines.append('    return 0')
        script_lines.append("}")
        script_lines.append("")
        
        # Add rollback function
        script_lines.append("rollback_deployment() {")
        script_lines.append('    log_message "Rolling back deployment for $CONFIG..."')
        script_lines.append('    $RESET_CMD')
        script_lines.append('    sleep 2')
        script_lines.append('    log_message "Rollback completed"')
        script_lines.append("}")
        script_lines.append("")
        
        # Add usage function
        script_lines.append("usage() {")
        script_lines.append('    echo "Usage: $0 <config> [action]"')
        script_lines.append('    echo "Actions: deploy, reset, status, validate"')
        script_lines.append('    echo ""')
        script_lines.append('    echo "Production-ready configurations:"')
        
        # Filter configurations by category
        if self.deployment_configs:
            for category in categories:
                if category in self.deployment_configs['categories']:
                    for config_name in self.deployment_configs['categories'][category]:
                        script_lines.append(f'    echo "    {config_name}"')
        
        script_lines.append('    exit 1')
        script_lines.append("}")
        script_lines.append("")
        
        # Add main script logic
        script_lines.append('if [ $# -lt 1 ]; then')
        script_lines.append('    usage')
        script_lines.append('fi')
        script_lines.append("")
        script_lines.append('CONFIG=$1')
        script_lines.append('ACTION=${2:-deploy}')
        script_lines.append("")
        script_lines.append('case "$CONFIG" in')
        
        # Add configuration cases (only for selected categories)
        if self.deployment_configs:
            for category in categories:
                if category in self.deployment_configs['categories']:
                    for config_name in self.deployment_configs['categories'][category]:
                        if config_name in self.deployment_configs['configurations']:
                            config = self.deployment_configs['configurations'][config_name]
                            
                            script_lines.append(f'    "{config_name}")')
                            script_lines.append(f'        DEPLOY_CMD="{config["nvidia_smi_command"]}"')
                            script_lines.append(f'        RESET_CMD="{config["reset_command"]}"')
                            script_lines.append(f'        OPTIMAL_FREQ={int(config["optimal_frequency"])}')
                            script_lines.append(f'        BASELINE_FREQ={int(config["baseline_frequency"])}')
                            script_lines.append('        ;;')
        
        script_lines.append('    *)')
        script_lines.append('        echo "❌ Unknown or unsupported configuration: $CONFIG"')
        script_lines.append('        usage')
        script_lines.append('        ;;')
        script_lines.append('esac')
        script_lines.append("")
        
        # Add action logic with safety features
        script_lines.append('case "$ACTION" in')
        script_lines.append('    "deploy")')
        script_lines.append('        log_message "Starting deployment for $CONFIG..."')
        script_lines.append('        echo "🚀 Deploying optimal frequency for $CONFIG..."')
        script_lines.append('        echo "Command: $DEPLOY_CMD"')
        script_lines.append('        ')
        script_lines.append('        # Execute deployment')
        script_lines.append('        if $DEPLOY_CMD; then')
        script_lines.append('            sleep 2  # Allow frequency to settle')
        script_lines.append('            if validate_deployment; then')
        script_lines.append('                echo "✅ Deployment successful and validated"')
        script_lines.append('                log_message "Deployment successful for $CONFIG"')
        script_lines.append('            else')
        script_lines.append('                echo "❌ Deployment validation failed"')
        script_lines.append('                rollback_deployment')
        script_lines.append('                exit 1')
        script_lines.append('            fi')
        script_lines.append('        else')
        script_lines.append('            echo "❌ Deployment command failed"')
        script_lines.append('            log_message "Deployment command failed for $CONFIG"')
        script_lines.append('            exit 1')
        script_lines.append('        fi')
        script_lines.append('        ;;')
        script_lines.append('    "reset")')
        script_lines.append('        rollback_deployment')
        script_lines.append('        ;;')
        script_lines.append('    "status")')
        script_lines.append('        echo "📊 Current GPU status:"')
        script_lines.append('        nvidia-smi --query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw --format=csv')
        script_lines.append('        ;;')
        script_lines.append('    "validate")')
        script_lines.append('        validate_deployment')
        script_lines.append('        ;;')
        script_lines.append('    *)')
        script_lines.append('        echo "❌ Unknown action: $ACTION"')
        script_lines.append('        usage')
        script_lines.append('        ;;')
        script_lines.append('esac')
        
        # Save script
        script_path = self.output_dir / "production_deployment.sh"
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Production deployment script created: {script_path}")
        return script_path
    
    def create_monitoring_dashboard_config(self) -> Path:
        """Create configuration for monitoring dashboard."""
        dashboard_config = {
            'dashboard_type': 'gpu_frequency_monitoring',
            'version': '1.0',
            'panels': [
                {
                    'title': 'GPU Frequencies',
                    'type': 'time_series',
                    'metrics': ['gpu_clock_frequency', 'memory_clock_frequency'],
                    'alert_thresholds': {
                        'gpu_clock_frequency': {'min': 500, 'max': 2000}
                    }
                },
                {
                    'title': 'GPU Temperature',
                    'type': 'gauge',
                    'metrics': ['gpu_temperature'],
                    'alert_thresholds': {
                        'gpu_temperature': {'max': 85}
                    }
                },
                {
                    'title': 'Power Consumption',
                    'type': 'time_series',
                    'metrics': ['gpu_power_draw'],
                    'alert_thresholds': {
                        'gpu_power_draw': {'max': 400}
                    }
                },
                {
                    'title': 'Application Performance',
                    'type': 'time_series',
                    'metrics': ['inference_latency', 'throughput'],
                    'alert_thresholds': {
                        'inference_latency': {'max': 5.0}  # 5 second max latency
                    }
                }
            ],
            'refresh_interval': '10s',
            'data_sources': [
                'nvidia_smi',
                'dcgmi',
                'application_metrics'
            ]
        }
        
        config_path = self.output_dir / "monitoring_dashboard_config.json"
        with open(config_path, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        logger.info(f"Monitoring dashboard config created: {config_path}")
        return config_path
    
    def run_complete_production_workflow(self) -> Dict[str, Any]:
        """Run complete production optimization workflow."""
        logger.info("🚀 Starting complete production optimization workflow...")
        
        try:
            # Step 1: Load and validate data
            self.load_and_validate_data()
            
            # Step 2: Run optimization
            optimization_results = self.run_optimization()
            
            # Step 3: Validate results
            validation_results = self.validate_optimization_results()
            
            if not validation_results['all_validations_passed']:
                logger.warning("⚠️ Some validations failed, but continuing with production workflow")
            
            # Step 4: Generate deployment configurations
            deployment_configs = self.generate_deployment_configurations()
            
            # Step 5: Create production deployment script
            prod_script = self.create_production_deployment_script(['production_ready'])
            all_script = self.create_production_deployment_script(['production_ready', 'a_b_testing'])
            
            # Step 6: Create monitoring configuration
            monitoring_config = self.create_monitoring_dashboard_config()
            
            # Step 7: Save comprehensive results
            final_results = {
                'workflow_type': 'production_optimization',
                'framework_version': get_framework_version(),
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_source': str(self.data_path),
                'optimization_results': optimization_results,
                'validation_results': validation_results,
                'deployment_configs': deployment_configs,
                'production_artifacts': {
                    'deployment_script_production': str(prod_script),
                    'deployment_script_all': str(all_script),
                    'monitoring_config': str(monitoring_config)
                }
            }
            
            results_path = self.output_dir / f"production_optimization_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_results(final_results, results_path)
            
            # Generate summary
            summary = create_deployment_summary(optimization_results)
            summary_path = self.output_dir / "production_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print("\n" + summary)
            
            logger.info("✅ Production optimization workflow completed successfully!")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Production deployment script: {prod_script}")
            logger.info(f"Monitoring configuration: {monitoring_config}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Production workflow failed: {e}")
            raise


def production_optimization(data_path: str,
                          config_path: Optional[str] = None,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for production optimization workflow.
    
    Args:
        data_path: Path to aggregated profiling data
        config_path: Optional path to configuration file
        output_dir: Optional output directory
        
    Returns:
        Complete production optimization results
    """
    # Set up logging
    setup_logging("INFO")
    
    # Initialize and run workflow
    workflow = ProductionOptimizationWorkflow(data_path, config_path, output_dir)
    return workflow.run_complete_production_workflow()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production GPU Frequency Optimization")
    parser.add_argument("data_path", help="Path to aggregated profiling data")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run production optimization
    results = production_optimization(
        data_path=args.data_path,
        config_path=args.config,
        output_dir=args.output_dir
    )
