"""
Quick Analysis Workflow

This module provides a streamlined workflow for rapid GPU frequency optimization analysis.
Perfect for quick insights and initial exploration of profiling data.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import sys

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    ProfilingDataLoader, FrequencyOptimizer, 
    load_config, save_results, create_deployment_summary,
    get_framework_version, setup_logging
)

logger = logging.getLogger(__name__)


class QuickAnalysisWorkflow:
    """
    Streamlined workflow for quick GPU frequency optimization analysis.
    """
    
    def __init__(self, 
                 data_path: str,
                 config_path: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize quick analysis workflow.
        
        Args:
            data_path: Path to aggregated profiling data
            config_path: Optional path to configuration file
            output_dir: Optional output directory for results
        """
        self.data_path = data_path
        self.config_path = config_path
        self.output_dir = Path(output_dir) if output_dir else Path("quick_analysis_results")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.optimizer = None
        self.data = None
        self.results = {}
        
        logger.info(f"Quick analysis initialized: {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate profiling data."""
        logger.info("Loading profiling data...")
        
        self.data_loader = ProfilingDataLoader(self.config_path)
        self.data = self.data_loader.load_aggregated_data(self.data_path)
        
        # Log data summary
        summary = self.data_loader.get_summary_statistics()
        logger.info(f"Loaded {summary['metadata']['total_records']} records")
        logger.info(f"Found {summary['metadata']['configurations']} configurations")
        logger.info(f"GPUs: {', '.join(summary['metadata']['gpus'])}")
        logger.info(f"Applications: {', '.join(summary['metadata']['applications'])}")
        
        return self.data
    
    def analyze_single_configuration(self, 
                                   gpu: str, 
                                   application: str) -> Dict[str, Any]:
        """
        Analyze a single GPU-application configuration.
        
        Args:
            gpu: GPU type
            application: Application name
            
        Returns:
            Analysis results dictionary
        """
        logger.info(f"Analyzing {gpu}+{application}...")
        
        if self.data is None:
            self.load_data()
        
        if self.optimizer is None:
            self.optimizer = FrequencyOptimizer(self.config_path)
        
        # Run optimization
        result = self.optimizer.optimize_single_configuration(
            self.data, gpu, application
        )
        
        # Store result
        config_key = f"{gpu}+{application}"
        self.results[config_key] = result
        
        return result
    
    def analyze_all_configurations(self) -> Dict[str, Any]:
        """
        Analyze all available configurations in the dataset.
        
        Returns:
            Complete analysis results
        """
        logger.info("Analyzing all configurations...")
        
        if self.data is None:
            self.load_data()
        
        if self.optimizer is None:
            self.optimizer = FrequencyOptimizer(self.config_path)
        
        # Run optimization for all configurations
        all_results = self.optimizer.optimize_all_configurations(self.data)
        
        # Store results
        self.results.update(all_results)
        
        return all_results
    
    def generate_summary(self) -> str:
        """
        Generate human-readable summary of analysis results.
        
        Returns:
            Formatted summary string
        """
        if not self.results:
            return "No analysis results available. Run analyze_* methods first."
        
        # Check if we have optimization results
        if 'configurations' in self.results:
            return create_deployment_summary(self.results)
        else:
            # Single configuration result
            summary_lines = []
            summary_lines.append("🎯 Quick Analysis Results")
            summary_lines.append("=" * 40)
            
            for config_key, result in self.results.items():
                if 'error' not in result:
                    summary_lines.append(f"\n📊 {config_key}:")
                    summary_lines.append(f"  Optimal Frequency: {result['optimal_frequency']:.0f} MHz")
                    summary_lines.append(f"  Energy Savings: {result['configuration_summary']['energy_savings_percent']:.1f}%")
                    summary_lines.append(f"  Performance Penalty: {abs(result['configuration_summary']['performance_penalty_percent']):.1f}%")
                    summary_lines.append(f"  Deployment: {result['deployment_recommendation']}")
            
            return "\n".join(summary_lines)
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save analysis results to file.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quick_analysis_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Prepare results for saving
        save_data = {
            'analysis_type': 'quick_analysis',
            'framework_version': get_framework_version(),
            'data_source': str(self.data_path),
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': self.results
        }
        
        # Add data metadata if available
        if self.data_loader:
            save_data['data_metadata'] = self.data_loader.get_summary_statistics()
        
        save_results(save_data, output_path)
        return output_path
    
    def create_deployment_script(self, 
                                category_filter: Optional[str] = None) -> Path:
        """
        Create deployment script for optimized configurations.
        
        Args:
            category_filter: Optional filter ('production_ready', 'a_b_testing', 'batch_only')
            
        Returns:
            Path to deployment script
        """
        script_lines = []
        script_lines.append("#!/bin/bash")
        script_lines.append("# GPU Frequency Deployment Script")
        script_lines.append("# Generated by EDP Analysis Framework")
        script_lines.append("")
        script_lines.append("set -e")
        script_lines.append("")
        
        # Add usage function
        script_lines.append("usage() {")
        script_lines.append('    echo "Usage: $0 <config> [action]"')
        script_lines.append('    echo "Actions: deploy, reset, status"')
        script_lines.append('    echo "Configs:"')
        
        # Extract deployment configurations
        if 'deployment_configs' in self.results:
            configs = self.results['deployment_configs']['configurations']
        elif 'configurations' in self.results:
            configs = self.results['configurations']
        else:
            configs = self.results
        
        # Filter configurations if specified
        if category_filter:
            filtered_configs = {}
            for key, config in configs.items():
                if 'error' not in config:
                    penalty = abs(config.get('configuration_summary', {}).get('performance_penalty_percent', 100))
                    if category_filter == 'production_ready' and penalty <= 20:
                        filtered_configs[key] = config
                    elif category_filter == 'a_b_testing' and 20 < penalty <= 50:
                        filtered_configs[key] = config
                    elif category_filter == 'batch_only' and penalty > 50:
                        filtered_configs[key] = config
            configs = filtered_configs
        
        # Add configuration options to usage
        for config_key in configs.keys():
            if 'error' not in configs[config_key]:
                script_lines.append(f'    echo "    {config_key}"')
        
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
        
        # Add configuration cases
        for config_key, config in configs.items():
            if 'error' not in config:
                optimal_freq = int(config['optimal_frequency'])
                baseline_freq = int(config['baseline_frequency'])
                
                # Determine memory frequency based on GPU
                if 'A100' in config['gpu']:
                    memory_freq = 1215
                elif 'V100' in config['gpu']:
                    memory_freq = 877
                else:
                    memory_freq = 1215  # Default
                
                script_lines.append(f'    "{config_key}")')
                script_lines.append(f'        DEPLOY_CMD="nvidia-smi -ac {memory_freq},{optimal_freq}"')
                script_lines.append(f'        RESET_CMD="nvidia-smi -ac {memory_freq},{baseline_freq}"')
                script_lines.append(f'        OPTIMAL_FREQ={optimal_freq}')
                script_lines.append(f'        BASELINE_FREQ={baseline_freq}')
                script_lines.append('        ;;')
        
        script_lines.append('    *)')
        script_lines.append('        echo "❌ Unknown configuration: $CONFIG"')
        script_lines.append('        usage')
        script_lines.append('        ;;')
        script_lines.append('esac')
        script_lines.append("")
        
        # Add action logic
        script_lines.append('case "$ACTION" in')
        script_lines.append('    "deploy")')
        script_lines.append('        echo "🚀 Deploying optimal frequency for $CONFIG..."')
        script_lines.append('        echo "Command: $DEPLOY_CMD"')
        script_lines.append('        $DEPLOY_CMD')
        script_lines.append('        echo "✅ Deployed successfully"')
        script_lines.append('        ;;')
        script_lines.append('    "reset")')
        script_lines.append('        echo "🔄 Resetting to baseline frequency for $CONFIG..."')
        script_lines.append('        echo "Command: $RESET_CMD"')
        script_lines.append('        $RESET_CMD')
        script_lines.append('        echo "✅ Reset successfully"')
        script_lines.append('        ;;')
        script_lines.append('    "status")')
        script_lines.append('        echo "📊 Current GPU status:"')
        script_lines.append('        nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader,nounits')
        script_lines.append('        ;;')
        script_lines.append('    *)')
        script_lines.append('        echo "❌ Unknown action: $ACTION"')
        script_lines.append('        usage')
        script_lines.append('        ;;')
        script_lines.append('esac')
        
        # Save script
        script_path = self.output_dir / "deploy_optimized_frequencies.sh"
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.info(f"Deployment script created: {script_path}")
        return script_path
    
    def run_complete_analysis(self, 
                            save_results: bool = True,
                            create_deployment: bool = True) -> Dict[str, Any]:
        """
        Run complete quick analysis workflow.
        
        Args:
            save_results: Whether to save results to file
            create_deployment: Whether to create deployment script
            
        Returns:
            Complete analysis results
        """
        logger.info("🚀 Starting complete quick analysis workflow...")
        
        # Load data
        self.load_data()
        
        # Run analysis
        results = self.analyze_all_configurations()
        
        # Generate summary
        summary_text = self.generate_summary()
        print("\n" + summary_text)
        
        # Save results if requested
        if save_results:
            results_path = self.save_results()
            logger.info(f"Results saved to: {results_path}")
        
        # Create deployment script if requested
        if create_deployment:
            # Create scripts for different categories
            prod_script = self.create_deployment_script('production_ready')
            logger.info(f"Production deployment script: {prod_script}")
            
            all_script = self.create_deployment_script()
            logger.info(f"Complete deployment script: {all_script}")
        
        logger.info("✅ Quick analysis workflow completed!")
        return results


def quick_analysis(data_path: str,
                  gpu: Optional[str] = None,
                  application: Optional[str] = None,
                  config_path: Optional[str] = None,
                  output_dir: Optional[str] = None,
                  save_results: bool = True) -> Dict[str, Any]:
    """
    Convenience function for quick analysis.
    
    Args:
        data_path: Path to aggregated profiling data
        gpu: Optional specific GPU to analyze
        application: Optional specific application to analyze
        config_path: Optional path to configuration file
        output_dir: Optional output directory
        save_results: Whether to save results to file
        
    Returns:
        Analysis results dictionary
    """
    # Set up logging
    setup_logging("INFO")
    
    # Initialize workflow
    workflow = QuickAnalysisWorkflow(data_path, config_path, output_dir)
    
    try:
        if gpu and application:
            # Single configuration analysis
            result = workflow.analyze_single_configuration(gpu, application)
            
            if save_results:
                workflow.save_results()
            
            # Print summary
            summary = workflow.generate_summary()
            print("\n" + summary)
            
            return result
        else:
            # Complete analysis
            return workflow.run_complete_analysis(save_results=save_results)
            
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick GPU Frequency Optimization Analysis")
    parser.add_argument("data_path", help="Path to aggregated profiling data")
    parser.add_argument("--gpu", help="Specific GPU to analyze")
    parser.add_argument("--application", help="Specific application to analyze")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Run quick analysis
    results = quick_analysis(
        data_path=args.data_path,
        gpu=args.gpu,
        application=args.application,
        config_path=args.config,
        output_dir=args.output_dir,
        save_results=not args.no_save
    )
