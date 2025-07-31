"""
EDP Analysis Workflows

This module provides high-level workflows for GPU frequency optimization analysis.
All workflows are designed to be comprehensive, production-ready, and easy to use.

Available Workflows:
- quick_analysis: Fast analysis with automated deployment scripts
- production_optimization: Production-ready optimization with safety features
- comparative_analysis: Compare optimization results across datasets and methods
- sensitivity_analysis: Analyze parameter sensitivity and robustness

Author: Mert Side
"""

from .quick_analysis import QuickAnalysisWorkflow, quick_analysis
from .production_optimization import ProductionOptimizationWorkflow, production_optimization
from .comparative_analysis import ComparativeAnalysisWorkflow, comparative_analysis
from .sensitivity_analysis import SensitivityAnalysisWorkflow, sensitivity_analysis

# High-level workflow functions
__all__ = [
    # Workflow classes
    'QuickAnalysisWorkflow',
    'ProductionOptimizationWorkflow', 
    'ComparativeAnalysisWorkflow',
    'SensitivityAnalysisWorkflow',
    
    # Convenience functions
    'quick_analysis',
    'production_optimization',
    'comparative_analysis', 
    'sensitivity_analysis',
    
    # High-level API
    'run_complete_analysis',
    'run_production_deployment',
    'run_comparative_study'
]


def run_complete_analysis(data_path: str, 
                         config_path: str = None,
                         output_dir: str = None,
                         include_sensitivity: bool = True) -> dict:
    """
    Run complete analysis workflow including quick analysis, production optimization,
    and optional sensitivity analysis.
    
    Args:
        data_path: Path to aggregated profiling data
        config_path: Optional path to configuration file
        output_dir: Optional output directory for results
        include_sensitivity: Whether to include sensitivity analysis (time-intensive)
        
    Returns:
        Dictionary containing all analysis results
        
    Example:
        >>> results = run_complete_analysis('data/aggregated_data.json')
        >>> print(f"Energy savings: {results['quick_analysis']['summary']['energy_savings']['mean']:.1f}%")
    """
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting complete analysis workflow...")
    
    # Ensure output directory
    if output_dir is None:
        output_dir = "complete_analysis_results"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        'workflow_type': 'complete_analysis',
        'data_path': data_path,
        'timestamp': None,
        'quick_analysis': None,
        'production_optimization': None,
        'sensitivity_analysis': None
    }
    
    try:
        # Step 1: Quick Analysis
        logger.info("Step 1/3: Running quick analysis...")
        quick_result = quick_analysis(
            data_path=data_path,
            config_path=config_path,
            output_dir=str(Path(output_dir) / "quick_analysis")
        )
        results['quick_analysis'] = quick_result
        
        # Step 2: Production Optimization
        logger.info("Step 2/3: Running production optimization...")
        prod_result = production_optimization(
            data_path=data_path,
            config_path=config_path,
            output_dir=str(Path(output_dir) / "production_optimization")
        )
        results['production_optimization'] = prod_result
        
        # Step 3: Sensitivity Analysis (optional)
        if include_sensitivity:
            logger.info("Step 3/3: Running sensitivity analysis...")
            sensitivity_result = sensitivity_analysis(
                data_path=data_path,
                config_path=config_path,
                output_dir=str(Path(output_dir) / "sensitivity_analysis")
            )
            results['sensitivity_analysis'] = sensitivity_result
        else:
            logger.info("Step 3/3: Skipping sensitivity analysis (include_sensitivity=False)")
        
        # Save combined results
        results['timestamp'] = quick_result.get('timestamp')
        
        from core.utils import save_results
        import pandas as pd
        
        final_path = Path(output_dir) / f"complete_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_results(results, final_path)
        
        logger.info("✅ Complete analysis workflow finished!")
        logger.info(f"Results saved to: {final_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Complete analysis workflow failed: {e}")
        raise


def run_production_deployment(data_path: str,
                            config_path: str = None,
                            categories: list = None,
                            validation_timeout: int = 30) -> dict:
    """
    Run production deployment workflow with enhanced safety features.
    
    Args:
        data_path: Path to aggregated profiling data
        config_path: Optional path to configuration file
        categories: List of deployment categories ['production_ready', 'a_b_testing', 'batch_only']
        validation_timeout: Timeout for deployment validation in seconds
        
    Returns:
        Production deployment results with scripts and monitoring configs
        
    Example:
        >>> deployment = run_production_deployment('data/aggregated_data.json')
        >>> print(f"Deployment script: {deployment['production_artifacts']['deployment_script_production']}")
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting production deployment workflow...")
    
    if categories is None:
        categories = ['production_ready']
    
    try:
        # Run production optimization
        workflow = ProductionOptimizationWorkflow(
            data_path=data_path,
            config_path=config_path,
            output_dir="production_deployment_results"
        )
        
        # Load and validate data
        workflow.load_and_validate_data()
        
        # Run optimization
        optimization_results = workflow.run_optimization()
        
        # Validate results
        validation_results = workflow.validate_optimization_results()
        
        if not validation_results['all_validations_passed']:
            logger.warning("⚠️ Some validations failed - review before production deployment")
        
        # Generate deployment configurations
        deployment_configs = workflow.generate_deployment_configurations()
        
        # Create production scripts
        prod_script = workflow.create_production_deployment_script(categories)
        monitoring_config = workflow.create_monitoring_dashboard_config()
        
        # Compile deployment package
        deployment_package = {
            'workflow_type': 'production_deployment',
            'validation_status': validation_results['all_validations_passed'],
            'deployment_categories': categories,
            'optimization_results': optimization_results,
            'deployment_configs': deployment_configs,
            'artifacts': {
                'deployment_script': str(prod_script),
                'monitoring_config': str(monitoring_config)
            },
            'safety_features': {
                'validation_timeout': validation_timeout,
                'rollback_procedures': deployment_configs.get('production_metadata', {}).get('rollback_procedures', {}),
                'monitoring_recommendations': deployment_configs.get('production_metadata', {}).get('monitoring_recommendations', {})
            }
        }
        
        logger.info("✅ Production deployment workflow completed!")
        logger.info(f"Deployment script: {prod_script}")
        logger.info(f"Monitoring config: {monitoring_config}")
        
        return deployment_package
        
    except Exception as e:
        logger.error(f"Production deployment workflow failed: {e}")
        raise


def run_comparative_study(datasets: dict = None,
                         optimization_results: dict = None,
                         config_path: str = None,
                         output_dir: str = None) -> dict:
    """
    Run comprehensive comparative study across multiple datasets and results.
    
    Args:
        datasets: Dict mapping names to dataset paths {'name': 'path'}
        optimization_results: Dict mapping names to result paths {'name': 'path'}
        config_path: Optional path to configuration file
        output_dir: Optional output directory for results
        
    Returns:
        Comprehensive comparative analysis results
        
    Example:
        >>> datasets = {'baseline': 'data/baseline.json', 'optimized': 'data/optimized.json'}
        >>> study = run_comparative_study(datasets=datasets)
        >>> print(f"Comparison completed: {len(study['comparisons_performed'])} analyses")
    """
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info("🔍 Starting comparative study workflow...")
    
    if not datasets and not optimization_results:
        raise ValueError("Must provide either datasets or optimization_results for comparison")
    
    try:
        # Initialize comparative analysis
        study = comparative_analysis(
            datasets=datasets,
            results=optimization_results,
            config_path=config_path,
            output_dir=output_dir or "comparative_study_results"
        )
        
        logger.info("✅ Comparative study completed!")
        return study
        
    except Exception as e:
        logger.error(f"Comparative study workflow failed: {e}")
        raise


# Workflow configuration templates
WORKFLOW_CONFIGS = {
    'quick_analysis': {
        'description': 'Fast analysis with automated deployment scripts',
        'typical_runtime': '5-15 minutes',
        'output_artifacts': ['optimization_results.json', 'deployment_script.sh', 'summary.txt']
    },
    
    'production_optimization': {
        'description': 'Production-ready optimization with safety features',
        'typical_runtime': '15-30 minutes', 
        'output_artifacts': ['production_deployment.sh', 'monitoring_config.json', 'validation_results.json']
    },
    
    'comparative_analysis': {
        'description': 'Compare optimization results across datasets and methods',
        'typical_runtime': '20-60 minutes',
        'output_artifacts': ['comparison_report.md', 'method_comparison.json', 'dataset_comparison.json']
    },
    
    'sensitivity_analysis': {
        'description': 'Analyze parameter sensitivity and robustness',
        'typical_runtime': '30-120 minutes',
        'output_artifacts': ['sensitivity_report.md', 'constraint_sensitivity.json', 'robustness_analysis.json']
    }
}


def get_workflow_info(workflow_name: str = None) -> dict:
    """
    Get information about available workflows.
    
    Args:
        workflow_name: Optional specific workflow name
        
    Returns:
        Workflow information
    """
    if workflow_name:
        return WORKFLOW_CONFIGS.get(workflow_name, {})
    return WORKFLOW_CONFIGS


# Version information
__version__ = "1.0.0"
__author__ = "EDP Analysis Framework"
