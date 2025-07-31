#!/usr/bin/env python3
"""
Example: Analyzing New Profiling Data for GPU Frequency Optimization

This example demonstrates how to apply the EDP analysis framework to new
profiling datasets, following best practices for reproducibility and 
production deployment.

Usage:
    python analyze_new_data.py --data-dir ./new_profiling_data --output ./results
"""

import argparse
import os
import sys
from pathlib import Path

# Add the edp_analysis module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Analyze new profiling data for GPU frequency optimization')
    parser.add_argument('--data-dir', required=True, help='Directory containing profiling data')
    parser.add_argument('--output', default='./analysis_results', help='Output directory for results')
    parser.add_argument('--config', help='Custom configuration file')
    parser.add_argument('--gpu', choices=['V100', 'A100', 'H100'], help='Filter by specific GPU')
    parser.add_argument('--app', choices=['LLAMA', 'STABLEDIFFUSION', 'VIT', 'WHISPER'], help='Filter by specific application')
    parser.add_argument('--deploy', action='store_true', help='Generate deployment scripts')
    parser.add_argument('--validate', action='store_true', help='Run comprehensive validation')
    
    args = parser.parse_args()
    
    print("🚀 Starting GPU Frequency Optimization Analysis")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Step 1: Data Quality Validation
        print("\n📊 Step 1: Validating data quality...")
        from edp_analysis.tools import data_validator
        
        validator = data_validator.DataQualityValidator()
        quality_report = validator.validate_profiling_directory(args.data_dir)
        
        print(f"✅ Found {quality_report.total_files} profiling files")
        print(f"✅ Data quality score: {quality_report.quality_score:.1f}/10")
        
        if quality_report.has_critical_issues:
            print("❌ Critical data quality issues found:")
            for issue in quality_report.critical_issues:
                print(f"  - {issue}")
            return 1
            
        # Step 2: Data Aggregation
        print("\n🔧 Step 2: Aggregating profiling data...")
        from edp_analysis.analysis.aggregation import DataAggregator
        
        aggregator = DataAggregator()
        aggregated_data = aggregator.aggregate_profiling_data(
            input_dir=args.data_dir,
            output_file=os.path.join(args.output, "aggregated_data.csv"),
            run_selection="warm_runs",  # Exclude cold start (run 1)
            gpu_filter=[args.gpu] if args.gpu else None,
            app_filter=[args.app] if args.app else None
        )
        
        print(f"✅ Aggregated {aggregated_data['total_configurations']} configurations")
        print(f"✅ Energy range: {aggregated_data['energy_range']['min']:.1f} - {aggregated_data['energy_range']['max']:.1f} J")
        print(f"✅ Execution time range: {aggregated_data['time_range']['min']:.1f} - {aggregated_data['time_range']['max']:.1f} s")
        
        # Step 3: Optimization Analysis
        print("\n🎯 Step 3: Finding optimal frequencies...")
        from edp_analysis.analysis.optimization import UnifiedOptimizer
        from edp_analysis.configs import load_config
        
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = load_config("configs/default.yaml")
            
        optimizer = UnifiedOptimizer(config)
        optimization_results = optimizer.optimize_all_configurations(
            data_file=os.path.join(args.output, "aggregated_data.csv")
        )
        
        print(f"✅ Optimized {len(optimization_results.configurations)} configurations")
        
        # Display results summary
        print("\n📋 Optimization Results Summary:")
        production_ready = [c for c in optimization_results.configurations if c.category == "production"]
        testing_needed = [c for c in optimization_results.configurations if c.category == "testing"]
        batch_only = [c for c in optimization_results.configurations if c.category == "batch_only"]
        
        print(f"  🟢 Production ready: {len(production_ready)} configurations")
        for config in production_ready:
            print(f"    - {config.configuration}: {config.optimal_frequency}MHz "
                  f"({config.performance_penalty:.1f}% slower, {config.energy_savings:.1f}% energy savings)")
                  
        print(f"  🟡 Needs A/B testing: {len(testing_needed)} configurations")
        for config in testing_needed:
            print(f"    - {config.configuration}: {config.optimal_frequency}MHz "
                  f"({config.performance_penalty:.1f}% slower, {config.energy_savings:.1f}% energy savings)")
                  
        print(f"  🔵 Batch processing only: {len(batch_only)} configurations")
        for config in batch_only:
            print(f"    - {config.configuration}: {config.optimal_frequency}MHz "
                  f"({config.performance_penalty:.1f}% slower, {config.energy_savings:.1f}% energy savings)")
        
        # Step 4: Generate Reports
        print("\n📝 Step 4: Generating reports...")
        from edp_analysis.analysis.reporting import ReportGenerator
        
        reporter = ReportGenerator()
        
        # Generate summary table
        summary_table = reporter.generate_summary_table(optimization_results.configurations)
        with open(os.path.join(args.output, "optimization_summary.txt"), "w") as f:
            f.write(summary_table)
            
        # Generate markdown report
        markdown_report = reporter.generate_markdown_report(
            results=optimization_results.configurations,
            include_deployment_guide=True,
            include_monitoring_recommendations=True
        )
        with open(os.path.join(args.output, "optimization_report.md"), "w") as f:
            f.write(markdown_report)
            
        print(f"✅ Summary table: {args.output}/optimization_summary.txt")
        print(f"✅ Detailed report: {args.output}/optimization_report.md")
        
        # Step 5: Generate Deployment Scripts (Optional)
        if args.deploy:
            print("\n🚀 Step 5: Generating deployment scripts...")
            
            deployment_script = reporter.generate_deployment_script(optimization_results.configurations)
            deployment_file = os.path.join(args.output, "deploy_optimal_frequencies.sh")
            
            with open(deployment_file, "w") as f:
                f.write(deployment_script)
            os.chmod(deployment_file, 0o755)  # Make executable
            
            print(f"✅ Deployment script: {deployment_file}")
            print("Usage examples:")
            print(f"  {deployment_file} A100+STABLEDIFFUSION deploy")
            print(f"  {deployment_file} V100+STABLEDIFFUSION status")
            print(f"  {deployment_file} A100+LLAMA reset")
        
        # Step 6: Validation (Optional)
        if args.validate:
            print("\n🔍 Step 6: Running validation...")
            from edp_analysis.analysis.validation import ResultValidator
            
            validator = ResultValidator()
            validation_result = validator.validate_optimization_results(
                results=optimization_results.configurations,
                data_file=os.path.join(args.output, "aggregated_data.csv")
            )
            
            if validation_result.passed:
                print("✅ All validation checks passed")
            else:
                print("⚠️ Validation warnings found:")
                for warning in validation_result.warnings:
                    print(f"  - {warning}")
                    
            validation_report = validator.generate_validation_report(validation_result)
            with open(os.path.join(args.output, "validation_report.txt"), "w") as f:
                f.write(validation_report)
                
            print(f"✅ Validation report: {args.output}/validation_report.txt")
        
        # Step 7: Visualization (if matplotlib available)
        try:
            print("\n📈 Step 7: Generating visualizations...")
            from edp_analysis.visualization import create_optimization_dashboard
            
            dashboard = create_optimization_dashboard(
                data=optimization_results,
                output_dir=os.path.join(args.output, "plots"),
                include_deployment_guide=True
            )
            
            print(f"✅ Visualization dashboard: {args.output}/plots/")
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping visualizations")
        
        # Final Summary
        print("\n🎉 Analysis Complete!")
        print(f"📁 All results saved to: {args.output}")
        
        if production_ready:
            print("\n🚀 Ready for immediate deployment:")
            best_config = min(production_ready, key=lambda x: x.performance_penalty)
            print(f"Recommended: {best_config.configuration}")
            print(f"Command: sudo nvidia-smi -ac {best_config.memory_frequency},{best_config.optimal_frequency}")
            print(f"Expected: {best_config.performance_penalty:.1f}% slower, {best_config.energy_savings:.1f}% energy savings")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
