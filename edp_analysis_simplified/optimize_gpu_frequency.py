#!/usr/bin/env python3
"""
GPU Frequency Optimization Tool

A unified tool for analyzing GPU frequency optimization with minimal performance
degradation and maximum energy savings. This tool provides a clean interface
to the frequency optimization framework.

Usage:
    python optimize_gpu_frequency.py --data data.csv --output results/
    python optimize_gpu_frequency.py --gpu A100 --app STABLEDIFFUSION --plots
    python optimize_gpu_frequency.py --max-degradation 10 --deploy
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import framework components
from core import (
    load_and_validate_data,
    calculate_efficiency_metrics,
    find_optimal_configurations,
    calculate_summary_statistics,
    create_output_directory,
    save_json_results,
    print_section_header,
    print_subsection_header,
    print_success,
    print_info,
    print_error,
    format_percentage,
    format_frequency,
    format_efficiency_ratio
)

from visualization import generate_all_plots
from frequency_optimization import create_deployment_package


def main():
    parser = argparse.ArgumentParser(
        description='GPU Frequency Optimization Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output options
    parser.add_argument('--data', default='data_aggregation/aggregated_data.csv',
                       help='Path to aggregated profiling data CSV file')
    parser.add_argument('--output', default='./optimization_results',
                       help='Output directory for results')
    
    # Analysis parameters
    parser.add_argument('--max-degradation', type=float, default=15.0,
                       help='Maximum acceptable performance degradation in percent')
    parser.add_argument('--min-efficiency', type=float, default=2.0,
                       help='Minimum energy efficiency ratio required')
    
    # Filtering options
    parser.add_argument('--gpu', choices=['A100', 'V100'],
                       help='Filter analysis to specific GPU type')
    parser.add_argument('--app', choices=['LLAMA', 'STABLEDIFFUSION', 'VIT', 'WHISPER'],
                       help='Filter analysis to specific application')
    
    # Output options
    parser.add_argument('--plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--deploy', action='store_true',
                       help='Generate deployment scripts and package')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print header
    print_section_header("🚀 GPU Frequency Optimization Tool")
    print_info(f"Data source: {args.data}")
    print_info(f"Output directory: {args.output}")
    print_info(f"Max performance degradation: {format_percentage(args.max_degradation)}")
    print_info(f"Min efficiency ratio: {format_efficiency_ratio(args.min_efficiency)}")
    
    if args.gpu:
        print_info(f"GPU filter: {args.gpu}")
    if args.app:
        print_info(f"Application filter: {args.app}")
    
    try:
        # Create output directory
        output_dir = create_output_directory(args.output)
        
        # Load and validate data
        print_subsection_header("📊 Loading and Validating Data")
        df, info_messages = load_and_validate_data(args.data, args.gpu, args.app)
        
        for message in info_messages:
            print_info(message)
        
        # Calculate efficiency metrics
        print_subsection_header("🧮 Calculating Efficiency Metrics")
        efficiency_df = calculate_efficiency_metrics(df)
        print_success(f"Calculated metrics for {len(efficiency_df)} data points")
        
        if args.verbose:
            print_info(f"Configurations analyzed: {efficiency_df['config'].nunique()}")
            print_info(f"Frequency range: {efficiency_df['gpu_frequency'].min():.0f}-{efficiency_df['gpu_frequency'].max():.0f} MHz")
        
        # Find optimal configurations
        print_subsection_header("🎯 Finding Optimal Configurations")
        optimal_df = find_optimal_configurations(
            efficiency_df, 
            args.max_degradation, 
            args.min_efficiency
        )
        
        if len(optimal_df) == 0:
            print_error("No optimal configurations found with current criteria")
            print_info("Try increasing --max-degradation or decreasing --min-efficiency")
            return 1
        
        print_success(f"Found {len(optimal_df)} optimal configurations")
        
        # Calculate and display summary statistics
        print_subsection_header("📋 Optimization Results Summary")
        summary_stats = calculate_summary_statistics(optimal_df)
        
        # Display category breakdown
        for category, count in summary_stats['configurations_by_category'].items():
            print_info(f"{category}: {count} configurations")
        
        print_info(f"Average energy savings: {format_percentage(summary_stats['average_energy_savings'])}")
        print_info(f"Average performance impact: {format_percentage(summary_stats['average_performance_impact'])}")
        print_info(f"Best efficiency ratio: {format_efficiency_ratio(summary_stats['best_efficiency_ratio'])}")
        
        # Display top recommendations
        print_subsection_header("🚀 Top Recommendations")
        
        # Sort by performance penalty (ascending) and efficiency ratio (descending)
        recommendations = optimal_df.sort_values(
            ['performance_penalty', 'efficiency_ratio'], 
            ascending=[True, False]
        )
        
        for idx, (_, config) in enumerate(recommendations.head(5).iterrows(), 1):
            print(f"{idx}. {config['config']}: {format_frequency(config['optimal_frequency'])}")
            print(f"   Performance: {format_percentage(config['performance_penalty'])} impact")
            print(f"   Energy: {format_percentage(config['energy_savings'])} savings")
            print(f"   Efficiency: {format_efficiency_ratio(config['efficiency_ratio'])} ratio")
            print(f"   Category: {config['category']}")
            print()
        
        # Save analysis data
        print_subsection_header("💾 Saving Analysis Results")
        
        # Save efficiency analysis data
        efficiency_path = os.path.join(output_dir, 'efficiency_analysis.csv')
        efficiency_df.to_csv(efficiency_path, index=False)
        print_success(f"Saved efficiency analysis: {efficiency_path}")
        
        # Save optimal configurations
        optimal_path = os.path.join(output_dir, 'optimal_configurations.csv')
        optimal_df.to_csv(optimal_path, index=False)
        print_success(f"Saved optimal configurations: {optimal_path}")
        
        # Save summary statistics as JSON
        analysis_params = {
            'data_source': args.data,
            'max_degradation': args.max_degradation,
            'min_efficiency': args.min_efficiency,
            'gpu_filter': args.gpu,
            'app_filter': args.app
        }
        
        results_data = {
            'analysis_parameters': analysis_params,
            'summary_statistics': summary_stats,
            'optimal_configurations': optimal_df.to_dict('records')
        }
        
        json_path = save_json_results(results_data, output_dir, 'analysis_results.json')
        print_success(f"Saved JSON results: {json_path}")
        
        # Generate visualizations
        if args.plots:
            print_subsection_header("📈 Generating Visualizations")
            try:
                plot_paths = generate_all_plots(efficiency_df, optimal_df, output_dir)
                for plot_path in plot_paths:
                    print_success(f"Generated plot: {os.path.basename(plot_path)}")
            except Exception as e:
                print_error(f"Failed to generate plots: {e}")
        
        # Generate deployment package
        if args.deploy:
            print_subsection_header("🚀 Creating Deployment Package")
            try:
                package_files = create_deployment_package(
                    efficiency_df, optimal_df, analysis_params, output_dir
                )
                for file_path in package_files:
                    print_success(f"Generated: {os.path.basename(file_path)}")
            except Exception as e:
                print_error(f"Failed to create deployment package: {e}")
        
        # Final summary
        print_section_header("🎉 Analysis Complete!")
        print_info(f"All results saved to: {output_dir}")
        
        # Show best recommendation
        if len(recommendations) > 0:
            best = recommendations.iloc[0]
            print()
            print("🌟 BEST RECOMMENDATION:")
            print(f"   Configuration: {best['config']}")
            print(f"   Frequency: {format_frequency(best['optimal_frequency'])}")
            print(f"   Performance impact: {format_percentage(best['performance_penalty'])}")
            print(f"   Energy savings: {format_percentage(best['energy_savings'])}")
            print(f"   Efficiency ratio: {format_efficiency_ratio(best['efficiency_ratio'])}")
            
            if args.deploy:
                print()
                print("📋 DEPLOYMENT COMMANDS:")
                print(f"   Check status: ./validate_deployment.sh")
                print(f"   Deploy: ./deploy_frequencies.sh \"{best['config']}\" deploy")
                print(f"   Monitor: ./deploy_frequencies.sh \"{best['config']}\" status")
                print(f"   Reset: ./deploy_frequencies.sh \"{best['config']}\" reset")
        
        return 0
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
