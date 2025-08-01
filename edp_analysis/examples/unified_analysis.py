#!/usr/bin/env python3
"""
Unified GPU Frequency Optimization Analysis Tool

This script combines the core frequency optimization with advanced visualization
using the refactored modules. Focuses on minimal performance degradation with
disproportionately high energy savings.

Usage:
    python unified_analysis.py --output ./results
    python unified_analysis.py --max-degradation 10 --gpu A100
    python unified_analysis.py --app STABLEDIFFUSION --comprehensive-plots
"""

import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.advanced_plotter import OptimizationVisualizer
from enhanced_frequency_analysis import calculate_efficiency_metrics, find_optimal_frequencies
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(
        description='Unified GPU Frequency Optimization Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output options
    parser.add_argument('--aggregated-data', 
                       default='../data_aggregation/complete_aggregation_run2.csv',
                       help='Path to aggregated profiling data')
    parser.add_argument('--output', default='./unified_analysis_results',
                       help='Output directory for all results')
    
    # Analysis parameters
    parser.add_argument('--max-degradation', type=float, default=15.0,
                       help='Maximum acceptable performance degradation (%)')
    parser.add_argument('--min-efficiency', type=float, default=2.0,
                       help='Minimum energy efficiency ratio')
    parser.add_argument('--target-savings', type=float, default=20.0,
                       help='Target energy savings (%)')
    
    # Filtering options
    parser.add_argument('--gpu', choices=['A100', 'V100'], 
                       help='Filter analysis to specific GPU type')
    parser.add_argument('--app', choices=['LLAMA', 'STABLEDIFFUSION', 'VIT', 'WHISPER'],
                       help='Filter analysis to specific application')
    
    # Visualization options
    parser.add_argument('--comprehensive-plots', action='store_true',
                       help='Generate comprehensive visualization report')
    parser.add_argument('--plot-style', default='default',
                       choices=['default', 'ggplot', 'seaborn', 'bmh'],
                       help='Matplotlib style for plots')
    
    # Output options
    parser.add_argument('--generate-deployment', action='store_true',
                       help='Generate deployment scripts for optimal configurations')
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON format')
    
    args = parser.parse_args()
    
    print("🚀 Unified GPU Frequency Optimization Analysis")
    print("=" * 60)
    print(f"📊 Data source: {args.aggregated_data}")
    print(f"📁 Output directory: {args.output}")
    print(f"🎯 Max performance degradation: {args.max_degradation}%")
    print(f"⚡ Min efficiency ratio: {args.min_efficiency}:1")
    print(f"💚 Target energy savings: {args.target_savings}%")
    
    if args.gpu:
        print(f"🖥️  GPU filter: {args.gpu}")
    if args.app:
        print(f"🤖 Application filter: {args.app}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load and process data
        print("\n📊 Loading and processing profiling data...")
        df = pd.read_csv(args.aggregated_data)
        print(f"✅ Loaded {len(df)} raw configurations")
        
        # Apply filters
        original_count = len(df)
        if args.gpu:
            df = df[df['gpu'].str.upper() == args.gpu.upper()]
            print(f"🔍 Filtered to {args.gpu}: {len(df)} configurations")
        
        if args.app:
            df = df[df['application'].str.upper() == args.app.upper()]
            print(f"🔍 Filtered to {args.app}: {len(df)} configurations")
        
        if len(df) == 0:
            print("❌ No data remaining after filtering")
            return 1
        
        # Calculate efficiency metrics
        print("\n🧮 Calculating efficiency metrics...")
        efficiency_df = calculate_efficiency_metrics(df)
        print(f"✅ Calculated metrics for {len(efficiency_df)} data points")
        
        # Find optimal configurations
        print(f"\n🎯 Finding optimal frequencies (≤{args.max_degradation}% degradation)...")
        optimal_df = find_optimal_frequencies(efficiency_df, args.max_degradation, args.min_efficiency)
        print(f"✅ Found {len(optimal_df)} optimal configurations")
        
        # Display results summary
        print("\n📋 Optimization Results Summary:")
        print("-" * 50)
        
        # Group results by category
        category_counts = optimal_df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"{category}: {count} configurations")
        
        print(f"\nAverage energy savings: {optimal_df['energy_savings'].mean():.1f}%")
        print(f"Average performance impact: {optimal_df['performance_penalty'].abs().mean():.1f}%")
        print(f"Best efficiency ratio: {optimal_df['efficiency_ratio'].max():.1f}:1")
        
        # Show top recommendations
        print(f"\n🚀 Top Recommendations (≤{args.max_degradation}% degradation):")
        print("-" * 50)
        
        # Sort by best efficiency ratio and lowest performance impact
        recommendations = optimal_df.sort_values(['performance_penalty', 'efficiency_ratio'], 
                                                ascending=[True, False])
        
        for idx, (_, config) in enumerate(recommendations.head(3).iterrows(), 1):
            print(f"{idx}. {config['config']}: {config['optimal_frequency']}MHz")
            print(f"   Performance: {config['performance_penalty']:.1f}% impact")
            print(f"   Energy: {config['energy_savings']:.1f}% savings")
            print(f"   Efficiency: {config['efficiency_ratio']:.1f}:1 ratio")
            print(f"   Category: {config['category']}")
            print()
        
        # Generate visualizations
        if args.comprehensive_plots:
            print("📈 Generating comprehensive visualization report...")
            visualizer = OptimizationVisualizer(style=args.plot_style)
            visualizer.generate_comprehensive_report(efficiency_df, optimal_df, args.output)
        else:
            print("📈 Generating core visualizations...")
            visualizer = OptimizationVisualizer(style=args.plot_style)
            visualizer.create_pareto_frontier_plot(efficiency_df, args.output)
            visualizer.create_optimal_selection_plot(optimal_df, args.output)
        
        # Save data files
        print("\n💾 Saving analysis data...")
        efficiency_df.to_csv(os.path.join(args.output, 'efficiency_analysis.csv'), index=False)
        optimal_df.to_csv(os.path.join(args.output, 'optimal_configurations.csv'), index=False)
        print("✅ Saved efficiency_analysis.csv and optimal_configurations.csv")
        
        # Generate deployment scripts
        if args.generate_deployment:
            print("\n🚀 Generating deployment scripts...")
            deployment_script_path = os.path.join(args.output, 'deploy_optimized_frequencies.sh')
            
            with open(deployment_script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# GPU Frequency Optimization Deployment Script\n")
                f.write(f"# Generated by Unified GPU Frequency Analysis\n")
                f.write(f"# Analysis date: {pd.Timestamp.now()}\n\n")
                
                f.write("if [ $# -ne 2 ]; then\n")
                f.write('    echo "Usage: $0 <CONFIG> <ACTION>"\n')
                f.write('    echo "Available configurations:"\n')
                for _, config in optimal_df.iterrows():
                    f.write(f'    echo "  {config["config"]} - {config["optimal_frequency"]}MHz"\n')
                f.write('    echo "Actions: deploy, reset, status"\n')
                f.write("    exit 1\n")
                f.write("fi\n\n")
                
                f.write("CONFIG=$1\nACTION=$2\n\n")
                
                for _, config in optimal_df.iterrows():
                    mem_freq = 1215 if 'A100' in config['config'] else 877
                    gpu_freq = config['optimal_frequency']
                    
                    f.write(f'if [ "$CONFIG" = "{config["config"]}" ]; then\n')
                    f.write(f'    case "$ACTION" in\n')
                    f.write(f'        deploy)\n')
                    f.write(f'            echo "Deploying {config["config"]}: {gpu_freq}MHz"\n')
                    f.write(f'            echo "Expected: {config["performance_penalty"]:.1f}% performance, {config["energy_savings"]:.1f}% energy savings"\n')
                    f.write(f'            sudo nvidia-smi -ac {mem_freq},{gpu_freq}\n')
                    f.write(f'            ;;\n')
                    f.write(f'        reset)\n')
                    f.write(f'            echo "Resetting {config["config"]} to default frequencies"\n')
                    f.write(f'            sudo nvidia-smi -rac\n')
                    f.write(f'            ;;\n')
                    f.write(f'        status)\n')
                    f.write(f'            echo "Configuration: {config["config"]}"\n')
                    f.write(f'            echo "Optimal frequency: {gpu_freq}MHz"\n')
                    f.write(f'            echo "Performance impact: {config["performance_penalty"]:.1f}%"\n')
                    f.write(f'            echo "Energy savings: {config["energy_savings"]:.1f}%"\n')
                    f.write(f'            echo "Efficiency ratio: {config["efficiency_ratio"]:.1f}:1"\n')
                    f.write(f'            nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv\n')
                    f.write(f'            ;;\n')
                    f.write(f'        *)\n')
                    f.write(f'            echo "Unknown action: $ACTION"\n')
                    f.write(f'            exit 1\n')
                    f.write(f'            ;;\n')
                    f.write(f'    esac\n')
                    f.write(f'    exit 0\n')
                    f.write(f'fi\n\n')
                
                f.write('echo "Unknown configuration: $CONFIG"\n')
                f.write('exit 1\n')
            
            os.chmod(deployment_script_path, 0o755)
            print(f"✅ Created deployment script: {deployment_script_path}")
        
        # Export to JSON
        if args.export_json:
            print("\n📄 Exporting results to JSON...")
            results = {
                'analysis_metadata': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'data_source': args.aggregated_data,
                    'filters': {
                        'gpu': args.gpu,
                        'application': args.app
                    },
                    'parameters': {
                        'max_performance_degradation': args.max_degradation,
                        'min_efficiency_ratio': args.min_efficiency,
                        'target_energy_savings': args.target_savings
                    }
                },
                'summary': {
                    'total_configurations_analyzed': len(efficiency_df),
                    'optimal_configurations_found': len(optimal_df),
                    'average_energy_savings': float(optimal_df['energy_savings'].mean()),
                    'average_performance_impact': float(optimal_df['performance_penalty'].abs().mean()),
                    'best_efficiency_ratio': float(optimal_df['efficiency_ratio'].max()),
                    'configurations_by_category': optimal_df['category'].value_counts().to_dict()
                },
                'optimal_configurations': optimal_df.to_dict('records'),
                'recommendations': {
                    'immediate_deployment': recommendations.iloc[0].to_dict() if len(recommendations) > 0 else None,
                    'top_3_recommendations': recommendations.head(3).to_dict('records')
                }
            }
            
            json_path = os.path.join(args.output, 'unified_analysis_results.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Exported to: {json_path}")
        
        print(f"\n🎉 Unified analysis complete!")
        print(f"📁 All results saved to: {args.output}")
        
        # Final recommendation
        if len(recommendations) > 0:
            best = recommendations.iloc[0]
            print(f"\n🌟 BEST RECOMMENDATION:")
            print(f"   Configuration: {best['config']}")
            print(f"   Frequency: {best['optimal_frequency']}MHz")
            print(f"   Performance impact: {best['performance_penalty']:.1f}%")
            print(f"   Energy savings: {best['energy_savings']:.1f}%")
            print(f"   Efficiency ratio: {best['efficiency_ratio']:.1f}:1")
            
            if args.generate_deployment:
                print(f"   Deploy command: ./deploy_optimized_frequencies.sh {best['config']} deploy")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
