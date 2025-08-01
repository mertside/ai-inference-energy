#!/usr/bin/env python3
"""
Simplified Unified GPU Frequency Optimization Analysis Tool

This script provides frequency optimization analysis without complex visualization
dependencies. Focuses on minimal performance degradation with energy savings.
"""

import sys
import os
import argparse
from pathlib import Path
import json

# Simple imports without visualization dependencies
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def calculate_efficiency_metrics(df):
    """Calculate efficiency metrics for frequency optimization"""
    # Group by configuration to find baseline (highest frequency) for each
    baseline_df = df.loc[df.groupby(['gpu', 'application'])['frequency'].idxmax()]
    baseline_dict = {}
    
    for _, baseline in baseline_df.iterrows():
        key = f"{baseline['gpu']}_{baseline['application']}"
        baseline_dict[key] = {
            'power': baseline['avg_power'],  # Use avg_power instead of power
            'execution_time': baseline['execution_time']
        }
    
    # Calculate metrics for each frequency point
    efficiency_data = []
    
    for _, row in df.iterrows():
        config_key = f"{row['gpu']}_{row['application']}"
        config_name = f"{row['gpu']}+{row['application']}"
        
        if config_key in baseline_dict:
            baseline = baseline_dict[config_key]
            
            # Calculate performance penalty (positive = slower)
            perf_penalty = ((row['execution_time'] - baseline['execution_time']) / 
                           baseline['execution_time']) * 100
            
            # Calculate energy savings (positive = less energy)
            energy_current = row['avg_power'] * row['execution_time']  # Use avg_power
            energy_baseline = baseline['power'] * baseline['execution_time']
            energy_savings = ((energy_baseline - energy_current) / energy_baseline) * 100
            
            # Calculate efficiency ratio (energy savings per % performance loss)
            if abs(perf_penalty) > 0.01:  # Avoid division by very small numbers
                efficiency_ratio = max(0, energy_savings) / max(0.01, abs(perf_penalty))
            else:
                # Performance improved or negligible change
                efficiency_ratio = max(0, energy_savings) * 100  # Very high efficiency
            
            efficiency_data.append({
                'config': config_name,
                'gpu': row['gpu'],
                'application': row['application'],
                'gpu_frequency': row['frequency'],  # Use frequency column
                'mem_frequency': row.get('avg_mmclk', 1215),  # Use avg_mmclk with fallback
                'power': row['avg_power'],  # Use avg_power
                'execution_time': row['execution_time'],
                'performance_penalty': perf_penalty,
                'energy_savings': energy_savings,
                'efficiency_ratio': efficiency_ratio
            })
    
    return pd.DataFrame(efficiency_data)

def find_optimal_frequencies(efficiency_df, max_degradation=15.0, min_efficiency=2.0):
    """Find optimal frequency configurations"""
    # Filter by performance criteria
    filtered_df = efficiency_df[
        (efficiency_df['performance_penalty'] <= max_degradation) &
        (efficiency_df['energy_savings'] > 0) &
        (efficiency_df['efficiency_ratio'] >= min_efficiency)
    ].copy()
    
    # Group by config and find best frequency for each
    optimal_configs = []
    
    for config in filtered_df['config'].unique():
        config_data = filtered_df[filtered_df['config'] == config]
        
        # Sort by efficiency ratio (descending) and performance penalty (ascending)
        best_config = config_data.sort_values(
            ['efficiency_ratio', 'performance_penalty'], 
            ascending=[False, True]
        ).iloc[0]
        
        # Categorize performance impact
        if best_config['performance_penalty'] <= 2:
            category = 'Minimal Impact'
        elif best_config['performance_penalty'] <= 5:
            category = 'Low Impact'
        elif best_config['performance_penalty'] <= 10:
            category = 'Moderate Impact'
        else:
            category = 'High Impact'
        
        optimal_configs.append({
            'config': best_config['config'],
            'gpu': best_config['gpu'],
            'application': best_config['application'],
            'optimal_frequency': best_config['gpu_frequency'],
            'performance_penalty': best_config['performance_penalty'],
            'energy_savings': best_config['energy_savings'],
            'efficiency_ratio': best_config['efficiency_ratio'],
            'category': category
        })
    
    return pd.DataFrame(optimal_configs)

def create_simple_plots(efficiency_df, optimal_df, output_dir):
    """Create simple visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pareto frontier plot
    plt.figure(figsize=(12, 8))
    
    for config in efficiency_df['config'].unique():
        config_data = efficiency_df[efficiency_df['config'] == config]
        plt.scatter(config_data['performance_penalty'], config_data['energy_savings'], 
                   alpha=0.6, label=config)
    
    # Highlight optimal points
    plt.scatter(optimal_df['performance_penalty'], optimal_df['energy_savings'], 
               color='red', s=100, marker='*', label='Optimal', zorder=5)
    
    plt.xlabel('Performance Penalty (%)')
    plt.ylabel('Energy Savings (%)')
    plt.title('GPU Frequency Optimization: Performance vs Energy Trade-off')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Optimal frequencies bar chart
    plt.figure(figsize=(14, 8))
    
    configs = optimal_df['config']
    frequencies = optimal_df['optimal_frequency']
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    
    bars = plt.bar(range(len(configs)), frequencies, color=colors)
    plt.xlabel('Configuration')
    plt.ylabel('Optimal Frequency (MHz)')
    plt.title('Optimal GPU Frequencies for Minimal Performance Impact')
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    
    # Add performance impact labels on bars
    for i, (bar, perf_penalty) in enumerate(zip(bars, optimal_df['performance_penalty'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{perf_penalty:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_frequencies.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Efficiency ratio comparison
    plt.figure(figsize=(14, 8))
    
    configs = optimal_df['config']
    efficiency_ratios = optimal_df['efficiency_ratio']
    
    bars = plt.bar(range(len(configs)), efficiency_ratios, color='green', alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Efficiency Ratio (Energy Savings / Performance Loss)')
    plt.title('Energy Efficiency Ratios for Optimal Configurations')
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    
    # Add energy savings labels
    for i, (bar, energy_savings) in enumerate(zip(bars, optimal_df['energy_savings'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{energy_savings:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_ratios.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated 3 visualization plots")

def main():
    parser = argparse.ArgumentParser(
        description='Simplified GPU Frequency Optimization Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output options
    parser.add_argument('--aggregated-data', 
                       default='../data_aggregation/complete_aggregation_run2.csv',
                       help='Path to aggregated profiling data')
    parser.add_argument('--output', default='./simple_analysis_results',
                       help='Output directory for all results')
    
    # Analysis parameters
    parser.add_argument('--max-degradation', type=float, default=15.0,
                       help='Maximum acceptable performance degradation (%)')
    parser.add_argument('--min-efficiency', type=float, default=2.0,
                       help='Minimum energy efficiency ratio')
    
    # Filtering options
    parser.add_argument('--gpu', choices=['A100', 'V100'], 
                       help='Filter analysis to specific GPU type')
    parser.add_argument('--app', choices=['LLAMA', 'STABLEDIFFUSION', 'VIT', 'WHISPER'],
                       help='Filter analysis to specific application')
    
    # Output options
    parser.add_argument('--generate-deployment', action='store_true',
                       help='Generate deployment scripts')
    parser.add_argument('--create-plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    print("🚀 Simplified GPU Frequency Optimization Analysis")
    print("=" * 60)
    print(f"📊 Data source: {args.aggregated_data}")
    print(f"📁 Output directory: {args.output}")
    print(f"🎯 Max performance degradation: {args.max_degradation}%")
    print(f"⚡ Min efficiency ratio: {args.min_efficiency}:1")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load and process data
        print("\n📊 Loading and processing profiling data...")
        df = pd.read_csv(args.aggregated_data)
        print(f"✅ Loaded {len(df)} raw configurations")
        
        # Apply filters
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
        
        if len(optimal_df) > 0:
            # Group results by category
            category_counts = optimal_df['category'].value_counts()
            for category, count in category_counts.items():
                print(f"{category}: {count} configurations")
            
            print(f"\nAverage energy savings: {optimal_df['energy_savings'].mean():.1f}%")
            print(f"Average performance impact: {optimal_df['performance_penalty'].abs().mean():.1f}%")
            print(f"Best efficiency ratio: {optimal_df['efficiency_ratio'].max():.1f}:1")
            
            # Show top recommendations
            print(f"\n🚀 Top Recommendations:")
            print("-" * 50)
            
            recommendations = optimal_df.sort_values(['performance_penalty', 'efficiency_ratio'], 
                                                    ascending=[True, False])
            
            for idx, (_, config) in enumerate(recommendations.head(3).iterrows(), 1):
                print(f"{idx}. {config['config']}: {config['optimal_frequency']}MHz")
                print(f"   Performance: {config['performance_penalty']:.1f}% impact")
                print(f"   Energy: {config['energy_savings']:.1f}% savings")
                print(f"   Efficiency: {config['efficiency_ratio']:.1f}:1 ratio")
                print(f"   Category: {config['category']}")
                print()
        else:
            print("❌ No optimal configurations found with current criteria")
            print("   Try increasing --max-degradation or decreasing --min-efficiency")
        
        # Generate visualizations
        if args.create_plots and len(optimal_df) > 0:
            print("📈 Generating visualization plots...")
            create_simple_plots(efficiency_df, optimal_df, args.output)
        
        # Save data files
        print("\n💾 Saving analysis data...")
        efficiency_df.to_csv(os.path.join(args.output, 'efficiency_analysis.csv'), index=False)
        if len(optimal_df) > 0:
            optimal_df.to_csv(os.path.join(args.output, 'optimal_configurations.csv'), index=False)
        print("✅ Saved analysis data files")
        
        # Generate deployment scripts
        if args.generate_deployment and len(optimal_df) > 0:
            print("\n🚀 Generating deployment script...")
            deployment_script_path = os.path.join(args.output, 'deploy_optimized_frequencies.sh')
            
            with open(deployment_script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# GPU Frequency Optimization Deployment Script\n")
                f.write(f"# Generated by Simplified GPU Frequency Analysis\n\n")
                
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
                    f.write(f'        status)\n')
                    f.write(f'            nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv\n')
                    f.write(f'            ;;\n')
                    f.write(f'        reset)\n')
                    f.write(f'            sudo nvidia-smi -rac\n')
                    f.write(f'            ;;\n')
                    f.write(f'    esac\n')
                    f.write(f'    exit 0\n')
                    f.write(f'fi\n\n')
                
                f.write('echo "Unknown configuration: $CONFIG"\n')
                f.write('exit 1\n')
            
            os.chmod(deployment_script_path, 0o755)
            print(f"✅ Created deployment script: {deployment_script_path}")
        
        # Export summary to JSON
        results = {
            'analysis_metadata': {
                'data_source': args.aggregated_data,
                'filters': {'gpu': args.gpu, 'application': args.app},
                'parameters': {
                    'max_performance_degradation': args.max_degradation,
                    'min_efficiency_ratio': args.min_efficiency
                }
            },
            'summary': {
                'total_configurations_analyzed': len(efficiency_df),
                'optimal_configurations_found': len(optimal_df),
                'configurations_by_category': optimal_df['category'].value_counts().to_dict() if len(optimal_df) > 0 else {}
            },
            'optimal_configurations': optimal_df.to_dict('records')
        }
        
        json_path = os.path.join(args.output, 'analysis_summary.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Exported summary to: {json_path}")
        
        print(f"\n🎉 Analysis complete! Results saved to: {args.output}")
        
        # Final recommendation
        if len(optimal_df) > 0:
            best = optimal_df.sort_values(['performance_penalty', 'efficiency_ratio'], 
                                         ascending=[True, False]).iloc[0]
            print(f"\n🌟 BEST RECOMMENDATION:")
            print(f"   Configuration: {best['config']}")
            print(f"   Frequency: {best['optimal_frequency']}MHz")
            print(f"   Performance impact: {best['performance_penalty']:.1f}%")
            print(f"   Energy savings: {best['energy_savings']:.1f}%")
            print(f"   Efficiency ratio: {best['efficiency_ratio']:.1f}:1")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
