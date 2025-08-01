#!/usr/bin/env python3
"""
Enhanced GPU Frequency Optimization with Advanced Visualization

This script provides sophisticated analysis and plotting focused on:
- Minimal performance degradation
- Disproportionately high energy savings
- Pareto frontier analysis
- Energy efficiency ratios
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def calculate_efficiency_metrics(df):
    """Calculate advanced efficiency metrics for optimization"""
    
    # Group by GPU and application
    results = []
    
    for (gpu, app), group in df.groupby(['gpu', 'application']):
        group = group.sort_values('frequency')
        
        # Find baseline (highest frequency)
        baseline = group.loc[group['frequency'].idxmax()]
        
        # Calculate metrics for each frequency point
        for idx, row in group.iterrows():
            # Performance degradation (positive = slower, negative = faster)
            perf_degradation = (row['execution_time'] / baseline['execution_time'] - 1) * 100
            
            # Energy savings (positive = savings)
            energy_savings = (1 - row['energy'] / baseline['energy']) * 100
            
            # Energy efficiency ratio (energy savings per unit performance loss)
            # Handle cases where performance improves (negative degradation)
            if abs(perf_degradation) > 0.01:  # Avoid division by near-zero
                if perf_degradation > 0:
                    # Performance degradation case
                    efficiency_ratio = energy_savings / perf_degradation
                else:
                    # Performance improvement case - use absolute values
                    efficiency_ratio = energy_savings / abs(perf_degradation)
            else:
                # Near-zero performance change
                efficiency_ratio = energy_savings * 100  # High efficiency for minimal perf impact
            
            # EDP improvement
            edp_improvement = (1 - row['edp'] / baseline['edp']) * 100
            
            # Power reduction
            power_reduction = (1 - row['avg_power'] / baseline['avg_power']) * 100
            
            results.append({
                'gpu': gpu.upper(),
                'application': app.upper(),
                'frequency': row['frequency'],
                'performance_degradation': perf_degradation,
                'energy_savings': energy_savings,
                'efficiency_ratio': efficiency_ratio,
                'edp_improvement': edp_improvement,
                'power_reduction': power_reduction,
                'execution_time': row['execution_time'],
                'energy': row['energy'],
                'power': row['avg_power'],
                'edp': row['edp'],
                'is_baseline': row['frequency'] == baseline['frequency']
            })
    
    return pd.DataFrame(results)

def find_optimal_frequencies(df, max_perf_degradation=15.0, min_efficiency_ratio=2.0):
    """Find optimal frequencies based on performance and efficiency constraints"""
    
    optimal_configs = []
    
    for (gpu, app), group in df.groupby(['gpu', 'application']):
        # Filter by performance constraint
        candidates = group[group['performance_degradation'] <= max_perf_degradation]
        
        if len(candidates) == 0:
            # If no candidates meet performance constraint, take best performance
            candidates = group.nsmallest(1, 'performance_degradation')
        
        # Find best efficiency ratio among candidates
        if len(candidates) > 0:
            optimal = candidates.loc[candidates['efficiency_ratio'].idxmax()]
            
            # Categorize based on performance degradation
            if optimal['performance_degradation'] <= 5:
                category = "🟢 Minimal Impact"
            elif optimal['performance_degradation'] <= 15:
                category = "🟡 Low Impact"
            elif optimal['performance_degradation'] <= 30:
                category = "🟠 Moderate Impact"
            else:
                category = "🔴 High Impact"
            
            optimal_configs.append({
                'config': f"{gpu.upper()}+{app.upper()}",
                'gpu': gpu.upper(),
                'application': app.upper(),
                'optimal_frequency': optimal['frequency'],
                'performance_degradation': optimal['performance_degradation'],
                'energy_savings': optimal['energy_savings'],
                'efficiency_ratio': optimal['efficiency_ratio'],
                'edp_improvement': optimal['edp_improvement'],
                'power_reduction': optimal['power_reduction'],
                'category': category
            })
    
    return pd.DataFrame(optimal_configs)

def create_pareto_analysis_plot(df, output_dir):
    """Create Pareto frontier analysis plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPU Frequency Optimization: Performance vs Energy Trade-offs', fontsize=16, fontweight='bold')
    
    # Color mapping for GPUs
    gpu_colors = {'A100': '#2E86AB', 'V100': '#A23B72'}
    app_markers = {'LLAMA': 'o', 'STABLEDIFFUSION': 's', 'VIT': '^', 'WHISPER': 'D'}
    
    # Plot 1: Energy Savings vs Performance Degradation
    ax1 = axes[0, 0]
    for (gpu, app), group in df.groupby(['gpu', 'application']):
        ax1.scatter(group['performance_degradation'], group['energy_savings'], 
                   c=gpu_colors[gpu.upper()], marker=app_markers[app.upper()], 
                   s=60, alpha=0.7, label=f'{gpu.upper()}+{app.upper()}')
    
    ax1.set_xlabel('Performance Degradation (%)')
    ax1.set_ylabel('Energy Savings (%)')
    ax1.set_title('Energy Savings vs Performance Degradation')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=15, color='red', linestyle='--', alpha=0.7, label='15% Performance Limit')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Efficiency Ratio vs Frequency
    ax2 = axes[0, 1]
    for (gpu, app), group in df.groupby(['gpu', 'application']):
        # Filter out infinite values for plotting
        plot_group = group[group['efficiency_ratio'] != float('inf')]
        if len(plot_group) > 0:
            ax2.plot(plot_group['frequency'], plot_group['efficiency_ratio'], 
                    marker=app_markers[app.upper()], color=gpu_colors[gpu.upper()],
                    linewidth=2, markersize=6, label=f'{gpu.upper()}+{app.upper()}')
    
    ax2.set_xlabel('GPU Frequency (MHz)')
    ax2.set_ylabel('Efficiency Ratio (Energy Savings / Performance Loss)')
    ax2.set_title('Energy Efficiency vs Frequency')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='2:1 Efficiency Target')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: EDP Improvement vs Performance Degradation
    ax3 = axes[1, 0]
    for (gpu, app), group in df.groupby(['gpu', 'application']):
        ax3.scatter(group['performance_degradation'], group['edp_improvement'], 
                   c=gpu_colors[gpu.upper()], marker=app_markers[app.upper()], 
                   s=60, alpha=0.7, label=f'{gpu.upper()}+{app.upper()}')
    
    ax3.set_xlabel('Performance Degradation (%)')
    ax3.set_ylabel('EDP Improvement (%)')
    ax3.set_title('EDP Improvement vs Performance Degradation')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=15, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Power vs Energy Reduction
    ax4 = axes[1, 1]
    for (gpu, app), group in df.groupby(['gpu', 'application']):
        ax4.scatter(group['power_reduction'], group['energy_savings'], 
                   c=gpu_colors[gpu.upper()], marker=app_markers[app.upper()], 
                   s=60, alpha=0.7, label=f'{gpu.upper()}+{app.upper()}')
    
    ax4.set_xlabel('Power Reduction (%)')
    ax4.set_ylabel('Energy Savings (%)')
    ax4.set_title('Energy vs Power Reduction')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'pareto_analysis.pdf'), bbox_inches='tight')
    print(f"✅ Saved Pareto analysis plot: {output_dir}/pareto_analysis.png")
    return fig

def create_optimal_selection_plot(optimal_df, output_dir):
    """Create visualization of optimal frequency selections"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Optimal GPU Frequency Selection Results', fontsize=16, fontweight='bold')
    
    # Color mapping by category
    category_colors = {
        "🟢 Minimal Impact": '#27AE60',
        "🟡 Low Impact": '#F39C12', 
        "🟠 Moderate Impact": '#E67E22',
        "🔴 High Impact": '#E74C3C'
    }
    
    # Plot 1: Performance vs Energy Trade-off
    ax1.scatter(optimal_df['performance_degradation'], optimal_df['energy_savings'], 
               c=[category_colors[cat] for cat in optimal_df['category']], 
               s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add labels for each point
    for idx, row in optimal_df.iterrows():
        ax1.annotate(f"{row['gpu']}+{row['application']}", 
                    (row['performance_degradation'], row['energy_savings']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Performance Degradation (%)')
    ax1.set_ylabel('Energy Savings (%)')
    ax1.set_title('Optimal Configurations: Performance vs Energy')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=15, color='red', linestyle='--', alpha=0.7, label='15% Performance Limit')
    
    # Add category legend
    for category, color in category_colors.items():
        ax1.scatter([], [], c=color, s=100, label=category.split(' ', 1)[1])
    ax1.legend()
    
    # Plot 2: Efficiency Ratio by Configuration
    configs = optimal_df['config'].tolist()
    efficiency_ratios = optimal_df['efficiency_ratio'].tolist()
    colors = [category_colors[cat] for cat in optimal_df['category']]
    
    bars = ax2.bar(range(len(configs)), efficiency_ratios, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('GPU + Application Configuration')
    ax2.set_ylabel('Efficiency Ratio (Energy Savings / Performance Loss)')
    ax2.set_title('Energy Efficiency Ratio by Configuration')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='2:1 Target Efficiency')
    
    # Add value labels on bars
    for bar, value in zip(bars, efficiency_ratios):
        if value != float('inf'):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimal_selection.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'optimal_selection.pdf'), bbox_inches='tight')
    print(f"✅ Saved optimal selection plot: {output_dir}/optimal_selection.png")
    return fig

def create_frequency_sweep_plot(df, output_dir):
    """Create detailed frequency sweep analysis"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Frequency Sweep Analysis: All GPU+Application Combinations', fontsize=16, fontweight='bold')
    
    combinations = df[['gpu', 'application']].drop_duplicates().sort_values(['gpu', 'application'])
    
    for idx, (_, row) in enumerate(combinations.iterrows()):
        gpu, app = row['gpu'], row['application']
        ax = axes[idx // 4, idx % 4]
        
        # Get data for this combination
        subset = df[(df['gpu'] == gpu) & (df['application'] == app)].sort_values('frequency')
        
        # Create twin axes for energy and performance
        ax2 = ax.twinx()
        
        # Plot energy savings
        line1 = ax.plot(subset['frequency'], subset['energy_savings'], 
                       'g-o', linewidth=2, markersize=4, label='Energy Savings')
        
        # Plot performance degradation  
        line2 = ax2.plot(subset['frequency'], subset['performance_degradation'], 
                        'r-s', linewidth=2, markersize=4, label='Performance Degradation')
        
        # Mark optimal point
        optimal_point = subset.loc[subset['efficiency_ratio'].idxmax()]
        ax.scatter(optimal_point['frequency'], optimal_point['energy_savings'], 
                  s=100, c='gold', marker='*', edgecolors='black', linewidth=2, 
                  label='Optimal', zorder=5)
        
        # Formatting
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Energy Savings (%)', color='green')
        ax2.set_ylabel('Performance Degradation (%)', color='red')
        ax.set_title(f'{gpu.upper()}+{app.upper()}')
        ax.grid(True, alpha=0.3)
        
        # Performance constraint line
        ax2.axhline(y=15, color='red', linestyle='--', alpha=0.5)
        
        # Color axes labels
        ax.tick_params(axis='y', labelcolor='green')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_sweep_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'frequency_sweep_analysis.pdf'), bbox_inches='tight')
    print(f"✅ Saved frequency sweep plot: {output_dir}/frequency_sweep_analysis.png")
    return fig

def main():
    parser = argparse.ArgumentParser(description='Enhanced GPU Frequency Optimization with Advanced Visualization')
    parser.add_argument('--aggregated-data', default='../data_aggregation/complete_aggregation_run2.csv',
                       help='Path to aggregated data file')
    parser.add_argument('--output', default='./enhanced_analysis_results', help='Output directory for results')
    parser.add_argument('--max-perf-degradation', type=float, default=15.0, 
                       help='Maximum acceptable performance degradation (%)')
    parser.add_argument('--min-efficiency-ratio', type=float, default=2.0,
                       help='Minimum energy efficiency ratio')
    parser.add_argument('--gpu', help='Filter by GPU type (V100, A100)')
    parser.add_argument('--app', help='Filter by application (LLAMA, STABLEDIFFUSION, VIT, WHISPER)')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced GPU Frequency Optimization Analysis")
    print(f"Using pre-aggregated data: {args.aggregated_data}")
    print(f"Output directory: {args.output}")
    print(f"Max performance degradation: {args.max_perf_degradation}%")
    print(f"Min efficiency ratio: {args.min_efficiency_ratio}:1")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load and process data
        print("\n📊 Loading and processing data...")
        df = pd.read_csv(args.aggregated_data)
        
        # Apply filters if specified
        if args.gpu:
            df = df[df['gpu'].str.upper() == args.gpu.upper()]
        if args.app:
            df = df[df['application'].str.upper() == args.app.upper()]
        
        print(f"✅ Loaded {len(df)} configurations")
        
        # Calculate efficiency metrics
        print("\n🧮 Calculating efficiency metrics...")
        efficiency_df = calculate_efficiency_metrics(df)
        print(f"✅ Calculated metrics for {len(efficiency_df)} data points")
        
        # Find optimal frequencies
        print(f"\n🎯 Finding optimal frequencies (max {args.max_perf_degradation}% degradation)...")
        optimal_df = find_optimal_frequencies(efficiency_df, args.max_perf_degradation, args.min_efficiency_ratio)
        print(f"✅ Found {len(optimal_df)} optimal configurations")
        
        # Display results
        print("\n📋 Optimal Configuration Results:")
        for _, row in optimal_df.iterrows():
            print(f"  {row['category']} {row['config']}: {row['optimal_frequency']}MHz")
            print(f"    Performance: {row['performance_degradation']:.1f}% degradation")
            print(f"    Energy: {row['energy_savings']:.1f}% savings")
            print(f"    Efficiency: {row['efficiency_ratio']:.1f}:1 ratio")
            print(f"    EDP: {row['edp_improvement']:.1f}% improvement")
        
        # Create visualizations
        print("\n📈 Creating advanced visualizations...")
        
        # Pareto analysis plot
        create_pareto_analysis_plot(efficiency_df, args.output)
        
        # Optimal selection plot
        create_optimal_selection_plot(optimal_df, args.output)
        
        # Frequency sweep plot
        create_frequency_sweep_plot(efficiency_df, args.output)
        
        # Save data
        efficiency_df.to_csv(os.path.join(args.output, 'efficiency_analysis.csv'), index=False)
        optimal_df.to_csv(os.path.join(args.output, 'optimal_configurations.csv'), index=False)
        
        # Save results JSON
        results = {
            'analysis_parameters': {
                'max_performance_degradation': args.max_perf_degradation,
                'min_efficiency_ratio': args.min_efficiency_ratio,
                'data_source': args.aggregated_data
            },
            'optimal_configurations': optimal_df.to_dict('records'),
            'summary': {
                'total_configurations': len(optimal_df),
                'minimal_impact': len(optimal_df[optimal_df['category'] == "🟢 Minimal Impact"]),
                'low_impact': len(optimal_df[optimal_df['category'] == "🟡 Low Impact"]),
                'moderate_impact': len(optimal_df[optimal_df['category'] == "🟠 Moderate Impact"]),
                'high_impact': len(optimal_df[optimal_df['category'] == "🔴 High Impact"])
            }
        }
        
        with open(os.path.join(args.output, 'enhanced_optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Enhanced analysis complete!")
        print(f"📁 All results saved to: {args.output}")
        
        # Recommendations
        best_minimal = optimal_df[optimal_df['category'] == "🟢 Minimal Impact"]
        if len(best_minimal) > 0:
            best = best_minimal.loc[best_minimal['efficiency_ratio'].idxmax()]
            print(f"\n🚀 Recommended for immediate deployment:")
            print(f"   {best['config']}: {best['optimal_frequency']}MHz")
            print(f"   Impact: {best['performance_degradation']:.1f}% performance, {best['energy_savings']:.1f}% energy savings")
            print(f"   Efficiency: {best['efficiency_ratio']:.1f}:1 ratio")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
