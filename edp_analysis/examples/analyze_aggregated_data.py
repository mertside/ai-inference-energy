#!/usr/bin/env python3
"""
Quick GPU Frequency Optimization Analysis using Pre-aggregated Data

This script uses the existing aggregated data to provide immediate results.
"""

import os
import sys
import argparse
import pandas as pd
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Analyze pre-aggregated GPU profiling data')
    parser.add_argument('--aggregated-data', default='../data_aggregation/complete_aggregation_run2.csv',
                       help='Path to aggregated data file')
    parser.add_argument('--output', default='./quick_analysis_results', help='Output directory for results')
    parser.add_argument('--gpu', help='Filter by GPU type (V100, A100)')
    parser.add_argument('--app', help='Filter by application (LLAMA, STABLEDIFFUSION, VIT, WHISPER)')
    parser.add_argument('--deploy', action='store_true', help='Generate deployment scripts')
    
    args = parser.parse_args()
    
    print("🚀 Quick GPU Frequency Optimization Analysis")
    print(f"Using pre-aggregated data: {args.aggregated_data}")
    print(f"Output directory: {args.output}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load pre-aggregated data
        print("\n📊 Loading pre-aggregated data...")
        df = pd.read_csv(args.aggregated_data)
        print(f"✅ Loaded {len(df)} configurations")
        print(f"Columns: {list(df.columns)}")
        
        # Apply filters if specified
        if args.gpu:
            df = df[df['gpu'].str.upper() == args.gpu.upper()]
            print(f"Filtered to {args.gpu}: {len(df)} configurations")
        
        if args.app:
            df = df[df['application'].str.upper() == args.app.upper()]
            print(f"Filtered to {args.app}: {len(df)} configurations")
        
        if len(df) == 0:
            print("❌ No data remaining after filtering")
            return
        
        print(f"\nConfigurations available:")
        for (gpu, app), group in df.groupby(['gpu', 'application']):
            print(f"  {gpu.upper()}+{app.upper()}: {len(group)} frequency points")
        
        # Step 2: Optimization
        print("\n🎯 Finding optimal frequencies...")
        
        results = {}
        configurations = df.groupby(['gpu', 'application'])
        
        # Check if we have the expected columns
        if 'edp' in df.columns:
            edp_col = 'edp'
        elif 'EDP' in df.columns:
            edp_col = 'EDP'
        else:
            print("Available columns:", list(df.columns))
            print("❌ No EDP column found in aggregated data")
            return
        
        for (gpu, app), group in configurations:
            if len(group) < 2:
                print(f"⚠️ Skipping {gpu}+{app}: insufficient data points")
                continue
                
            # Find baseline (highest frequency) for performance reference
            baseline = group.loc[group['frequency'].idxmax()]
            
            # Find optimal frequency (minimum EDP)
            optimal_idx = group[edp_col].idxmin()
            optimal = group.loc[optimal_idx]
            
            # Calculate metrics from the aggregated data
            if 'Performance_Penalty' in group.columns and 'Energy_Savings' in group.columns:
                perf_penalty = optimal['Performance_Penalty']
                energy_savings = optimal['Energy_Savings']
            else:
                # Calculate basic metrics from available data
                baseline_energy = baseline['energy']
                optimal_energy = optimal['energy']
                baseline_time = baseline['execution_time']
                optimal_time = optimal['execution_time']
                
                # Performance penalty (how much slower)
                perf_penalty = (optimal_time / baseline_time - 1) * 100
                
                # Energy savings
                energy_savings = (1 - optimal_energy / baseline_energy) * 100
            
            # Categorize the result based on performance penalty
            abs_perf_penalty = abs(perf_penalty)  # Use absolute value for categorization
            if abs_perf_penalty <= 15:
                category = "🟢 Production ready"
            elif abs_perf_penalty <= 50:
                category = "🟡 Needs A/B testing"
            else:
                category = "🔵 Batch processing only"
            
            config_key = f"{gpu.upper()}+{app.upper()}"
            results[config_key] = {
                'gpu': gpu.upper(),
                'app': app.upper(),
                'optimal_frequency': int(optimal['frequency']),
                'performance_penalty': float(perf_penalty),
                'energy_savings': float(energy_savings),
                'category': category,
                'baseline_frequency': int(baseline['frequency']),
                'optimal_power': float(optimal['avg_power']),
                'baseline_power': float(baseline['avg_power']),
                'optimal_edp': float(optimal[edp_col]),
                'baseline_edp': float(baseline[edp_col]),
                'optimal_execution_time': float(optimal['execution_time']),
                'baseline_execution_time': float(baseline['execution_time'])
            }
        
        print(f"✅ Optimized {len(results)} configurations")
        
        # Step 3: Results analysis and reporting
        print("\n📋 Analyzing results...")
        
        # Categorize results
        production_ready = []
        ab_testing = []
        batch_only = []
        
        for config_key, result in results.items():
            if result['category'] == "🟢 Production ready":
                production_ready.append((config_key, result))
            elif result['category'] == "🟡 Needs A/B testing":
                ab_testing.append((config_key, result))
            else:
                batch_only.append((config_key, result))
        
        # Print summary
        print(f"  🟢 Production ready: {len(production_ready)} configurations")
        for config_key, result in production_ready:
            print(f"    - {config_key}: {result['optimal_frequency']}MHz "
                  f"({result['performance_penalty']:.1f}% slower, {result['energy_savings']:.1f}% energy savings)")
        
        print(f"  🟡 Needs A/B testing: {len(ab_testing)} configurations")
        for config_key, result in ab_testing:
            print(f"    - {config_key}: {result['optimal_frequency']}MHz "
                  f"({result['performance_penalty']:.1f}% slower, {result['energy_savings']:.1f}% energy savings)")
        
        print(f"  🔵 Batch processing only: {len(batch_only)} configurations")
        for config_key, result in batch_only:
            print(f"    - {config_key}: {result['optimal_frequency']}MHz "
                  f"({result['performance_penalty']:.1f}% slower, {result['energy_savings']:.1f}% energy savings)")
        
        # Save aggregated data to output
        output_data_file = os.path.join(args.output, 'aggregated_data.csv')
        df.to_csv(output_data_file, index=False)
        
        # Save detailed results
        results_file = os.path.join(args.output, 'optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        summary_file = os.path.join(args.output, 'optimization_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("GPU Frequency Optimization Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
            f.write(f"Data Source: {args.aggregated_data}\n")
            f.write(f"Total Configurations: {len(results)}\n")
            f.write(f"Production Ready: {len(production_ready)}\n")
            f.write(f"A/B Testing Recommended: {len(ab_testing)}\n")
            f.write(f"Batch Processing Only: {len(batch_only)}\n\n")
            
            for config_key, result in results.items():
                f.write(f"{config_key}:\n")
                f.write(f"  Optimal Frequency: {result['optimal_frequency']} MHz\n")
                f.write(f"  Baseline Frequency: {result['baseline_frequency']} MHz\n")
                f.write(f"  Performance Penalty: {result['performance_penalty']:.1f}%\n")
                f.write(f"  Energy Savings: {result['energy_savings']:.1f}%\n")
                f.write(f"  Category: {result['category']}\n")
                f.write(f"  Optimal EDP: {result['optimal_edp']:.6f}\n")
                f.write(f"  Baseline EDP: {result['baseline_edp']:.6f}\n\n")
        
        # Generate deployment script if requested
        if args.deploy and production_ready:
            deploy_file = os.path.join(args.output, 'deploy_optimal_frequencies.sh')
            with open(deploy_file, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# GPU Frequency Deployment Script\n")
                f.write("# Generated by GPU Frequency Optimization Analysis\n\n")
                f.write("if [ $# -ne 2 ]; then\n")
                f.write('    echo "Usage: $0 <CONFIG> <ACTION>"\n')
                f.write('    echo "CONFIG: ' + ', '.join([k for k, _ in production_ready]) + '"\n')
                f.write('    echo "ACTION: deploy, reset"\n')
                f.write("    exit 1\n")
                f.write("fi\n\n")
                f.write("CONFIG=$1\nACTION=$2\n\n")
                
                for config_key, result in production_ready:
                    mem_freq = 1215 if 'A100' in config_key else 877  # Default memory frequencies
                    gpu_freq = result['optimal_frequency']
                    f.write(f'if [ "$CONFIG" = "{config_key}" ]; then\n')
                    f.write(f'    if [ "$ACTION" = "deploy" ]; then\n')
                    f.write(f'        echo "Deploying {config_key}: {gpu_freq}MHz"\n')
                    f.write(f'        sudo nvidia-smi -ac {mem_freq},{gpu_freq}\n')
                    f.write(f'    elif [ "$ACTION" = "reset" ]; then\n')
                    f.write(f'        echo "Resetting {config_key} to default"\n')
                    f.write(f'        sudo nvidia-smi -rac\n')
                    f.write(f'    fi\n')
                    f.write(f'fi\n\n')
            
            os.chmod(deploy_file, 0o755)
            print(f"✅ Created deployment script: {deploy_file}")
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📁 All results saved to: {args.output}")
        
        # Recommend next steps
        if production_ready:
            best_config = min(production_ready, key=lambda x: x[1]['performance_penalty'])
            config_key, result = best_config
            print(f"\n🚀 Ready for immediate deployment:")
            print(f"Recommended: {config_key}")
            mem_freq = 1215 if 'A100' in config_key else 877
            print(f"Command: sudo nvidia-smi -ac {mem_freq},{result['optimal_frequency']}")
            print(f"Expected: {result['performance_penalty']:.1f}% slower, {result['energy_savings']:.1f}% energy savings")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
