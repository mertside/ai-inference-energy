#!/usr/bin/env python3
"""
Simple GPU Frequency Optimization Analysis Tool

A simplified version that works with the current framework structure
and provides immediate results for testing the new tools.
"""

import os
import sys
import argparse
import pandas as pd
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Analyze GPU profiling data for optimal frequencies')
    parser.add_argument('--data-dir', required=True, help='Directory containing profiling data')
    parser.add_argument('--output', required=True, help='Output directory for results')
    parser.add_argument('--gpu', help='Filter by GPU type (V100, A100)')
    parser.add_argument('--app', help='Filter by application (LLAMA, STABLEDIFFUSION, VIT, WHISPER)')
    parser.add_argument('--run', type=int, default=2, help='Run number to use (default: 2)')
    parser.add_argument('--deploy', action='store_true', help='Generate deployment scripts')
    
    args = parser.parse_args()
    
    print("🚀 Starting GPU Frequency Optimization Analysis")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Using run {args.run} data {'(excludes cold start if run > 1)' if args.run > 1 else ''}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Step 1: Simple data discovery and aggregation
        print("\n📊 Step 1: Discovering and aggregating profiling data...")
        
        # Find result directories
        data_dir = Path(args.data_dir)
        result_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
        
        if not result_dirs:
            print("❌ No result directories found!")
            return
            
        print(f"✅ Found {len(result_dirs)} result directories")
        
        aggregated_data = []
        
        for result_dir in result_dirs:
            # Parse directory name to extract GPU, app, and job info
            dir_name = result_dir.name
            if 'results_' in dir_name:
                parts = dir_name.replace('results_', '').split('_')
                if len(parts) >= 2:
                    gpu_type = parts[0].upper()
                    app_name = parts[1].upper()
                    
                    # Apply filters if specified
                    if args.gpu and gpu_type != args.gpu.upper():
                        continue
                    if args.app and app_name != args.app.upper():
                        continue
                    
                    print(f"Processing {gpu_type}+{app_name}...")
                    
                    # Find CSV files for the specified run
                    csv_files = list(result_dir.glob(f'run_*_{args.run:02d}_freq_*_profile.csv'))
                    
                    if not csv_files:
                        print(f"  ⚠️ No run {args.run} data found, checking available runs...")
                        all_csv_files = list(result_dir.glob('run_*_profile.csv'))
                        if all_csv_files:
                            available_runs = set()
                            for f in all_csv_files:
                                parts = f.stem.split('_')
                                if len(parts) >= 3:
                                    available_runs.add(int(parts[2]))
                            print(f"  Available runs: {sorted(available_runs)}")
                            # Use the highest available run
                            if available_runs:
                                use_run = max(available_runs)
                                csv_files = list(result_dir.glob(f'run_*_{use_run:02d}_freq_*_profile.csv'))
                                print(f"  Using run {use_run} instead")
                    
                    freq_data = []
                    for csv_file in csv_files:
                        try:
                            # Extract frequency from filename: run_XXX_02_freq_YYY_profile.csv
                            filename = csv_file.stem
                            parts = filename.split('_')
                            
                            # Look for the frequency value which comes after 'freq'
                            frequency = None
                            for i, part in enumerate(parts):
                                if part == 'freq' and i + 1 < len(parts):
                                    frequency = int(parts[i + 1])
                                    break
                            
                            if frequency is None:
                                print(f"  ⚠️ Could not extract frequency from {csv_file.name}")
                                continue
                            
                            # Read CSV and calculate basic metrics
                            df = pd.read_csv(csv_file, comment='#', skiprows=1)  # Skip units row
                            
                            # Clean column names (remove extra spaces)
                            df.columns = df.columns.str.strip()
                            
                            if 'POWER' in df.columns and 'GPUTL' in df.columns:
                                # Simple aggregation - use median of middle 50% of data
                                n = len(df)
                                start_idx = n // 4
                                end_idx = 3 * n // 4
                                stable_data = df.iloc[start_idx:end_idx]
                                
                                power = stable_data['POWER'].median()
                                utilization = stable_data['GPUTL'].median()
                                
                                # Simple runtime estimation (inverse of utilization)
                                runtime_factor = 1.0 / max(utilization / 100.0, 0.01)
                                
                                # Calculate EDP (Energy-Delay Product)
                                energy = power * runtime_factor
                                edp = energy * runtime_factor
                                
                                freq_data.append({
                                    'GPU': gpu_type,
                                    'App': app_name,
                                    'Frequency': frequency,
                                    'Power': power,
                                    'Utilization': utilization,
                                    'Runtime_Factor': runtime_factor,
                                    'Energy': energy,
                                    'EDP': edp
                                })
                        except Exception as e:
                            print(f"  ⚠️ Error processing {csv_file.name}: {e}")
                            continue
                    
                    if freq_data:
                        print(f"  ✅ Processed {len(freq_data)} frequency points")
                        aggregated_data.extend(freq_data)
                    else:
                        print(f"  ❌ No valid data found")
        
        if not aggregated_data:
            print("❌ No data found to process")
            return
        
        # Save aggregated data
        df_aggregated = pd.DataFrame(aggregated_data)
        aggregated_file = os.path.join(args.output, 'aggregated_data.csv')
        df_aggregated.to_csv(aggregated_file, index=False)
        print(f"✅ Aggregated {len(aggregated_data)} total configurations")
        print(f"✅ Saved aggregated data to: {aggregated_file}")
        
        # Step 2: Simple optimization
        print("\n🎯 Step 2: Finding optimal frequencies...")
        
        results = {}
        configurations = df_aggregated.groupby(['GPU', 'App'])
        
        for (gpu, app), group in configurations:
            # Find baseline (highest frequency) for performance reference
            baseline = group.loc[group['Frequency'].idxmax()]
            baseline_runtime = baseline['Runtime_Factor']
            baseline_power = baseline['Power']
            baseline_energy = baseline_power * baseline_runtime
            
            # Find optimal frequency (minimum EDP)
            optimal_idx = group['EDP'].idxmin()
            optimal = group.loc[optimal_idx]
            
            # Calculate performance penalty and energy savings
            perf_penalty = (optimal['Runtime_Factor'] / baseline_runtime - 1) * 100
            energy_savings = (1 - optimal['Energy'] / baseline_energy) * 100
            
            # Categorize the result
            if perf_penalty <= 15:
                category = "🟢 Production ready"
            elif perf_penalty <= 50:
                category = "🟡 Needs A/B testing"
            else:
                category = "🔵 Batch processing only"
            
            config_key = f"{gpu}+{app}"
            results[config_key] = {
                'gpu': gpu,
                'app': app,
                'optimal_frequency': optimal['Frequency'],
                'performance_penalty': perf_penalty,
                'energy_savings': energy_savings,
                'category': category,
                'baseline_frequency': baseline['Frequency'],
                'optimal_power': optimal['Power'],
                'baseline_power': baseline_power
            }
        
        print(f"✅ Optimized {len(results)} configurations")
        
        # Step 3: Results analysis and reporting
        print("\n📋 Step 3: Analyzing results...")
        
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
                f.write(f"  Category: {result['category']}\n\n")
        
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
            print(f"Command: sudo nvidia-smi -ac 1215,{result['optimal_frequency']}")
            print(f"Expected: {result['performance_penalty']:.1f}% slower, {result['energy_savings']:.1f}% energy savings")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
