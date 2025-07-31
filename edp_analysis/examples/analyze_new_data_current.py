#!/usr/bin/env python3
"""
Example: Analyzing New Profiling Data for GPU Frequency Optimization

This example demonstrates how to apply the current EDP analysis framework to new
profiling datasets using the existing optimization pipeline.

Usage:
    python analyze_new_data_current.py --data-dir ./new_profiling_data --output ./results
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add the edp_analysis module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description='Analyze new profiling data for GPU frequency optimization')
    parser.add_argument('--data-dir', required=True, help='Directory containing profiling data')
    parser.add_argument('--output', default='./analysis_results', help='Output directory for results')
    parser.add_argument('--gpu', choices=['V100', 'A100', 'H100'], help='Filter by specific GPU')
    parser.add_argument('--app', choices=['LLAMA', 'STABLEDIFFUSION', 'VIT', 'WHISPER'], help='Filter by specific application')
    parser.add_argument('--run', type=int, default=2, help='Run number to use (default: 2, first warm run)')
    parser.add_argument('--deploy', action='store_true', help='Generate deployment scripts')
    
    args = parser.parse_args()
    
    print("🚀 Starting GPU Frequency Optimization Analysis")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Using run {args.run} data (excludes cold start if run > 1)")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Step 1: Data Aggregation using existing module
        print("\n📊 Step 1: Aggregating profiling data...")
        
        # Use the existing data aggregation module
        from edp_analysis.data_aggregation.aggregate_profiling_data import (
            discover_result_directories, process_result_directory, save_aggregated_data
        )
        
        # Discover available data
        result_dirs = discover_result_directories(args.data_dir)
        print(f"✅ Found {len(result_dirs)} result directories")
        
        # Filter by GPU/app if specified
        if args.gpu or args.app:
            filtered_dirs = []
            for dir_path in result_dirs:
                dir_name = os.path.basename(dir_path)
                if args.gpu and args.gpu not in dir_name:
                    continue
                if args.app and args.app not in dir_name:
                    continue
                filtered_dirs.append(dir_path)
            result_dirs = filtered_dirs
            print(f"✅ Filtered to {len(result_dirs)} directories matching criteria")
        
        # Process each directory
        all_data = []
        for result_dir in result_dirs:
            try:
                print(f"Processing {os.path.basename(result_dir)}...")
                data = process_result_directory(result_dir, run_number=args.run)
                if not data.empty:
                    all_data.append(data)
                    print(f"  ✅ Processed {len(data)} frequency points")
                else:
                    print(f"  ⚠️ No data found")
            except Exception as e:
                print(f"  ❌ Error processing {result_dir}: {e}")
        
        if not all_data:
            print("❌ No data found to process")
            return 1
            
        # Combine all data
        aggregated_df = pd.concat(all_data, ignore_index=True)
        aggregated_file = os.path.join(args.output, "aggregated_data.csv")
        aggregated_df.to_csv(aggregated_file, index=False)
        
        print(f"✅ Aggregated {len(aggregated_df)} total configurations")
        print(f"✅ Saved aggregated data to: {aggregated_file}")
        
        # Step 2: Optimization using existing production optimizer
        print("\n🎯 Step 2: Finding optimal frequencies...")
        
        from edp_analysis.optimization.production_optimizer import ProductionOptimizer
        
        optimizer = ProductionOptimizer()
        optimization_results = optimizer.optimize_with_constraints(aggregated_df)
        
        print(f"✅ Optimized {len(optimization_results)} configurations")
        
        # Step 3: Categorize results and generate summary
        print("\n📋 Step 3: Analyzing results...")
        
        production_ready = []
        testing_needed = []
        batch_only = []
        
        for config_name, result in optimization_results.items():
            performance_penalty = result.get('performance_penalty_percent', 0)
            energy_savings = result.get('energy_savings_percent', 0)
            optimal_freq = result.get('optimal_frequency', 0)
            
            config_info = {
                'configuration': config_name,
                'optimal_frequency': optimal_freq,
                'performance_penalty': performance_penalty,
                'energy_savings': energy_savings
            }
            
            # Categorize based on performance penalty
            if performance_penalty <= 20:
                config_info['category'] = 'production'
                production_ready.append(config_info)
            elif performance_penalty <= 50:
                config_info['category'] = 'testing'
                testing_needed.append(config_info)
            else:
                config_info['category'] = 'batch_only'
                batch_only.append(config_info)
        
        # Display results summary
        print(f"  🟢 Production ready: {len(production_ready)} configurations")
        for config in production_ready:
            print(f"    - {config['configuration']}: {config['optimal_frequency']:.0f}MHz "
                  f"({config['performance_penalty']:.1f}% slower, {config['energy_savings']:.1f}% energy savings)")
                  
        print(f"  🟡 Needs A/B testing: {len(testing_needed)} configurations") 
        for config in testing_needed:
            print(f"    - {config['configuration']}: {config['optimal_frequency']:.0f}MHz "
                  f"({config['performance_penalty']:.1f}% slower, {config['energy_savings']:.1f}% energy savings)")
                  
        print(f"  🔵 Batch processing only: {len(batch_only)} configurations")
        for config in batch_only:
            print(f"    - {config['configuration']}: {config['optimal_frequency']:.0f}MHz "
                  f"({config['performance_penalty']:.1f}% slower, {config['energy_savings']:.1f}% energy savings)")
        
        # Step 4: Save detailed results
        print("\n📝 Step 4: Saving results...")
        
        # Save optimization results
        import json
        results_file = os.path.join(args.output, "optimization_results.json")
        with open(results_file, "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for k, v in optimization_results.items():
                json_results[k] = {key: float(val) if hasattr(val, 'item') else val 
                                 for key, val in v.items()}
            json.dump(json_results, f, indent=2)
        
        # Generate summary table
        summary_lines = [
            "GPU Frequency Optimization Results Summary",
            "=" * 50,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Source: {args.data_dir}",
            f"Run Number: {args.run}",
            f"Total Configurations: {len(optimization_results)}",
            "",
            "PRODUCTION READY (≤20% performance penalty):",
        ]
        
        for config in production_ready:
            summary_lines.append(
                f"  {config['configuration']}: {config['optimal_frequency']:.0f}MHz "
                f"({config['performance_penalty']:.1f}% slower, {config['energy_savings']:.1f}% energy savings)"
            )
            
        summary_lines.extend([
            "",
            "NEEDS A/B TESTING (20-50% performance penalty):",
        ])
        
        for config in testing_needed:
            summary_lines.append(
                f"  {config['configuration']}: {config['optimal_frequency']:.0f}MHz "
                f"({config['performance_penalty']:.1f}% slower, {config['energy_savings']:.1f}% energy savings)"
            )
            
        summary_lines.extend([
            "",
            "BATCH PROCESSING ONLY (>50% performance penalty):",
        ])
        
        for config in batch_only:
            summary_lines.append(
                f"  {config['configuration']}: {config['optimal_frequency']:.0f}MHz "
                f"({config['performance_penalty']:.1f}% slower, {config['energy_savings']:.1f}% energy savings)"
            )
        
        summary_file = os.path.join(args.output, "optimization_summary.txt")
        with open(summary_file, "w") as f:
            f.write("\n".join(summary_lines))
            
        print(f"✅ Optimization results: {results_file}")
        print(f"✅ Summary report: {summary_file}")
        
        # Step 5: Generate deployment script (Optional)
        if args.deploy:
            print("\n🚀 Step 5: Generating deployment script...")
            
            deployment_lines = [
                "#!/bin/bash",
                "# GPU Frequency Deployment Script",
                "# Generated automatically from optimization analysis",
                "",
                "set -e",
                "",
                "show_usage() {",
                "    echo \"Usage: $0 <configuration> <action>\"",
                "    echo \"Configurations: " + " ".join([c['configuration'] for c in production_ready + testing_needed + batch_only]) + "\"",
                "    echo \"Actions: deploy, status, reset\"",
                "}",
                "",
                "if [ $# -lt 2 ]; then",
                "    show_usage",
                "    exit 1",
                "fi",
                "",
                "CONFIG=$1",
                "ACTION=$2",
                "",
                "case $CONFIG in"
            ]
            
            # Add deployment commands for each configuration
            for config in production_ready + testing_needed + batch_only:
                gpu_type = config['configuration'].split('+')[0]
                freq = int(config['optimal_frequency'])
                
                # Determine memory frequency based on GPU type
                if gpu_type == "A100":
                    mem_freq = 1215
                elif gpu_type == "V100":
                    mem_freq = 877
                elif gpu_type == "H100":
                    mem_freq = 1593
                else:
                    mem_freq = 1215  # Default
                    
                deployment_lines.extend([
                    f"    {config['configuration']})",
                    f"        GPU_FREQ={freq}",
                    f"        MEM_FREQ={mem_freq}",
                    f"        ;;",
                ])
                
            deployment_lines.extend([
                "    *)",
                "        echo \"Unknown configuration: $CONFIG\"",
                "        show_usage",
                "        exit 1",
                "        ;;",
                "esac",
                "",
                "case $ACTION in",
                "    deploy)",
                "        echo \"Deploying $CONFIG with frequency ${GPU_FREQ}MHz...\"",
                "        sudo nvidia-smi -ac $MEM_FREQ,$GPU_FREQ",
                "        echo \"✅ Deployed successfully\"",
                "        ;;",
                "    status)",
                "        echo \"Current GPU status:\"",
                "        nvidia-smi --query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw --format=csv,noheader,nounits",
                "        ;;",
                "    reset)",
                "        echo \"Resetting to baseline frequency...\"",
                "        if [[ $CONFIG == A100* ]]; then",
                "            sudo nvidia-smi -ac 1215,1410",
                "        elif [[ $CONFIG == V100* ]]; then",
                "            sudo nvidia-smi -ac 877,1380",
                "        elif [[ $CONFIG == H100* ]]; then",
                "            sudo nvidia-smi -ac 1593,1785",
                "        fi",
                "        echo \"✅ Reset to baseline\"",
                "        ;;",
                "    *)",
                "        echo \"Unknown action: $ACTION\"",
                "        show_usage",
                "        exit 1",
                "        ;;",
                "esac"
            ])
            
            deployment_file = os.path.join(args.output, "deploy_optimal_frequencies.sh")
            with open(deployment_file, "w") as f:
                f.write("\n".join(deployment_lines))
            os.chmod(deployment_file, 0o755)  # Make executable
            
            print(f"✅ Deployment script: {deployment_file}")
            print("Usage examples:")
            if production_ready:
                example_config = production_ready[0]['configuration']
                print(f"  {deployment_file} {example_config} deploy")
                print(f"  {deployment_file} {example_config} status")
                print(f"  {deployment_file} {example_config} reset")
        
        # Final Summary
        print("\n🎉 Analysis Complete!")
        print(f"📁 All results saved to: {args.output}")
        
        if production_ready:
            print("\n🚀 Ready for immediate deployment:")
            best_config = min(production_ready, key=lambda x: x['performance_penalty'])
            gpu_type = best_config['configuration'].split('+')[0] 
            mem_freq = 1215 if gpu_type == "A100" else 877 if gpu_type == "V100" else 1593
            
            print(f"Recommended: {best_config['configuration']}")
            print(f"Command: sudo nvidia-smi -ac {mem_freq},{best_config['optimal_frequency']:.0f}")
            print(f"Expected: {best_config['performance_penalty']:.1f}% slower, {best_config['energy_savings']:.1f}% energy savings")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
