#!/usr/bin/env python3
"""
Compare performance metrics across different runs to analyze cold start effects.
"""

import pandas as pd
import numpy as np

def compare_run_performance():
    print("🔍 Analyzing Cold Start Effects Across Runs")
    print("=" * 60)
    
    # Load data from different runs
    try:
        run1_df = pd.read_csv('complete_aggregation.csv')  # Original (run 1)
        run2_df = pd.read_csv('complete_aggregation_run2.csv')  # First warm run
        run3_df = pd.read_csv('complete_aggregation_run3.csv')  # Second warm run
        
        print(f"Run 1 (Cold): {len(run1_df)} configurations")
        print(f"Run 2 (Warm): {len(run2_df)} configurations") 
        print(f"Run 3 (Warm): {len(run3_df)} configurations")
        print()
        
        # Compare specific configurations
        for gpu in ['A100', 'V100']:
            for app in ['LLAMA', 'STABLEDIFFUSION']:
                print(f"📊 {gpu} + {app} Comparison:")
                print("-" * 40)
                
                # Get maximum frequency data for comparison
                run1_subset = run1_df[(run1_df['gpu'] == gpu) & (run1_df['application'] == app)]
                run2_subset = run2_df[(run2_df['gpu'] == gpu) & (run2_df['application'] == app)]
                run3_subset = run3_df[(run3_df['gpu'] == gpu) & (run3_df['application'] == app)]
                
                if len(run1_subset) > 0 and len(run2_subset) > 0:
                    # Get max frequency performance
                    max_freq = run1_subset['frequency'].max()
                    
                    run1_max = run1_subset[run1_subset['frequency'] == max_freq]['execution_time'].iloc[0]
                    run2_max = run2_subset[run2_subset['frequency'] == max_freq]['execution_time'].iloc[0]
                    
                    cold_start_penalty = ((run1_max - run2_max) / run2_max) * 100
                    
                    print(f"   Max Frequency ({max_freq} MHz):")
                    print(f"   Run 1 (Cold): {run1_max:.2f}s")
                    print(f"   Run 2 (Warm): {run2_max:.2f}s")
                    print(f"   Cold Start Penalty: {cold_start_penalty:.1f}%")
                    
                    if len(run3_subset) > 0:
                        run3_max = run3_subset[run3_subset['frequency'] == max_freq]['execution_time'].iloc[0]
                        run2_vs_run3 = ((abs(run3_max - run2_max)) / run2_max) * 100
                        print(f"   Run 3 (Warm): {run3_max:.2f}s")
                        print(f"   Run 2 vs Run 3 Variation: {run2_vs_run3:.1f}%")
                    
                    print()
                    
                    # Performance stability analysis
                    if cold_start_penalty > 50:
                        print("   🚨 SEVERE cold start effect detected!")
                    elif cold_start_penalty > 20:
                        print("   ⚠️  Significant cold start effect")
                    elif cold_start_penalty > 5:
                        print("   📝 Moderate cold start effect")
                    else:
                        print("   ✅ Minimal cold start effect")
                    print()
                else:
                    print(f"   No data available for comparison")
                    print()
        
        # Recommend strategy
        print("💡 Recommendations:")
        print("-" * 20)
        print("1. ✅ EXCLUDE Run 1 from optimization analysis")
        print("2. ✅ Use Run 2 or Run 3 for reliable performance metrics")
        print("3. ✅ Consider averaging Run 2 and Run 3 for robust statistics")
        print("4. 📊 Use median aggregation to handle outliers")
        print("5. 🔍 Monitor coefficient of variation (CV) for data quality")
        
    except FileNotFoundError as e:
        print(f"❌ Could not load data files: {e}")
        print("Make sure to run aggregation for all runs first")

if __name__ == "__main__":
    compare_run_performance()
