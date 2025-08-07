#!/usr/bin/env python
"""
Test script for real data-driven optimal frequency selection
"""

import os
import sys

def test_data_loading():
    """Test loading and processing the aggregated data"""
    print("Testing data loading and optimal frequency selection...")
    
    # Check if aggregated data exists
    data_file = "aggregated_results/ai_inference_aggregated_data_20250807_134913.csv"
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        return False
        
    print(f"Found data file: {data_file}")
    
    # Try to import required modules
    try:
        import pandas as pd
        import numpy as np
        print("✓ Required modules imported successfully")
    except ImportError as e:
        print(f"ERROR: Missing required module: {e}")
        print("Please install required packages:")
        print("pip install pandas numpy scikit-learn matplotlib seaborn")
        return False
    
    # Load the data
    try:
        df = pd.read_csv(data_file)
        print(f"✓ Loaded {len(df)} data points")
        
        # Check required columns
        required_cols = ['gpu', 'workload', 'frequency_mhz', 'duration_seconds', 
                        'avg_power_watts', 'total_energy_joules']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            return False
            
        print("✓ All required columns present")
        
        # Show data overview
        print(f"\nData Overview:")
        print(f"  GPUs: {list(df['gpu'].unique())}")
        print(f"  Workloads: {list(df['workload'].unique())}")
        print(f"  Frequency range: {df['frequency_mhz'].min():.0f} - {df['frequency_mhz'].max():.0f} MHz")
        print(f"  Power range: {df['avg_power_watts'].min():.1f} - {df['avg_power_watts'].max():.1f} W")
        print(f"  Duration range: {df['duration_seconds'].min():.1f} - {df['duration_seconds'].max():.1f} s")
        
        # Calculate EDP if not present
        if 'edp' not in df.columns:
            df['edp'] = df['total_energy_joules'] * df['duration_seconds']
            print("✓ Calculated EDP values")
        
        # Find optimal frequencies using direct approach
        print(f"\nFinding optimal frequencies...")
        
        results = []
        for gpu in df['gpu'].unique():
            for workload in df['workload'].unique():
                subset = df[(df['gpu'] == gpu) & (df['workload'] == workload)]
                
                if subset.empty:
                    continue
                    
                # Find baseline (max frequency)
                max_freq = subset['frequency_mhz'].max()
                baseline = subset[subset['frequency_mhz'] == max_freq]
                
                if baseline.empty:
                    continue
                    
                baseline_time = baseline['duration_seconds'].mean()
                baseline_energy = baseline['total_energy_joules'].mean()
                
                # Performance constraint (5% degradation)
                max_acceptable_time = baseline_time * 1.05
                
                # Apply constraint
                feasible = subset[subset['duration_seconds'] <= max_acceptable_time]
                
                if feasible.empty:
                    print(f"  {gpu} {workload}: No feasible solutions within 5% constraint")
                    continue
                
                # Find optimal EDP
                optimal_idx = feasible['edp'].idxmin()
                optimal = feasible.loc[optimal_idx]
                
                energy_savings = (1 - optimal['total_energy_joules'] / baseline_energy) * 100
                perf_impact = (optimal['duration_seconds'] / baseline_time - 1) * 100
                
                result = {
                    'gpu': gpu,
                    'workload': workload,
                    'optimal_freq_mhz': int(optimal['frequency_mhz']),
                    'energy_savings_pct': energy_savings,
                    'perf_impact_pct': perf_impact
                }
                results.append(result)
                
                print(f"  {gpu} {workload:>15}: {int(optimal['frequency_mhz']):>4}MHz "
                      f"({energy_savings:+5.1f}% energy, {perf_impact:+4.1f}% perf)")
        
        print(f"\n✓ Found optimal frequencies for {len(results)} GPU-workload combinations")
        
        # Summary statistics
        if results:
            avg_energy_savings = sum(r['energy_savings_pct'] for r in results) / len(results)
            avg_perf_impact = sum(r['perf_impact_pct'] for r in results) / len(results)
            
            print(f"\nSummary:")
            print(f"  Average energy savings: {avg_energy_savings:.1f}%")
            print(f"  Average performance impact: {avg_perf_impact:.1f}%")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to process data: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 Real data-driven optimal frequency selection test PASSED!")
    else:
        print("\n❌ Test FAILED!")
        sys.exit(1)
