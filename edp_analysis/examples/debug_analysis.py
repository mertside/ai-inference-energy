#!/usr/bin/env python3
"""
Debug version of the GPU analysis script to identify issues
"""

import os
import sys
import pandas as pd
from pathlib import Path

def debug_analysis():
    data_dir = Path("../../sample-collection-scripts")
    run_number = 1
    
    print("🔍 Debug Analysis")
    print(f"Data directory: {data_dir}")
    print(f"Run number: {run_number}")
    
    # Find result directories
    result_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('results_')]
    print(f"Found {len(result_dirs)} result directories:")
    
    for result_dir in result_dirs:
        print(f"\n📁 Processing: {result_dir.name}")
        
        # Parse directory name
        dir_name = result_dir.name
        if 'results_' in dir_name:
            parts = dir_name.replace('results_', '').split('_')
            if len(parts) >= 2:
                gpu_type = parts[0].upper()
                app_name = parts[1].upper()
                print(f"  GPU: {gpu_type}, App: {app_name}")
                
                # Find CSV files
                pattern = f'run_*_{run_number:02d}_freq_*_profile.csv'
                csv_files = list(result_dir.glob(pattern))
                print(f"  Pattern: {pattern}")
                print(f"  Found {len(csv_files)} CSV files")
                
                if len(csv_files) > 0:
                    # Test first file
                    test_file = csv_files[0]
                    print(f"  Testing file: {test_file.name}")
                    
                    try:
                        # Extract frequency
                        filename = test_file.stem
                        parts = filename.split('_')
                        frequency = None
                        for i, part in enumerate(parts):
                            if part == 'freq' and i + 1 < len(parts):
                                frequency = int(parts[i + 1])
                                break
                        print(f"  Extracted frequency: {frequency}")
                        
                        # Test CSV reading - these are space-separated files, not CSV
                        print(f"  Reading space-separated file...")
                        
                        # Read the header line (skip comment, use first line as header)
                        with open(test_file, 'r') as f:
                            lines = f.readlines()
                            header_line = lines[0].strip().lstrip('#')  # Remove # from first line
                            print(f"  Header line: {repr(header_line[:100])}")
                        
                        # Read with proper header
                        df = pd.read_csv(test_file, comment='#', sep=r'\s+', skiprows=1)
                        
                        # Set proper column names from header
                        header_cols = header_line.split()
                        if len(header_cols) == len(df.columns):
                            df.columns = header_cols
                        df.columns = df.columns.str.strip()
                        print(f"  Columns: {list(df.columns)}")
                        print(f"  Shape: {df.shape}")
                        
                        if 'POWER' in df.columns and 'GPUTL' in df.columns:
                            print(f"  ✅ Required columns found")
                            print(f"  POWER range: {df['POWER'].min():.2f} - {df['POWER'].max():.2f}")
                            print(f"  GPUTL range: {df['GPUTL'].min():.2f} - {df['GPUTL'].max():.2f}")
                        else:
                            print(f"  ❌ Missing required columns")
                            
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        
                else:
                    # Check what files do exist
                    all_files = list(result_dir.glob('*.csv'))
                    print(f"  All CSV files ({len(all_files)}):")
                    for f in all_files[:3]:  # Show first 3
                        print(f"    {f.name}")

if __name__ == "__main__":
    debug_analysis()
