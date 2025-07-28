#!/usr/bin/env python3
"""
Test script to verify H100 all-GPU monitoring functionality.
This directly tests the enhanced profiler with H100-specific all-GPU monitoring.
"""

import sys
import os
import time

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile import profile_application

def test_h100_all_gpus():
    """Test all-GPU monitoring for H100 systems."""
    print("Testing H100 all-GPU monitoring...")
    
    # Test with all-GPU monitoring enabled (H100 style)
    result = profile_application(
        command="sleep 3",  # Simple test command
        output_file="test_h100_all_gpus_output.csv",
        interval_ms=1000,   # 1 second sampling
        gpu_id=0,           # Ignored when monitor_all_gpus=True
        monitor_all_gpus=True  # Enable all-GPU monitoring
    )
    
    print(f"Profiling completed: {result}")
    print(f"Results saved to: test_h100_all_gpus_output.csv")
    
    # Check if the output file was created and contains data
    if os.path.exists("test_h100_all_gpus_output.csv"):
        with open("test_h100_all_gpus_output.csv", "r") as f:
            lines = f.readlines()
            print(f"Output file contains {len(lines)} lines")
            
            # Show first few lines to verify all GPUs are captured
            print("First 10 lines of output:")
            for i, line in enumerate(lines[:10]):
                print(f"  {i+1}: {line.strip()}")
                
            # Count unique GPU entries
            gpu_ids = set()
            for line in lines[1:]:  # Skip header
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) > 1 and parts[0].startswith('GPU'):
                        gpu_ids.add(parts[1])  # GPU ID is second column
            
            print(f"Detected GPUs in output: {sorted(gpu_ids)}")
            print(f"Total unique GPUs: {len(gpu_ids)}")
            
            if len(gpu_ids) > 1:
                print("✅ SUCCESS: Multiple GPUs detected in monitoring output!")
            else:
                print("⚠️  WARNING: Only one GPU detected. Check DCGMI configuration.")
    else:
        print("❌ ERROR: Output file not created")

if __name__ == "__main__":
    test_h100_all_gpus()
