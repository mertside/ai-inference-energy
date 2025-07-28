#!/usr/bin/env python3
"""
H100 N/A Values Diagnostic Script

This script systematically investigates why certain DCGM fields report N/A values
on H100 systems but not on A100/V100 systems.
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_command(cmd, capture_output=True, timeout=30):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"⚠️  Command timeout: {cmd}")
        return None
    except Exception as e:
        print(f"❌ Command failed: {cmd} - {e}")
        return None

def check_dcgm_version():
    """Check DCGM version and build info."""
    print("🔍 Checking DCGM Version...")
    result = run_command("dcgmi --version")
    if result and result.returncode == 0:
        print(f"DCGM Version Info:\n{result.stdout}")
        return result.stdout
    else:
        print("❌ Failed to get DCGM version")
        return None

def check_nvidia_driver():
    """Check NVIDIA driver version."""
    print("\n🔍 Checking NVIDIA Driver Version...")
    result = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if result and result.returncode == 0:
        driver_version = result.stdout.strip()
        print(f"NVIDIA Driver Version: {driver_version}")
        return driver_version
    else:
        print("❌ Failed to get driver version")
        return None

def test_individual_fields():
    """Test each DCGM field individually to identify which ones cause N/A."""
    print("\n🔍 Testing Individual DCGM Fields...")
    
    # Core fields that should always work
    core_fields = [52, 50, 155, 160, 150, 203, 204, 250, 251, 252, 100, 101]
    
    # Extended fields that might be problematic
    extended_fields = [156, 140, 110, 111, 190, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
    
    print("Testing core fields...")
    for field in core_fields:
        print(f"  Testing field {field}...")
        result = run_command(f"timeout 3s dcgmi dmon -i 0 -e {field} -d 1000 -c 1", timeout=5)
        if result and result.returncode == 0:
            # Check if output contains N/A
            if "N/A" in result.stdout:
                print(f"    ⚠️  Field {field}: Contains N/A values")
            else:
                print(f"    ✅ Field {field}: Working properly")
        else:
            print(f"    ❌ Field {field}: Failed to query")
    
    print("\nTesting extended fields...")
    for field in extended_fields:
        print(f"  Testing field {field}...")
        result = run_command(f"timeout 3s dcgmi dmon -i 0 -e {field} -d 1000 -c 1", timeout=5)
        if result and result.returncode == 0:
            # Check if output contains N/A
            if "N/A" in result.stdout:
                print(f"    ⚠️  Field {field}: Contains N/A values")
            else:
                print(f"    ✅ Field {field}: Working properly")
        else:
            print(f"    ❌ Field {field}: Failed to query or not supported")

def test_field_combinations():
    """Test different combinations of fields to identify interaction issues."""
    print("\n🔍 Testing Field Combinations...")
    
    # Test core fields only
    core_fields = "52,50,155,160,150,203,204,250,251,252,100,101"
    print("Testing core fields combination...")
    result = run_command(f"timeout 5s dcgmi dmon -i 0 -e {core_fields} -d 1000 -c 2", timeout=10)
    if result and result.returncode == 0:
        na_count = result.stdout.count("N/A")
        print(f"  Core fields: {na_count} N/A values found")
        if na_count > 0:
            print("  ⚠️  Even core fields contain N/A values")
    
    # Test without extended fields that are known problematic
    reduced_fields = "52,50,155,160,150,203,204,250,251,252,100,101,156,140"
    print("Testing reduced field set...")
    result = run_command(f"timeout 5s dcgmi dmon -i 0 -e {reduced_fields} -d 1000 -c 2", timeout=10)
    if result and result.returncode == 0:
        na_count = result.stdout.count("N/A")
        print(f"  Reduced fields: {na_count} N/A values found")

def check_gpu_state():
    """Check GPU state and configuration."""
    print("\n🔍 Checking GPU State...")
    
    # Check GPU power state
    result = run_command("nvidia-smi --query-gpu=pstate --format=csv,noheader")
    if result and result.returncode == 0:
        print(f"GPU Power State: {result.stdout.strip()}")
    
    # Check GPU persistence mode
    result = run_command("nvidia-smi --query-gpu=persistence_mode --format=csv,noheader")
    if result and result.returncode == 0:
        print(f"Persistence Mode: {result.stdout.strip()}")
    
    # Check GPU compute mode
    result = run_command("nvidia-smi --query-gpu=compute_mode --format=csv,noheader")
    if result and result.returncode == 0:
        print(f"Compute Mode: {result.stdout.strip()}")
    
    # Check if any processes are running
    result = run_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv")
    if result and result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            print(f"Running processes: {len(lines)-1} processes detected")
            print(result.stdout)
        else:
            print("No GPU processes running")

def check_dcgm_service():
    """Check DCGM service status."""
    print("\n🔍 Checking DCGM Service...")
    
    # Check if nv-hostengine is running
    result = run_command("pgrep -f nv-hostengine")
    if result and result.returncode == 0:
        print("✅ DCGM host engine is running")
    else:
        print("⚠️  DCGM host engine may not be running")
    
    # Try to get DCGM service info
    result = run_command("dcgmi discovery --list")
    if result and result.returncode == 0:
        print("DCGM Discovery:")
        print(result.stdout)

def compare_with_working_gpu():
    """Compare field support between different GPU types if available."""
    print("\n🔍 Comparing GPU Field Support...")
    
    # Get list of all GPUs
    result = run_command("dcgmi discovery --list")
    if result and result.returncode == 0:
        print("Available GPUs:")
        print(result.stdout)
        
        # Test the same fields on each GPU individually
        for gpu_id in range(4):  # Test all 4 H100 GPUs
            print(f"\nTesting GPU {gpu_id}:")
            result = run_command(f"timeout 3s dcgmi dmon -i {gpu_id} -e 1001,1002,1003 -d 1000 -c 1", timeout=5)
            if result and result.returncode == 0:
                na_count = result.stdout.count("N/A")
                print(f"  GPU {gpu_id}: {na_count} N/A values in advanced fields")

def test_sampling_rates():
    """Test if sampling rate affects N/A occurrence."""
    print("\n🔍 Testing Sampling Rate Impact...")
    
    rates = [100, 500, 1000, 2000]  # Different sampling rates in ms
    
    for rate in rates:
        print(f"Testing {rate}ms sampling rate...")
        result = run_command(f"timeout 3s dcgmi dmon -i 0 -e 1001,1002,1003 -d {rate} -c 2", timeout=5)
        if result and result.returncode == 0:
            na_count = result.stdout.count("N/A")
            total_values = result.stdout.count("GPU")
            if total_values > 0:
                na_percentage = (na_count / (total_values * 3)) * 100  # 3 fields tested
                print(f"  {rate}ms: {na_count} N/A values ({na_percentage:.1f}%)")

def save_diagnostic_data():
    """Save comprehensive diagnostic data to file."""
    print("\n💾 Saving Diagnostic Data...")
    
    diagnostic_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {},
        "dcgm_info": {},
        "nvidia_info": {},
        "test_results": {}
    }
    
    # Collect system info
    result = run_command("uname -a")
    if result and result.returncode == 0:
        diagnostic_data["system_info"]["uname"] = result.stdout.strip()
    
    # Collect DCGM info
    result = run_command("dcgmi --version")
    if result and result.returncode == 0:
        diagnostic_data["dcgm_info"]["version"] = result.stdout.strip()
    
    # Collect NVIDIA info
    result = run_command("nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv")
    if result and result.returncode == 0:
        diagnostic_data["nvidia_info"]["gpu_info"] = result.stdout.strip()
    
    # Save to file
    with open("h100_na_diagnostic.json", "w") as f:
        json.dump(diagnostic_data, f, indent=2)
    
    print("Diagnostic data saved to: h100_na_diagnostic.json")

def main():
    """Main diagnostic function."""
    print("🔍 H100 N/A Values Diagnostic Tool")
    print("=" * 50)
    
    # Run all diagnostic tests
    check_dcgm_version()
    check_nvidia_driver()
    check_gpu_state()
    check_dcgm_service()
    test_individual_fields()
    test_field_combinations()
    compare_with_working_gpu()
    test_sampling_rates()
    save_diagnostic_data()
    
    print("\n" + "=" * 50)
    print("🎯 Diagnostic Summary")
    print("=" * 50)
    print("Check the output above for:")
    print("1. Fields that consistently return N/A")
    print("2. Driver/DCGM version compatibility issues")
    print("3. GPU state configuration problems")
    print("4. Sampling rate effects on N/A occurrence")
    print("\nNext steps:")
    print("- Compare with A100/V100 DCGM versions")
    print("- Check NVIDIA documentation for H100-specific field support")
    print("- Consider driver updates if version mismatch found")

if __name__ == "__main__":
    main()
