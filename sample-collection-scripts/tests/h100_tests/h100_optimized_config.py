#!/usr/bin/env python3
"""
Optimized H100 Profiling Configuration

This script provides H100-optimized DCGM field configurations that exclude
problematic fields causing N/A values while maintaining comprehensive monitoring.
"""

# H100-optimized DCGM field configurations
H100_OPTIMIZED_FIELDS = {
    "core_stable": [
        52,   # GPU Index (DCGM_FI_DEV_NVML_INDEX)
        50,   # GPU Name (DCGM_FI_DEV_NAME)
        155,  # Power Usage (DCGM_FI_DEV_POWER_USAGE) 
        160,  # Power Limit (DCGM_FI_DEV_POWER_MGMT_LIMIT)
        150,  # GPU Temperature (DCGM_FI_DEV_GPU_TEMP)
        203,  # GPU Utilization (DCGM_FI_DEV_GPU_UTIL)
        204,  # Memory Utilization (DCGM_FI_DEV_MEM_COPY_UTIL)
        250,  # Frame Buffer Total (DCGM_FI_DEV_FB_TOTAL)
        251,  # Frame Buffer Free (DCGM_FI_DEV_FB_FREE)
        252,  # Frame Buffer Used (DCGM_FI_DEV_FB_USED)
        100,  # SM Clock (DCGM_FI_DEV_SM_CLOCK)
        101,  # Memory Clock (DCGM_FI_DEV_MEM_CLOCK)
    ],
    
    "extended_safe": [
        52, 50, 155, 160, 150, 203, 204, 250, 251, 252, 100, 101,  # Core fields
        156,  # Total Energy Consumption (DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION)
        140,  # Memory Temperature (DCGM_FI_DEV_MEMORY_TEMP)
        110,  # Graphics Clock (DCGM_FI_DEV_APP_SM_CLOCK)
        111,  # Memory Clock (DCGM_FI_DEV_APP_MEM_CLOCK)
        190,  # Power State (DCGM_FI_DEV_PSTATE)
    ],
    
    # Fields that cause N/A on H100 - avoid these
    "problematic_h100": [
        1001, # GRACT (Graphics Activity)
        1002, # SMACT (SM Activity)
        1003, # SMOCC (SM Occupancy) 
        1004, # TENSO (Tensor Activity)
        1005, # DRAMA (DRAM Activity)
        1006, # FP64A (FP64 Activity)
        1007, # FP32A (FP32 Activity)
        1008, # FP16A (FP16 Activity)
    ]
}

def get_h100_optimized_fields(level="extended_safe"):
    """
    Get optimized DCGM field list for H100 monitoring.
    
    Args:
        level: "core_stable" for minimal reliable fields,
               "extended_safe" for comprehensive monitoring without N/A issues
    
    Returns:
        List of DCGM field IDs
    """
    return H100_OPTIMIZED_FIELDS.get(level, H100_OPTIMIZED_FIELDS["core_stable"])

def test_optimized_configuration():
    """Test the optimized H100 configuration."""
    import subprocess
    import time
    
    print("🧪 Testing H100 Optimized Configuration")
    print("=" * 50)
    
    # Test core stable fields
    core_fields = ",".join(str(f) for f in H100_OPTIMIZED_FIELDS["core_stable"])
    print(f"Testing core stable fields: {core_fields}")
    
    result = subprocess.run(
        f"timeout 5s dcgmi dmon -i -1 -e {core_fields} -d 1000 -c 3",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        na_count = result.stdout.count("N/A")
        total_lines = result.stdout.count("GPU")
        print(f"  ✅ Core fields: {na_count} N/A values out of {total_lines} GPU readings")
        if na_count == 0:
            print("  🎉 Perfect! No N/A values in core fields")
    else:
        print(f"  ❌ Test failed: {result.stderr}")
    
    # Test extended safe fields
    extended_fields = ",".join(str(f) for f in H100_OPTIMIZED_FIELDS["extended_safe"])
    print(f"\nTesting extended safe fields: {extended_fields}")
    
    result = subprocess.run(
        f"timeout 5s dcgmi dmon -i -1 -e {extended_fields} -d 1000 -c 3",
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        na_count = result.stdout.count("N/A")
        total_lines = result.stdout.count("GPU")
        print(f"  ✅ Extended fields: {na_count} N/A values out of {total_lines} GPU readings")
        if na_count == 0:
            print("  🎉 Perfect! No N/A values in extended fields")
    else:
        print(f"  ❌ Test failed: {result.stderr}")

if __name__ == "__main__":
    test_optimized_configuration()
    
    print("\n" + "=" * 50)
    print("📋 H100 Profiling Recommendations")
    print("=" * 50)
    print("1. Use 'core_stable' fields for guaranteed reliability")
    print("2. Use 'extended_safe' fields for comprehensive monitoring")
    print("3. Avoid fields 1001-1008 on H100 systems")
    print("4. Consider updating to newer NVIDIA drivers/DCGM if available")
    print("5. For A100/V100 systems, all fields including 1001-1008 work fine")
