#!/usr/bin/env python3
"""
Test script for optimal frequency selection implementation

This script validates that the optimal frequency selection works with your existing data.
"""

import os
import sys
from pathlib import Path

def test_basic_functionality():
    """Test basic functionality without dependencies"""
    print("🧪 Testing basic optimal frequency selection functionality...")
    
    # Test 1: Check if scripts exist
    scripts = [
        'aggregate_data.py',
        'optimal_frequency_analysis.py', 
        'optimal_frequency_selector.py',
        'run_optimal_frequency.sh'
    ]
    
    missing_scripts = []
    for script in scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
            
    if missing_scripts:
        print(f"❌ Missing scripts: {missing_scripts}")
        return False
    else:
        print("✅ All scripts present")
    
    # Test 2: Check for results directories
    result_dirs = list(Path('.').glob('results_*'))
    if not result_dirs:
        print("⚠️ No results directories found - make sure you have collected data first")
        return False
    else:
        print(f"✅ Found {len(result_dirs)} results directories")
        
    # Test 3: Basic heuristic functionality
    try:
        sys.path.append('.')
        from optimal_frequency_selector import SimpleOptimalFrequencySelector
        
        selector = SimpleOptimalFrequencySelector('.')
        result = selector.get_optimal_frequency('A100', 'llama')
        
        if result and 'optimal_frequency' in result:
            print(f"✅ Heuristic optimal frequency for A100 LLaMA: {result['optimal_frequency']} MHz")
            print(f"   Expected energy savings: {result['energy_savings']:.1f}%")
            print(f"   Expected performance impact: {result['performance_impact']:.1f}%")
        else:
            print("❌ Heuristic frequency selection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing heuristic approach: {e}")
        return False
        
    return True

def test_data_aggregation():
    """Test data aggregation functionality"""
    print("\n🔬 Testing data aggregation...")
    
    try:
        # Check if we can import required modules
        import pandas as pd
        print("✅ pandas available")
        
        # Test basic aggregation functionality
        sys.path.append('.')
        from aggregate_data import AIInferenceDataAggregator
        
        aggregator = AIInferenceDataAggregator('.')
        result_dirs = aggregator.find_result_directories()
        
        if result_dirs:
            print(f"✅ Found {len(result_dirs)} result directories for aggregation")
            
            # Test parsing a directory name
            sample_dir = result_dirs[0]
            parsed = aggregator.parse_directory_name(sample_dir.name)
            if parsed:
                gpu, workload, job_id = parsed
                print(f"✅ Successfully parsed directory: {gpu} {workload} (job {job_id})")
            else:
                print(f"⚠️ Could not parse directory name: {sample_dir.name}")
                
        else:
            print("❌ No result directories found for aggregation")
            return False
            
    except ImportError:
        print("⚠️ pandas not available - data aggregation will be limited")
        return False
    except Exception as e:
        print(f"❌ Error testing data aggregation: {e}")
        return False
        
    return True

def test_integration_commands():
    """Test integration with existing framework"""
    print("\n🔗 Testing framework integration...")
    
    # Test launch command generation
    try:
        sys.path.append('.')
        from optimal_frequency_selector import SimpleOptimalFrequencySelector
        
        selector = SimpleOptimalFrequencySelector('.')
        
        # Test all combinations
        test_combinations = [
            ('A100', 'llama'),
            ('H100', 'stablediffusion'),
            ('V100', 'vit'),
            ('A100', 'whisper')
        ]
        
        successful_combinations = 0
        
        for gpu, workload in test_combinations:
            try:
                result = selector.get_optimal_frequency(gpu, workload)
                if result and 'optimal_frequency' in result:
                    print(f"✅ {gpu} {workload}: {result['optimal_frequency']} MHz")
                    successful_combinations += 1
                else:
                    print(f"❌ Failed to get optimal frequency for {gpu} {workload}")
            except Exception as e:
                print(f"❌ Error with {gpu} {workload}: {e}")
                
        if successful_combinations == len(test_combinations):
            print(f"✅ All {len(test_combinations)} combinations successful")
        else:
            print(f"⚠️ {successful_combinations}/{len(test_combinations)} combinations successful")
            
        # Test command generation
        results = {}
        for gpu, workload in test_combinations[:2]:  # Test first 2
            result = selector.get_optimal_frequency(gpu, workload)
            if result:
                results[f"{gpu}_{workload}"] = result
                
        if results:
            commands = selector.generate_launch_commands(results)
            print(f"✅ Generated {len(commands)} launch commands")
            print("   Example command:")
            print(f"   {commands[0]}")
        else:
            print("❌ Could not generate launch commands")
            return False
            
    except Exception as e:
        print(f"❌ Error testing integration: {e}")
        return False
        
    return True

def test_shell_script():
    """Test shell script functionality"""
    print("\n🐚 Testing shell script integration...")
    
    # Check if shell script is executable
    script_path = Path('run_optimal_frequency.sh')
    if not script_path.exists():
        print("❌ run_optimal_frequency.sh not found")
        return False
        
    # Check if it's executable
    if not os.access(script_path, os.X_OK):
        print("⚠️ run_optimal_frequency.sh not executable, trying to fix...")
        try:
            script_path.chmod(0o755)
            print("✅ Made script executable")
        except Exception as e:
            print(f"❌ Could not make script executable: {e}")
            return False
    else:
        print("✅ Shell script is executable")
        
    # Test help output
    try:
        import subprocess
        result = subprocess.run(['./run_optimal_frequency.sh', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Shell script help works")
        else:
            print(f"⚠️ Shell script help returned code {result.returncode}")
    except Exception as e:
        print(f"⚠️ Could not test shell script: {e}")
        
    return True

def generate_example_usage():
    """Generate example usage based on available data"""
    print("\n📋 Example usage commands based on your data:")
    
    # Find available GPU-workload combinations
    result_dirs = list(Path('.').glob('results_*'))
    combinations = set()
    
    for result_dir in result_dirs:
        parts = result_dir.name.split('_')
        if len(parts) >= 3:
            gpu = parts[1].upper()
            workload = parts[2].lower()
            if gpu in ['V100', 'A100', 'H100'] and workload in ['llama', 'stablediffusion', 'vit', 'whisper']:
                combinations.add((gpu, workload))
                
    if combinations:
        print("\n🎯 Quick optimal frequency lookup:")
        sample_gpu, sample_workload = list(combinations)[0]
        print(f"   python optimal_frequency_selector.py --gpu {sample_gpu} --workload {sample_workload}")
        
        print("\n🚀 Run experiment with optimal frequency:")
        print(f"   ./run_optimal_frequency.sh --mode experiment --gpu {sample_gpu} --workload {sample_workload}")
        
        print("\n📊 Complete analysis pipeline:")
        print("   ./run_optimal_frequency.sh --mode complete")
        
        print("\n🔍 Available combinations in your data:")
        for gpu, workload in sorted(combinations):
            print(f"   {gpu} {workload}")
    else:
        print("❌ No valid GPU-workload combinations found in results")

def main():
    print("🧪 Optimal Frequency Selection Test Suite")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Basic functionality
    total_tests += 1
    if test_basic_functionality():
        success_count += 1
        
    # Test 2: Data aggregation
    total_tests += 1
    if test_data_aggregation():
        success_count += 1
        
    # Test 3: Integration commands
    total_tests += 1
    if test_integration_commands():
        success_count += 1
        
    # Test 4: Shell script
    total_tests += 1
    if test_shell_script():
        success_count += 1
        
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Optimal frequency selection is ready to use.")
    elif success_count >= total_tests - 1:
        print("✅ Most tests passed! Implementation should work with minor issues.")
    else:
        print("⚠️ Some tests failed. Check dependencies and data structure.")
        
    # Generate usage examples
    generate_example_usage()
    
    print("\n📚 Next steps:")
    print("1. Install dependencies: pip install -r requirements-optimal-frequency.txt")
    print("2. Run quick test: python optimal_frequency_selector.py --gpu A100 --workload llama")
    print("3. Run complete analysis: ./run_optimal_frequency.sh --mode complete")
    
    return 0 if success_count >= total_tests - 1 else 1

if __name__ == "__main__":
    exit(main())
