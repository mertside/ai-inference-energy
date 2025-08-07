#!/usr/bin/env python3
"""
Real-time Optimal Frequency Selector

This script provides a simple interface for optimal frequency selection
that can be integrated with your existing profiling framework.

Usage examples:
  # Single workload optimization
  python optimal_frequency_selector.py --gpu A100 --workload llama
  
  # Batch optimization for all combinations
  python optimal_frequency_selector.py --run-all
  
  # Integration with launch_v2.sh
  OPTIMAL_FREQ=$(python optimal_frequency_selector.py --gpu H100 --workload stablediffusion --quiet)
  ./launch_v2.sh --gpu-type H100 --profiling-mode custom --custom-frequencies "$OPTIMAL_FREQ"

Author: Mert Side  
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class SimpleOptimalFrequencySelector:
    """
    Simple optimal frequency selector that works with pre-computed results
    or uses heuristic approaches when models aren't available
    """
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.precomputed_results = {}
        self.gpu_frequencies = {
            'V100': list(range(510, 1381, 30)),  # 117 frequencies (your data)
            'A100': list(range(510, 1411, 15)),  # 61 frequencies (your data)  
            'H100': list(range(510, 1786, 15))   # 86 frequencies (your data)
        }
        
        # Load any existing optimal frequency results
        self.load_existing_results()
        
    def load_existing_results(self):
        """Load any existing optimal frequency analysis results"""
        # Look for optimal frequency results files
        result_files = list(self.results_dir.glob("optimal_frequencies_*.csv"))
        
        if result_files:
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Loading existing results from {latest_file.name}")
            
            try:
                import pandas as pd
                df = pd.read_csv(latest_file)
                
                for _, row in df.iterrows():
                    key = f"{row['gpu']}_{row['workload']}"
                    self.precomputed_results[key] = {
                        'optimal_frequency': int(row['frequency_mhz']),
                        'energy_savings': float(row.get('energy_savings_pct', 0)),
                        'performance_impact': float(row.get('performance_degradation_pct', 0)),
                        'source': 'precomputed'
                    }
                    
                logger.info(f"Loaded {len(self.precomputed_results)} precomputed results")
                
            except ImportError:
                logger.warning("pandas not available, cannot load precomputed results")
            except Exception as e:
                logger.warning(f"Could not load results: {e}")
                
    def get_heuristic_optimal_frequency(self, gpu: str, workload: str) -> Dict:
        """
        Get optimal frequency using heuristic approach based on your research insights
        """
        frequencies = self.gpu_frequencies.get(gpu, [])
        if not frequencies:
            logger.error(f"No frequency data for {gpu}")
            return {}
            
        max_freq = max(frequencies)
        min_freq = min(frequencies)
        freq_range = max_freq - min_freq
        
        # Heuristic rules based on your AI workload analysis
        if workload == 'llama':
            # Text generation: balanced compute/memory, favor medium-high frequencies
            optimal_freq = min_freq + int(0.7 * freq_range)
            expected_savings = 25.0
            expected_impact = 3.0
            
        elif workload == 'stablediffusion':
            # Image generation: iterative process, good for frequency reduction
            optimal_freq = min_freq + int(0.6 * freq_range)
            expected_savings = 30.0
            expected_impact = 4.0
            
        elif workload == 'vit':
            # Vision transformer: attention-heavy, batch-dependent
            optimal_freq = min_freq + int(0.65 * freq_range)
            expected_savings = 22.0
            expected_impact = 2.5
            
        elif workload == 'whisper':
            # Audio processing: encoder-decoder, variable length
            optimal_freq = min_freq + int(0.68 * freq_range)
            expected_savings = 27.0
            expected_impact = 3.5
            
        else:
            # Conservative default
            optimal_freq = min_freq + int(0.75 * freq_range)
            expected_savings = 20.0
            expected_impact = 2.0
            
        # Round to nearest valid frequency
        optimal_freq = min(frequencies, key=lambda f: abs(f - optimal_freq))
        
        return {
            'optimal_frequency': optimal_freq,
            'energy_savings': expected_savings,
            'performance_impact': expected_impact,
            'source': 'heuristic'
        }
        
    def get_optimal_frequency(self, gpu: str, workload: str) -> Dict:
        """Get optimal frequency for given GPU and workload"""
        gpu = gpu.upper()
        workload = workload.lower()
        
        key = f"{gpu}_{workload}"
        
        # Check for precomputed results first
        if key in self.precomputed_results:
            result = self.precomputed_results[key].copy()
            logger.info(f"Using precomputed optimal frequency: {result['optimal_frequency']} MHz")
            return result
            
        # Fall back to heuristic approach
        logger.info("Using heuristic optimal frequency selection")
        return self.get_heuristic_optimal_frequency(gpu, workload)
        
    def get_frequency_range(self, gpu: str, constraint_pct: float = 5.0) -> List[int]:
        """Get frequency range for experimentation around optimal frequency"""
        gpu = gpu.upper()
        
        all_frequencies = self.gpu_frequencies.get(gpu, [])
        if not all_frequencies:
            return []
            
        # For experimentation, return a subset around likely optimal range
        max_freq = max(all_frequencies)
        min_freq = min(all_frequencies)
        freq_range = max_freq - min_freq
        
        # Focus on 60-80% of frequency range (where optimal usually lies)
        start_freq = min_freq + int(0.6 * freq_range)
        end_freq = min_freq + int(0.8 * freq_range)
        
        selected_frequencies = [f for f in all_frequencies if start_freq <= f <= end_freq]
        
        # Include max frequency for baseline comparison
        if max_freq not in selected_frequencies:
            selected_frequencies.append(max_freq)
            
        return sorted(selected_frequencies)
        
    def run_all_combinations(self) -> Dict:
        """Run optimal frequency selection for all GPU-workload combinations"""
        results = {}
        
        logger.info("Running optimal frequency selection for all combinations...")
        
        for gpu in ['V100', 'A100', 'H100']:
            for workload in ['llama', 'stablediffusion', 'vit', 'whisper']:
                result = self.get_optimal_frequency(gpu, workload)
                if result:
                    results[f"{gpu}_{workload}"] = result
                    logger.info(f"{gpu} {workload}: {result['optimal_frequency']} MHz "
                              f"({result['energy_savings']:.1f}% energy, "
                              f"{result['performance_impact']:.1f}% perf impact)")
                              
        return results
        
    def generate_launch_commands(self, results: Dict) -> List[str]:
        """Generate launch_v2.sh commands with optimal frequencies"""
        commands = []
        
        app_mapping = {
            'llama': '../app-llama/LlamaViaHF.py',
            'stablediffusion': '../app-stable-diffusion/StableDiffusionViaHF.py', 
            'vit': '../app-vision-transformer/ViTViaHF.py',
            'whisper': '../app-whisper/WhisperViaHF.py'
        }
        
        for key, result in results.items():
            gpu, workload = key.split('_')
            app_executable = app_mapping.get(workload, f'../app-{workload}/app.py')
            
            cmd = (f"./launch_v2.sh --gpu-type {gpu} --profiling-mode custom "
                  f"--custom-frequencies '{result['optimal_frequency']}' "
                  f"--app-name {workload} --app-executable {app_executable} "
                  f"--num-runs 3 --sleep-interval 2")
            commands.append(cmd)
            
        return commands

def main():
    parser = argparse.ArgumentParser(description="Optimal frequency selection for AI inference")
    parser.add_argument("--gpu", help="GPU type (V100, A100, H100)")
    parser.add_argument("--workload", help="Workload (llama, stablediffusion, vit, whisper)")
    parser.add_argument("--run-all", action="store_true", 
                       help="Run optimization for all GPU-workload combinations")
    parser.add_argument("--generate-commands", action="store_true",
                       help="Generate launch_v2.sh commands with optimal frequencies")
    parser.add_argument("--frequency-range", action="store_true",
                       help="Get frequency range for experimentation")
    parser.add_argument("--results-dir", default=".",
                       help="Directory containing results")
    parser.add_argument("--quiet", action="store_true",
                       help="Only output the optimal frequency value")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        
    # Initialize selector
    selector = SimpleOptimalFrequencySelector(args.results_dir)
    
    if args.run_all:
        # Run optimization for all combinations
        results = selector.run_all_combinations()
        
        if args.generate_commands:
            commands = selector.generate_launch_commands(results)
            print("\n# Optimal frequency launch commands:")
            for cmd in commands:
                print(cmd)
                
    elif args.gpu and args.workload:
        # Single workload optimization
        result = selector.get_optimal_frequency(args.gpu, args.workload)
        
        if args.quiet:
            # Just output the frequency for script integration
            print(result.get('optimal_frequency', ''))
        elif args.frequency_range:
            # Output frequency range for experimentation
            frequencies = selector.get_frequency_range(args.gpu)
            print(','.join(map(str, frequencies)))
        else:
            # Full result output
            if result:
                print(f"Optimal frequency for {args.gpu} {args.workload}: {result['optimal_frequency']} MHz")
                print(f"Expected energy savings: {result['energy_savings']:.1f}%")
                print(f"Expected performance impact: {result['performance_impact']:.1f}%")
                print(f"Source: {result['source']}")
                
                # Generate launch command
                app_mapping = {
                    'llama': '../app-llama/LlamaViaHF.py',
                    'stablediffusion': '../app-stable-diffusion/StableDiffusionViaHF.py',
                    'vit': '../app-vision-transformer/ViTViaHF.py',
                    'whisper': '../app-whisper/WhisperViaHF.py'
                }
                
                app_executable = app_mapping.get(args.workload.lower(), f'../app-{args.workload}/app.py')
                
                print(f"\nLaunch command:")
                print(f"./launch_v2.sh --gpu-type {args.gpu} --profiling-mode custom "
                     f"--custom-frequencies '{result['optimal_frequency']}' "
                     f"--app-name {args.workload} --app-executable {app_executable} "
                     f"--num-runs 3 --sleep-interval 2")
            else:
                print(f"Could not determine optimal frequency for {args.gpu} {args.workload}")
                return 1
                
    else:
        parser.print_help()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
