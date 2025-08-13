#!/usr/bin/env python3
"""
AI Inference Energy Data Aggregation Script

This script consolidates all collected DVFS profiling results across:
- All GPU types (V100, A100, H100)  
- All AI workloads (LLaMA, StableDiffusion, ViT, Whisper)
- All frequencies and runs

It processes DCGMI profiling data and timing information to create:
1. Consolidated dataset for modeling
2. Performance baselines for 5% constraint analysis
3. Feature extraction for optimal frequency selection

Author: Mert Side
Based on proven FGCS/ICPP methodology
"""

import os
import pandas as pd
import numpy as np
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIInferenceDataAggregator:
    """
    Aggregate AI inference energy profiling data from collected results
    """
    
    def __init__(self, results_base_dir: str = "."):
        self.results_base_dir = Path(results_base_dir)
        self.gpus = ['V100', 'A100', 'H100']
        self.workloads = ['llama', 'stablediffusion', 'vit', 'whisper']
        
        # DCGMI field mapping (from your proven approach)
        self.dcgmi_fields = {
            'POWER': 'power_watts',           # Field 155 - Power usage
            'PMLMT': 'power_limit_watts',     # Field 160 - Power management limit  
            'TOTEC': 'total_energy_mj',       # Field 156 - Total energy consumption
            'GPUTL': 'gpu_utilization_pct',   # Field 203 - GPU utilization
            'MCUTL': 'memory_utilization_pct', # Field 204 - Memory copy utilization
            'SMCLK': 'sm_clock_mhz',          # Field 100 - SM clock
            'MMCLK': 'memory_clock_mhz',      # Field 101 - Memory clock
            'SACLK': 'graphics_clock_mhz',    # Field 110 - Graphics/SM application clock  
            'MACLK': 'memory_app_clock_mhz',  # Field 111 - Memory application clock
            'FBTTL': 'memory_total_mb',       # Field 250 - Framebuffer total
            'FBFRE': 'memory_free_mb',        # Field 251 - Framebuffer free
            'FBUSD': 'memory_used_mb',        # Field 252 - Framebuffer used
            'GRACT': 'graphics_active_pct',   # Field 1001 - Graphics active
            'SMACT': 'sm_active_pct',         # Field 1002 - SM active
            'SMOCC': 'sm_occupancy_pct',      # Field 1003 - SM occupancy  
            'TENSO': 'tensor_active_pct',     # Field 1004 - Tensor pipe active
            'DRAMA': 'dram_active_pct',       # Field 1005 - DRAM active
            'FP64A': 'fp64_active_pct',       # Field 1006 - FP64 active
            'FP32A': 'fp32_active_pct',       # Field 1007 - FP32 active
            'FP16A': 'fp16_active_pct'        # Field 1008 - FP16 active
        }
        
        # Your proven critical features (from FGCS/ICPP papers)
        self.critical_features = {
            'fp_active': 'fp32_active_pct',    # Your proven feature
            'dram_active': 'dram_active_pct',  # Your proven feature  
            'sm_app_clock': 'graphics_clock_mhz' # Your proven feature
        }
        
        logger.info(f"Initialized aggregator for {len(self.gpus)} GPUs and {len(self.workloads)} workloads")
        
    def find_result_directories(self) -> List[Path]:
        """Find all results directories matching the pattern"""
        pattern = "results_*_*_job_*"
        result_dirs = list(self.results_base_dir.glob(pattern))
        
        # Also check for non-SLURM pattern
        pattern_no_job = "results_*_*"
        result_dirs.extend([d for d in self.results_base_dir.glob(pattern_no_job) 
                           if not d.name.endswith("_job")])
        
        logger.info(f"Found {len(result_dirs)} result directories")
        for d in sorted(result_dirs):
            logger.debug(f"  {d.name}")
            
        return result_dirs
        
    def parse_directory_name(self, dir_name: str) -> Optional[Tuple[str, str, str]]:
        """Parse directory name to extract GPU, workload, and job info"""
        # Pattern: results_gpu_workload_job_number or results_gpu_workload
        parts = dir_name.split('_')
        
        if len(parts) >= 3:
            gpu = parts[1].upper()
            workload = parts[2].lower()
            
            # Validate GPU and workload
            if gpu in self.gpus and workload in self.workloads:
                job_id = parts[4] if len(parts) >= 5 else "unknown"
                return gpu, workload, job_id
                
        logger.warning(f"Could not parse directory name: {dir_name}")
        return None
        
    def load_timing_data(self, result_dir: Path) -> pd.DataFrame:
        """Load timing summary data from results directory"""
        timing_file = result_dir / "timing_summary.log"
        
        if not timing_file.exists():
            logger.warning(f"No timing file found in {result_dir.name}")
            return pd.DataFrame()
            
        timing_data = []
        
        try:
            with open(timing_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                        
                    # Format: run_id,frequency_mhz,duration_seconds,exit_code,status
                    parts = line.split(',')
                    if len(parts) >= 5:
                        timing_data.append({
                            'run_id': parts[0],
                            'frequency': int(parts[1]),
                            'duration_seconds': float(parts[2]),
                            'exit_code': int(parts[3]),
                            'status': parts[4]
                        })
                        
        except Exception as e:
            logger.error(f"Error loading timing data from {timing_file}: {e}")
            return pd.DataFrame()
            
        df = pd.DataFrame(timing_data)
        logger.debug(f"Loaded {len(df)} timing records from {result_dir.name}")
        return df
        
    def load_profile_data(self, profile_file: Path) -> pd.DataFrame:
        """Load DCGMI profiling data from CSV file"""
        if not profile_file.exists():
            logger.warning(f"Profile file does not exist: {profile_file}")
            return pd.DataFrame()
            
        try:
            # Read the CSV file, skipping the header lines
            df = pd.read_csv(profile_file, comment='#', delim_whitespace=True)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Convert numeric columns
            numeric_columns = ['POWER', 'PMLMT', 'TOTEC', 'TMPTR', 'MMTMP', 'GPUTL', 'MCUTL',
                             'FBTTL', 'FBFRE', 'FBUSD', 'SMCLK', 'MMCLK', 'SACLK', 'MACLK', 
                             'PSTAT', 'GRACT', 'SMACT', 'SMOCC', 'TENSO', 'DRAMA', 'FP64A', 
                             'FP32A', 'FP16A']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            logger.debug(f"Loaded {len(df)} profiling samples from {profile_file.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading profile data from {profile_file}: {e}")
            return pd.DataFrame()
            
    def extract_features(self, profile_df: pd.DataFrame, timing_data: dict) -> dict:
        """Extract features using your proven FGCS/ICPP methodology"""
        if profile_df.empty:
            return {}
            
        features = {}
        
        # Basic run information
        features['frequency'] = timing_data.get('frequency', 0)
        features['duration_seconds'] = timing_data.get('duration_seconds', 0)
        features['exit_code'] = timing_data.get('exit_code', -1)
        features['status'] = timing_data.get('status', 'unknown')
        
        # Your proven critical features (FGCS/ICPP methodology)
        if 'FP32A' in profile_df.columns:
            features['fp_active'] = profile_df['FP32A'].mean()  # Your proven feature
        if 'DRAMA' in profile_df.columns:
            features['dram_active'] = profile_df['DRAMA'].mean()  # Your proven feature
        if 'SACLK' in profile_df.columns:
            features['sm_app_clock'] = profile_df['SACLK'].mean()  # Your proven feature
            
        # Power and energy metrics
        if 'POWER' in profile_df.columns:
            features['avg_power_watts'] = profile_df['POWER'].mean()
            features['max_power_watts'] = profile_df['POWER'].max()
            features['min_power_watts'] = profile_df['POWER'].min()
            
        # Calculate total energy consumption
        if features['duration_seconds'] > 0 and 'avg_power_watts' in features:
            features['total_energy_joules'] = features['avg_power_watts'] * features['duration_seconds']
            
        # GPU utilization metrics
        if 'GPUTL' in profile_df.columns:
            features['avg_gpu_util'] = profile_df['GPUTL'].mean()
        if 'MCUTL' in profile_df.columns:
            features['avg_memory_util'] = profile_df['MCUTL'].mean()
            
        # Advanced activity metrics (from your FGCS approach)
        if 'SMACT' in profile_df.columns:
            features['sm_active'] = profile_df['SMACT'].mean()
        if 'SMOCC' in profile_df.columns:
            features['sm_occupancy'] = profile_df['SMOCC'].mean()
        if 'TENSO' in profile_df.columns:
            features['tensor_active'] = profile_df['TENSO'].mean()
            
        # Clock frequencies
        if 'SMCLK' in profile_df.columns:
            features['sm_clock'] = profile_df['SMCLK'].mean()
        if 'MMCLK' in profile_df.columns:
            features['memory_clock'] = profile_df['MMCLK'].mean()
            
        # Memory usage
        if all(col in profile_df.columns for col in ['FBTTL', 'FBUSD']):
            features['memory_usage_pct'] = (profile_df['FBUSD'] / profile_df['FBTTL'] * 100).mean()
            
        # Temperature metrics
        if 'TMPTR' in profile_df.columns:
            features['avg_gpu_temp'] = profile_df['TMPTR'].mean()
        if 'MMTMP' in profile_df.columns:
            features['avg_memory_temp'] = profile_df['MMTMP'].mean()
            
        return features
        
    def calculate_energy_metrics(self, profile_df: pd.DataFrame, duration_seconds: float) -> Dict[str, float]:
        """Calculate energy and power metrics"""
        if profile_df.empty:
            return {}
            
        metrics = {}
        
        # Power metrics (Watts)
        if 'POWER' in profile_df.columns:
            metrics['avg_power_watts'] = profile_df['POWER'].mean()
            metrics['max_power_watts'] = profile_df['POWER'].max()
            metrics['min_power_watts'] = profile_df['POWER'].min()
        else:
            metrics['avg_power_watts'] = 0.0
            metrics['max_power_watts'] = 0.0
            metrics['min_power_watts'] = 0.0
            
        # Energy calculation (Joules)
        # Method 1: From total energy if available
        if 'TOTEC' in profile_df.columns and not profile_df['TOTEC'].isna().all():
            # TOTEC is in millijoules, convert to joules
            energy_start = profile_df['TOTEC'].iloc[0] / 1000.0
            energy_end = profile_df['TOTEC'].iloc[-1] / 1000.0
            metrics['total_energy_joules'] = energy_end - energy_start
        else:
            # Method 2: From average power * duration
            metrics['total_energy_joules'] = metrics['avg_power_watts'] * duration_seconds
            
        # Energy efficiency metrics
        if duration_seconds > 0:
            metrics['energy_per_second'] = metrics['total_energy_joules'] / duration_seconds
            
        return metrics
        
    def process_single_result_directory(self, result_dir: Path) -> List[Dict]:
        """Process a single results directory"""
        parsed = self.parse_directory_name(result_dir.name)
        if not parsed:
            return []
            
        gpu, workload, job_id = parsed
        logger.info(f"Processing {gpu} {workload} (job {job_id})")
        
        # Load timing data
        timing_df = self.load_timing_data(result_dir)
        if timing_df.empty:
            logger.warning(f"No timing data for {result_dir.name}")
            return []
            
        # Process each run
        aggregated_data = []
        
        for _, timing_row in timing_df.iterrows():
            run_id = timing_row['run_id']
            frequency = timing_row['frequency']
            duration = timing_row['duration_seconds']
            status = timing_row['status']
            
            # Only process successful runs
            if status != 'success':
                logger.debug(f"Skipping failed run {run_id} (status: {status})")
                continue
                
            # Find corresponding profile file
            profile_pattern = f"run_{run_id}_freq_{frequency}_profile.csv"
            profile_files = list(result_dir.glob(profile_pattern))
            
            if not profile_files:
                logger.warning(f"No profile file found for {run_id} at {frequency}MHz")
                continue
                
            profile_file = profile_files[0]
            
            # Load profile data
            profile_df = self.load_profile_data(profile_file)
            if profile_df.empty:
                logger.warning(f"Empty profile data for {run_id}")
                continue
                
            # Extract features using your proven approach
            features = self.extract_features(profile_df)
            
            # Calculate energy metrics
            energy_metrics = self.calculate_energy_metrics(profile_df, duration)
            
            # Combine all data
            run_data = {
                # Metadata
                'gpu': gpu,
                'workload': workload,
                'job_id': job_id,
                'run_id': run_id,
                'frequency_mhz': frequency,
                'duration_seconds': duration,
                
                # Your proven features
                'fp_active': features.get('fp_active', 0.0),
                'dram_active': features.get('dram_active', 0.0), 
                'sm_app_clock': features.get('sm_app_clock', frequency),
                
                # AI-specific features
                'tensor_core_utilization': features.get('tensor_core_utilization', 0.0),
                'memory_bandwidth_utilization': features.get('memory_bandwidth_utilization', 0.0),
                'gpu_utilization': features.get('gpu_utilization', 0.0),
                'mixed_precision_ratio': features.get('mixed_precision_ratio', 0.0),
                
                # Energy and performance metrics
                'avg_power_watts': energy_metrics.get('avg_power_watts', 0.0),
                'total_energy_joules': energy_metrics.get('total_energy_joules', 0.0),
                'energy_per_second': energy_metrics.get('energy_per_second', 0.0),
                
                # EDP calculation (your proven objective)
                'edp': energy_metrics.get('total_energy_joules', 0.0) * duration,
                'ed2p': energy_metrics.get('total_energy_joules', 0.0) * (duration ** 2),
                
                # Raw file paths for reference
                'profile_file': str(profile_file),
                'result_directory': str(result_dir)
            }
            
            aggregated_data.append(run_data)
            
        logger.info(f"Processed {len(aggregated_data)} successful runs from {result_dir.name}")
        return aggregated_data
        
    def aggregate_all_data(self) -> pd.DataFrame:
        """Aggregate data from all result directories"""
        logger.info("Starting comprehensive data aggregation...")
        
        result_dirs = self.find_result_directories()
        if not result_dirs:
            logger.error("No result directories found!")
            return pd.DataFrame()
            
        all_data = []
        
        for result_dir in result_dirs:
            try:
                dir_data = self.process_single_result_directory(result_dir)
                all_data.extend(dir_data)
            except Exception as e:
                logger.error(f"Error processing {result_dir.name}: {e}")
                continue
                
        if not all_data:
            logger.error("No data was successfully aggregated!")
            return pd.DataFrame()
            
        # Create consolidated DataFrame
        df = pd.DataFrame(all_data)
        
        # Data validation and cleanup
        df = df.dropna(subset=['frequency_mhz', 'duration_seconds', 'avg_power_watts'])
        
        # Sort by GPU, workload, frequency for easier analysis
        df = df.sort_values(['gpu', 'workload', 'frequency_mhz', 'run_id'])
        
        logger.info(f"Successfully aggregated {len(df)} data points:")
        logger.info(f"  GPUs: {sorted(df['gpu'].unique())}")
        logger.info(f"  Workloads: {sorted(df['workload'].unique())}")
        logger.info(f"  Frequency range: {df['frequency_mhz'].min()}-{df['frequency_mhz'].max()} MHz")
        logger.info(f"  Total energy range: {df['total_energy_joules'].min():.2f}-{df['total_energy_joules'].max():.2f} J")
        
        return df
        
    def establish_performance_baselines(self, df: pd.DataFrame, constraint_pct: float = 5.0) -> Dict:
        """Establish performance baselines for constraint analysis"""
        baselines = {}
        
        for gpu in df['gpu'].unique():
            for workload in df['workload'].unique():
                subset = df[(df['gpu'] == gpu) & (df['workload'] == workload)]
                
                if subset.empty:
                    continue
                    
                # Find maximum frequency performance (baseline)
                max_freq = subset['frequency_mhz'].max()
                baseline_subset = subset[subset['frequency_mhz'] == max_freq]
                
                if baseline_subset.empty:
                    continue
                    
                baseline_time = baseline_subset['duration_seconds'].mean()
                baseline_energy = baseline_subset['total_energy_joules'].mean()
                
                # Calculate constraint threshold (5% performance degradation)
                max_acceptable_time = baseline_time * (1 + constraint_pct / 100.0)
                
                baselines[f"{gpu}_{workload}"] = {
                    'gpu': gpu,
                    'workload': workload,
                    'baseline_time_seconds': baseline_time,
                    'baseline_energy_joules': baseline_energy,
                    'max_frequency_mhz': max_freq,
                    'max_acceptable_time_seconds': max_acceptable_time,
                    'performance_constraint_pct': constraint_pct
                }
                
        logger.info(f"Established baselines for {len(baselines)} GPU-workload combinations")
        return baselines
        
    def save_results(self, df: pd.DataFrame, baselines: Dict, output_dir: str = "aggregated_results"):
        """Save aggregated results and baselines"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main dataset
        main_file = output_path / f"ai_inference_aggregated_data_{timestamp}.csv"
        df.to_csv(main_file, index=False)
        logger.info(f"Saved aggregated dataset: {main_file}")
        
        # Save baselines
        baselines_file = output_path / f"performance_baselines_{timestamp}.json"
        with open(baselines_file, 'w') as f:
            json.dump(baselines, f, indent=2)
        logger.info(f"Saved performance baselines: {baselines_file}")
        
        # Generate summary statistics
        summary_file = output_path / f"aggregation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("AI Inference Energy Data Aggregation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            f.write(f"Total data points: {len(df)}\n")
            f.write(f"GPUs: {sorted(df['gpu'].unique())}\n")
            f.write(f"Workloads: {sorted(df['workload'].unique())}\n")
            f.write(f"Frequency range: {df['frequency_mhz'].min()}-{df['frequency_mhz'].max()} MHz\n\n")
            
            # Per-GPU statistics
            for gpu in sorted(df['gpu'].unique()):
                gpu_data = df[df['gpu'] == gpu]
                f.write(f"{gpu} Statistics:\n")
                f.write(f"  Data points: {len(gpu_data)}\n")
                f.write(f"  Frequencies: {len(gpu_data['frequency_mhz'].unique())}\n")
                f.write(f"  Avg power: {gpu_data['avg_power_watts'].mean():.2f}W\n")
                f.write(f"  Energy range: {gpu_data['total_energy_joules'].min():.2f}-{gpu_data['total_energy_joules'].max():.2f}J\n\n")
                
        logger.info(f"Saved summary: {summary_file}")
        
        return main_file, baselines_file, summary_file

def main():
    parser = argparse.ArgumentParser(description="Aggregate AI inference energy profiling data")
    parser.add_argument("--results-dir", default=".", 
                       help="Directory containing results_* subdirectories")
    parser.add_argument("--output-dir", default="aggregated_results",
                       help="Output directory for aggregated data")
    parser.add_argument("--constraint-pct", type=float, default=5.0,
                       help="Performance degradation constraint percentage")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize aggregator
    aggregator = AIInferenceDataAggregator(args.results_dir)
    
    # Aggregate all data
    df = aggregator.aggregate_all_data()
    
    if df.empty:
        logger.error("No data was aggregated. Check your results directories.")
        return 1
        
    # Establish performance baselines
    baselines = aggregator.establish_performance_baselines(df, args.constraint_pct)
    
    # Save results
    files = aggregator.save_results(df, baselines, args.output_dir)
    
    logger.info("Data aggregation completed successfully!")
    logger.info(f"Files created: {[str(f) for f in files]}")
    
    return 0

if __name__ == "__main__":
    exit(main())
