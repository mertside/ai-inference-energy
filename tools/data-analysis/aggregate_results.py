#!/usr/bin/env python3
"""
Results Aggregation Script for AI Inference Energy Optimization.

This script consolidates DVFS experimental results across multiple AI workloads,
GPU architectures, and frequency configurations into unified datasets for
optimal frequency selection research.

Features:
- Aggregates DCGMI and nvidia-smi profiling data
- Extracts power, performance, and feature metrics
- Calculates energy consumption and EDP/ED2P metrics
- Establishes performance baselines for constraint optimization
- Supports multi-GPU and multi-workload consolidation

Requirements:
    - Existing results directories from launch_v2.sh experiments
    - DCGMI or nvidia-smi profiling data in CSV format
    - Python 3.8+ with pandas, numpy

Author: Mert Side
"""

import argparse
import glob
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from config import profiling_config
    from utils import setup_logging
except ImportError:
    # Fallback configuration
    class ProfilingConfig:
        DCGMI_FIELDS = [
            52, 50, 155, 160, 156, 150, 140, 203, 204, 250, 251, 252,
            100, 101, 110, 111, 190, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008
        ]
    
    profiling_config = ProfilingConfig()
    
    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)


class AIWorkloadDataAggregator:
    """
    Comprehensive data aggregator for AI inference energy optimization experiments.
    
    Consolidates DVFS experimental results across multiple workloads, GPU architectures,
    and frequency configurations into unified datasets suitable for model training
    and optimal frequency selection research.
    """

    def __init__(self, results_base_dir: str = ".", logger: Optional[logging.Logger] = None):
        """
        Initialize the data aggregator.
        
        Args:
            results_base_dir: Base directory containing results_* directories
            logger: Optional logger instance
        """
        self.results_base_dir = Path(results_base_dir)
        self.logger = logger or setup_logging()
        
        # AI workloads and GPU architectures
        self.workloads = ['llama', 'stablediffusion', 'vit', 'whisper', 'lstm']
        self.gpu_types = ['v100', 'a100', 'h100']
        
        # Results patterns
        self.results_pattern = "results_*"
        
        # DCGMI field mapping (based on your proven profiling configuration)
        self.dcgmi_field_mapping = {
            0: 'timestamp',           # Host timestamp
            52: 'gpu_index',          # NVML GPU index  
            50: 'gpu_name',           # Product name
            155: 'power_usage',       # Instantaneous power (W)
            160: 'power_limit',       # Software power cap (W)
            156: 'energy_consumption', # Total energy (mJ)
            150: 'gpu_temp',          # GPU temperature (°C)
            140: 'memory_temp',       # Memory temperature (°C)
            203: 'gpu_util',          # GPU utilization (%)
            204: 'mem_copy_util',     # Memory copy utilization (%)
            250: 'fb_total',          # FB memory total (MB)
            251: 'fb_free',           # FB memory free (MB)
            252: 'fb_used',           # FB memory used (MB)
            100: 'sm_clock',          # SM frequency (MHz)
            101: 'mem_clock',         # Memory frequency (MHz)
            110: 'app_sm_clock',      # Application SM clock (MHz)
            111: 'app_mem_clock',     # Application memory clock (MHz)
            190: 'pstate',            # Performance state
            1001: 'gr_active',        # Graphics active (%)
            1002: 'sm_active',        # SM active (%)
            1003: 'sm_occupancy',     # SM occupancy (%)
            1004: 'tensor_active',    # Tensor pipe active (%)
            1005: 'dram_active',      # DRAM active (%) - Key feature
            1006: 'fp64_active',      # FP64 active (%)
            1007: 'fp32_active',      # FP32 active (%)
            1008: 'fp16_active'       # FP16 active (%) - Key for AI workloads
        }
        
        # nvidia-smi field mapping (alternative profiling method)
        self.nvidia_smi_field_mapping = {
            0: 'timestamp',
            1: 'gpu_index',
            2: 'gpu_name',
            3: 'power_draw',
            4: 'power_limit',
            5: 'gpu_temp',
            6: 'gpu_util',
            7: 'memory_util',
            8: 'memory_total',
            9: 'memory_free',
            10: 'memory_used',
            11: 'sm_clock',
            12: 'mem_clock',
            13: 'gr_clock',
            14: 'app_gr_clock',
            15: 'app_mem_clock',
            16: 'pstate'
        }

    def find_results_directories(self) -> List[Path]:
        """
        Find all results directories matching the expected pattern.
        
        Returns:
            List of results directory paths
        """
        pattern = str(self.results_base_dir / self.results_pattern)
        results_dirs = [Path(d) for d in glob.glob(pattern)]
        
        self.logger.info(f"Found {len(results_dirs)} results directories")
        for results_dir in sorted(results_dirs):
            self.logger.debug(f"  - {results_dir.name}")
            
        return results_dirs

    def parse_results_directory_name(self, results_dir: Path) -> Dict[str, str]:
        """
        Parse results directory name to extract metadata.
        
        Expected formats:
        - results_h100_stablediffusion_job_12345
        - results_a100_llama
        - results_v100_whisper_custom
        
        Args:
            results_dir: Results directory path
            
        Returns:
            Dictionary with parsed metadata
        """
        dir_name = results_dir.name
        
        # Match pattern: results_{gpu}_{workload}[_job_{id}][_suffix]
        pattern = r'results_([^_]+)_([^_]+)(?:_job_(\d+))?(?:_(.+))?'
        match = re.match(pattern, dir_name)
        
        if not match:
            self.logger.warning(f"Could not parse directory name: {dir_name}")
            return {'gpu': 'unknown', 'workload': 'unknown', 'job_id': None, 'suffix': None}
        
        gpu, workload, job_id, suffix = match.groups()
        
        return {
            'gpu': gpu.lower(),
            'workload': workload.lower(),
            'job_id': job_id,
            'suffix': suffix,
            'directory': str(results_dir)
        }

    def extract_frequency_from_filename(self, csv_file: Path) -> Optional[int]:
        """
        Extract frequency from CSV filename.
        
        Expected formats:
        - A100_freq_1410_run_1_profile.csv
        - H100_1620MHz_run_2_dcgmi.csv
        - profile_v100_885_run_3.csv
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Frequency in MHz or None if not found
        """
        filename = csv_file.name
        
        # Try multiple patterns
        patterns = [
            r'freq_(\d+)',           # _freq_1410_
            r'(\d+)MHz',             # _1620MHz_
            r'(\d{3,4})_run',        # _885_run_
            r'_(\d{3,4})_',          # Generic _FREQ_
            r'freq(\d+)',            # freq1410
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                freq = int(match.group(1))
                # Validate frequency range (typical GPU frequencies: 300-2100 MHz)
                if 300 <= freq <= 2100:
                    return freq
        
        self.logger.debug(f"Could not extract frequency from: {filename}")
        return None

    def extract_run_number(self, csv_file: Path) -> int:
        """
        Extract run number from CSV filename.
        
        Args:
            csv_file: CSV file path
            
        Returns:
            Run number (defaults to 1 if not found)
        """
        filename = csv_file.name
        
        # Try to find run number
        pattern = r'run_?(\d+)'
        match = re.search(pattern, filename)
        
        if match:
            return int(match.group(1))
        
        return 1  # Default run number

    def load_dcgmi_metrics(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """
        Load and parse DCGMI profiling data.
        
        Args:
            csv_file: Path to DCGMI CSV file
            
        Returns:
            DataFrame with parsed metrics or None if failed
        """
        try:
            # Read CSV file (DCGMI outputs without headers)
            df = pd.read_csv(csv_file, header=None)
            
            # Check if we have the expected number of columns
            expected_cols = len(self.dcgmi_field_mapping)
            if len(df.columns) != expected_cols:
                self.logger.warning(
                    f"Unexpected column count in {csv_file}: "
                    f"got {len(df.columns)}, expected {expected_cols}"
                )
            
            # Map column indices to field names
            column_mapping = {}
            for i, col_idx in enumerate(df.columns):
                if i in self.dcgmi_field_mapping:
                    column_mapping[col_idx] = self.dcgmi_field_mapping[i]
                else:
                    column_mapping[col_idx] = f"field_{i}"
            
            df = df.rename(columns=column_mapping)
            
            # Convert numeric columns
            numeric_cols = [col for col in df.columns if col not in ['timestamp', 'gpu_name']]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter out invalid rows (e.g., headers that might be included)
            if 'power_usage' in df.columns:
                df = df[df['power_usage'].notna() & (df['power_usage'] > 0)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load DCGMI data from {csv_file}: {e}")
            return None

    def load_nvidia_smi_metrics(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """
        Load and parse nvidia-smi profiling data.
        
        Args:
            csv_file: Path to nvidia-smi CSV file
            
        Returns:
            DataFrame with parsed metrics or None if failed
        """
        try:
            # Read CSV file (nvidia-smi outputs without headers when using --format=csv,noheader)
            df = pd.read_csv(csv_file, header=None)
            
            # Map column indices to field names
            column_mapping = {}
            for i, col_idx in enumerate(df.columns):
                if i in self.nvidia_smi_field_mapping:
                    column_mapping[col_idx] = self.nvidia_smi_field_mapping[i]
                else:
                    column_mapping[col_idx] = f"field_{i}"
            
            df = df.rename(columns=column_mapping)
            
            # Convert numeric columns  
            numeric_cols = [col for col in df.columns if col not in ['timestamp', 'gpu_name']]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter out invalid rows
            if 'power_draw' in df.columns:
                df = df[df['power_draw'].notna() & (df['power_draw'] > 0)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load nvidia-smi data from {csv_file}: {e}")
            return None

    def extract_ai_features(self, metrics_df: pd.DataFrame, workload: str) -> Dict[str, float]:
        """
        Extract AI workload-specific features from profiling metrics.
        
        Based on your FGCS/ICPP proven features plus AI-specific extensions.
        
        Args:
            metrics_df: DataFrame with profiling metrics
            workload: Workload name
            
        Returns:
            Dictionary of extracted features
        """
        # Your proven base features from FGCS/ICPP papers
        base_features = {}
        
        # Core features that remain constant across frequencies (your key insight)
        if 'dram_active' in metrics_df.columns:
            base_features['dram_active'] = metrics_df['dram_active'].mean()
        elif 'memory_util' in metrics_df.columns:
            # nvidia-smi equivalent
            base_features['dram_active'] = metrics_df['memory_util'].mean()
        
        if 'sm_active' in metrics_df.columns:
            base_features['fp_active'] = metrics_df['sm_active'].mean()
        elif 'gpu_util' in metrics_df.columns:
            # nvidia-smi equivalent
            base_features['fp_active'] = metrics_df['gpu_util'].mean()
        
        if 'sm_clock' in metrics_df.columns:
            base_features['sm_app_clock'] = metrics_df['sm_clock'].mean()
        
        # AI workload-specific features
        ai_features = {}
        
        if workload == 'llama':
            # LLaMA-specific patterns
            if 'tensor_active' in metrics_df.columns:
                ai_features['tensor_core_util'] = metrics_df['tensor_active'].mean()
            if 'fp16_active' in metrics_df.columns:
                ai_features['mixed_precision_ratio'] = metrics_df['fp16_active'].mean()
            # Sequence processing efficiency
            if 'sm_occupancy' in metrics_df.columns:
                ai_features['sequence_processing_ratio'] = metrics_df['sm_occupancy'].mean()
                
        elif workload == 'stablediffusion':
            # Stable Diffusion-specific patterns
            if 'tensor_active' in metrics_df.columns:
                ai_features['attention_compute_ratio'] = metrics_df['tensor_active'].mean()
            if 'gr_active' in metrics_df.columns:
                ai_features['denoising_compute_ratio'] = metrics_df['gr_active'].mean()
                
        elif workload == 'vit':
            # Vision Transformer-specific patterns
            if 'tensor_active' in metrics_df.columns:
                ai_features['attention_compute_ratio'] = metrics_df['tensor_active'].mean()
            if 'sm_occupancy' in metrics_df.columns:
                ai_features['patch_processing_efficiency'] = metrics_df['sm_occupancy'].mean()
                
        elif workload == 'whisper':
            # Whisper-specific patterns
            if 'tensor_active' in metrics_df.columns:
                ai_features['encoder_decoder_ratio'] = metrics_df['tensor_active'].mean()
            if 'dram_active' in metrics_df.columns:
                ai_features['beam_search_memory'] = metrics_df['dram_active'].mean()
        
        return {**base_features, **ai_features}

    def calculate_energy_metrics(self, metrics_df: pd.DataFrame, execution_time: float) -> Dict[str, float]:
        """
        Calculate energy consumption and efficiency metrics.
        
        Args:
            metrics_df: DataFrame with profiling metrics
            execution_time: Total execution time in seconds
            
        Returns:
            Dictionary of energy metrics
        """
        energy_metrics = {}
        
        # Power-based energy calculation (preferred method)
        if 'power_usage' in metrics_df.columns:
            power_data = metrics_df['power_usage'].dropna()
            if not power_data.empty:
                avg_power = power_data.mean()
                max_power = power_data.max()
                min_power = power_data.min()
                
                # Total energy consumption (J)
                energy_metrics['total_energy'] = avg_power * execution_time
                energy_metrics['avg_power'] = avg_power
                energy_metrics['max_power'] = max_power
                energy_metrics['min_power'] = min_power
                energy_metrics['power_std'] = power_data.std()
                
        elif 'power_draw' in metrics_df.columns:
            # nvidia-smi equivalent
            power_data = metrics_df['power_draw'].dropna()
            if not power_data.empty:
                avg_power = power_data.mean()
                energy_metrics['total_energy'] = avg_power * execution_time
                energy_metrics['avg_power'] = avg_power
                energy_metrics['max_power'] = power_data.max()
                energy_metrics['min_power'] = power_data.min()
                energy_metrics['power_std'] = power_data.std()
        
        # Direct energy measurement (if available from DCGMI field 156)
        if 'energy_consumption' in metrics_df.columns:
            energy_data = metrics_df['energy_consumption'].dropna()
            if not energy_data.empty and len(energy_data) > 1:
                # Energy consumption is cumulative, calculate difference
                total_energy_mj = energy_data.iloc[-1] - energy_data.iloc[0]
                energy_metrics['direct_energy'] = total_energy_mj / 1000.0  # Convert mJ to J
        
        return energy_metrics

    def process_workload_results(self, results_dir: Path, metadata: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Process all CSV files in a results directory for a specific workload.
        
        Args:
            results_dir: Results directory path
            metadata: Metadata extracted from directory name
            
        Returns:
            List of processed experiment records
        """
        workload_results = []
        
        # Find all CSV files in the results directory
        csv_files = list(results_dir.glob("*.csv"))
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in {results_dir}")
            return workload_results
        
        self.logger.info(f"Processing {len(csv_files)} CSV files from {results_dir.name}")
        
        for csv_file in csv_files:
            try:
                # Extract metadata from filename
                frequency = self.extract_frequency_from_filename(csv_file)
                run_number = self.extract_run_number(csv_file)
                
                if frequency is None:
                    self.logger.debug(f"Skipping {csv_file.name} - no frequency found")
                    continue
                
                # Load profiling metrics (try DCGMI first, fallback to nvidia-smi)
                metrics_df = self.load_dcgmi_metrics(csv_file)
                if metrics_df is None or metrics_df.empty:
                    metrics_df = self.load_nvidia_smi_metrics(csv_file)
                
                if metrics_df is None or metrics_df.empty:
                    self.logger.warning(f"Could not load metrics from {csv_file}")
                    continue
                
                # Calculate execution time (from profiling duration)
                if 'timestamp' in metrics_df.columns:
                    # Try to parse timestamps to get execution time
                    try:
                        timestamps = pd.to_datetime(metrics_df['timestamp'], errors='coerce')
                        valid_timestamps = timestamps.dropna()
                        if len(valid_timestamps) > 1:
                            execution_time = (valid_timestamps.iloc[-1] - valid_timestamps.iloc[0]).total_seconds()
                        else:
                            execution_time = len(metrics_df) * 0.05  # Fallback: assume 50ms sampling
                    except:
                        execution_time = len(metrics_df) * 0.05  # Fallback: assume 50ms sampling
                else:
                    execution_time = len(metrics_df) * 0.05  # Fallback: assume 50ms sampling
                
                # Extract AI workload features
                features = self.extract_ai_features(metrics_df, metadata['workload'])
                
                # Calculate energy metrics
                energy_metrics = self.calculate_energy_metrics(metrics_df, execution_time)
                
                # Compile experiment record
                record = {
                    # Metadata
                    'gpu': metadata['gpu'],
                    'workload': metadata['workload'],
                    'job_id': metadata['job_id'],
                    'frequency': frequency,
                    'run': run_number,
                    'csv_file': str(csv_file),
                    'results_directory': metadata['directory'],
                    
                    # Performance metrics
                    'execution_time': execution_time,
                    'samples_collected': len(metrics_df),
                    
                    # Energy metrics
                    **energy_metrics,
                    
                    # Features (your proven approach)
                    **features,
                    
                    # Multi-objective metrics (your proven EDP/ED2P approach)
                    'edp': energy_metrics.get('total_energy', 0) * execution_time,  # Energy-Delay Product
                    'ed2p': energy_metrics.get('total_energy', 0) * (execution_time ** 2),  # Energy-Delay^2 Product
                }
                
                workload_results.append(record)
                
            except Exception as e:
                self.logger.error(f"Error processing {csv_file}: {e}")
                continue
        
        self.logger.info(f"Successfully processed {len(workload_results)} experiments from {results_dir.name}")
        return workload_results

    def aggregate_all_results(self) -> pd.DataFrame:
        """
        Aggregate all experimental results into a unified dataset.
        
        Returns:
            DataFrame with all experimental results
        """
        self.logger.info("Starting comprehensive results aggregation")
        
        results_dirs = self.find_results_directories()
        all_results = []
        
        for results_dir in results_dirs:
            # Parse directory metadata
            metadata = self.parse_results_directory_name(results_dir)
            
            # Process workload results
            workload_results = self.process_workload_results(results_dir, metadata)
            all_results.extend(workload_results)
        
        if not all_results:
            self.logger.error("No experimental results found!")
            return pd.DataFrame()
        
        # Create unified DataFrame
        df = pd.DataFrame(all_results)
        
        # Add derived metrics
        self.add_derived_metrics(df)
        
        self.logger.info(f"Aggregated {len(df)} experimental records")
        self.logger.info(f"  GPUs: {sorted(df['gpu'].unique())}")
        self.logger.info(f"  Workloads: {sorted(df['workload'].unique())}")
        self.logger.info(f"  Frequency range: {df['frequency'].min()}-{df['frequency'].max()} MHz")
        
        return df

    def add_derived_metrics(self, df: pd.DataFrame) -> None:
        """
        Add derived metrics to the aggregated dataset.
        
        Args:
            df: DataFrame to modify in-place
        """
        # Performance efficiency metrics
        if 'total_energy' in df.columns and 'execution_time' in df.columns:
            df['energy_efficiency'] = 1.0 / (df['total_energy'] * df['execution_time'])
            
        # Frequency scaling metrics
        if 'avg_power' in df.columns and 'frequency' in df.columns:
            df['power_frequency_ratio'] = df['avg_power'] / df['frequency']
            
        # Workload intensity metrics
        if 'fp_active' in df.columns and 'dram_active' in df.columns:
            df['compute_memory_ratio'] = df['fp_active'] / (df['dram_active'] + 1e-6)  # Avoid division by zero

    def establish_performance_baselines(self, df: pd.DataFrame, constraint_pct: float = 5.0) -> Dict[str, Dict[str, float]]:
        """
        Establish performance baselines for constrained optimization.
        
        Based on maximum frequency performance for each GPU-workload combination.
        
        Args:
            df: Aggregated results DataFrame
            constraint_pct: Performance degradation constraint percentage
            
        Returns:
            Dictionary of baseline metrics
        """
        self.logger.info(f"Establishing performance baselines with {constraint_pct}% constraint")
        
        baselines = {}
        
        for gpu in df['gpu'].unique():
            for workload in df['workload'].unique():
                subset = df[(df['gpu'] == gpu) & (df['workload'] == workload)]
                
                if subset.empty:
                    continue
                
                # Find maximum frequency experiments
                max_freq = subset['frequency'].max()
                max_freq_experiments = subset[subset['frequency'] == max_freq]
                
                if max_freq_experiments.empty:
                    continue
                
                # Calculate baseline performance (mean of max frequency experiments)
                baseline_time = max_freq_experiments['execution_time'].mean()
                baseline_energy = max_freq_experiments['total_energy'].mean()
                
                # Performance constraint
                max_acceptable_time = baseline_time * (1 + constraint_pct / 100)
                
                baseline_key = f"{gpu}_{workload}"
                baselines[baseline_key] = {
                    'baseline_time': baseline_time,
                    'baseline_energy': baseline_energy,
                    'max_acceptable_time': max_acceptable_time,
                    'max_frequency': max_freq,
                    'constraint_pct': constraint_pct,
                    'num_baseline_experiments': len(max_freq_experiments)
                }
                
                self.logger.debug(
                    f"{baseline_key}: baseline_time={baseline_time:.3f}s, "
                    f"max_acceptable_time={max_acceptable_time:.3f}s"
                )
        
        self.logger.info(f"Established baselines for {len(baselines)} GPU-workload combinations")
        return baselines

    def save_aggregated_results(self, df: pd.DataFrame, output_file: str) -> None:
        """
        Save aggregated results to CSV file.
        
        Args:
            df: Aggregated results DataFrame
            output_file: Output CSV file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved aggregated results to {output_path}")
        self.logger.info(f"Dataset shape: {df.shape}")

    def save_baselines(self, baselines: Dict[str, Dict[str, float]], output_file: str) -> None:
        """
        Save performance baselines to JSON file.
        
        Args:
            baselines: Performance baselines dictionary
            output_file: Output JSON file path
        """
        import json
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(baselines, f, indent=2)
        
        self.logger.info(f"Saved performance baselines to {output_path}")

    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        Generate a summary report of the aggregated data.
        
        Args:
            df: Aggregated results DataFrame
            
        Returns:
            Summary report as string
        """
        report = []
        report.append("=" * 60)
        report.append("AI INFERENCE ENERGY OPTIMIZATION - DATA SUMMARY")
        report.append("=" * 60)
        
        # Dataset overview
        report.append(f"\nDataset Overview:")
        report.append(f"  Total experiments: {len(df)}")
        report.append(f"  GPU architectures: {len(df['gpu'].unique())}")
        report.append(f"  AI workloads: {len(df['workload'].unique())}")
        report.append(f"  Frequency configurations: {len(df['frequency'].unique())}")
        
        # GPU breakdown
        report.append(f"\nGPU Architecture Breakdown:")
        for gpu in sorted(df['gpu'].unique()):
            gpu_data = df[df['gpu'] == gpu]
            freq_range = f"{gpu_data['frequency'].min()}-{gpu_data['frequency'].max()}"
            report.append(f"  {gpu.upper()}: {len(gpu_data)} experiments, {freq_range} MHz")
        
        # Workload breakdown
        report.append(f"\nAI Workload Breakdown:")
        for workload in sorted(df['workload'].unique()):
            workload_data = df[df['workload'] == workload]
            report.append(f"  {workload}: {len(workload_data)} experiments")
        
        # Energy metrics
        if 'total_energy' in df.columns:
            report.append(f"\nEnergy Consumption Summary:")
            report.append(f"  Mean energy: {df['total_energy'].mean():.2f} J")
            report.append(f"  Energy range: {df['total_energy'].min():.2f} - {df['total_energy'].max():.2f} J")
            report.append(f"  Energy std: {df['total_energy'].std():.2f} J")
        
        # Performance metrics
        if 'execution_time' in df.columns:
            report.append(f"\nPerformance Summary:")
            report.append(f"  Mean execution time: {df['execution_time'].mean():.2f} s")
            report.append(f"  Time range: {df['execution_time'].min():.2f} - {df['execution_time'].max():.2f} s")
        
        # Feature availability
        feature_cols = [col for col in df.columns if 'active' in col or 'ratio' in col or 'util' in col]
        if feature_cols:
            report.append(f"\nFeature Availability:")
            for col in sorted(feature_cols):
                non_null_count = df[col].notna().sum()
                report.append(f"  {col}: {non_null_count}/{len(df)} ({100*non_null_count/len(df):.1f}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Aggregate AI inference energy optimization experimental results"
    )
    parser.add_argument(
        "-d", "--directory", 
        default=".",
        help="Base directory containing results_* directories (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default="aggregated_ai_inference_results.csv",
        help="Output CSV file for aggregated results (default: aggregated_ai_inference_results.csv)"
    )
    parser.add_argument(
        "-b", "--baselines",
        default="performance_baselines.json",
        help="Output JSON file for performance baselines (default: performance_baselines.json)"
    )
    parser.add_argument(
        "-c", "--constraint",
        type=float,
        default=5.0,
        help="Performance degradation constraint percentage (default: 5.0)"
    )
    parser.add_argument(
        "-r", "--report",
        default="aggregation_summary.txt",
        help="Output file for summary report (default: aggregation_summary.txt)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        # Initialize aggregator
        aggregator = AIWorkloadDataAggregator(
            results_base_dir=args.directory,
            logger=logger
        )
        
        # Aggregate all results
        logger.info("Starting data aggregation process")
        df = aggregator.aggregate_all_results()
        
        if df.empty:
            logger.error("No data to aggregate. Check your results directories.")
            return 1
        
        # Establish performance baselines
        baselines = aggregator.establish_performance_baselines(df, args.constraint)
        
        # Save results
        aggregator.save_aggregated_results(df, args.output)
        aggregator.save_baselines(baselines, args.baselines)
        
        # Generate and save summary report
        summary = aggregator.generate_summary_report(df)
        with open(args.report, 'w') as f:
            f.write(summary)
        
        # Display summary
        print(summary)
        
        logger.info("Data aggregation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Data aggregation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
