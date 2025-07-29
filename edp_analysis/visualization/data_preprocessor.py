"""
Data Preprocessing Module for Profiling Visualization

This module provides data preprocessing utilities specifically designed for
the AI Inference Energy Profiling Framework's DCGMI profiling data.

Key Features:
- DCGMI CSV data loading and processing
- Field mapping from numeric IDs to readable names
- Time-series data preparation
- Multi-experiment data aggregation
- Derived metrics calculation
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProfilingDataPreprocessor:
    """Preprocess profiling data for visualization."""
    
    # Mapping from DCGMI field numbers to readable names
    DCGMI_FIELD_MAPPING = {
        # GPU identification and basic info
        52: "gpu_index",
        50: "gpu_name", 
        
        # Power metrics
        155: "power_draw",
        160: "power_limit",
        150: "gpu_temperature",
        156: "total_energy_consumption_mj",
        140: "total_energy_consumption",
        
        # Utilization metrics
        203: "gpu_utilization",
        204: "memory_utilization",
        
        # Memory metrics
        250: "memory_total",
        251: "memory_free", 
        252: "memory_used",
        
        # Clock frequencies
        100: "sm_clock",
        101: "memory_clock",
        110: "graphics_clock",
        111: "memory_app_clock",
        
        # Performance state
        190: "pstate",
        
        # Activity metrics
        1001: "graphics_pipe_active",
        1002: "sm_active", 
        1003: "sm_occupancy",
        1004: "tensor_pipe_active",
        1005: "dram_active",
        1006: "fp64_active",
        1007: "fp32_active",
        1008: "fp16_active"
    }
    
    @classmethod
    def load_and_process_csv(
        cls, 
        csv_path: str,
        frequency: Optional[int] = None,
        app_name: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Load and process a DCGMI CSV file."""
        try:
            # Read CSV, skipping comment lines that start with #
            df = pd.read_csv(csv_path, comment='#', skipinitialspace=True)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Add metadata if provided
            if frequency is not None:
                df['frequency'] = frequency
            if app_name is not None:
                df['app_name'] = app_name
            if run_id is not None:
                df['run_id'] = run_id
            
            # Add timestamp if not present (assuming regular sampling)
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(
                    start='2024-01-01', 
                    periods=len(df), 
                    freq='50ms'  # Default 50ms sampling
                )
            else:
                # Convert timestamp to datetime if it's not already
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Rename columns using field mapping
            df = cls.rename_dcgmi_columns(df)
            
            # Calculate derived metrics
            df = cls.calculate_derived_metrics(df)
            
            logger.info(f"Loaded {len(df)} rows from {csv_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
            raise
    
    @classmethod
    def rename_dcgmi_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Rename DCGMI field numbers to readable names."""
        df_renamed = df.copy()
        
        # Map numeric column names to readable names
        column_mapping = {}
        for col in df.columns:
            if str(col).isdigit():
                field_id = int(col)
                if field_id in cls.DCGMI_FIELD_MAPPING:
                    column_mapping[col] = cls.DCGMI_FIELD_MAPPING[field_id]
        
        df_renamed = df_renamed.rename(columns=column_mapping)
        return df_renamed
    
    @classmethod
    def calculate_derived_metrics(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics from raw profiling data."""
        df_derived = df.copy()
        
        # Power efficiency metrics
        if all(col in df.columns for col in ['power_draw', 'sm_clock']):
            df_derived['power_per_mhz'] = df_derived['power_draw'] / (df_derived['sm_clock'] + 1e-6)
        
        # Memory metrics
        if all(col in df.columns for col in ['memory_total', 'memory_used']):
            df_derived['memory_usage_percent'] = (df_derived['memory_used'] / df_derived['memory_total']) * 100
        
        # Activity ratios
        activity_cols = ['graphics_pipe_active', 'sm_active', 'tensor_pipe_active', 'dram_active']
        available_activity_cols = [col for col in activity_cols if col in df.columns]
        if available_activity_cols:
            df_derived['total_activity'] = df_derived[available_activity_cols].sum(axis=1)
        
        # Compute intensity
        if all(col in df.columns for col in ['fp32_active', 'fp64_active']):
            df_derived['compute_intensity'] = df_derived['fp32_active'] + df_derived['fp64_active'] * 2
        
        # Thermal efficiency
        if all(col in df.columns for col in ['power_draw', 'gpu_temperature']):
            df_derived['thermal_efficiency'] = df_derived['power_draw'] / (df_derived['gpu_temperature'] + 1e-6)
        
        return df_derived
    
    @classmethod
    def load_multiple_experiments(
        cls,
        file_paths: Dict[str, str],  # {experiment_name: file_path}
        frequencies: Optional[Dict[str, int]] = None,
        app_names: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Load and combine multiple experiment CSV files."""
        dfs = []
        
        for exp_name, file_path in file_paths.items():
            freq = frequencies.get(exp_name) if frequencies else None
            app = app_names.get(exp_name) if app_names else exp_name
            
            df = cls.load_and_process_csv(file_path, frequency=freq, app_name=app, run_id=exp_name)
            df['experiment'] = exp_name
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} experiments into DataFrame with {len(combined_df)} total rows")
        
        return combined_df
    
    @classmethod
    def extract_frequency_from_filename(cls, filename: str) -> Optional[int]:
        """Extract frequency from filename pattern like 'run_1_01_freq_1410_profile.csv'."""
        match = re.search(r'freq_(\d+)', filename)
        return int(match.group(1)) if match else None
    
    @classmethod
    def load_result_directory(
        cls,
        result_dir: str,
        pattern: str = "*_profile.csv",
        app_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Load all profiling CSV files from a result directory."""
        result_path = Path(result_dir)
        csv_files = list(result_path.glob(pattern))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {result_dir} with pattern {pattern}")
            return pd.DataFrame()
        
        dfs = []
        for csv_file in csv_files:
            frequency = cls.extract_frequency_from_filename(csv_file.name)
            run_id = csv_file.stem
            
            df = cls.load_and_process_csv(
                str(csv_file), 
                frequency=frequency, 
                app_name=app_name,
                run_id=run_id
            )
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(dfs)} files from {result_dir}, total {len(combined_df)} rows")
        
        return combined_df
    
    @classmethod
    def prepare_for_time_series(
        cls,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        sort_by_time: bool = True
    ) -> pd.DataFrame:
        """Prepare data for time-series visualization."""
        df_prepared = df.copy()
        
        # Ensure timestamp is datetime
        if time_col in df_prepared.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_prepared[time_col]):
                df_prepared[time_col] = pd.to_datetime(df_prepared[time_col])
        
        # Sort by time if requested
        if sort_by_time and time_col in df_prepared.columns:
            df_prepared = df_prepared.sort_values(time_col)
        
        # Add relative time (seconds from start)
        if time_col in df_prepared.columns:
            df_prepared['relative_time_seconds'] = (
                df_prepared[time_col] - df_prepared[time_col].min()
            ).dt.total_seconds()
        
        return df_prepared


def create_synthetic_profiling_data(
    frequencies: List[int] = [900, 1200, 1410],
    duration_seconds: int = 10,
    sampling_rate_ms: int = 50,
    app_name: str = "Demo Application"
) -> pd.DataFrame:
    """Create synthetic profiling data for demonstration purposes."""
    np.random.seed(42)
    data = []
    
    for freq in frequencies:
        n_samples = int(duration_seconds * 1000 / sampling_rate_ms)
        time_points = np.linspace(0, duration_seconds, n_samples)
        
        # Simulate realistic GPU behavior over time
        base_power = 150 + (freq - 900) * 0.1
        power_variation = 20 * np.sin(0.5 * time_points) + 10 * np.random.normal(0, 1, n_samples)
        power_draw = np.clip(base_power + power_variation, 100, 300)
        
        # GPU utilization with some workload phases
        utilization_base = 85 + 10 * np.sin(0.3 * time_points)
        gpu_utilization = np.clip(utilization_base + 5 * np.random.normal(0, 1, n_samples), 0, 100)
        
        # Temperature follows power with some lag
        gpu_temperature = 30 + power_draw * 0.2 + 5 * np.random.normal(0, 1, n_samples)
        gpu_temperature = np.clip(gpu_temperature, 30, 90)
        
        # Memory utilization
        memory_utilization = np.random.uniform(60, 90, n_samples)
        
        # Activity metrics
        sm_active = np.random.uniform(0.7, 0.95, n_samples)
        dram_active = np.random.uniform(0.3, 0.8, n_samples)
        
        for i in range(n_samples):
            data.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(seconds=time_points[i]),
                'frequency': freq,
                'power_draw': power_draw[i],
                'gpu_utilization': gpu_utilization[i],
                'gpu_temperature': gpu_temperature[i],
                'memory_utilization': memory_utilization[i],
                'sm_active': sm_active[i],
                'dram_active': dram_active[i],
                'app_name': app_name,
                'sm_clock': freq,
                'memory_clock': 1215,  # Typical memory clock
                'memory_usage_percent': memory_utilization[i],
                'power_per_mhz': power_draw[i] / freq
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic profiling data: {len(df)} samples across {len(frequencies)} frequencies")
    return df
