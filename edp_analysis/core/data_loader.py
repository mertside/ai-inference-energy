"""
Unified Data Loader for EDP Analysis Framework

This module provides a standardized interface for loading and validating
profiling data from various sources, with built-in cold start handling
and data quality validation.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml

logger = logging.getLogger(__name__)


class ProfilingDataLoader:
    """
    Unified data loader for GPU profiling data with quality validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data loader with configuration."""
        self.config = self._load_config(config_path)
        self.data = None
        self.metadata = {}
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if file not found."""
        return {
            'data': {
                'exclude_cold_start': True,
                'default_run': 2,
                'validation_threshold': 0.05,
                'min_data_points': 5,
                'max_cv_threshold': 0.3
            }
        }
    
    def load_aggregated_data(self, 
                           data_path: str,
                           exclude_cold_start: Optional[bool] = None,
                           run_number: Optional[int] = None) -> pd.DataFrame:
        """
        Load aggregated profiling data with optional cold start exclusion.
        
        Args:
            data_path: Path to aggregated CSV file
            exclude_cold_start: Whether to exclude run 1 (cold start) data
            run_number: Specific run number to load (overrides exclude_cold_start)
            
        Returns:
            Loaded and validated DataFrame
        """
        logger.info(f"Loading data from: {data_path}")
        
        # Use config defaults if not specified
        if exclude_cold_start is None:
            exclude_cold_start = self.config['data']['exclude_cold_start']
        if run_number is None and exclude_cold_start:
            run_number = self.config['data']['default_run']
        
        # Load data
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(data)} records from {data_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        # Filter by run number if specified
        if run_number is not None:
            if 'run' in data.columns:
                original_size = len(data)
                data = data[data['run'] == run_number]
                logger.info(f"Filtered to run {run_number}: {len(data)} records (was {original_size})")
            else:
                logger.warning("Run column not found, cannot filter by run number")
        
        # Validate data quality
        data = self._validate_data_quality(data)
        
        # Store metadata
        self._update_metadata(data, data_path, run_number)
        
        self.data = data
        return data
    
    def _validate_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and remove outliers."""
        logger.info("Validating data quality...")
        
        # Check required columns
        required_columns = ['gpu', 'application', 'frequency', 'execution_time', 'power']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate derived metrics if not present
        if 'energy' not in data.columns:
            data['energy'] = data['power'] * data['execution_time']
            logger.info("Calculated energy from power and execution time")
        
        if 'edp' not in data.columns:
            data['edp'] = data['energy'] * data['execution_time']
            logger.info("Calculated EDP from energy and execution time")
        
        # Remove rows with invalid values
        original_size = len(data)
        data = data.dropna(subset=required_columns)
        data = data[data['execution_time'] > 0]
        data = data[data['power'] > 0]
        data = data[data['frequency'] > 0]
        
        if len(data) < original_size:
            logger.warning(f"Removed {original_size - len(data)} invalid records")
        
        # Check minimum data points per configuration
        min_points = self.config['data']['min_data_points']
        config_counts = data.groupby(['gpu', 'application']).size()
        insufficient_configs = config_counts[config_counts < min_points]
        
        if len(insufficient_configs) > 0:
            logger.warning(f"Configurations with <{min_points} data points: {list(insufficient_configs.index)}")
        
        # Validate coefficient of variation for key metrics
        max_cv = self.config['data']['max_cv_threshold']
        for config, group in data.groupby(['gpu', 'application', 'frequency']):
            for metric in ['execution_time', 'power']:
                if len(group) > 1:
                    cv = group[metric].std() / group[metric].mean()
                    if cv > max_cv:
                        logger.warning(f"High CV ({cv:.3f}) for {config} - {metric}")
        
        return data
    
    def _update_metadata(self, data: pd.DataFrame, data_path: str, run_number: Optional[int]):
        """Update metadata about loaded data."""
        self.metadata = {
            'source_file': data_path,
            'run_number': run_number,
            'total_records': len(data),
            'configurations': len(data.groupby(['gpu', 'application'])),
            'gpus': list(data['gpu'].unique()),
            'applications': list(data['application'].unique()),
            'frequency_range': {
                'min': data['frequency'].min(),
                'max': data['frequency'].max(),
                'unique_count': data['frequency'].nunique()
            },
            'loaded_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_configuration_data(self, gpu: str, application: str) -> pd.DataFrame:
        """Get data for a specific GPU-application configuration."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_aggregated_data() first.")
        
        config_data = self.data[
            (self.data['gpu'] == gpu) & 
            (self.data['application'] == application)
        ]
        
        if len(config_data) == 0:
            raise ValueError(f"No data found for {gpu}+{application}")
        
        return config_data.sort_values('frequency')
    
    def get_available_configurations(self) -> List[Tuple[str, str]]:
        """Get list of available GPU-application configurations."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_aggregated_data() first.")
        
        configs = self.data.groupby(['gpu', 'application']).size().index.tolist()
        return configs
    
    def get_baseline_performance(self, gpu: str, application: str) -> Dict[str, float]:
        """Get baseline (maximum frequency) performance metrics."""
        config_data = self.get_configuration_data(gpu, application)
        
        # Get maximum frequency data
        max_freq = config_data['frequency'].max()
        baseline = config_data[config_data['frequency'] == max_freq].iloc[0]
        
        return {
            'frequency': baseline['frequency'],
            'execution_time': baseline['execution_time'],
            'power': baseline['power'],
            'energy': baseline['energy'],
            'edp': baseline['edp']
        }
    
    def validate_data_consistency(self) -> Dict[str, bool]:
        """Validate data consistency across runs and configurations."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_aggregated_data() first.")
        
        results = {}
        
        # Check frequency consistency across configurations
        freq_by_config = self.data.groupby(['gpu', 'application'])['frequency'].apply(set)
        all_frequencies = set()
        for freqs in freq_by_config:
            all_frequencies.update(freqs)
        
        results['frequency_consistency'] = all(
            len(freqs.intersection(all_frequencies)) >= len(all_frequencies) * 0.8
            for freqs in freq_by_config
        )
        
        # Check for missing data points
        expected_combinations = len(self.get_available_configurations()) * len(all_frequencies)
        actual_combinations = len(self.data.groupby(['gpu', 'application', 'frequency']))
        results['completeness'] = actual_combinations >= expected_combinations * 0.9
        
        # Check data ranges are reasonable
        results['reasonable_ranges'] = (
            self.data['execution_time'].between(0.1, 10000).all() and
            self.data['power'].between(10, 1000).all() and
            self.data['frequency'].between(500, 2000).all()
        )
        
        return results
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_aggregated_data() first.")
        
        summary = {
            'metadata': self.metadata,
            'data_quality': self.validate_data_consistency(),
            'statistics': {
                'execution_time': self.data['execution_time'].describe().to_dict(),
                'power': self.data['power'].describe().to_dict(),
                'energy': self.data['energy'].describe().to_dict(),
                'edp': self.data['edp'].describe().to_dict()
            },
            'configurations': {
                str(config): len(group) 
                for config, group in self.data.groupby(['gpu', 'application'])
            }
        }
        
        return summary


def load_profiling_data(data_path: str, 
                       config_path: Optional[str] = None,
                       **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load profiling data with metadata.
    
    Args:
        data_path: Path to aggregated CSV file
        config_path: Path to configuration file
        **kwargs: Additional arguments for data loading
        
    Returns:
        Tuple of (DataFrame, metadata dictionary)
    """
    loader = ProfilingDataLoader(config_path)
    data = loader.load_aggregated_data(data_path, **kwargs)
    metadata = loader.get_summary_statistics()
    
    return data, metadata


def validate_profiling_data(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Quick validation function for profiling data.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        Dictionary of validation results
    """
    loader = ProfilingDataLoader()
    loader.data = data
    return loader.validate_data_consistency()
