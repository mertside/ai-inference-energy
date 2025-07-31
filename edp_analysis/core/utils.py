"""
Core Utilities for EDP Analysis Framework

This module provides common utilities, helper functions, and shared functionality
used across the EDP analysis framework.

Author: Mert Side
"""

import pandas as pd
import numpy as np
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration for the framework.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    
    logger.info(f"Logging configured: level={level}, file={log_file}")


def load_config(config_path: Union[str, Path]) -> Dict:
    """
    Load configuration from YAML file with error handling.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def save_config(config: Dict, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=2, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")


def save_results(data: Dict, 
                file_path: Union[str, Path],
                format: str = "json") -> None:
    """
    Save analysis results to file with metadata.
    
    Args:
        data: Data to save
        file_path: Output file path
        format: File format ("json" or "yaml")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    if isinstance(data, dict):
        data['_metadata'] = {
            'saved_timestamp': datetime.now().isoformat(),
            'framework_version': get_framework_version(),
            'file_format': format
        }
    
    if format.lower() == "json":
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format.lower() == "yaml":
        with open(file_path, 'w') as f:
            yaml.dump(data, f, indent=2, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {file_path} ({format} format)")


def load_results(file_path: Union[str, Path]) -> Dict:
    """
    Load analysis results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Loaded data dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Results loaded from {file_path}")
    return data


def get_framework_version() -> str:
    """Get framework version string."""
    return "2.0.0-refactored"


def calculate_data_hash(data: pd.DataFrame) -> str:
    """
    Calculate hash of DataFrame for data integrity checking.
    
    Args:
        data: DataFrame to hash
        
    Returns:
        SHA256 hash string
    """
    # Create string representation of data
    data_string = data.to_string()
    
    # Calculate hash
    hash_object = hashlib.sha256(data_string.encode())
    return hash_object.hexdigest()


def validate_gpu_name(gpu: str) -> str:
    """
    Validate and normalize GPU name.
    
    Args:
        gpu: GPU name to validate
        
    Returns:
        Normalized GPU name
    """
    gpu_lower = gpu.lower().strip()
    
    # Known GPU mappings
    gpu_mappings = {
        'a100': 'A100',
        'v100': 'V100',
        'h100': 'H100',
        'tesla_a100': 'A100',
        'tesla_v100': 'V100',
        'tesla_h100': 'H100'
    }
    
    # Try direct mapping
    if gpu_lower in gpu_mappings:
        return gpu_mappings[gpu_lower]
    
    # Try partial matches
    for key, value in gpu_mappings.items():
        if key in gpu_lower or gpu_lower in key:
            return value
    
    # Return original if no mapping found
    logger.warning(f"Unknown GPU type: {gpu}. Using as-is.")
    return gpu.upper()


def validate_application_name(application: str) -> str:
    """
    Validate and normalize application name.
    
    Args:
        application: Application name to validate
        
    Returns:
        Normalized application name
    """
    app_lower = application.lower().strip()
    
    # Known application mappings
    app_mappings = {
        'llama': 'LLAMA',
        'stable_diffusion': 'STABLEDIFFUSION',
        'stable-diffusion': 'STABLEDIFFUSION',
        'stablediffusion': 'STABLEDIFFUSION',
        'vit': 'VIT',
        'vision_transformer': 'VIT',
        'whisper': 'WHISPER',
        'lstm': 'LSTM'
    }
    
    # Try direct mapping
    if app_lower in app_mappings:
        return app_mappings[app_lower]
    
    # Try partial matches
    for key, value in app_mappings.items():
        if key in app_lower or app_lower in key:
            return value
    
    # Return normalized original
    return application.upper()


def create_configuration_key(gpu: str, application: str) -> str:
    """
    Create standardized configuration key.
    
    Args:
        gpu: GPU name
        application: Application name
        
    Returns:
        Standardized configuration key
    """
    gpu_norm = validate_gpu_name(gpu)
    app_norm = validate_application_name(application)
    return f"{gpu_norm}+{app_norm}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format percentage value for display.
    
    Args:
        value: Percentage value (0-100)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimal_places}f}%"


def format_frequency(frequency: float) -> str:
    """
    Format frequency value for display.
    
    Args:
        frequency: Frequency in MHz
        
    Returns:
        Formatted frequency string
    """
    return f"{int(frequency)} MHz"


def format_time(time_seconds: float) -> str:
    """
    Format time value for display.
    
    Args:
        time_seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if time_seconds < 1:
        return f"{time_seconds*1000:.1f} ms"
    elif time_seconds < 60:
        return f"{time_seconds:.2f} s"
    else:
        minutes = int(time_seconds // 60)
        seconds = time_seconds % 60
        return f"{minutes}m {seconds:.1f}s"


def format_energy(energy_joules: float) -> str:
    """
    Format energy value for display.
    
    Args:
        energy_joules: Energy in joules
        
    Returns:
        Formatted energy string
    """
    if energy_joules < 1:
        return f"{energy_joules*1000:.1f} mJ"
    elif energy_joules < 1000:
        return f"{energy_joules:.1f} J"
    else:
        return f"{energy_joules/1000:.2f} kJ"


def format_power(power_watts: float) -> str:
    """
    Format power value for display.
    
    Args:
        power_watts: Power in watts
        
    Returns:
        Formatted power string
    """
    if power_watts < 1:
        return f"{power_watts*1000:.0f} mW"
    else:
        return f"{power_watts:.1f} W"


def create_summary_table(results: Dict, 
                        columns: List[str],
                        format_functions: Optional[Dict[str, callable]] = None) -> pd.DataFrame:
    """
    Create formatted summary table from results.
    
    Args:
        results: Results dictionary
        columns: Columns to include in table
        format_functions: Optional formatting functions for columns
        
    Returns:
        Formatted DataFrame
    """
    if format_functions is None:
        format_functions = {}
    
    # Extract data for table
    table_data = []
    for config_key, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            row = {'Configuration': config_key}
            for col in columns:
                value = result.get(col, 'N/A')
                if col in format_functions and value != 'N/A':
                    value = format_functions[col](value)
                row[col] = value
            table_data.append(row)
    
    return pd.DataFrame(table_data)


def validate_data_ranges(data: pd.DataFrame, 
                        column_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
    """
    Validate that data values are within expected ranges.
    
    Args:
        data: DataFrame to validate
        column_ranges: Dictionary mapping column names to (min, max) ranges
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    for column, (min_val, max_val) in column_ranges.items():
        if column in data.columns:
            in_range = data[column].between(min_val, max_val, inclusive='both').all()
            validation_results[f"{column}_in_range"] = in_range
            
            if not in_range:
                out_of_range = data[~data[column].between(min_val, max_val, inclusive='both')]
                logger.warning(f"{column} values out of range [{min_val}, {max_val}]: {len(out_of_range)} records")
        else:
            validation_results[f"{column}_in_range"] = False
            logger.warning(f"Column {column} not found in data")
    
    return validation_results


def calculate_statistics(data: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a data series.
    
    Args:
        data: Pandas Series
        
    Returns:
        Dictionary of statistics
    """
    return {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
        'cv': data.std() / data.mean() if data.mean() != 0 else np.inf
    }


def create_deployment_summary(optimization_results: Dict) -> str:
    """
    Create human-readable deployment summary.
    
    Args:
        optimization_results: Results from frequency optimization
        
    Returns:
        Formatted summary string
    """
    summary_lines = []
    summary_lines.append("🚀 GPU Frequency Optimization Deployment Summary")
    summary_lines.append("=" * 60)
    
    if 'summary' in optimization_results:
        summary = optimization_results['summary']
        summary_lines.append(f"Total Configurations: {summary.get('successful_optimizations', 'N/A')}")
        
        if 'energy_savings' in summary:
            energy_stats = summary['energy_savings']
            summary_lines.append(f"Energy Savings: {energy_stats['mean']:.1f}% ± {energy_stats['std']:.1f}%")
        
        if 'performance_penalties' in summary:
            perf_stats = summary['performance_penalties']
            summary_lines.append(f"Performance Penalty: {perf_stats['mean']:.1f}% ± {perf_stats['std']:.1f}%")
    
    summary_lines.append("")
    summary_lines.append("📊 Configuration Details:")
    summary_lines.append("-" * 40)
    
    if 'configurations' in optimization_results:
        for config_key, result in optimization_results['configurations'].items():
            if 'error' not in result:
                freq = result.get('optimal_frequency', 'N/A')
                energy = result.get('configuration_summary', {}).get('energy_savings_percent', 'N/A')
                penalty = result.get('configuration_summary', {}).get('performance_penalty_percent', 'N/A')
                
                summary_lines.append(f"  {config_key}:")
                summary_lines.append(f"    Frequency: {format_frequency(freq) if freq != 'N/A' else 'N/A'}")
                summary_lines.append(f"    Energy Savings: {format_percentage(energy) if energy != 'N/A' else 'N/A'}")
                summary_lines.append(f"    Performance: {format_percentage(abs(penalty)) if penalty != 'N/A' else 'N/A'} penalty")
                summary_lines.append("")
    
    return "\n".join(summary_lines)


def get_file_timestamp() -> str:
    """Get timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Progress"):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
        
    def finish(self) -> None:
        """Mark progress as complete."""
        logger.info(f"{self.description}: Complete! ({self.total}/{self.total})")


def retry_on_failure(func, max_attempts: int = 3, delay: float = 1.0):
    """
    Retry decorator for functions that might fail.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Returns:
        Decorated function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    
    return wrapper
