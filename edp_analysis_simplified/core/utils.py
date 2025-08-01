"""
Core utilities and helper functions for the frequency optimization framework.
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path


def create_output_directory(output_path: str) -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to output directory
        
    Returns:
        Absolute path to created directory
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def save_json_results(data: Dict[str, Any], output_path: str, filename: str = "results.json") -> str:
    """
    Save results to JSON file.
    
    Args:
        data: Data to save
        output_path: Output directory path
        filename: JSON filename
        
    Returns:
        Path to saved file
    """
    file_path = os.path.join(output_path, filename)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    return file_path


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage value for display."""
    return f"{value:.{decimals}f}%"


def format_frequency(value: float) -> str:
    """Format frequency value for display."""
    return f"{int(value)}MHz"


def format_efficiency_ratio(value: float, decimals: int = 1) -> str:
    """Format efficiency ratio for display."""
    return f"{value:.{decimals}f}:1"


def print_section_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{title}")
    print("=" * width)


def print_subsection_header(title: str, width: int = 50) -> None:
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * width)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"📊 {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def get_memory_frequency(gpu_type: str) -> int:
    """
    Get default memory frequency for GPU type.
    
    Args:
        gpu_type: GPU type ('A100', 'V100')
        
    Returns:
        Memory frequency in MHz
    """
    gpu_mem_frequencies = {
        'A100': 1215,
        'V100': 877
    }
    return gpu_mem_frequencies.get(gpu_type.upper(), 1215)


def validate_frequency_range(frequency: float, gpu_type: str) -> bool:
    """
    Validate that frequency is within reasonable range for GPU type.
    
    Args:
        frequency: GPU frequency in MHz
        gpu_type: GPU type
        
    Returns:
        True if frequency is valid
    """
    frequency_ranges = {
        'A100': (400, 1400),  # Reasonable range for A100
        'V100': (400, 1300)   # Reasonable range for V100
    }
    
    min_freq, max_freq = frequency_ranges.get(gpu_type.upper(), (400, 1500))
    return min_freq <= frequency <= max_freq
