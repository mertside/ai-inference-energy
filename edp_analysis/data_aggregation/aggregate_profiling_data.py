#!/usr/bin/env python3
"""
Profiling Data Aggregation Pipeline

This script converts time-series profiling data into aggregated statistics suitable for EDP analysis.
It processes DCGMI profiling CSV files and creates per-frequency statistics for energy optimization.

Usage:
    python aggregate_profiling_data.py --input-dir ../../sample-collection-scripts --output aggregated_data.csv
    python aggregate_profiling_data.py --gpu V100 --app LLAMA --run 1 --output llama_data.csv
    python aggregate_profiling_data.py --help

Author: Mert Side
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Comment out the problematic import for now
    # from edp_analysis.visualization.data_preprocessor import ProfilingDataPreprocessor
    HAS_PREPROCESSOR = False
except ImportError:
    HAS_PREPROCESSOR = False
    logger.warning("ProfilingDataPreprocessor not available. Using fallback implementation.")


class ProfilingDataAggregator:
    """Aggregate time-series profiling data for EDP analysis."""
    
    def __init__(self, data_dir: str = "../../sample-collection-scripts"):
        self.data_dir = Path(data_dir)
        self.supported_gpus = ["V100", "A100", "H100"]
        self.supported_apps = ["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER"]
        
        # Standard frequency sets for each GPU - updated to use auto-discovery
        # We'll discover frequencies from the actual data rather than hardcode them
        self.gpu_frequencies = {
            "V100": None,  # Will be auto-discovered
            "A100": None,  # Will be auto-discovered
            "H100": None   # Will be auto-discovered
        }
        
        # DCGMI sampling interval (50ms)
        self.sampling_interval_ms = 50
        self.sampling_interval_s = self.sampling_interval_ms / 1000.0
    
    def find_result_directories(self, gpu: str = None, app: str = None) -> List[Path]:
        """Find all result directories matching the criteria."""
        pattern_parts = ["results"]
        
        if gpu:
            pattern_parts.append(gpu.lower())
        else:
            pattern_parts.append("*")
            
        if app:
            pattern_parts.append(app.lower())
        else:
            pattern_parts.append("*")
            
        pattern_parts.append("job_*")
        pattern = "_".join(pattern_parts)
        
        matching_dirs = list(self.data_dir.glob(pattern))
        logger.info(f"Found {len(matching_dirs)} result directories with pattern: {pattern}")
        
        return sorted(matching_dirs)
    
    def extract_metadata_from_path(self, result_dir: Path) -> Dict[str, str]:
        """Extract GPU, app, and job info from directory path."""
        # Pattern: results_{gpu}_{app}_job_{jobid}
        dir_name = result_dir.name
        parts = dir_name.split('_')
        
        if len(parts) >= 4 and parts[0] == 'results':
            return {
                'gpu': parts[1].upper(),
                'app': parts[2].upper(),
                'job_id': parts[3] if len(parts) > 3 else 'unknown'
            }
        else:
            logger.warning(f"Could not parse directory name: {dir_name}")
            return {'gpu': 'UNKNOWN', 'app': 'UNKNOWN', 'job_id': 'unknown'}
    
    def find_profile_files(self, result_dir: Path, run_number: int = None) -> Dict[int, Path]:
        """Find profile CSV files in a result directory."""
        # Pattern: run_{run_id}_{run_number}_freq_{frequency}_profile.csv
        if run_number:
            pattern = f"run_*_{run_number:02d}_freq_*_profile.csv"
        else:
            pattern = "run_*_freq_*_profile.csv"
        
        matching_files = list(result_dir.glob(pattern))
        
        # Extract frequency from filename
        freq_files = {}
        for file_path in matching_files:
            # Extract run number and frequency from filename using regex
            # Pattern: run_{run_id}_{run_number}_freq_{frequency}_profile.csv
            freq_match = re.search(r'run_\d+_(\d+)_freq_(\d+)_profile\.csv', file_path.name)
            if freq_match:
                file_run_number = int(freq_match.group(1))
                frequency = int(freq_match.group(2))
                
                # Filter by run number if specified
                if run_number is None or file_run_number == run_number:
                    # Store only the first file for each frequency (if multiple runs exist)
                    if frequency not in freq_files:
                        freq_files[frequency] = file_path
        
        logger.debug(f"Found {len(freq_files)} profile files in {result_dir.name} for run {run_number}")
        return freq_files
    
    def load_and_parse_csv(self, csv_path: Path, frequency: int) -> pd.DataFrame:
        """Load and parse a DCGMI CSV file with robust error handling."""
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # Find header line that starts with # Entity
            header_line = None
            data_start_idx = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('# Entity') or line.strip().startswith('#Entity'):
                    header_line = line.strip()[1:].strip()  # Remove the # prefix
                    # Find the next GPU data line
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith('GPU'):
                            data_start_idx = j
                            break
                    break
            
            if not header_line:
                logger.error(f"Could not find header line in {csv_path}")
                return pd.DataFrame()
            
            # Parse column names from header
            columns = header_line.split()
            
            # Read data lines and handle the device name parsing correctly
            data_lines = []
            
            for line in lines[data_start_idx:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith('GPU'):
                    parts = line.split()
                    
                    # Handle device name split issue
                    if len(parts) >= 7:  # Need at least 7 parts for basic data
                        # Reconstruct properly aligned data
                        reconstructed = []
                        
                        # Entity, NVIDX
                        reconstructed.append(parts[0])  # GPU
                        reconstructed.append(parts[1])  # 0
                        
                        # Device name - join parts 2,3,4 as needed
                        # Look for the power value (should be a decimal around 20-500)
                        power_idx = None
                        for idx in range(2, min(len(parts), 8)):
                            try:
                                val = float(parts[idx])
                                if 15.0 <= val <= 500.0:  # Reasonable power range
                                    power_idx = idx
                                    break
                            except ValueError:
                                continue
                        
                        if power_idx is not None:
                            # Everything from index 2 to power_idx-1 is device name
                            device_name = ' '.join(parts[2:power_idx])
                            reconstructed.append(device_name)
                            
                            # Add the rest of the numeric data
                            reconstructed.extend(parts[power_idx:])
                            
                            # Trim to expected number of columns
                            if len(reconstructed) >= len(columns):
                                data_lines.append(reconstructed[:len(columns)])
                        else:
                            # Fallback - assume standard format
                            if len(parts) >= len(columns):
                                data_lines.append(parts[:len(columns)])
            
            if not data_lines:
                logger.error(f"No data lines found in {csv_path}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(data_lines, columns=columns)
            
            # Convert numeric columns (DCGMI metrics) 
            numeric_columns = ['POWER', 'PMLMT', 'TOTEC', 'TMPTR', 'MMTMP', 'GPUTL', 'MCUTL', 
                             'FBTTL', 'FBFRE', 'FBUSD', 'SMCLK', 'MMCLK', 'SACLK', 'MACLK', 
                             'PSTAT', 'GRACT', 'SMACT', 'SMOCC', 'TENSO', 'DRAMA', 'FP64A', 
                             'FP32A', 'FP16A']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add frequency and normalized time
            df['frequency'] = frequency
            df['normalized_time'] = np.linspace(0, 1, len(df))
            
            # Calculate execution time from number of samples
            execution_time = len(df) * self.sampling_interval_s
            df['execution_time'] = execution_time
            
            logger.debug(f"Loaded {len(df)} samples from {csv_path.name} (execution_time: {execution_time:.2f}s)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
            return pd.DataFrame()
    
    def calculate_aggregated_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate aggregated statistics for a single frequency run."""
        if df.empty:
            return {}
        
        # Core metrics for EDP analysis
        stats = {
            'frequency': df['frequency'].iloc[0],
            'execution_time': df['execution_time'].iloc[0],  # Total execution time
            'num_samples': len(df)
        }
        
        # Power metrics
        if 'POWER' in df.columns and not df['POWER'].isna().all():
            power_vals = df['POWER'].dropna()
            stats.update({
                'avg_power': power_vals.mean(),
                'std_power': power_vals.std(),
                'min_power': power_vals.min(),
                'max_power': power_vals.max(),
                'energy': power_vals.mean() * stats['execution_time']  # Energy = Power × Time
            })
        
        # Utilization metrics
        for metric in ['GPUTL', 'MCUTL', 'DRAMA', 'GRACT', 'SMACT']:
            if metric in df.columns and not df[metric].isna().all():
                vals = df[metric].dropna()
                if len(vals) > 0:
                    # Convert to percentage if values are 0-1
                    if vals.max() <= 1.0:
                        vals = vals * 100
                    stats[f'avg_{metric.lower()}'] = vals.mean()
                    stats[f'std_{metric.lower()}'] = vals.std()
        
        # Temperature metrics
        if 'TMPTR' in df.columns and not df['TMPTR'].isna().all():
            temp_vals = df['TMPTR'].dropna()
            stats.update({
                'avg_temp': temp_vals.mean(),
                'max_temp': temp_vals.max()
            })
        
        # Clock frequencies
        for clock_metric in ['SMCLK', 'MMCLK']:
            if clock_metric in df.columns and not df[clock_metric].isna().all():
                clock_vals = df[clock_metric].dropna()
                if len(clock_vals) > 0:
                    stats[f'avg_{clock_metric.lower()}'] = clock_vals.mean()
        
        return stats
    
    def aggregate_single_configuration(self, result_dir: Path, frequency: int, run_number: int = 1) -> Optional[Dict]:
        """Aggregate data for a single configuration (GPU, app, frequency, run)."""
        
        # Extract metadata from path
        metadata = self.extract_metadata_from_path(result_dir)
        
        # Find profile files for this run
        profile_files = self.find_profile_files(result_dir, run_number)
        
        if frequency not in profile_files:
            logger.debug(f"No profile file found for {frequency}MHz in {result_dir.name}")
            return None
        
        # Load and parse the CSV
        df = self.load_and_parse_csv(profile_files[frequency], frequency)
        
        if df.empty:
            logger.warning(f"Empty dataframe for {result_dir.name}, {frequency}MHz")
            return None
        
        # Calculate statistics
        stats = self.calculate_aggregated_stats(df)
        
        if not stats:
            return None
        
        # Add metadata
        stats.update({
            'gpu': metadata['gpu'],
            'application': metadata['app'],
            'run': run_number,
            'job_id': metadata['job_id'],
            'result_dir': result_dir.name
        })
        
        return stats
    
    def aggregate_all_data(self, gpu: str = None, app: str = None, run_number: int = 1) -> pd.DataFrame:
        """Aggregate data for all configurations."""
        
        # Find all result directories
        result_dirs = self.find_result_directories(gpu, app)
        
        if not result_dirs:
            logger.error("No result directories found!")
            return pd.DataFrame()
        
        all_stats = []
        
        for result_dir in result_dirs:
            metadata = self.extract_metadata_from_path(result_dir)
            current_gpu = metadata['gpu']
            
            # Auto-discover frequencies from the actual data
            all_profile_files = self.find_profile_files(result_dir, run_number)
            frequencies = list(all_profile_files.keys())
            
            if not frequencies:
                logger.warning(f"No profile files found in {result_dir.name}")
                continue
                
            logger.info(f"Processing {result_dir.name} with {len(frequencies)} frequencies: {sorted(frequencies)}")
            
            for frequency in frequencies:
                stats = self.aggregate_single_configuration(result_dir, frequency, run_number)
                if stats:
                    all_stats.append(stats)
                    logger.debug(f"✓ {metadata['gpu']} {metadata['app']} {frequency}MHz: "
                               f"E={stats.get('energy', 0):.1f}J, T={stats.get('execution_time', 0):.1f}s")
        
        if not all_stats:
            logger.error("No valid data was aggregated!")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_stats)
        
        # Calculate EDP and ED²P
        if 'energy' in df.columns and 'execution_time' in df.columns:
            df['edp'] = df['energy'] * df['execution_time']
            df['ed2p'] = df['energy'] * (df['execution_time'] ** 2)
        
        logger.info(f"Successfully aggregated {len(df)} configurations")
        
        return df
    
    def save_aggregated_data(self, df: pd.DataFrame, output_path: str):
        """Save aggregated data to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved aggregated data to {output_path}")
        
        # Print summary
        self.print_summary(df)
    
    def print_summary(self, df: pd.DataFrame):
        """Print a summary of the aggregated data."""
        print("\n" + "="*60)
        print("AGGREGATED DATA SUMMARY")
        print("="*60)
        
        if df.empty:
            print("No data available")
            return
        
        print(f"Total configurations: {len(df)}")
        
        if 'gpu' in df.columns:
            print(f"GPUs: {sorted(df['gpu'].unique())}")
        
        if 'application' in df.columns:
            print(f"Applications: {sorted(df['application'].unique())}")
        
        if 'frequency' in df.columns:
            print(f"Frequencies: {sorted(df['frequency'].unique())} MHz")
        
        if 'run' in df.columns:
            print(f"Runs: {sorted(df['run'].unique())}")
        
        # Energy and time ranges
        if 'energy' in df.columns:
            print(f"Energy range: {df['energy'].min():.1f} - {df['energy'].max():.1f} J")
        
        if 'execution_time' in df.columns:
            print(f"Execution time range: {df['execution_time'].min():.1f} - {df['execution_time'].max():.1f} s")
        
        if 'avg_power' in df.columns:
            print(f"Power range: {df['avg_power'].min():.1f} - {df['avg_power'].max():.1f} W")
        
        print()


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Aggregate profiling data for EDP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Aggregate all data
    python aggregate_profiling_data.py --input-dir ../../sample-collection-scripts --output aggregated_data.csv
    
    # Aggregate specific GPU and application
    python aggregate_profiling_data.py --gpu V100 --app LLAMA --output llama_v100_data.csv
    
    # Aggregate with specific run number
    python aggregate_profiling_data.py --run 2 --output run2_data.csv
        """
    )
    
    parser.add_argument("--input-dir", type=str, default="../../sample-collection-scripts",
                       help="Directory containing result folders (default: ../../sample-collection-scripts)")
    parser.add_argument("--output", type=str, default="aggregated_profiling_data.csv",
                       help="Output CSV file path (default: aggregated_profiling_data.csv)")
    parser.add_argument("--gpu", type=str, choices=["V100", "A100", "H100"],
                       help="Filter by GPU type")
    parser.add_argument("--app", type=str, choices=["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER"],
                       help="Filter by application")
    parser.add_argument("--run", type=int, default=1,
                       help="Run number to process (default: 1)")
    
    args = parser.parse_args()
    
    # Initialize aggregator
    aggregator = ProfilingDataAggregator(args.input_dir)
    
    logger.info(f"🚀 Starting data aggregation from {args.input_dir}")
    
    # Aggregate data
    df = aggregator.aggregate_all_data(
        gpu=args.gpu,
        app=args.app,
        run_number=args.run
    )
    
    if df.empty:
        logger.error("No data was aggregated!")
        return 1
    
    # Save results
    aggregator.save_aggregated_data(df, args.output)
    
    logger.info("✅ Data aggregation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
