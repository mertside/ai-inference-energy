#!/usr/bin/env python3
"""
GPU Profiling Metric Visualization Tool

This script plots any profiling metric against normalized time for specified
GPU types, applications, frequencies, and run numbers.

Usage:
    python plot_metric_vs_time.py --gpu V100 --app LLAMA --frequencies 510,960,1380 --metric GPUTL --run 2
    python plot_metric_vs_time.py --gpu A100 --app VIT --frequencies 1200,1410 --metric POWER
    python plot_metric_vs_time.py --help

Author: AI Inference Energy Profiling Framework
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.error("Matplotlib not available. Please install with: pip install matplotlib")
    sys.exit(1)


class ProfilingDataLoader:
    """Load and process GPU profiling data from job result directories."""
    
    def __init__(self, data_dir: str = "../../sample-collection-scripts"):
        self.data_dir = Path(data_dir)
        
        # Supported configurations
        self.supported_gpus = ["V100", "A100", "H100"]
        self.supported_apps = ["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER", "LSTM"]
        
        # Application name mapping (handle variations)
        self.app_mapping = {
            "LLAMA": ["llama"],
            "VIT": ["vit"],
            "STABLEDIFFUSION": ["stablediffusion"],
            "WHISPER": ["whisper"], 
            "LSTM": ["lstm"]
        }
    
    def find_result_directory(self, gpu: str, app: str) -> Optional[Path]:
        """Find the result directory for given GPU and application."""
        gpu_lower = gpu.lower()
        app_lower = app.lower()
        
        # Look for result directories matching pattern: results_{gpu}_{app}_job_{jobid}
        pattern = f"results_{gpu_lower}_{app_lower}_job_*"
        
        matching_dirs = list(self.data_dir.glob(pattern))
        
        if not matching_dirs:
            logger.error(f"No result directory found for {gpu} + {app}")
            logger.info(f"Searched for pattern: {pattern} in {self.data_dir}")
            self._list_available_directories()
            return None
        
        if len(matching_dirs) > 1:
            logger.warning(f"Multiple directories found for {gpu} + {app}, using: {matching_dirs[0]}")
        
        logger.info(f"Found result directory: {matching_dirs[0]}")
        return matching_dirs[0]
    
    def _list_available_directories(self):
        """List available result directories for debugging."""
        result_dirs = list(self.data_dir.glob("results_*"))
        if result_dirs:
            logger.info("Available result directories:")
            for d in sorted(result_dirs):
                logger.info(f"  {d.name}")
        else:
            logger.info(f"No result directories found in {self.data_dir}")
    
    def find_profile_files(self, result_dir: Path, frequencies: List[int], run_number: int) -> Dict[int, Path]:
        """Find profile CSV files for specified frequencies and run number."""
        found_files = {}
        
        for freq in frequencies:
            # Pattern: run_{run_number}_{run_index}_freq_{frequency}_profile.csv
            pattern = f"run_*_{run_number:02d}_freq_{freq}_profile.csv"
            matching_files = list(result_dir.glob(pattern))
            
            if matching_files:
                found_files[freq] = matching_files[0]
                logger.debug(f"Found file for {freq}MHz: {matching_files[0].name}")
            else:
                logger.warning(f"No profile file found for frequency {freq}MHz, run {run_number}")
                logger.debug(f"Searched for pattern: {pattern}")
        
        if not found_files:
            logger.error("No profile files found for any requested frequency")
            self._list_available_files(result_dir)
        
        return found_files
    
    def _list_available_files(self, result_dir: Path):
        """List available CSV files for debugging."""
        csv_files = list(result_dir.glob("*_profile.csv"))
        if csv_files:
            logger.info(f"Available profile files in {result_dir.name}:")
            for f in sorted(csv_files)[:10]:  # Show first 10
                logger.info(f"  {f.name}")
            if len(csv_files) > 10:
                logger.info(f"  ... and {len(csv_files) - 10} more files")
        
    def load_csv_file(self, csv_path: Path, frequency: int) -> pd.DataFrame:
        """Load and parse a single DCGMI CSV file."""
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
                    # Expected structure: GPU 0 0 Tesla V100-PCIE-32GB 25.584 250.000 ...
                    # But device name gets split into: ['GPU', '0', '0', 'Tesla', 'V100-PCIE-32GB', '25.584', '250.000', ...]
                    
                    if len(parts) >= 7:  # Need at least 7 parts for basic data
                        # Reconstruct properly aligned data
                        reconstructed = []
                        
                        # Entity, NVIDX
                        reconstructed.append(parts[0])  # GPU
                        reconstructed.append(parts[1])  # 0
                        
                        # Device name - join parts 2,3,4 as needed
                        # Look for the power value (should be a decimal around 20-50)
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
            
            # Add actual timestamp (for reference, 50ms intervals)
            df['timestamp'] = pd.date_range(
                start='2024-01-01', 
                periods=len(df), 
                freq='50ms'
            )
            
            logger.info(f"Loaded {len(df)} samples from {csv_path.name}")
            
            # Log some basic stats for verification
            if 'POWER' in df.columns and not df['POWER'].isna().all():
                power_range = f"{df['POWER'].min():.1f}-{df['POWER'].max():.1f}W"
                logger.debug(f"Power range: {power_range}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
            return pd.DataFrame()
    
    def load_data(self, gpu: str, app: str, frequencies: List[int], run_number: int) -> pd.DataFrame:
        """Load profiling data for specified parameters."""
        
        # Find result directory
        result_dir = self.find_result_directory(gpu, app)
        if not result_dir:
            return pd.DataFrame()
        
        # Find profile files
        profile_files = self.find_profile_files(result_dir, frequencies, run_number)
        if not profile_files:
            return pd.DataFrame()
        
        # Load all files and combine
        dfs = []
        for freq, file_path in profile_files.items():
            df = self.load_csv_file(file_path, freq)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            logger.error("No data was successfully loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df)} total samples across {len(dfs)} frequencies")
        
        return combined_df


class MetricPlotter:
    """Create metric vs time plots with multiple frequency overlays."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = plt.cm.viridis  # Use viridis colormap for frequency colors 
        # plt.cm.plasma  # Use plasma colormap for frequency colors
                    

    def plot_metric_vs_time(
        self, 
        df: pd.DataFrame, 
        metric: str,
        gpu: str,
        app: str,
        run_number: int,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Plot metric vs normalized time for multiple frequencies."""
        
        if metric not in df.columns:
            available_metrics = [col for col in df.columns if col not in ['Entity', 'NVIDX', 'DVNAM', 'frequency', 'normalized_time', 'timestamp']]
            logger.error(f"Metric '{metric}' not found in data")
            logger.info(f"Available metrics: {available_metrics}")
            return None
        
        if df.empty:
            logger.error("No data to plot")
            return None
        
        # Check if metric has valid data
        if df[metric].isna().all():
            logger.error(f"All values for metric '{metric}' are NaN - check data parsing")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Check if metric values are between 0-1 and should be converted to percentage
        metric_data = df[metric].dropna()
        convert_to_percentage = False
        if len(metric_data) > 0:
            data_min, data_max = metric_data.min(), metric_data.max()
            # Convert to percentage if all values are between 0 and 1
            if data_min >= 0 and data_max <= 1:
                convert_to_percentage = True
                logger.info(f"Converting {metric} from 0-1 range to 0-100% for better readability")
        
        # Get unique frequencies and sort them
        frequencies = sorted(df['frequency'].unique())
        colors = self.colors(np.linspace(0, 1, len(frequencies)))
        
        # Plot each frequency as a separate line
        for freq, color in zip(frequencies, colors):
            freq_data = df[df['frequency'] == freq].copy()
            freq_data = freq_data.sort_values('normalized_time')
            
            # Filter out NaN values for this frequency
            valid_data = freq_data.dropna(subset=[metric])
            
            if len(valid_data) > 0:
                # Convert to percentage if needed
                plot_values = valid_data[metric] * 100 if convert_to_percentage else valid_data[metric]
                
                ax.plot(
                    valid_data['normalized_time'], 
                    plot_values,
                    color=color,
                    linewidth=3,
                    alpha=0.9,
                    label=f"{freq} MHz"
                )
                
                # Log some stats for this frequency
                metric_values = valid_data[metric]
                if convert_to_percentage:
                    logger.debug(f"{freq}MHz {metric}: {metric_values.min()*100:.1f}%-{metric_values.max()*100:.1f}% (mean: {metric_values.mean()*100:.1f}%)")
                else:
                    logger.debug(f"{freq}MHz {metric}: {metric_values.min():.2f}-{metric_values.max():.2f} (mean: {metric_values.mean():.2f})")
        
        # Customize plot with larger, bolder text
        ax.set_xlabel('Normalized Time', fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=16, fontweight='bold')
        ax.set_title(title or f'{metric} vs Time - {gpu} {app} (Run {run_number})', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Style legend with larger, bold text
        legend = ax.legend(title="Frequency", fontsize=14, title_fontsize=14)
        legend.get_title().set_fontweight('bold')
        
        # Make tick labels larger and bold
        ax.tick_params(axis='both', which='major', labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Set x-axis to always be 0 to 1
        ax.set_xlim(0, 1)
        
        # Set appropriate y-axis limits based on metric type
        metric_data = df[metric].dropna()
        if len(metric_data) > 0:
            if metric == 'POWER':
                # For power, use PMLMT (power limit/TDP) as the upper bound
                if 'PMLMT' in df.columns and not df['PMLMT'].isna().all():
                    power_limit = df['PMLMT'].dropna().iloc[0]
                    y_max = power_limit  # Use the actual TDP as upper bound
                    logger.info(f"Using power limit (TDP) as y-axis maximum: {power_limit:.0f}W")
                else:
                    # Fallback if PMLMT not available
                    power_max = metric_data.max()
                    y_max = power_max * 1.2
                    logger.warning(f"PMLMT not available, using 120% of max power: {y_max:.0f}W")
                ax.set_ylim(0, y_max)
                ax.set_ylabel('Power (W)', fontsize=16, fontweight='bold')
            elif metric in ['GPUTL', 'MCUTL', 'GRACT', 'SMACT'] or convert_to_percentage:
                # For utilization metrics or converted percentage metrics, set 0-100%
                ax.set_ylim(0, 100)
                if convert_to_percentage:
                    ax.set_ylabel(f'{metric} (%)', fontsize=16, fontweight='bold')
                else:
                    ax.set_ylabel(f'{metric} (%)', fontsize=16, fontweight='bold')
            elif metric == 'TMPTR':
                # For temperature, set reasonable range starting from 0
                temp_max = metric_data.max() + 10
                ax.set_ylim(0, temp_max)
                ax.set_ylabel('Temperature (°C)', fontsize=16, fontweight='bold')
            else:
                # For other metrics, use data-driven range with some padding, starting from 0
                data_min = 0  # Always start from 0
                data_max = metric_data.max()
                data_range = data_max - data_min
                padding = data_range * 0.1 if data_range > 0 else 1
                ax.set_ylim(0, data_max + padding)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        
        return fig
    
    def get_metric_info(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get information about available metrics."""
        if df.empty:
            return {}
        
        # Common DCGMI metrics with descriptions
        metric_descriptions = {
            'POWER': 'Power Draw (W)',
            'GPUTL': 'GPU Utilization (%)',
            'MCUTL': 'Memory Controller Utilization (%)', 
            'TMPTR': 'GPU Temperature (°C)',
            'MMTMP': 'Memory Temperature (°C)',
            'SMCLK': 'SM Clock (MHz)',
            'MMCLK': 'Memory Clock (MHz)',
            'FBTTL': 'Frame Buffer Total (MB)',
            'FBUSD': 'Frame Buffer Used (MB)',
            'GRACT': 'Graphics Active (%)',
            'SMACT': 'SM Active (%)',
            'TENSO': 'Tensor Active (%)',
            'DRAMA': 'DRAM Active (%)',
            'FP32A': 'FP32 Active (%)',
            'FP64A': 'FP64 Active (%)',
            'FP16A': 'FP16 Active (%)'
        }
        
        available_metrics = {}
        for col in df.columns:
            if col in metric_descriptions and col in df.columns:
                available_metrics[col] = metric_descriptions[col]
        
        return available_metrics


def get_default_frequencies(gpu: str) -> str:
    """Get default frequencies for each GPU type with consistent low/mid frequencies.
    
    Uses standardized frequency points for comparison across GPUs:
    - 510 MHz: Minimum frequency (energy efficiency baseline)
    - 750 MHz: Low-medium frequency (often optimal EDP point)
    - 960 MHz: Medium frequency (balanced performance-energy)
    - 1200 MHz: High-medium frequency (performance focus)
    - GPU-specific maximum: Peak performance reference
    """
    
    gpu_frequencies = {
        "V100": "510,750,960,1200,1380",    # V100 max: 1380 MHz
        "A100": "510,750,960,1200,1410",    # A100 max: 1410 MHz  
        "H100": "510,750,960,1200,1830"     # H100 max: 1830 MHz
    }
    
    return gpu_frequencies.get(gpu.upper(), "510,750,960,1200,1380")  # Default to V100


def parse_frequencies(freq_string: str) -> List[int]:
    """Parse comma-separated frequency string into list of integers."""
    try:
        frequencies = [int(f.strip()) for f in freq_string.split(',')]
        return frequencies
    except ValueError as e:
        logger.error(f"Error parsing frequencies '{freq_string}': {e}")
        return []


def validate_inputs(gpu: str, app: str, frequencies: List[int], metric: str, run_number: int) -> bool:
    """Validate input parameters."""
    
    # Validate GPU
    supported_gpus = ["V100", "A100", "H100"]
    if gpu.upper() not in supported_gpus:
        logger.error(f"Unsupported GPU: {gpu}. Supported: {supported_gpus}")
        return False
    
    # Validate application
    supported_apps = ["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER", "LSTM"]
    if app.upper() not in supported_apps:
        logger.error(f"Unsupported application: {app}. Supported: {supported_apps}")
        return False
    
    # Validate frequencies
    if not frequencies:
        logger.error("No valid frequencies provided")
        return False
    
    for freq in frequencies:
        if freq < 100 or freq > 2000:  # Reasonable bounds
            logger.error(f"Frequency {freq}MHz seems out of reasonable range (100-2000 MHz)")
            return False
    
    # Validate run number
    if run_number < 1 or run_number > 10:  # Reasonable bounds
        logger.error(f"Run number {run_number} out of reasonable range (1-10)")
        return False
    
    return True


def generate_plot_filename(gpu: str, app: str, metric: str, frequencies: List[int], run_number: int, plots_dir: str = "plots") -> str:
    """Generate an appropriate filename for the plot based on parameters."""
    
    # Create plots directory if it doesn't exist
    plots_path = Path(plots_dir)
    plots_path.mkdir(exist_ok=True)
    
    # Format frequencies for filename
    freq_str = "_".join(map(str, sorted(frequencies)))
    
    # Generate filename: {gpu}_{app}_{metric}_freq{frequencies}_run{run}.png
    filename = f"{gpu.lower()}_{app.lower()}_{metric.lower()}_freq{freq_str}_run{run_number:02d}.png"
    
    # Return full path
    full_path = plots_path / filename
    
    return str(full_path)


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Plot profiling metrics vs normalized time for GPU applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot GPU utilization for LLAMA on V100 (auto-saves to plots/v100_llama_gputl_freq510_960_1380_run02.png)
    python plot_metric_vs_time.py --gpu V100 --app LLAMA --frequencies 510,960,1380 --metric GPUTL --run 2
    
    # Plot power consumption for Vision Transformer on A100 (auto-saves to plots/a100_vit_power_freq1200_1410_run01.png)
    python plot_metric_vs_time.py --gpu A100 --app VIT --frequencies 1200,1410 --metric POWER
    
    # Plot DRAM activity for Stable Diffusion (auto-saves to plots/v100_stablediffusion_drama_freq800_1200_run01.png)
    python plot_metric_vs_time.py --gpu V100 --app STABLEDIFFUSION --frequencies 800,1200 --metric DRAMA --run 1
    
    # Save to custom location
    python plot_metric_vs_time.py --gpu V100 --app LLAMA --metric POWER --save custom_power_plot.png
        """
    )
    
    # Required parameters with defaults
    parser.add_argument("--gpu", type=str, default="V100",
                       help="GPU type (default: V100)")
    parser.add_argument("--app", type=str, default="LLAMA", 
                       help="Application name (default: LLAMA)")
    parser.add_argument("--frequencies", type=str, default=None,
                       help="Comma-separated frequencies in MHz (default: GPU-specific optimal set)")
    parser.add_argument("--metric", type=str, default="GPUTL",
                       help="Profiling metric to plot (default: GPUTL)")
    parser.add_argument("--run", type=int, default=1,
                       help="Run number to read from (default: 1)")
    
    # Optional parameters
    parser.add_argument("--data-dir", type=str, default="../../sample-collection-scripts",
                       help="Directory containing result folders (default: ../../sample-collection-scripts)")
    parser.add_argument("--save", type=str, default=None,
                       help="Path to save the plot (default: auto-generate filename in plots/ folder)")
    parser.add_argument("--title", type=str, default=None,
                       help="Custom plot title")
    parser.add_argument("--list-metrics", action="store_true",
                       help="List available metrics for the specified configuration")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show the plot (useful when saving)")
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        logger.error("Matplotlib is required but not available")
        return 1
    
    # Set GPU-specific default frequencies if not provided
    if args.frequencies is None:
        args.frequencies = get_default_frequencies(args.gpu)
        logger.info(f"Using default frequencies for {args.gpu}: {args.frequencies}")
    
    # Parse frequencies
    frequencies = parse_frequencies(args.frequencies)
    if not frequencies:
        return 1
    
    # Validate inputs
    if not validate_inputs(args.gpu.upper(), args.app.upper(), frequencies, args.metric, args.run):
        return 1
    
    # Initialize components
    loader = ProfilingDataLoader(args.data_dir)
    plotter = MetricPlotter()
    
    logger.info(f"🚀 Loading data: {args.gpu.upper()} + {args.app.upper()} @ {frequencies} MHz (Run {args.run})")
    
    # Load data
    df = loader.load_data(args.gpu.upper(), args.app.upper(), frequencies, args.run)
    
    if df.empty:
        logger.error("Failed to load data")
        return 1
    
    # List metrics if requested
    if args.list_metrics:
        metric_info = plotter.get_metric_info(df)
        print("\n📊 Available metrics:")
        for metric, description in metric_info.items():
            print(f"  {metric:<8} - {description}")
        print()
        return 0
    
    # Validate metric exists
    if args.metric not in df.columns:
        logger.error(f"Metric '{args.metric}' not found in data")
        metric_info = plotter.get_metric_info(df)
        print("\n📊 Available metrics:")
        for metric, description in metric_info.items():
            print(f"  {metric:<8} - {description}")
        return 1
    
    # Create plot
    logger.info(f"📈 Creating plot: {args.metric} vs normalized time")
    
    # Generate automatic filename if no save path provided
    save_path = args.save
    if save_path is None:
        save_path = generate_plot_filename(
            gpu=args.gpu.upper(),
            app=args.app.upper(),
            metric=args.metric,
            frequencies=frequencies,
            run_number=args.run
        )
        logger.info(f"💾 Auto-generating filename: {save_path}")
    
    fig = plotter.plot_metric_vs_time(
        df=df,
        metric=args.metric,
        gpu=args.gpu.upper(),
        app=args.app.upper(), 
        run_number=args.run,
        title=args.title,
        save_path=save_path,
        show_plot=not args.no_show
    )
    
    if fig is None:
        return 1
    
    logger.info("✅ Plot created successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
