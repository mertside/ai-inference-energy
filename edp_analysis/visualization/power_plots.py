"""
Power Visualization Module

This module provides visualization functions for power consumption analysis,
including power vs frequency plots, power efficiency analysis, and power modeling validation.

Key Features:
- Power consumption vs frequency plots
- Power efficiency analysis
- Power model validation plots
- Multi-application power comparison
- Power prediction vs actual plots
- Time-series power visualization
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Conditional imports for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning(
        "Matplotlib not available. Power plotting functions will be limited."
    )

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning(
        "Seaborn not available. Some advanced plotting features will be limited."
    )

try:
    from mpl_toolkits.mplot3d import Axes3D

    HAS_3D = True
except ImportError:
    HAS_3D = False
    logger.warning("3D plotting not available.")


def check_plotting_dependencies():
    """Check if required plotting libraries are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for power plotting functionality")


class PowerPlotter:
    """
    Power visualization class providing comprehensive plotting capabilities for
    power consumption analysis and power model validation.
    """

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (10, 6)):
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for PowerPlotter")
        
        self.figsize = figsize
        if style != "default" and HAS_MATPLOTLIB:
            plt.style.use(style)
        
        logger.info(f"PowerPlotter initialized: {style} style, {figsize} figsize")

    def plot_power_vs_time(
        self,
        df: pd.DataFrame,
        power_col: str = "power_draw",
        time_col: str = "timestamp",
        frequency_filter: Optional[Union[int, List[int]]] = None,
        app_name: str = "Application",
        normalize_time: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot power consumption over time.
        
        Args:
            df: DataFrame with power and time data
            power_col: Column name for power values
            time_col: Column name for time data
            frequency_filter: Single frequency or list of frequencies to include
            app_name: Application name for labeling
            normalize_time: Whether to normalize time to [0,1] range
            title: Custom plot title
            save_path: Path to save the plot
        
        Returns:
            Matplotlib Figure object
        """
        check_plotting_dependencies()
        
        df_work = df.copy()
        
        # Filter by frequency if specified
        if frequency_filter is not None:
            if isinstance(frequency_filter, int):
                frequency_filter = [frequency_filter]
            df_work = df_work[df_work['frequency'].isin(frequency_filter)]
        
        if df_work.empty:
            raise ValueError("No data available after filtering")
        
        # Normalize time if requested
        if normalize_time:
            if pd.api.types.is_datetime64_any_dtype(df_work[time_col]):
                time_numeric = df_work[time_col].astype('int64') / 1e9
                time_min = time_numeric.min()
                time_max = time_numeric.max()
                df_work['normalized_time'] = (time_numeric - time_min) / (time_max - time_min)
            else:
                time_min = df_work[time_col].min()
                time_max = df_work[time_col].max()
                df_work['normalized_time'] = (df_work[time_col] - time_min) / (time_max - time_min)
            x_col = 'normalized_time'
            x_label = "Normalized Time"
        else:
            x_col = time_col
            x_label = "Time"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot different frequencies with different colors
        if 'frequency' in df_work.columns and len(df_work['frequency'].unique()) > 1:
            frequencies = sorted(df_work['frequency'].unique())
            colors = plt.cm.plasma(np.linspace(0, 1, len(frequencies)))
            
            for freq, color in zip(frequencies, colors):
                freq_data = df_work[df_work['frequency'] == freq].sort_values(x_col)
                ax.plot(freq_data[x_col], freq_data[power_col], 
                       color=color, linewidth=2, alpha=0.8, 
                       label=f"{freq} MHz")
        else:
            df_work = df_work.sort_values(x_col)
            ax.plot(df_work[x_col], df_work[power_col], 
                   'r-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{power_col.replace('_', ' ').title()} (W)")
        ax.set_title(title or f"Power Consumption vs {x_label} - {app_name}")
        ax.grid(True, alpha=0.3)
        
        if 'frequency' in df_work.columns and len(df_work['frequency'].unique()) > 1:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power time-series plot saved to {save_path}")
        
        return fig

    def plot_power_temperature_correlation(
        self,
        df: pd.DataFrame,
        power_col: str = "power_draw",
        time_col: str = "timestamp",
        frequency_filter: Optional[Union[int, List[int]]] = None,
        app_name: str = "Application",
        normalize_time: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot power consumption over time.
        
        Args:
            df: DataFrame with power and time data
            power_col: Column name for power values
            time_col: Column name for time data
            frequency_filter: Single frequency or list of frequencies to include
            app_name: Application name for labeling
            normalize_time: Whether to normalize time to [0,1] range
            title: Custom plot title
            save_path: Path to save the plot
        
        Returns:
            Matplotlib Figure object
        """
        check_plotting_dependencies()
        
        df_work = df.copy()
        
        # Filter by frequency if specified
        if frequency_filter is not None:
            if isinstance(frequency_filter, int):
                frequency_filter = [frequency_filter]
            df_work = df_work[df_work['frequency'].isin(frequency_filter)]
        
        if df_work.empty:
            raise ValueError("No data available after filtering")
        
        # Normalize time if requested
        if normalize_time:
            if pd.api.types.is_datetime64_any_dtype(df_work[time_col]):
                time_numeric = df_work[time_col].astype('int64') / 1e9
                time_min = time_numeric.min()
                time_max = time_numeric.max()
                df_work['normalized_time'] = (time_numeric - time_min) / (time_max - time_min)
            else:
                time_min = df_work[time_col].min()
                time_max = df_work[time_col].max()
                df_work['normalized_time'] = (df_work[time_col] - time_min) / (time_max - time_min)
            x_col = 'normalized_time'
            x_label = "Normalized Time"
        else:
            x_col = time_col
            x_label = "Time"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot different frequencies with different colors
        if 'frequency' in df_work.columns and len(df_work['frequency'].unique()) > 1:
            frequencies = sorted(df_work['frequency'].unique())
            colors = plt.cm.plasma(np.linspace(0, 1, len(frequencies)))
            
            for freq, color in zip(frequencies, colors):
                freq_data = df_work[df_work['frequency'] == freq].sort_values(x_col)
                ax.plot(freq_data[x_col], freq_data[power_col], 
                       color=color, linewidth=2, alpha=0.8, 
                       label=f"{freq} MHz")
        else:
            df_work = df_work.sort_values(x_col)
            ax.plot(df_work[x_col], df_work[power_col], 
                   'r-', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{power_col.replace('_', ' ').title()} (W)")
        ax.set_title(title or f"Power Consumption vs {x_label} - {app_name}")
        ax.grid(True, alpha=0.3)
        
        if 'frequency' in df_work.columns and len(df_work['frequency'].unique()) > 1:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power time-series plot saved to {save_path}")
        
        return fig

    def plot_power_temperature_correlation(
        self,
        df: pd.DataFrame,
        power_col: str = "power_draw",
        temperature_col: str = "gpu_temperature",
        frequency_col: str = "frequency",
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot correlation between power consumption and temperature.
        """
        check_plotting_dependencies()
        
        if temperature_col not in df.columns:
            logger.warning(f"Temperature column '{temperature_col}' not found")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot by frequency if available
        if frequency_col in df.columns:
            frequencies = sorted(df[frequency_col].unique())
            colors = plt.cm.plasma(np.linspace(0, 1, len(frequencies)))
            
            for freq, color in zip(frequencies, colors):
                freq_data = df[df[frequency_col] == freq]
                ax.scatter(freq_data[power_col], freq_data[temperature_col], 
                          color=color, alpha=0.6, s=20, 
                          label=f"{freq} MHz")
        else:
            ax.scatter(df[power_col], df[temperature_col], 
                      alpha=0.6, s=20, color='red')
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df[power_col], df[temperature_col], 1)
            p = np.poly1d(z)
            ax.plot(df[power_col], p(df[power_col]), "k--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel(f"{power_col.replace('_', ' ').title()} (W)")
        ax.set_ylabel(f"{temperature_col.replace('_', ' ').title()} (°C)")
        ax.set_title(title or f"Power vs Temperature Correlation - {app_name}")
        ax.grid(True, alpha=0.3)
        
        if frequency_col in df.columns:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power-temperature correlation plot saved to {save_path}")
        
        return fig

    def plot_energy_efficiency_analysis(
        self,
        df: pd.DataFrame,
        frequency_col: str = "frequency",
        power_col: str = "power_draw",
        energy_col: str = "energy",
        performance_col: str = "throughput",
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot comprehensive energy efficiency analysis.

        Args:
            df: DataFrame with power, energy, and performance data
            frequency_col: Column name for frequency values
            power_col: Column name for power values
            energy_col: Column name for energy values
            performance_col: Column name for performance/throughput values
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        check_plotting_dependencies()
        
        if title is None:
            title = f"Energy Efficiency Analysis - {app_name}"

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Calculate efficiency metrics
        energy_per_op = (
            df[energy_col] / df[performance_col]
            if performance_col in df.columns
            else None
        )
        power_efficiency = (
            df[performance_col] / df[power_col]
            if performance_col in df.columns
            else None
        )

        # Plot implementation would continue here...
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Energy efficiency analysis plot saved to {save_path}")

        return fig


def plot_power_histogram(
    df: pd.DataFrame,
    power_col: str = "power",
    bins: int = 20,
    title: str = "Power Consumption Distribution",
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot histogram of power consumption values.

    Args:
        df: DataFrame with power data
        power_col: Column name for power values
        bins: Number of histogram bins
        title: Plot title
        save_path: Path to save the plot

    Returns:
        Matplotlib Figure object
    """
    check_plotting_dependencies()

    fig, ax = plt.subplots(figsize=(10, 6))

    n, bins_edges, patches = ax.hist(
        df[power_col], bins=bins, alpha=0.7, edgecolor="black"
    )

    # Color bars by value
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.viridis(i / len(patches)))

    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_power = df[power_col].mean()
    std_power = df[power_col].std()
    ax.axvline(
        mean_power,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_power:.2f}W",
    )
    ax.axvline(
        mean_power + std_power,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"+1σ: {mean_power + std_power:.2f}W",
    )
    ax.axvline(
        mean_power - std_power,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"-1σ: {mean_power - std_power:.2f}W",
    )

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Power histogram saved to {save_path}")

    return fig
