"""
Performance Visualization Module

This module provides visualization functions for performance analysis,
including execution time plots, throughput analysis, and performance scaling visualization.

Key Features:
- Execution time vs frequency plots
- Performance scaling analysis
- Throughput and latency visualization
- Performance model validation
- Multi-application performance comparison
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conditional imports for plotting libraries
try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning(
        "Matplotlib not available. Performance plotting functions will be limited."
    )

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning(
        "Seaborn not available. Some advanced plotting features will be limited."
    )


def check_plotting_dependencies():
    """Check if required plotting libraries are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Matplotlib is required for performance plotting. Install with: pip install matplotlib"
        )


class PerformancePlotter:
    """
    Performance visualization class providing comprehensive plotting capabilities for
    execution time analysis, throughput visualization, and performance model validation.
    """

    def __init__(self, style: str = "default", figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize performance plotter.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        check_plotting_dependencies()

        self.style = style
        self.figsize = figsize

        if HAS_SEABORN:
            sns.set_style("whitegrid")

        plt.style.use(style)
        logger.info(f"Performance plotter initialized with style: {style}")

    def plot_execution_time_vs_frequency(
        self,
        df: pd.DataFrame,
        frequency_col: str = "frequency",
        time_col: str = "execution_time",
        app_name: str = "Application",
        optimal_freq: Optional[int] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot execution time vs frequency.

        Args:
            df: DataFrame with frequency and execution time data
            frequency_col: Column name for frequency values
            time_col: Column name for execution time values
            app_name: Application name for labeling
            optimal_freq: Optimal frequency to highlight
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Execution Time vs Frequency - {app_name}"

        fig, ax = plt.subplots(figsize=self.figsize)

        # Main execution time curve
        ax.plot(
            df[frequency_col],
            df[time_col],
            "b-o",
            linewidth=2,
            markersize=6,
            label="Execution Time",
        )

        # Highlight optimal frequency if provided
        if optimal_freq is not None:
            optimal_row = df[df[frequency_col] == optimal_freq]
            if not optimal_row.empty:
                optimal_time = optimal_row[time_col].iloc[0]
                ax.plot(
                    optimal_freq,
                    optimal_time,
                    "ro",
                    markersize=10,
                    label=f"Optimal ({optimal_freq} MHz)",
                )

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Execution Time (s)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add min/max annotations
        min_time_idx = df[time_col].idxmin()
        max_time_idx = df[time_col].idxmax()

        min_freq = df.loc[min_time_idx, frequency_col]
        min_time = df.loc[min_time_idx, time_col]
        max_freq = df.loc[max_time_idx, frequency_col]
        max_time = df.loc[max_time_idx, time_col]

        ax.annotate(
            f"Fastest: {min_time:.2f}s",
            xy=(min_freq, min_time),
            xytext=(10, 10),
            textcoords="offset points",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        ax.annotate(
            f"Slowest: {max_time:.2f}s",
            xy=(max_freq, max_time),
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Execution time vs frequency plot saved to {save_path}")

        return fig

    def plot_performance_scaling(
        self,
        df: pd.DataFrame,
        frequency_col: str = "frequency",
        time_col: str = "execution_time",
        baseline_freq: Optional[int] = None,
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot performance scaling relative to baseline frequency.

        Args:
            df: DataFrame with frequency and execution time data
            frequency_col: Column name for frequency values
            time_col: Column name for execution time values
            baseline_freq: Baseline frequency for scaling calculation (uses max if None)
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Performance Scaling - {app_name}"

        df_work = df.copy()

        # Use max frequency as baseline if not specified
        if baseline_freq is None:
            baseline_freq = df_work[frequency_col].max()

        # Find baseline execution time
        baseline_row = df_work[df_work[frequency_col] == baseline_freq]
        if baseline_row.empty:
            # Use closest frequency
            baseline_idx = (df_work[frequency_col] - baseline_freq).abs().idxmin()
            baseline_time = df_work.loc[baseline_idx, time_col]
        else:
            baseline_time = baseline_row[time_col].iloc[0]

        # Calculate performance scaling (speedup)
        df_work["speedup"] = baseline_time / df_work[time_col]
        df_work["ideal_speedup"] = df_work[frequency_col] / baseline_freq

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5)
        )

        # Plot 1: Speedup vs frequency
        ax1.plot(
            df_work[frequency_col],
            df_work["speedup"],
            "b-o",
            linewidth=2,
            markersize=6,
            label="Actual Speedup",
        )
        ax1.plot(
            df_work[frequency_col],
            df_work["ideal_speedup"],
            "r--",
            linewidth=2,
            label="Ideal Speedup",
        )

        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Speedup")
        ax1.set_title("Performance Speedup vs Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Scaling efficiency
        df_work["efficiency"] = df_work["speedup"] / df_work["ideal_speedup"] * 100

        ax2.plot(
            df_work[frequency_col],
            df_work["efficiency"],
            "g-o",
            linewidth=2,
            markersize=6,
            label="Scaling Efficiency",
        )
        ax2.axhline(
            y=100, color="r", linestyle="--", alpha=0.7, label="Perfect Scaling"
        )

        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Scaling Efficiency (%)")
        ax2.set_title("Scaling Efficiency vs Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Performance scaling plot saved to {save_path}")

        return fig

    def plot_throughput_analysis(
        self,
        df: pd.DataFrame,
        frequency_col: str = "frequency",
        time_col: str = "execution_time",
        workload_size: float = 1.0,
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot throughput analysis (work units per time).

        Args:
            df: DataFrame with frequency and execution time data
            frequency_col: Column name for frequency values
            time_col: Column name for execution time values
            workload_size: Size of workload unit
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Throughput Analysis - {app_name}"

        df_work = df.copy()
        df_work["throughput"] = workload_size / df_work[time_col]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Throughput vs frequency
        ax1.plot(
            df_work[frequency_col],
            df_work["throughput"],
            "b-o",
            linewidth=2,
            markersize=6,
            label="Throughput",
        )

        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Throughput (Work Units/s)")
        ax1.set_title("Throughput vs Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Highlight max throughput
        max_throughput_idx = df_work["throughput"].idxmax()
        max_freq = df_work.loc[max_throughput_idx, frequency_col]
        max_throughput = df_work.loc[max_throughput_idx, "throughput"]

        ax1.plot(
            max_freq,
            max_throughput,
            "ro",
            markersize=10,
            label=f"Max Throughput ({max_freq} MHz)",
        )
        ax1.legend()

        # Plot 2: Throughput vs execution time scatter
        scatter = ax2.scatter(
            df_work[time_col],
            df_work["throughput"],
            c=df_work[frequency_col],
            cmap="viridis",
            s=60,
        )

        ax2.set_xlabel("Execution Time (s)")
        ax2.set_ylabel("Throughput (Work Units/s)")
        ax2.set_title("Throughput vs Execution Time")
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Frequency (MHz)")

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Throughput analysis plot saved to {save_path}")

        return fig

    def plot_multi_application_performance(
        self,
        data_dict: Dict[str, pd.DataFrame],
        frequency_col: str = "frequency",
        time_col: str = "execution_time",
        normalize: bool = True,
        title: str = "Multi-Application Performance Comparison",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Compare performance across multiple applications.

        Args:
            data_dict: Dictionary with app names as keys and DataFrames as values
            frequency_col: Column name for frequency values
            time_col: Column name for execution time values
            normalize: Whether to normalize execution times
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

        # Plot 1: Execution time comparison
        for i, (app_name, df) in enumerate(data_dict.items()):
            if normalize:
                # Normalize to max frequency execution time
                max_freq_time = df.loc[df[frequency_col].idxmax(), time_col]
                normalized_time = df[time_col] / max_freq_time
                ax1.plot(
                    df[frequency_col],
                    normalized_time,
                    "o-",
                    color=colors[i],
                    linewidth=2,
                    markersize=6,
                    label=app_name,
                )
                ax1.set_ylabel("Normalized Execution Time")
            else:
                ax1.plot(
                    df[frequency_col],
                    df[time_col],
                    "o-",
                    color=colors[i],
                    linewidth=2,
                    markersize=6,
                    label=app_name,
                )
                ax1.set_ylabel("Execution Time (s)")

        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_title("Execution Time vs Frequency")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Performance efficiency (inverse of normalized time)
        for i, (app_name, df) in enumerate(data_dict.items()):
            max_freq_time = df.loc[df[frequency_col].idxmax(), time_col]
            efficiency = max_freq_time / df[time_col]
            ax2.plot(
                df[frequency_col],
                efficiency,
                "o-",
                color=colors[i],
                linewidth=2,
                markersize=6,
                label=app_name,
            )

        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Performance Efficiency")
        ax2.set_title("Performance Efficiency vs Frequency")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Multi-application performance plot saved to {save_path}")

        return fig

    def plot_performance_model_validation(
        self,
        actual_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
        frequency_col: str = "frequency",
        actual_time_col: str = "actual_time",
        predicted_time_col: str = "predicted_time",
        title: str = "Performance Model Validation",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Validate performance model predictions against actual measurements.

        Args:
            actual_df: DataFrame with actual execution time measurements
            predicted_df: DataFrame with predicted execution time values
            frequency_col: Column name for frequency values
            actual_time_col: Column name for actual time values
            predicted_time_col: Column name for predicted time values
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Merge dataframes on frequency for comparison
        merged_df = pd.merge(actual_df, predicted_df, on=frequency_col, how="inner")

        # Plot 1: Actual vs Predicted scatter
        ax1.scatter(
            merged_df[actual_time_col], merged_df[predicted_time_col], alpha=0.7
        )

        # Add perfect prediction line
        min_val = min(
            merged_df[actual_time_col].min(), merged_df[predicted_time_col].min()
        )
        max_val = max(
            merged_df[actual_time_col].max(), merged_df[predicted_time_col].max()
        )
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        ax1.set_xlabel("Actual Execution Time (s)")
        ax1.set_ylabel("Predicted Execution Time (s)")
        ax1.set_title("Predicted vs Actual Execution Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calculate and display R²
        correlation_matrix = np.corrcoef(
            merged_df[actual_time_col], merged_df[predicted_time_col]
        )
        r_squared = correlation_matrix[0, 1] ** 2
        ax1.text(
            0.05,
            0.95,
            f"R² = {r_squared:.3f}",
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Plot 2: Residuals vs Frequency
        residuals = merged_df[predicted_time_col] - merged_df[actual_time_col]
        ax2.scatter(merged_df[frequency_col], residuals, alpha=0.7)
        ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Residuals (Predicted - Actual)")
        ax2.set_title("Residuals vs Frequency")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Both curves vs frequency
        ax3.plot(
            actual_df[frequency_col],
            actual_df[actual_time_col],
            "b-o",
            linewidth=2,
            markersize=6,
            label="Actual",
        )
        ax3.plot(
            predicted_df[frequency_col],
            predicted_df[predicted_time_col],
            "r-s",
            linewidth=2,
            markersize=6,
            label="Predicted",
        )
        ax3.set_xlabel("Frequency (MHz)")
        ax3.set_ylabel("Execution Time (s)")
        ax3.set_title("Execution Time vs Frequency Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Error statistics
        ax4.axis("off")

        # Calculate error metrics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / merged_df[actual_time_col])) * 100

        error_text = f"Error Metrics:\n\n"
        error_text += f"MAE: {mae:.3f} s\n"
        error_text += f"RMSE: {rmse:.3f} s\n"
        error_text += f"MAPE: {mape:.1f}%\n"
        error_text += f"R²: {r_squared:.3f}\n\n"
        error_text += f"Data Points: {len(merged_df)}\n"
        error_text += f"Frequency Range: {merged_df[frequency_col].min()}-{merged_df[frequency_col].max()} MHz"

        ax4.text(
            0.1,
            0.9,
            error_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Performance model validation plot saved to {save_path}")

        return fig

    def plot_latency_distribution(
        self,
        df: pd.DataFrame,
        time_col: str = "execution_time",
        frequency_groups: Optional[List[int]] = None,
        title: str = "Execution Time Distribution",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot distribution of execution times, optionally grouped by frequency.

        Args:
            df: DataFrame with execution time data
            time_col: Column name for execution time values
            frequency_groups: List of frequency values to group by
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if frequency_groups and "frequency" in df.columns:
            # Plot distribution for each frequency group
            for freq in frequency_groups:
                freq_data = df[df["frequency"] == freq][time_col]
                if not freq_data.empty:
                    ax.hist(freq_data, alpha=0.6, label=f"{freq} MHz", bins=20)
        else:
            # Plot overall distribution
            ax.hist(df[time_col], alpha=0.7, bins=20, edgecolor="black")

        ax.set_xlabel("Execution Time (s)")
        ax.set_ylabel("Frequency")
        ax.set_title(title)

        if frequency_groups:
            ax.legend()

        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_time = df[time_col].mean()
        std_time = df[time_col].std()
        ax.axvline(
            mean_time,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_time:.3f}s",
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Latency distribution plot saved to {save_path}")

        return fig

    def plot_fgcs_performance_validation(
        self,
        df: pd.DataFrame,
        predicted_col: str = "predicted_runtime",
        actual_col: str = "execution_time",
        frequency_col: str = "frequency",
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot FGCS performance model validation comparing predicted vs actual runtime.

        Args:
            df: DataFrame with predicted and actual runtime values
            predicted_col: Column name for predicted runtime values
            actual_col: Column name for actual runtime values
            frequency_col: Column name for frequency values
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"FGCS Performance Model Validation - {app_name}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Predicted vs Actual Runtime
        ax1.scatter(df[actual_col], df[predicted_col], alpha=0.7, s=60)

        # Perfect prediction line
        min_val = min(df[actual_col].min(), df[predicted_col].min())
        max_val = max(df[actual_col].max(), df[predicted_col].max())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        # Calculate metrics if sklearn is available
        try:
            from sklearn.metrics import mean_squared_error, r2_score

            r2 = r2_score(df[actual_col], df[predicted_col])
            rmse = np.sqrt(mean_squared_error(df[actual_col], df[predicted_col]))
            subtitle = f"Predicted vs Actual Runtime\nR² = {r2:.3f}, RMSE = {rmse:.3f}s"
        except ImportError:
            subtitle = "Predicted vs Actual Runtime"

        ax1.set_xlabel("Actual Runtime (s)")
        ax1.set_ylabel("Predicted Runtime (s)")
        ax1.set_title(subtitle)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Runtime vs Frequency (both predicted and actual)
        ax2.plot(
            df[frequency_col],
            df[actual_col],
            "bo-",
            linewidth=2,
            markersize=6,
            label="Actual Runtime",
        )
        ax2.plot(
            df[frequency_col],
            df[predicted_col],
            "ro-",
            linewidth=2,
            markersize=6,
            label="Predicted Runtime",
        )

        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Runtime (s)")
        ax2.set_title("Runtime vs Frequency Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"FGCS performance validation plot saved to {save_path}")

        return fig

    def plot_throughput_analysis(
        self,
        df: pd.DataFrame,
        frequency_col: str = "frequency",
        throughput_col: str = "throughput",
        latency_col: str = "execution_time",
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot comprehensive throughput and latency analysis.

        Args:
            df: DataFrame with throughput and latency data
            frequency_col: Column name for frequency values
            throughput_col: Column name for throughput values
            latency_col: Column name for latency/execution time values
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Throughput and Latency Analysis - {app_name}"

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Throughput vs Frequency
        if throughput_col in df.columns:
            ax1.plot(
                df[frequency_col], df[throughput_col], "g-o", linewidth=2, markersize=6
            )
            ax1.set_xlabel("Frequency (MHz)")
            ax1.set_ylabel("Throughput (ops/s)")
            ax1.set_title("Throughput vs Frequency")
            ax1.grid(True, alpha=0.3)

            # Highlight max throughput
            max_throughput_idx = df[throughput_col].idxmax()
            max_freq = df.loc[max_throughput_idx, frequency_col]
            max_throughput = df.loc[max_throughput_idx, throughput_col]
            ax1.plot(
                max_freq,
                max_throughput,
                "ro",
                markersize=10,
                label=f"Max: {max_throughput:.1f} ops/s",
            )
            ax1.legend()

        # Plot 2: Latency vs Frequency
        ax2.plot(df[frequency_col], df[latency_col], "b-o", linewidth=2, markersize=6)
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Latency (s)")
        ax2.set_title("Latency vs Frequency")
        ax2.grid(True, alpha=0.3)

        # Highlight min latency
        min_latency_idx = df[latency_col].idxmin()
        min_freq = df.loc[min_latency_idx, frequency_col]
        min_latency = df.loc[min_latency_idx, latency_col]
        ax2.plot(
            min_freq, min_latency, "ro", markersize=10, label=f"Min: {min_latency:.3f}s"
        )
        ax2.legend()

        # Plot 3: Throughput-Latency Trade-off
        if throughput_col in df.columns:
            scatter = ax3.scatter(
                df[latency_col],
                df[throughput_col],
                c=df[frequency_col],
                cmap="viridis",
                s=80,
                alpha=0.7,
            )
            ax3.set_xlabel("Latency (s)")
            ax3.set_ylabel("Throughput (ops/s)")
            ax3.set_title("Throughput-Latency Trade-off")
            plt.colorbar(scatter, ax=ax3, label="Frequency (MHz)")

        # Plot 4: Performance Scaling Efficiency
        if throughput_col in df.columns:
            # Calculate relative performance scaling
            baseline_freq = df[frequency_col].min()
            baseline_throughput = df[df[frequency_col] == baseline_freq][
                throughput_col
            ].iloc[0]

            performance_scaling = df[throughput_col] / baseline_throughput
            frequency_scaling = df[frequency_col] / baseline_freq
            efficiency = performance_scaling / frequency_scaling

            ax4.plot(
                df[frequency_col],
                efficiency,
                "purple",
                linewidth=2,
                marker="o",
                markersize=6,
            )
            ax4.axhline(
                y=1.0, color="r", linestyle="--", alpha=0.7, label="Perfect Scaling"
            )
            ax4.set_xlabel("Frequency (MHz)")
            ax4.set_ylabel("Scaling Efficiency")
            ax4.set_title("Performance Scaling Efficiency")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Throughput analysis plot saved to {save_path}")

        return fig

    def plot_performance_breakdown(
        self,
        df: pd.DataFrame,
        frequency_col: str = "frequency",
        execution_time_col: str = "execution_time",
        compute_time_col: Optional[str] = None,
        memory_time_col: Optional[str] = None,
        io_time_col: Optional[str] = None,
        app_name: str = "Application",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot performance breakdown showing different components of execution time.

        Args:
            df: DataFrame with performance breakdown data
            frequency_col: Column name for frequency values
            execution_time_col: Column name for total execution time
            compute_time_col: Column name for compute time (optional)
            memory_time_col: Column name for memory time (optional)
            io_time_col: Column name for I/O time (optional)
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot

        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Performance Breakdown Analysis - {app_name}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Stacked bar chart of time components
        frequencies = df[frequency_col]
        total_time = df[execution_time_col]

        # Prepare data for stacking
        bottom = np.zeros(len(frequencies))
        colors = ["skyblue", "lightcoral", "lightgreen", "gold"]
        components = []
        labels = []

        if compute_time_col and compute_time_col in df.columns:
            components.append(df[compute_time_col])
            labels.append("Compute Time")

        if memory_time_col and memory_time_col in df.columns:
            components.append(df[memory_time_col])
            labels.append("Memory Time")

        if io_time_col and io_time_col in df.columns:
            components.append(df[io_time_col])
            labels.append("I/O Time")

        # If no breakdown available, show total time
        if not components:
            components = [total_time]
            labels = ["Total Execution Time"]

        # Create stacked bars
        for i, (component, label, color) in enumerate(
            zip(components, labels, colors[: len(components)])
        ):
            ax1.bar(
                frequencies,
                component,
                bottom=bottom,
                label=label,
                alpha=0.8,
                color=color,
            )
            bottom += component

        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Time (s)")
        ax1.set_title("Execution Time Breakdown")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Relative performance breakdown (percentages)
        if len(components) > 1:
            percentages = []
            for component in components:
                percentages.append((component / total_time) * 100)

            # Create stacked area plot
            ax2.stackplot(
                frequencies,
                *percentages,
                labels=labels,
                alpha=0.7,
                colors=colors[: len(components)],
            )
            ax2.set_xlabel("Frequency (MHz)")
            ax2.set_ylabel("Percentage (%)")
            ax2.set_title("Relative Time Breakdown")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)
        else:
            # Show performance scaling instead
            baseline_time = total_time.max()  # Use slowest as baseline
            performance_improvement = (
                (baseline_time - total_time) / baseline_time
            ) * 100
            ax2.plot(
                frequencies, performance_improvement, "b-o", markersize=6, linewidth=2
            )
            ax2.axhline(y=0, color="r", linestyle="--", alpha=0.7)
            ax2.set_xlabel("Frequency (MHz)")
            ax2.set_ylabel("Performance Improvement (%)")
            ax2.set_title("Performance Scaling vs Frequency")
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Performance breakdown plot saved to {save_path}")

        return fig
