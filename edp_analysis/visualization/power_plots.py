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
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

# Conditional imports for plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available. Power plotting functions will be limited.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn not available. Some advanced plotting features will be limited.")

try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except ImportError:
    HAS_3D = False
    logger.warning("3D plotting not available.")


def check_plotting_dependencies():
    """Check if required plotting libraries are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for power plotting. Install with: pip install matplotlib")


class PowerPlotter:
    """
    Power visualization class providing comprehensive plotting capabilities for
    power consumption analysis and power model validation.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize power plotter.
        
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
        logger.info(f"Power plotter initialized with style: {style}")
    
    def plot_power_vs_frequency(self,
                              df: pd.DataFrame,
                              frequency_col: str = 'frequency',
                              power_col: str = 'power',
                              app_name: str = "Application",
                              optimal_freq: Optional[int] = None,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> Figure:
        """
        Plot power consumption vs frequency.
        
        Args:
            df: DataFrame with frequency and power data
            frequency_col: Column name for frequency values
            power_col: Column name for power values
            app_name: Application name for labeling
            optimal_freq: Optimal frequency to highlight
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Power Consumption vs Frequency - {app_name}"
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Main power curve
        ax.plot(df[frequency_col], df[power_col], 'b-o', linewidth=2, markersize=6, label='Power Consumption')
        
        # Highlight optimal frequency if provided
        if optimal_freq is not None:
            optimal_row = df[df[frequency_col] == optimal_freq]
            if not optimal_row.empty:
                optimal_power = optimal_row[power_col].iloc[0]
                ax.plot(optimal_freq, optimal_power, 'ro', markersize=10, 
                       label=f'Optimal ({optimal_freq} MHz)')
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (W)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add min/max annotations
        min_power_idx = df[power_col].idxmin()
        max_power_idx = df[power_col].idxmax()
        
        min_freq = df.loc[min_power_idx, frequency_col]
        min_power = df.loc[min_power_idx, power_col]
        max_freq = df.loc[max_power_idx, frequency_col]
        max_power = df.loc[max_power_idx, power_col]
        
        ax.annotate(f'Min: {min_power:.1f}W', 
                   xy=(min_freq, min_power), xytext=(10, 10),
                   textcoords='offset points', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Max: {max_power:.1f}W', 
                   xy=(max_freq, max_power), xytext=(10, -10),
                   textcoords='offset points', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power vs frequency plot saved to {save_path}")
        
        return fig
    
    def plot_power_efficiency(self,
                            df: pd.DataFrame,
                            frequency_col: str = 'frequency',
                            power_col: str = 'power',
                            performance_col: str = 'performance',
                            app_name: str = "Application",
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> Figure:
        """
        Plot power efficiency (performance per watt).
        
        Args:
            df: DataFrame with frequency, power, and performance data
            frequency_col: Column name for frequency values
            power_col: Column name for power values
            performance_col: Column name for performance metric
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Power Efficiency - {app_name}"
        
        # Calculate efficiency (performance per watt)
        df_work = df.copy()
        df_work['efficiency'] = df_work[performance_col] / df_work[power_col]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        # Plot 1: Power efficiency vs frequency
        ax1.plot(df_work[frequency_col], df_work['efficiency'], 'g-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Efficiency (Performance/Watt)')
        ax1.set_title('Power Efficiency vs Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Find and highlight most efficient configuration
        max_eff_idx = df_work['efficiency'].idxmax()
        max_eff_freq = df_work.loc[max_eff_idx, frequency_col]
        max_eff_val = df_work.loc[max_eff_idx, 'efficiency']
        
        ax1.plot(max_eff_freq, max_eff_val, 'ro', markersize=10, 
                label=f'Most Efficient ({max_eff_freq} MHz)')
        ax1.legend()
        
        # Plot 2: Performance vs Power scatter
        ax2.scatter(df_work[power_col], df_work[performance_col], 
                   c=df_work[frequency_col], cmap='viridis', s=60)
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Performance')
        ax2.set_title('Performance vs Power')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for frequency
        scatter = ax2.scatter(df_work[power_col], df_work[performance_col], 
                            c=df_work[frequency_col], cmap='viridis', s=60)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Frequency (MHz)')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power efficiency plot saved to {save_path}")
        
        return fig
    
    def plot_multi_application_power(self,
                                   data_dict: Dict[str, pd.DataFrame],
                                   frequency_col: str = 'frequency',
                                   power_col: str = 'power',
                                   title: str = "Power Consumption Comparison",
                                   save_path: Optional[str] = None) -> Figure:
        """
        Compare power consumption across multiple applications.
        
        Args:
            data_dict: Dictionary with app names as keys and DataFrames as values
            frequency_col: Column name for frequency values
            power_col: Column name for power values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
        
        for i, (app_name, df) in enumerate(data_dict.items()):
            ax.plot(df[frequency_col], df[power_col], 'o-', 
                   color=colors[i], linewidth=2, markersize=6, label=app_name)
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (W)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = "Statistics:\n"
        for app_name, df in data_dict.items():
            min_power = df[power_col].min()
            max_power = df[power_col].max()
            avg_power = df[power_col].mean()
            stats_text += f"{app_name}: {min_power:.1f}-{max_power:.1f}W (avg: {avg_power:.1f}W)\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Multi-application power plot saved to {save_path}")
        
        return fig
    
    def plot_power_model_validation(self,
                                  actual_df: pd.DataFrame,
                                  predicted_df: pd.DataFrame,
                                  frequency_col: str = 'frequency',
                                  actual_power_col: str = 'actual_power',
                                  predicted_power_col: str = 'predicted_power',
                                  title: str = "Power Model Validation",
                                  save_path: Optional[str] = None) -> Figure:
        """
        Validate power model predictions against actual measurements.
        
        Args:
            actual_df: DataFrame with actual power measurements
            predicted_df: DataFrame with predicted power values
            frequency_col: Column name for frequency values
            actual_power_col: Column name for actual power values
            predicted_power_col: Column name for predicted power values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Merge dataframes on frequency for comparison
        merged_df = pd.merge(actual_df, predicted_df, on=frequency_col, how='inner')
        
        # Plot 1: Actual vs Predicted scatter
        ax1.scatter(merged_df[actual_power_col], merged_df[predicted_power_col], alpha=0.7)
        
        # Add perfect prediction line
        min_val = min(merged_df[actual_power_col].min(), merged_df[predicted_power_col].min())
        max_val = max(merged_df[actual_power_col].max(), merged_df[predicted_power_col].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Power (W)')
        ax1.set_ylabel('Predicted Power (W)')
        ax1.set_title('Predicted vs Actual Power')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate and display R²
        correlation_matrix = np.corrcoef(merged_df[actual_power_col], merged_df[predicted_power_col])
        r_squared = correlation_matrix[0, 1] ** 2
        ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot 2: Residuals vs Frequency
        residuals = merged_df[predicted_power_col] - merged_df[actual_power_col]
        ax2.scatter(merged_df[frequency_col], residuals, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel('Residuals (Predicted - Actual)')
        ax2.set_title('Residuals vs Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Both curves vs frequency
        ax3.plot(actual_df[frequency_col], actual_df[actual_power_col], 
                'b-o', linewidth=2, markersize=6, label='Actual')
        ax3.plot(predicted_df[frequency_col], predicted_df[predicted_power_col], 
                'r-s', linewidth=2, markersize=6, label='Predicted')
        ax3.set_xlabel('Frequency (MHz)')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power vs Frequency Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error statistics
        ax4.axis('off')
        
        # Calculate error metrics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / merged_df[actual_power_col])) * 100
        
        error_text = f"Error Metrics:\n\n"
        error_text += f"MAE: {mae:.3f} W\n"
        error_text += f"RMSE: {rmse:.3f} W\n"
        error_text += f"MAPE: {mape:.1f}%\n"
        error_text += f"R²: {r_squared:.3f}\n\n"
        error_text += f"Data Points: {len(merged_df)}\n"
        error_text += f"Frequency Range: {merged_df[frequency_col].min()}-{merged_df[frequency_col].max()} MHz"
        
        ax4.text(0.1, 0.9, error_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power model validation plot saved to {save_path}")
        
        return fig
    
    def plot_power_breakdown(self,
                           df: pd.DataFrame,
                           frequency_col: str = 'frequency',
                           components: List[str] = ['gpu_power', 'memory_power', 'total_power'],
                           title: str = "Power Breakdown by Component",
                           save_path: Optional[str] = None) -> Figure:
        """
        Plot power breakdown by components.
        
        Args:
            df: DataFrame with component power data
            frequency_col: Column name for frequency values
            components: List of component power columns
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        
        # Plot 1: Stacked area chart
        bottom = np.zeros(len(df))
        for i, component in enumerate(components):
            if component in df.columns:
                ax1.fill_between(df[frequency_col], bottom, bottom + df[component],
                               label=component.replace('_', ' ').title(), 
                               color=colors[i], alpha=0.7)
                bottom += df[component]
        
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Power (W)')
        ax1.set_title('Power Breakdown - Stacked')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual component lines
        for i, component in enumerate(components):
            if component in df.columns:
                ax2.plot(df[frequency_col], df[component], 'o-',
                        color=colors[i], linewidth=2, markersize=6,
                        label=component.replace('_', ' ').title())
        
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power Breakdown - Individual Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power breakdown plot saved to {save_path}")
        
        return fig

    def plot_fgcs_power_validation(self,
                                 df: pd.DataFrame,
                                 predicted_col: str = 'predicted_power',
                                 actual_col: str = 'power',
                                 frequency_col: str = 'frequency',
                                 app_name: str = "Application",
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> Figure:
        """
        Plot FGCS power model validation comparing predicted vs actual power.
        
        Args:
            df: DataFrame with predicted and actual power values
            predicted_col: Column name for predicted power values
            actual_col: Column name for actual power values
            frequency_col: Column name for frequency values
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"FGCS Power Model Validation - {app_name}"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Predicted vs Actual Power
        ax1.scatter(df[actual_col], df[predicted_col], alpha=0.7, s=60)
        
        # Perfect prediction line
        min_val = min(df[actual_col].min(), df[predicted_col].min())
        max_val = max(df[actual_col].max(), df[predicted_col].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R² and RMSE
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(df[actual_col], df[predicted_col])
        rmse = np.sqrt(mean_squared_error(df[actual_col], df[predicted_col]))
        
        ax1.set_xlabel('Actual Power (W)')
        ax1.set_ylabel('Predicted Power (W)')
        ax1.set_title(f'Predicted vs Actual Power\nR² = {r2:.3f}, RMSE = {rmse:.2f}W')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power vs Frequency (both predicted and actual)
        ax2.plot(df[frequency_col], df[actual_col], 'bo-', linewidth=2, markersize=6, label='Actual Power')
        ax2.plot(df[frequency_col], df[predicted_col], 'ro-', linewidth=2, markersize=6, label='Predicted Power')
        
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power vs Frequency Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"FGCS power validation plot saved to {save_path}")
        
        return fig

    def plot_power_breakdown_analysis(self,
                                    df: pd.DataFrame,
                                    frequency_col: str = 'frequency',
                                    power_col: str = 'power',
                                    fp_activity_col: str = 'fp_activity',
                                    dram_activity_col: str = 'dram_activity',
                                    app_name: str = "Application",
                                    title: Optional[str] = None,
                                    save_path: Optional[str] = None) -> Figure:
        """
        Plot power breakdown analysis showing contributions from FP and DRAM activities.
        
        Args:
            df: DataFrame with power and activity data
            frequency_col: Column name for frequency values
            power_col: Column name for power values
            fp_activity_col: Column name for FP activity values
            dram_activity_col: Column name for DRAM activity values
            app_name: Application name for labeling
            title: Plot title (auto-generated if None)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        if title is None:
            title = f"Power Breakdown Analysis - {app_name}"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Power vs Frequency
        ax1.plot(df[frequency_col], df[power_col], 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Power (W)')
        ax1.set_title('Power vs Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power vs FP Activity
        ax2.scatter(df[fp_activity_col], df[power_col], alpha=0.7, s=60, c=df[frequency_col], cmap='viridis')
        ax2.set_xlabel('FP Activity')
        ax2.set_ylabel('Power (W)')
        ax2.set_title('Power vs FP Activity')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Power vs DRAM Activity
        ax3.scatter(df[dram_activity_col], df[power_col], alpha=0.7, s=60, c=df[frequency_col], cmap='viridis')
        ax3.set_xlabel('DRAM Activity')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power vs DRAM Activity')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: 3D Power Surface (if 3D available)
        if HAS_3D:
            ax4 = fig.add_subplot(224, projection='3d')
            scatter = ax4.scatter(df[fp_activity_col], df[dram_activity_col], df[power_col], 
                                c=df[frequency_col], cmap='viridis', s=60, alpha=0.7)
            ax4.set_xlabel('FP Activity')
            ax4.set_ylabel('DRAM Activity')
            ax4.set_zlabel('Power (W)')
            ax4.set_title('3D Power Surface')
        else:
            # Fallback to 2D heatmap
            scatter = ax4.scatter(df[fp_activity_col], df[dram_activity_col], 
                                c=df[power_col], cmap='viridis', s=80, alpha=0.7)
            ax4.set_xlabel('FP Activity')
            ax4.set_ylabel('DRAM Activity')
            ax4.set_title('Power Heatmap')
            plt.colorbar(scatter, ax=ax4, label='Power (W)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Power breakdown analysis plot saved to {save_path}")
        
        return fig

    def plot_energy_efficiency_analysis(self,
                                      df: pd.DataFrame,
                                      frequency_col: str = 'frequency',
                                      power_col: str = 'power',
                                      energy_col: str = 'energy',
                                      performance_col: str = 'throughput',
                                      app_name: str = "Application",
                                      title: Optional[str] = None,
                                      save_path: Optional[str] = None) -> Figure:
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
        if title is None:
            title = f"Energy Efficiency Analysis - {app_name}"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Calculate efficiency metrics
        energy_per_op = df[energy_col] / df[performance_col] if performance_col in df.columns else None
        power_efficiency = df[performance_col] / df[power_col] if performance_col in df.columns else None
        
        # Plot 1: Power Efficiency vs Frequency
        if power_efficiency is not None:
            ax1.plot(df[frequency_col], power_efficiency, 'g-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Frequency (MHz)')
            ax1.set_ylabel('Performance/Power (ops/W)')
            ax1.set_title('Power Efficiency vs Frequency')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy per Operation vs Frequency
        if energy_per_op is not None:
            ax2.plot(df[frequency_col], energy_per_op, 'r-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Frequency (MHz)')
            ax2.set_ylabel('Energy per Operation (J/op)')
            ax2.set_title('Energy per Operation vs Frequency')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Energy vs Performance Trade-off
        if performance_col in df.columns:
            scatter = ax3.scatter(df[energy_col], df[performance_col], 
                                c=df[frequency_col], cmap='viridis', s=80, alpha=0.7)
            ax3.set_xlabel('Energy (J)')
            ax3.set_ylabel('Performance (ops/s)')
            ax3.set_title('Energy-Performance Trade-off')
            plt.colorbar(scatter, ax=ax3, label='Frequency (MHz)')
        
        # Plot 4: Efficiency Contour Plot
        if power_efficiency is not None and energy_per_op is not None:
            ax4.scatter(power_efficiency, energy_per_op, 
                       c=df[frequency_col], cmap='viridis', s=80, alpha=0.7)
            ax4.set_xlabel('Power Efficiency (ops/W)')
            ax4.set_ylabel('Energy per Operation (J/op)')
            ax4.set_title('Efficiency Space')
            
            # Add Pareto frontier approximation
            combined_efficiency = power_efficiency * (1.0 / energy_per_op)  # Higher is better
            pareto_mask = np.zeros(len(combined_efficiency), dtype=bool)
            for i in range(len(combined_efficiency)):
                is_pareto = True
                for j in range(len(combined_efficiency)):
                    if (power_efficiency[j] >= power_efficiency[i] and 
                        energy_per_op[j] <= energy_per_op[i] and
                        (power_efficiency[j] > power_efficiency[i] or energy_per_op[j] < energy_per_op[i])):
                        is_pareto = False
                        break
                pareto_mask[i] = is_pareto
            
            if np.any(pareto_mask):
                pareto_power_eff = power_efficiency[pareto_mask]
                pareto_energy_per_op = energy_per_op[pareto_mask]
                sorted_indices = np.argsort(pareto_power_eff)
                ax4.plot(pareto_power_eff[sorted_indices], pareto_energy_per_op[sorted_indices], 
                        'r-', linewidth=2, alpha=0.7, label='Pareto Frontier')
                ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Energy efficiency analysis plot saved to {save_path}")
        
        return fig
def plot_power_histogram(df: pd.DataFrame,
                        power_col: str = 'power',
                        bins: int = 20,
                        title: str = "Power Consumption Distribution",
                        save_path: Optional[str] = None) -> Figure:
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
    
    n, bins_edges, patches = ax.hist(df[power_col], bins=bins, alpha=0.7, edgecolor='black')
    
    # Color bars by value
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.viridis(i / len(patches)))
    
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_power = df[power_col].mean()
    std_power = df[power_col].std()
    ax.axvline(mean_power, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_power:.2f}W')
    ax.axvline(mean_power + std_power, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_power + std_power:.2f}W')
    ax.axvline(mean_power - std_power, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_power - std_power:.2f}W')
    
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Power histogram saved to {save_path}")
    
    return fig
