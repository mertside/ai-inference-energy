"""
EDP Visualization Module

This module provides visualization functions for Energy-Delay Product (EDP) analysis,
including EDP plots, trade-off analysis, and optimization result visualization.

Key Features:
- EDP vs frequency plots
- Pareto frontier visualization
- Optimization comparison plots
- Energy-delay trade-off scatter plots
- 3D surface plots for multi-parameter analysis
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
    logger.warning("Matplotlib not available. EDP plotting functions will be limited.")

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
        raise ImportError("Matplotlib is required for EDP plotting. Install with: pip install matplotlib")


class EDPPlotter:
    """
    EDP visualization class providing comprehensive plotting capabilities for
    energy-delay product analysis and optimization results.
    """
    
    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize EDP plotter.
        
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
        logger.info(f"EDP plotter initialized with style: {style}")
    
    def plot_edp_vs_frequency(self,
                            df: pd.DataFrame,
                            frequency_col: str = 'frequency',
                            energy_col: str = 'energy',
                            time_col: str = 'execution_time',
                            optimal_freq: Optional[int] = None,
                            title: str = "EDP vs Frequency",
                            save_path: Optional[str] = None) -> Figure:
        """
        Plot Energy-Delay Product vs frequency.
        
        Args:
            df: DataFrame with frequency sweep data
            frequency_col: Column name for frequency values
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            optimal_freq: Optimal frequency to highlight
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Calculate EDP
        edp_values = df[energy_col] * df[time_col]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Main EDP curve
        ax.plot(df[frequency_col], edp_values, 'b-o', linewidth=2, markersize=6, label='EDP')
        
        # Highlight optimal frequency if provided
        if optimal_freq is not None:
            optimal_row = df[df[frequency_col] == optimal_freq]
            if not optimal_row.empty:
                optimal_edp = optimal_row[energy_col].iloc[0] * optimal_row[time_col].iloc[0]
                ax.plot(optimal_freq, optimal_edp, 'ro', markersize=10, label=f'Optimal ({optimal_freq} MHz)')
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Energy-Delay Product (J·s)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"EDP plot saved to {save_path}")
        
        return fig
    
    def plot_energy_delay_tradeoff(self,
                                 df: pd.DataFrame,
                                 energy_col: str = 'energy',
                                 time_col: str = 'execution_time',
                                 frequency_col: str = 'frequency',
                                 pareto_df: Optional[pd.DataFrame] = None,
                                 optimal_points: Optional[Dict] = None,
                                 title: str = "Energy-Delay Trade-off",
                                 save_path: Optional[str] = None) -> Figure:
        """
        Plot energy vs execution time trade-off scatter plot.
        
        Args:
            df: DataFrame with energy and time data
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            frequency_col: Column name for frequency values
            pareto_df: DataFrame with Pareto-optimal points
            optimal_points: Dictionary with optimization results
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Main scatter plot colored by frequency
        scatter = ax.scatter(df[time_col], df[energy_col], 
                           c=df[frequency_col], cmap='viridis', 
                           alpha=0.7, s=60, label='Configurations')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frequency (MHz)')
        
        # Plot Pareto frontier if provided
        if pareto_df is not None and not pareto_df.empty:
            # Sort by time for proper line connection
            pareto_sorted = pareto_df.sort_values(time_col)
            ax.plot(pareto_sorted[time_col], pareto_sorted[energy_col], 
                   'r-o', linewidth=2, markersize=8, label='Pareto Frontier')
        
        # Highlight optimal points if provided
        if optimal_points:
            colors = ['red', 'orange', 'purple']
            markers = ['s', '^', 'D']
            labels = ['EDP Optimal', 'ED²P Optimal', 'Weighted Optimal']
            
            for i, (key, result) in enumerate(optimal_points.items()):
                if hasattr(result, 'energy') and hasattr(result, 'execution_time'):
                    ax.plot(result.execution_time, result.energy, 
                           color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                           markersize=10, label=labels[i % len(labels)])
        
        ax.set_xlabel('Execution Time (s)')
        ax.set_ylabel('Energy (J)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Energy-delay trade-off plot saved to {save_path}")
        
        return fig
    
    def plot_optimization_comparison(self,
                                   optimization_results: Dict,
                                   metrics: List[str] = ['energy', 'execution_time', 'frequency'],
                                   title: str = "Optimization Methods Comparison",
                                   save_path: Optional[str] = None) -> Figure:
        """
        Compare different optimization methods in a radar/bar chart.
        
        Args:
            optimization_results: Results from optimization analysis
            metrics: List of metrics to compare
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract data for comparison
        methods = ['EDP Optimal', 'ED²P Optimal', 'Min Energy', 'Min Time']
        data = []
        
        # Get results
        edp_result = optimization_results.get('edp_optimal')
        ed2p_result = optimization_results.get('ed2p_optimal')
        min_energy = optimization_results.get('min_energy_config', {})
        min_time = optimization_results.get('min_time_config', {})
        
        results = [edp_result, ed2p_result, min_energy, min_time]
        
        # Prepare data matrix
        for result in results:
            row = []
            for metric in metrics:
                if hasattr(result, metric):
                    row.append(getattr(result, metric))
                elif isinstance(result, dict) and metric in result:
                    row.append(result[metric])
                else:
                    row.append(0)
            data.append(row)
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        colors = ['blue', 'orange', 'green', 'red']
        
        for i, metric in enumerate(metrics):
            values = [row[i] for row in data]
            bars = axes[i].bar(methods, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization comparison plot saved to {save_path}")
        
        return fig
    
    def plot_3d_edp_surface(self,
                          df: pd.DataFrame,
                          x_col: str = 'frequency',
                          y_col: str = 'power',
                          energy_col: str = 'energy',
                          time_col: str = 'execution_time',
                          title: str = "3D EDP Surface",
                          save_path: Optional[str] = None) -> Figure:
        """
        Create 3D surface plot for EDP analysis.
        
        Args:
            df: DataFrame with data for 3D plotting
            x_col: Column for X-axis
            y_col: Column for Y-axis
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        if not HAS_3D:
            raise ImportError("3D plotting requires mpl_toolkits.mplot3d")
        
        # Calculate EDP
        edp_values = df[energy_col] * df[time_col]
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        scatter = ax.scatter(df[x_col], df[y_col], edp_values, 
                           c=edp_values, cmap='viridis', s=60)
        
        ax.set_xlabel(f'{x_col.replace("_", " ").title()}')
        ax.set_ylabel(f'{y_col.replace("_", " ").title()}')
        ax.set_zlabel('EDP (J·s)')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('EDP (J·s)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D EDP surface plot saved to {save_path}")
        
        return fig
    
    def plot_frequency_sweep_analysis(self,
                                    df: pd.DataFrame,
                                    frequency_col: str = 'frequency',
                                    energy_col: str = 'energy',
                                    time_col: str = 'execution_time',
                                    power_col: str = 'power',
                                    optimal_results: Optional[Dict] = None,
                                    title: str = "Frequency Sweep Analysis",
                                    save_path: Optional[str] = None) -> Figure:
        """
        Create comprehensive frequency sweep analysis plot.
        
        Args:
            df: DataFrame with frequency sweep data
            frequency_col: Column name for frequency values
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            power_col: Column name for power values
            optimal_results: Optimization results to highlight
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate EDP and ED²P
        edp_values = df[energy_col] * df[time_col]
        ed2p_values = df[energy_col] * (df[time_col] ** 2)
        
        # Plot 1: Energy vs Frequency
        ax1.plot(df[frequency_col], df[energy_col], 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Energy (J)')
        ax1.set_title('Energy vs Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time vs Frequency
        ax2.plot(df[frequency_col], df[time_col], 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Frequency (MHz)')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Execution Time vs Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: EDP vs Frequency
        ax3.plot(df[frequency_col], edp_values, 'r-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Frequency (MHz)')
        ax3.set_ylabel('EDP (J·s)')
        ax3.set_title('EDP vs Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Power vs Frequency
        ax4.plot(df[frequency_col], df[power_col], 'm-o', linewidth=2, markersize=4)
        ax4.set_xlabel('Frequency (MHz)')
        ax4.set_ylabel('Power (W)')
        ax4.set_title('Power vs Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Highlight optimal points if provided
        if optimal_results:
            for result_name, result in optimal_results.items():
                if hasattr(result, 'frequency'):
                    freq = result.frequency
                    # Find corresponding row in dataframe
                    result_row = df[df[frequency_col] == freq]
                    if not result_row.empty:
                        row = result_row.iloc[0]
                        edp_val = row[energy_col] * row[time_col]
                        
                        # Mark optimal points
                        ax1.plot(freq, row[energy_col], 'ro', markersize=8)
                        ax2.plot(freq, row[time_col], 'ro', markersize=8)
                        ax3.plot(freq, edp_val, 'ro', markersize=8)
                        ax4.plot(freq, row[power_col], 'ro', markersize=8)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Frequency sweep analysis plot saved to {save_path}")
        
        return fig


def plot_edp_heatmap(df: pd.DataFrame,
                    x_col: str,
                    y_col: str,
                    energy_col: str = 'energy',
                    time_col: str = 'execution_time',
                    title: str = "EDP Heatmap",
                    save_path: Optional[str] = None) -> Figure:
    """
    Create a heatmap of EDP values across two parameters.
    
    Args:
        df: DataFrame with parameter sweep data
        x_col: Column for X-axis parameter
        y_col: Column for Y-axis parameter
        energy_col: Column name for energy values
        time_col: Column name for execution time values
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    check_plotting_dependencies()
    
    if not HAS_SEABORN:
        logger.warning("Seaborn not available. Using basic matplotlib heatmap.")
    
    # Calculate EDP
    df_work = df.copy()
    df_work['edp'] = df_work[energy_col] * df_work[time_col]
    
    # Pivot for heatmap
    heatmap_data = df_work.pivot_table(values='edp', index=y_col, columns=x_col, aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
    else:
        im = ax.imshow(heatmap_data.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticklabels(heatmap_data.index)
        plt.colorbar(im, ax=ax, label='EDP (J·s)')
    
    ax.set_xlabel(f'{x_col.replace("_", " ").title()}')
    ax.set_ylabel(f'{y_col.replace("_", " ").title()}')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"EDP heatmap saved to {save_path}")
    
    return fig


def create_optimization_summary_plot(optimization_results: Dict,
                                   app_name: str = "Application",
                                   save_path: Optional[str] = None) -> Figure:
    """
    Create a comprehensive summary plot of optimization results.
    
    Args:
        optimization_results: Results from optimization analysis
        app_name: Application name for the title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib Figure object
    """
    check_plotting_dependencies()
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main energy-delay trade-off plot
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Get data
    pareto_frontier = optimization_results.get('pareto_frontier', pd.DataFrame())
    edp_optimal = optimization_results.get('edp_optimal')
    ed2p_optimal = optimization_results.get('ed2p_optimal')
    
    # Plot summary text
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    summary_text = f"Optimization Summary: {app_name}\n\n"
    if edp_optimal:
        summary_text += f"EDP Optimal:\n"
        summary_text += f"  Frequency: {edp_optimal.frequency} MHz\n"
        summary_text += f"  Energy: {edp_optimal.energy:.2f} J\n"
        summary_text += f"  Time: {edp_optimal.execution_time:.2f} s\n\n"
    
    if ed2p_optimal:
        summary_text += f"ED²P Optimal:\n"
        summary_text += f"  Frequency: {ed2p_optimal.frequency} MHz\n"
        summary_text += f"  Energy: {ed2p_optimal.energy:.2f} J\n"
        summary_text += f"  Time: {ed2p_optimal.execution_time:.2f} s\n"
    
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # Statistics plot
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    stats = optimization_results.get('statistics', {})
    stats_text = "Statistics:\n\n"
    stats_text += f"Total Configs: {stats.get('total_configurations', 'N/A')}\n"
    stats_text += f"Pareto Points: {stats.get('pareto_points', 'N/A')}\n"
    stats_text += f"Energy Range: {stats.get('energy_range', 0):.2f} J\n"
    stats_text += f"Time Range: {stats.get('time_range', 0):.2f} s\n"
    
    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # Improvement bar chart
    ax4 = fig.add_subplot(gs[2, :])
    
    improvements = []
    labels = []
    
    if edp_optimal and hasattr(edp_optimal, 'energy_improvement') and edp_optimal.energy_improvement:
        improvements.append(edp_optimal.energy_improvement)
        labels.append('EDP Energy\nImprovement (%)')
    
    if edp_optimal and hasattr(edp_optimal, 'time_improvement') and edp_optimal.time_improvement:
        improvements.append(edp_optimal.time_improvement)
        labels.append('EDP Time\nImprovement (%)')
    
    if improvements:
        bars = ax4.bar(labels, improvements, color=['green', 'blue'], alpha=0.7)
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Performance Improvements')
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
    
    plt.suptitle(f"Optimization Analysis Summary: {app_name}", fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Optimization summary plot saved to {save_path}")
    
    return fig

    def plot_feature_importance_for_edp(self,
                                      feature_importance: Dict[str, float],
                                      title: str = "Feature Importance for EDP Optimization",
                                      max_features: int = 20,
                                      save_path: Optional[str] = None) -> Figure:
        """
        Plot feature importance for EDP optimization.
        
        Args:
            feature_importance: Dictionary with feature names and importance scores
            title: Plot title
            max_features: Maximum number of features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        sorted_features = sorted_features[:max_features]
        
        features, importance = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, alpha=0.7)
        
        # Color bars based on importance
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Highest importance at top
        ax.set_xlabel('Importance Score')
        ax.set_title(title)
        
        # Add value labels on bars
        for i, v in enumerate(importance):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', ha='left')
        
        ax.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig

    def plot_fgcs_optimization_results(self,
                                     optimization_results: Dict[str, Any],
                                     title: str = "FGCS EDP Optimization Results",
                                     save_path: Optional[str] = None) -> Figure:
        """
        Plot FGCS-style optimization results showing EDP and ED²P optimal points.
        
        Args:
            optimization_results: Results from FGCS EDP optimizer
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data from results
        if 'frequency_sweep_data' in optimization_results:
            sweep_data = optimization_results['frequency_sweep_data']
            frequencies = sweep_data['sm_app_clock']
            energies = sweep_data['predicted_n_to_r_energy']
            times = sweep_data['predicted_n_to_r_run_time']
            powers = sweep_data['predicted_n_to_r_power_usage']
            edp_values = energies * times
            ed2p_values = energies * (times ** 2)
            
            # Plot 1: Energy vs Frequency
            ax1.plot(frequencies, energies, 'b-o', linewidth=2, markersize=4)
            ax1.set_xlabel('Frequency (MHz)')
            ax1.set_ylabel('Energy (J)')
            ax1.set_title('Energy vs Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Execution Time vs Frequency
            ax2.plot(frequencies, times, 'g-o', linewidth=2, markersize=4)
            ax2.set_xlabel('Frequency (MHz)')
            ax2.set_ylabel('Execution Time (s)')
            ax2.set_title('Execution Time vs Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: EDP vs Frequency
            ax3.plot(frequencies, edp_values, 'r-o', linewidth=2, markersize=4)
            if 'edp_optimal' in optimization_results:
                edp_opt = optimization_results['edp_optimal']
                ax3.plot(edp_opt['frequency'], edp_opt['edp'], 'ro', markersize=10, 
                        label=f"EDP Optimal ({edp_opt['frequency']} MHz)")
            ax3.set_xlabel('Frequency (MHz)')
            ax3.set_ylabel('EDP (J·s)')
            ax3.set_title('Energy-Delay Product vs Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: ED²P vs Frequency
            ax4.plot(frequencies, ed2p_values, 'm-o', linewidth=2, markersize=4)
            if 'ed2p_optimal' in optimization_results:
                ed2p_opt = optimization_results['ed2p_optimal']
                ax4.plot(ed2p_opt['frequency'], ed2p_opt['ed2p'], 'mo', markersize=10,
                        label=f"ED²P Optimal ({ed2p_opt['frequency']} MHz)")
            ax4.set_xlabel('Frequency (MHz)')
            ax4.set_ylabel('ED²P (J·s²)')
            ax4.set_title('Energy-Delay² Product vs Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"FGCS optimization results plot saved to {save_path}")
        
        return fig

    def create_comprehensive_edp_dashboard(self,
                                         profiling_data: pd.DataFrame,
                                         optimization_results: Dict[str, Any],
                                         feature_importance: Optional[Dict[str, float]] = None,
                                         app_name: str = "Application",
                                         save_path: Optional[str] = None) -> Figure:
        """
        Create comprehensive EDP analysis dashboard combining all visualizations.
        
        Args:
            profiling_data: DataFrame with profiling data
            optimization_results: Results from optimization analysis
            feature_importance: Feature importance scores (optional)
            app_name: Application name for labeling
            save_path: Path to save the dashboard
            
        Returns:
            Matplotlib Figure object
        """
        # Determine layout based on available data
        if feature_importance is not None:
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(15, 12))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Calculate EDP values
        energy_col = 'energy' if 'energy' in profiling_data.columns else 'energy_joules'
        time_col = 'execution_time'
        freq_col = 'frequency'
        
        edp_values = profiling_data[energy_col] * profiling_data[time_col]
        
        # Plot 1: EDP vs Frequency
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(profiling_data[freq_col], edp_values, 'b-o', linewidth=2, markersize=6)
        if 'edp_optimal' in optimization_results:
            opt_freq = optimization_results['edp_optimal']['frequency']
            opt_row = profiling_data[profiling_data[freq_col] == opt_freq]
            if not opt_row.empty:
                opt_edp = opt_row[energy_col].iloc[0] * opt_row[time_col].iloc[0]
                ax1.plot(opt_freq, opt_edp, 'ro', markersize=10, label='EDP Optimal')
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('EDP (J·s)')
        ax1.set_title(f'EDP vs Frequency - {app_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy vs Time Trade-off
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(profiling_data[energy_col], profiling_data[time_col], 
                            c=profiling_data[freq_col], cmap='viridis', s=80, alpha=0.7)
        ax2.set_xlabel('Energy (J)')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Energy-Time Trade-off')
        plt.colorbar(scatter, ax=ax2, label='Frequency (MHz)')
        
        # Plot 3: Power vs Frequency
        ax3 = fig.add_subplot(gs[0, 2])
        if 'power' in profiling_data.columns:
            ax3.plot(profiling_data[freq_col], profiling_data['power'], 'g-o', linewidth=2, markersize=6)
            ax3.set_xlabel('Frequency (MHz)')
            ax3.set_ylabel('Power (W)')
            ax3.set_title('Power vs Frequency')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Optimization Summary
        ax4 = fig.add_subplot(gs[1, 0])
        if 'edp_optimal' in optimization_results and 'ed2p_optimal' in optimization_results:
            metrics = ['EDP Optimal', 'ED²P Optimal']
            frequencies = [optimization_results['edp_optimal']['frequency'],
                          optimization_results['ed2p_optimal']['frequency']]
            colors = ['red', 'purple']
            bars = ax4.bar(metrics, frequencies, color=colors, alpha=0.7)
            ax4.set_ylabel('Optimal Frequency (MHz)')
            ax4.set_title('Optimization Summary')
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{freq} MHz', ha='center', va='bottom')
        
        # Plot 5: Pareto Frontier (if we have multiple objectives)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(profiling_data[energy_col], profiling_data[time_col], 
                   alpha=0.6, s=60, label='All Points')
        
        # Highlight optimal points
        if 'edp_optimal' in optimization_results:
            edp_opt = optimization_results['edp_optimal']
            ax5.scatter([edp_opt['energy']], [edp_opt['runtime']], 
                       color='red', s=120, marker='*', label='EDP Optimal')
        
        if 'ed2p_optimal' in optimization_results:
            ed2p_opt = optimization_results['ed2p_optimal']
            ax5.scatter([ed2p_opt['energy']], [ed2p_opt['runtime']], 
                       color='purple', s=120, marker='*', label='ED²P Optimal')
        
        ax5.set_xlabel('Energy (J)')
        ax5.set_ylabel('Execution Time (s)')
        ax5.set_title('Pareto Frontier Analysis')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Energy Efficiency
        ax6 = fig.add_subplot(gs[1, 2])
        efficiency = 1.0 / (profiling_data[energy_col] * profiling_data[time_col])  # Inverse of EDP
        ax6.plot(profiling_data[freq_col], efficiency, 'orange', linewidth=2, marker='o', markersize=6)
        ax6.set_xlabel('Frequency (MHz)')
        ax6.set_ylabel('Energy Efficiency (1/EDP)')
        ax6.set_title('Energy Efficiency vs Frequency')
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Feature Importance (if available)
        if feature_importance is not None:
            ax7 = fig.add_subplot(gs[2, :])
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            sorted_features = sorted_features[:15]  # Top 15 features
            
            features, importance = zip(*sorted_features)
            y_pos = np.arange(len(features))
            bars = ax7.barh(y_pos, importance, alpha=0.7)
            
            # Color bars based on importance
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels(features)
            ax7.invert_yaxis()
            ax7.set_xlabel('Importance Score')
            ax7.set_title('Feature Importance for EDP Optimization')
            
            # Add value labels
            for i, v in enumerate(importance):
                ax7.text(v + 0.01, i, f'{v:.3f}', va='center', ha='left', fontsize=9)
        
        # Add overall title
        fig.suptitle(f'Comprehensive EDP Analysis Dashboard - {app_name}', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive EDP dashboard saved to {save_path}")
        
        return fig

# Convenience functions for quick plotting
