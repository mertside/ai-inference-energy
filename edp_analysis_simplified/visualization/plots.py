"""
Visualization functions for GPU frequency optimization analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple


def create_pareto_plot(efficiency_df: pd.DataFrame, 
                      optimal_df: pd.DataFrame,
                      output_dir: str,
                      filename: str = "pareto_analysis") -> str:
    """
    Create Pareto frontier plot showing performance vs energy trade-offs.
    
    Args:
        efficiency_df: DataFrame with all efficiency data points
        optimal_df: DataFrame with optimal configurations
        output_dir: Output directory for plot
        filename: Base filename (without extension)
        
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all configurations
    for config in efficiency_df['config'].unique():
        config_data = efficiency_df[efficiency_df['config'] == config]
        plt.scatter(config_data['performance_penalty'], config_data['energy_savings'], 
                   alpha=0.6, label=config, s=20)
    
    # Highlight optimal points
    if len(optimal_df) > 0:
        plt.scatter(optimal_df['performance_penalty'], optimal_df['energy_savings'], 
                   color='red', s=100, marker='*', label='Optimal Configurations', 
                   zorder=5, edgecolors='black', linewidth=1)
    
    plt.xlabel('Performance Penalty (%)', fontsize=12)
    plt.ylabel('Energy Savings (%)', fontsize=12)
    plt.title('GPU Frequency Optimization: Performance vs Energy Trade-off', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add zero lines for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    png_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return png_path


def create_frequency_bar_plot(optimal_df: pd.DataFrame,
                             output_dir: str,
                             filename: str = "optimal_frequencies") -> str:
    """
    Create bar plot of optimal frequencies with performance impact labels.
    
    Args:
        optimal_df: DataFrame with optimal configurations
        output_dir: Output directory for plot
        filename: Base filename (without extension)
        
    Returns:
        Path to saved plot
    """
    if len(optimal_df) == 0:
        raise ValueError("No optimal configurations to plot")
    
    plt.figure(figsize=(14, 8))
    
    configs = optimal_df['config']
    frequencies = optimal_df['optimal_frequency']
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    
    bars = plt.bar(range(len(configs)), frequencies, color=colors)
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Optimal Frequency (MHz)', fontsize=12)
    plt.title('Optimal GPU Frequencies for Energy Efficiency', fontsize=14)
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    
    # Add performance impact and energy savings labels on bars
    for i, (bar, row) in enumerate(zip(bars, optimal_df.itertuples())):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{row.performance_penalty:.1f}%\n{row.energy_savings:.1f}% energy',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    png_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return png_path


def create_efficiency_ratio_plot(optimal_df: pd.DataFrame,
                                output_dir: str,
                                filename: str = "efficiency_ratios") -> str:
    """
    Create plot of efficiency ratios for optimal configurations.
    
    Args:
        optimal_df: DataFrame with optimal configurations
        output_dir: Output directory for plot
        filename: Base filename (without extension)
        
    Returns:
        Path to saved plot
    """
    if len(optimal_df) == 0:
        raise ValueError("No optimal configurations to plot")
    
    plt.figure(figsize=(14, 8))
    
    configs = optimal_df['config']
    efficiency_ratios = optimal_df['efficiency_ratio']
    
    # Color bars by efficiency ratio (green gradient)
    colors = plt.cm.Greens(0.4 + 0.6 * (efficiency_ratios / efficiency_ratios.max()))
    
    bars = plt.bar(range(len(configs)), efficiency_ratios, color=colors)
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Efficiency Ratio (Energy Savings / Performance Impact)', fontsize=12)
    plt.title('Energy Efficiency Ratios for Optimal Configurations', fontsize=14)
    plt.xticks(range(len(configs)), configs, rotation=45, ha='right')
    
    # Add energy savings labels on bars
    for i, (bar, row) in enumerate(zip(bars, optimal_df.itertuples())):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{row.energy_savings:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    png_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return png_path


def create_summary_dashboard(efficiency_df: pd.DataFrame,
                           optimal_df: pd.DataFrame,
                           output_dir: str,
                           filename: str = "optimization_dashboard") -> str:
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        efficiency_df: DataFrame with all efficiency data points
        optimal_df: DataFrame with optimal configurations
        output_dir: Output directory for plot
        filename: Base filename (without extension)
        
    Returns:
        Path to saved plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Pareto frontier plot
    for config in efficiency_df['config'].unique():
        config_data = efficiency_df[efficiency_df['config'] == config]
        ax1.scatter(config_data['performance_penalty'], config_data['energy_savings'], 
                   alpha=0.6, label=config, s=15)
    
    if len(optimal_df) > 0:
        ax1.scatter(optimal_df['performance_penalty'], optimal_df['energy_savings'], 
                   color='red', s=60, marker='*', label='Optimal', zorder=5)
    
    ax1.set_xlabel('Performance Penalty (%)')
    ax1.set_ylabel('Energy Savings (%)')
    ax1.set_title('Performance vs Energy Trade-off')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Frequency distribution
    if len(optimal_df) > 0:
        ax2.bar(range(len(optimal_df)), optimal_df['optimal_frequency'], 
               color='skyblue', alpha=0.7)
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('Frequency (MHz)')
        ax2.set_title('Optimal Frequencies')
        ax2.set_xticks(range(len(optimal_df)))
        ax2.set_xticklabels(optimal_df['config'], rotation=45, ha='right')
    
    # 3. Energy savings distribution
    if len(optimal_df) > 0:
        ax3.bar(range(len(optimal_df)), optimal_df['energy_savings'], 
               color='green', alpha=0.7)
        ax3.set_xlabel('Configuration')
        ax3.set_ylabel('Energy Savings (%)')
        ax3.set_title('Energy Savings by Configuration')
        ax3.set_xticks(range(len(optimal_df)))
        ax3.set_xticklabels(optimal_df['config'], rotation=45, ha='right')
    
    # 4. Category distribution pie chart
    if len(optimal_df) > 0:
        category_counts = optimal_df['category'].value_counts()
        colors = ['lightgreen', 'yellow', 'orange', 'red', 'darkred'][:len(category_counts)]
        ax4.pie(category_counts.values, labels=category_counts.index, 
               autopct='%1.0f%%', colors=colors)
        ax4.set_title('Performance Impact Categories')
    
    plt.tight_layout()
    
    # Save plot
    png_path = os.path.join(output_dir, f'{filename}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return png_path


def generate_all_plots(efficiency_df: pd.DataFrame,
                      optimal_df: pd.DataFrame,
                      output_dir: str) -> list:
    """
    Generate all standard visualization plots.
    
    Args:
        efficiency_df: DataFrame with all efficiency data points
        optimal_df: DataFrame with optimal configurations
        output_dir: Output directory for plots
        
    Returns:
        List of paths to generated plot files
    """
    plot_paths = []
    
    # Create pareto plot
    plot_paths.append(create_pareto_plot(efficiency_df, optimal_df, output_dir))
    
    # Create frequency and efficiency plots only if we have optimal configurations
    if len(optimal_df) > 0:
        plot_paths.append(create_frequency_bar_plot(optimal_df, output_dir))
        plot_paths.append(create_efficiency_ratio_plot(optimal_df, output_dir))
        plot_paths.append(create_summary_dashboard(efficiency_df, optimal_df, output_dir))
    
    return plot_paths
