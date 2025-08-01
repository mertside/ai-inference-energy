"""
Advanced GPU Frequency Optimization Visualizer

This module provides sophisticated visualization capabilities for GPU frequency
optimization analysis, focusing on the trade-offs between performance and energy
efficiency. Designed for research-quality plots with comprehensive analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Try to import seaborn, fall back to matplotlib-only if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: Seaborn not available, using matplotlib-only plotting")

class OptimizationVisualizer:
    """Advanced visualization for frequency optimization results"""
    
    def __init__(self, style='default', figure_size=(12, 8)):
        """
        Initialize the OptimizationVisualizer
        
        Args:
            style: Matplotlib style ('default', 'ggplot', 'bmh', etc.)
            figure_size: Default figure size as (width, height)
        """
        self.figure_size = figure_size
        
        # Set matplotlib style with fallback
        try:
            if style == 'seaborn' and HAS_SEABORN:
                sns.set_style("whitegrid")
            else:
                plt.style.use(style)
        except Exception as e:
            print(f"Warning: Could not set style '{style}', using default: {e}")
            plt.style.use('default')
        
        # Define color palettes for different categories
        self.category_colors = {
            'Minimal Impact': '#2E8B57',    # Sea Green
            'Low Impact': '#32CD32',        # Lime Green  
            'Moderate Impact': '#FFA500',   # Orange
            'High Impact': '#FF6347',       # Tomato
            'Extreme Impact': '#DC143C'     # Crimson
        }
        
        # GPU and application colors
        self.gpu_colors = {
            'A100': '#1f77b4',  # Blue
            'V100': '#ff7f0e'   # Orange
        }
        
        self.app_colors = {
            'LLAMA': '#2ca02c',         # Green
            'STABLEDIFFUSION': '#d62728', # Red
            'VIT': '#9467bd',           # Purple
            'WHISPER': '#8c564b'        # Brown
        }
    
    def create_pareto_frontier_plot(self, efficiency_df: pd.DataFrame, output_dir: str) -> plt.Figure:
        """
        Create comprehensive Pareto frontier analysis
        
        Args:
            efficiency_df: DataFrame with efficiency metrics
            output_dir: Directory to save plots
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GPU Frequency Optimization: Pareto Frontier Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Energy Savings vs Performance Degradation (main trade-off)
        ax1 = axes[0, 0]
        for (gpu, app), group in efficiency_df.groupby(['gpu', 'application']):
            gpu_upper = gpu.upper()
            app_upper = app.upper()
            ax1.scatter(group['performance_degradation'], group['energy_savings'], 
                       c=self.gpu_colors.get(gpu_upper, '#333333'), 
                       marker=self.app_markers.get(app_upper, 'o'), 
                       s=60, alpha=0.7, label=f'{gpu_upper}+{app_upper}')
        
        ax1.set_xlabel('Performance Degradation (%)')
        ax1.set_ylabel('Energy Savings (%)')
        ax1.set_title('Energy Savings vs Performance Degradation')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=15, color='red', linestyle='--', alpha=0.7, label='15% Performance Limit')
        ax1.axvline(x=5, color='green', linestyle='--', alpha=0.7, label='5% Minimal Impact')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Add Pareto frontier line
        pareto_points = self._find_pareto_frontier(
            efficiency_df['performance_degradation'], 
            efficiency_df['energy_savings']
        )
        if len(pareto_points) > 1:
            pareto_x, pareto_y = zip(*pareto_points)
            ax1.plot(pareto_x, pareto_y, 'r-', linewidth=2, alpha=0.8, label='Pareto Frontier')
        
        # Plot 2: Efficiency Ratio vs Frequency
        ax2 = axes[0, 1]
        for (gpu, app), group in efficiency_df.groupby(['gpu', 'application']):
            gpu_upper = gpu.upper()
            app_upper = app.upper()
            # Filter out extreme values for better visualization
            plot_group = group[group['efficiency_ratio'] < 100]
            if len(plot_group) > 0:
                ax2.plot(plot_group['frequency'], plot_group['efficiency_ratio'], 
                        marker=self.app_markers.get(app_upper, 'o'), 
                        color=self.gpu_colors.get(gpu_upper, '#333333'),
                        linewidth=2, markersize=6, label=f'{gpu_upper}+{app_upper}')
        
        ax2.set_xlabel('GPU Frequency (MHz)')
        ax2.set_ylabel('Efficiency Ratio (Energy Savings / Performance Loss)')
        ax2.set_title('Energy Efficiency vs Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='2:1 Efficiency Target')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 3: EDP Improvement vs Performance Degradation
        ax3 = axes[1, 0]
        for (gpu, app), group in efficiency_df.groupby(['gpu', 'application']):
            gpu_upper = gpu.upper()
            app_upper = app.upper()
            ax3.scatter(group['performance_degradation'], group['edp_improvement'], 
                       c=self.gpu_colors.get(gpu_upper, '#333333'), 
                       marker=self.app_markers.get(app_upper, 'o'), 
                       s=60, alpha=0.7, label=f'{gpu_upper}+{app_upper}')
        
        ax3.set_xlabel('Performance Degradation (%)')
        ax3.set_ylabel('EDP Improvement (%)')
        ax3.set_title('EDP Improvement vs Performance Degradation')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=15, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(x=5, color='green', linestyle='--', alpha=0.7)
        
        # Plot 4: Energy vs Power Trade-off
        ax4 = axes[1, 1]
        for (gpu, app), group in efficiency_df.groupby(['gpu', 'application']):
            gpu_upper = gpu.upper()
            app_upper = app.upper()
            ax4.scatter(group['power_reduction'], group['energy_savings'], 
                       c=self.gpu_colors.get(gpu_upper, '#333333'), 
                       marker=self.app_markers.get(app_upper, 'o'), 
                       s=60, alpha=0.7, label=f'{gpu_upper}+{app_upper}')
        
        ax4.set_xlabel('Power Reduction (%)')
        ax4.set_ylabel('Energy Savings (%)')
        ax4.set_title('Energy Savings vs Power Reduction')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        png_path = os.path.join(output_dir, 'pareto_frontier_analysis.png')
        pdf_path = os.path.join(output_dir, 'pareto_frontier_analysis.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        return fig
    
    def create_optimal_selection_plot(self, optimal_df: pd.DataFrame, output_dir: str) -> plt.Figure:
        """
        Create visualization of optimal frequency selections with efficiency focus
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Optimal GPU Frequency Selection: Minimal Degradation, Maximum Efficiency', 
                    fontsize=14, fontweight='bold')
        
        # Plot 1: Performance vs Energy Trade-off for optimal points
        ax1.scatter(optimal_df['performance_degradation'], optimal_df['energy_savings'], 
                   c=[self.category_colors.get(cat, '#333333') for cat in optimal_df['category']], 
                   s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Add labels for each point
        for idx, row in optimal_df.iterrows():
            ax1.annotate(f"{row['gpu']}+{row['application']}", 
                        (row['performance_degradation'], row['energy_savings']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax1.set_xlabel('Performance Degradation (%)')
        ax1.set_ylabel('Energy Savings (%)')
        ax1.set_title('Optimal Configurations: Performance vs Energy Trade-off')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=15, color='red', linestyle='--', alpha=0.7, label='15% Performance Limit')
        ax1.axvline(x=5, color='green', linestyle='--', alpha=0.7, label='5% Minimal Impact')
        
        # Add category legend
        legend_elements = []
        for category, color in self.category_colors.items():
            if category in optimal_df['category'].values:
                legend_elements.append(plt.scatter([], [], c=color, s=100, label=category.split(' ', 1)[1]))
        ax1.legend(handles=legend_elements)
        
        # Plot 2: Efficiency Ratio by Configuration
        configs = optimal_df['config'].tolist()
        efficiency_ratios = optimal_df['efficiency_ratio'].tolist()
        colors = [self.category_colors.get(cat, '#333333') for cat in optimal_df['category']]
        
        # Cap efficiency ratios for better visualization
        capped_ratios = [min(ratio, 50) if ratio != float('inf') else 50 for ratio in efficiency_ratios]
        
        bars = ax2.bar(range(len(configs)), capped_ratios, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('GPU + Application Configuration')
        ax2.set_ylabel('Efficiency Ratio (Energy Savings / Performance Loss)')
        ax2.set_title('Energy Efficiency by Configuration')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='2:1 Target Efficiency')
        ax2.axhline(y=5.0, color='blue', linestyle='--', alpha=0.7, label='5:1 Excellent Efficiency')
        
        # Add value labels on bars
        for bar, orig_value, capped_value in zip(bars, efficiency_ratios, capped_ratios):
            if orig_value == float('inf'):
                label = '∞'
            elif orig_value > 50:
                label = f'{orig_value:.0f}*'
            else:
                label = f'{orig_value:.1f}'
            
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    label, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plots
        png_path = os.path.join(output_dir, 'optimal_selection_detailed.png')
        pdf_path = os.path.join(output_dir, 'optimal_selection_detailed.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        return fig
    
    def create_frequency_sweep_analysis(self, efficiency_df: pd.DataFrame, output_dir: str) -> plt.Figure:
        """
        Create detailed frequency sweep analysis for all configurations
        """
        combinations = efficiency_df[['gpu', 'application']].drop_duplicates().sort_values(['gpu', 'application'])
        n_combinations = len(combinations)
        
        # Calculate grid dimensions
        cols = min(4, n_combinations)
        rows = (n_combinations + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Frequency Sweep Analysis: Energy vs Performance Trade-offs', 
                    fontsize=16, fontweight='bold')
        
        # Handle single plot case
        if n_combinations == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (_, row) in enumerate(combinations.iterrows()):
            if idx >= len(axes):
                break
                
            gpu, app = row['gpu'], row['application']
            ax = axes[idx]
            
            # Get data for this combination
            subset = efficiency_df[
                (efficiency_df['gpu'] == gpu) & 
                (efficiency_df['application'] == app)
            ].sort_values('frequency')
            
            # Create twin axes for energy and performance
            ax2 = ax.twinx()
            
            # Plot energy savings
            line1 = ax.plot(subset['frequency'], subset['energy_savings'], 
                           'g-o', linewidth=2, markersize=4, label='Energy Savings')
            
            # Plot performance degradation (use absolute value for better visualization)
            abs_perf_deg = subset['performance_degradation'].abs()
            line2 = ax2.plot(subset['frequency'], abs_perf_deg, 
                            'r-s', linewidth=2, markersize=4, label='|Performance Degradation|')
            
            # Mark optimal point
            optimal_idx = subset['efficiency_ratio'].idxmax()
            optimal_point = subset.loc[optimal_idx]
            ax.scatter(optimal_point['frequency'], optimal_point['energy_savings'], 
                      s=120, c='gold', marker='*', edgecolors='black', linewidth=2, 
                      label='Optimal', zorder=5)
            
            # Mark points within performance constraint
            constraint_points = subset[subset['performance_degradation'].abs() <= 15]
            if len(constraint_points) > 0:
                ax.scatter(constraint_points['frequency'], constraint_points['energy_savings'], 
                          s=30, c='lightblue', alpha=0.6, label='≤15% Degradation', zorder=3)
            
            # Formatting
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Energy Savings (%)', color='green')
            ax2.set_ylabel('|Performance Degradation| (%)', color='red')
            ax.set_title(f'{gpu.upper()}+{app.upper()}')
            ax.grid(True, alpha=0.3)
            
            # Performance constraint line
            ax2.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='15% Limit')
            ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% Minimal')
            
            # Color axes labels
            ax.tick_params(axis='y', labelcolor='green')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_combinations, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plots
        png_path = os.path.join(output_dir, 'frequency_sweep_detailed.png')
        pdf_path = os.path.join(output_dir, 'frequency_sweep_detailed.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        return fig
    
    def create_efficiency_heatmap(self, efficiency_df: pd.DataFrame, output_dir: str) -> plt.Figure:
        """
        Create efficiency heatmap showing energy savings vs performance degradation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('GPU Frequency Optimization Efficiency Heatmaps', fontsize=14, fontweight='bold')
        
        # Create pivot tables for heatmaps
        # Heatmap 1: Energy Savings
        pivot_energy = efficiency_df.pivot_table(
            values='energy_savings', 
            index='application', 
            columns='gpu', 
            aggfunc='max'  # Take maximum energy savings for each combination
        )
        
        im1 = ax1.imshow(pivot_energy.values, cmap='Greens', aspect='auto')
        ax1.set_xticks(range(len(pivot_energy.columns)))
        ax1.set_yticks(range(len(pivot_energy.index)))
        ax1.set_xticklabels(pivot_energy.columns)
        ax1.set_yticklabels(pivot_energy.index)
        ax1.set_title('Maximum Energy Savings (%)')
        
        # Add text annotations
        for i in range(len(pivot_energy.index)):
            for j in range(len(pivot_energy.columns)):
                text = ax1.text(j, i, f'{pivot_energy.iloc[i, j]:.1f}%',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Heatmap 2: Efficiency Ratio (capped for visualization)
        efficiency_capped = efficiency_df.copy()
        efficiency_capped['efficiency_ratio_capped'] = efficiency_capped['efficiency_ratio'].apply(
            lambda x: min(x, 20) if x != float('inf') else 20
        )
        
        pivot_efficiency = efficiency_capped.pivot_table(
            values='efficiency_ratio_capped', 
            index='application', 
            columns='gpu', 
            aggfunc='max'
        )
        
        im2 = ax2.imshow(pivot_efficiency.values, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(pivot_efficiency.columns)))
        ax2.set_yticks(range(len(pivot_efficiency.index)))
        ax2.set_xticklabels(pivot_efficiency.columns)
        ax2.set_yticklabels(pivot_efficiency.index)
        ax2.set_title('Maximum Efficiency Ratio (capped at 20:1)')
        
        # Add text annotations
        for i in range(len(pivot_efficiency.index)):
            for j in range(len(pivot_efficiency.columns)):
                orig_value = efficiency_df[
                    (efficiency_df['gpu'] == pivot_efficiency.columns[j]) & 
                    (efficiency_df['application'] == pivot_efficiency.index[i])
                ]['efficiency_ratio'].max()
                
                if orig_value == float('inf'):
                    text_val = '∞'
                elif orig_value > 20:
                    text_val = f'{orig_value:.0f}*'
                else:
                    text_val = f'{orig_value:.1f}'
                
                text = ax2.text(j, i, text_val,
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.tight_layout()
        
        # Save plots
        png_path = os.path.join(output_dir, 'efficiency_heatmap.png')
        pdf_path = os.path.join(output_dir, 'efficiency_heatmap.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        return fig
    
    def _find_pareto_frontier(self, x_values, y_values):
        """Find Pareto frontier points for x vs y optimization"""
        points = list(zip(x_values, y_values))
        points.sort()
        
        pareto_points = []
        max_y = float('-inf')
        
        for x, y in points:
            if y > max_y:
                pareto_points.append((x, y))
                max_y = y
        
        return pareto_points
    
    def generate_comprehensive_report(self, efficiency_df: pd.DataFrame, optimal_df: pd.DataFrame, 
                                    output_dir: str) -> None:
        """
        Generate comprehensive visualization report
        """
        print("📊 Generating comprehensive visualization report...")
        
        # Create all plots
        self.create_pareto_frontier_plot(efficiency_df, output_dir)
        self.create_optimal_selection_plot(optimal_df, output_dir)
        self.create_frequency_sweep_analysis(efficiency_df, output_dir)
        self.create_efficiency_heatmap(efficiency_df, output_dir)
        
        print(f"✅ Comprehensive visualization report saved to: {output_dir}")
        print("📊 Generated plots:")
        print("   - pareto_frontier_analysis.png/pdf")
        print("   - optimal_selection_detailed.png/pdf")
        print("   - frequency_sweep_detailed.png/pdf")
        print("   - efficiency_heatmap.png/pdf")
