#!/usr/bin/env python3
"""
EDP Optimization Results Visualization

This script creates visualizations for the optimal frequency analysis results,
showing EDP curves, energy savings, and optimization comparisons.

Usage:
    python plot_edp_optimization.py --data ../data_aggregation/complete_aggregation.csv --optimal optimal_frequencies.json
    python plot_edp_optimization.py --data ../data_aggregation/complete_aggregation.csv --gpu V100 --app LLAMA

Author: Mert Side
"""

import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class EDPVisualizer:
    """Create visualizations for EDP optimization results."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = {
            'edp': '#1f77b4',
            'ed2p': '#ff7f0e', 
            'energy': '#2ca02c',
            'performance': '#d62728'
        }
    
    def load_data(self, data_path: str, optimal_path: str = None):
        """Load aggregated data and optimal results."""
        
        # Load aggregated data
        self.df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.df)} configurations")
        
        # Calculate EDP and ED²P if not present
        if 'edp' not in self.df.columns:
            self.df['edp'] = self.df['energy'] * self.df['execution_time']
        if 'ed2p' not in self.df.columns:
            self.df['ed2p'] = self.df['energy'] * (self.df['execution_time'] ** 2)
        
        # Load optimal results if provided
        if optimal_path:
            with open(optimal_path, 'r') as f:
                self.optimal_results = json.load(f)
            logger.info(f"Loaded optimal results from {optimal_path}")
        else:
            self.optimal_results = None
    
    def plot_edp_curves(self, gpu: str, app: str, save_dir: str = "plots"):
        """Plot EDP and ED²P curves for a specific GPU+application."""
        
        # Filter data
        data = self.df[(self.df['gpu'] == gpu) & (self.df['application'] == app)].copy()
        data = data.sort_values('frequency')
        
        if data.empty:
            logger.warning(f"No data for {gpu} + {app}")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{gpu} + {app} - EDP Optimization Analysis', fontsize=16, fontweight='bold')
        
        # Normalize metrics for comparison (baseline = max frequency)
        baseline_idx = data['frequency'].idxmax()
        baseline_energy = data.loc[baseline_idx, 'energy']
        baseline_time = data.loc[baseline_idx, 'execution_time']
        baseline_edp = data.loc[baseline_idx, 'edp']
        baseline_ed2p = data.loc[baseline_idx, 'ed2p']
        
        # Plot 1: Energy vs Frequency
        ax1.plot(data['frequency'], data['energy'], 'o-', color=self.colors['energy'], linewidth=2, markersize=4)
        ax1.set_xlabel('Frequency (MHz)', fontweight='bold')
        ax1.set_ylabel('Energy (J)', fontweight='bold')
        ax1.set_title('Energy Consumption', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Execution Time vs Frequency
        ax2.plot(data['frequency'], data['execution_time'], 'o-', color=self.colors['performance'], linewidth=2, markersize=4)
        ax2.set_xlabel('Frequency (MHz)', fontweight='bold')
        ax2.set_ylabel('Execution Time (s)', fontweight='bold')
        ax2.set_title('Execution Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: EDP vs Frequency
        ax3.plot(data['frequency'], data['edp'], 'o-', color=self.colors['edp'], linewidth=2, markersize=4, label='EDP')
        if self.optimal_results:
            config_key = f"{gpu}_{app}"
            if config_key in self.optimal_results['configurations']:
                edp_optimal = self.optimal_results['configurations'][config_key]['optimizations']['edp']['frequency']
                edp_value = data[data['frequency'] == edp_optimal]['edp'].values[0]
                ax3.axvline(x=edp_optimal, color=self.colors['edp'], linestyle='--', alpha=0.7)
                ax3.plot(edp_optimal, edp_value, 'o', color=self.colors['edp'], markersize=8, markerfacecolor='white', markeredgewidth=2)
                ax3.text(edp_optimal, edp_value, f'  EDP Optimal\n  {edp_optimal} MHz', fontweight='bold', ha='left')
        
        ax3.set_xlabel('Frequency (MHz)', fontweight='bold')
        ax3.set_ylabel('EDP (J⋅s)', fontweight='bold')
        ax3.set_title('Energy-Delay Product (EDP)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: ED²P vs Frequency
        ax4.plot(data['frequency'], data['ed2p'], 'o-', color=self.colors['ed2p'], linewidth=2, markersize=4, label='ED²P')
        if self.optimal_results:
            config_key = f"{gpu}_{app}"
            if config_key in self.optimal_results['configurations']:
                ed2p_optimal = self.optimal_results['configurations'][config_key]['optimizations']['ed2p']['frequency']
                ed2p_value = data[data['frequency'] == ed2p_optimal]['ed2p'].values[0]
                ax4.axvline(x=ed2p_optimal, color=self.colors['ed2p'], linestyle='--', alpha=0.7)
                ax4.plot(ed2p_optimal, ed2p_value, 'o', color=self.colors['ed2p'], markersize=8, markerfacecolor='white', markeredgewidth=2)
                ax4.text(ed2p_optimal, ed2p_value, f'  ED²P Optimal\n  {ed2p_optimal} MHz', fontweight='bold', ha='left')
        
        ax4.set_xlabel('Frequency (MHz)', fontweight='bold')
        ax4.set_ylabel('ED²P (J⋅s²)', fontweight='bold')
        ax4.set_title('Energy-Delay² Product (ED²P)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        filename = f"edp_analysis_{gpu.lower()}_{app.lower()}.png"
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {save_path / filename}")
        
        plt.show()
        
        return fig
    
    def plot_energy_savings_summary(self, save_dir: str = "plots"):
        """Create a summary plot of energy savings across all configurations."""
        
        if not self.optimal_results:
            logger.error("No optimal results loaded for summary plot")
            return
        
        # Extract data for plotting
        configs = []
        edp_savings = []
        ed2p_savings = []
        edp_freqs = []
        ed2p_freqs = []
        
        for config_key, config in self.optimal_results['configurations'].items():
            gpu, app = config_key.split('_')
            configs.append(f"{gpu}\n{app}")
            
            edp_data = config['optimizations']['edp']['baseline_comparison']
            ed2p_data = config['optimizations']['ed2p']['baseline_comparison']
            
            edp_savings.append(edp_data['energy_savings_percent'])
            ed2p_savings.append(ed2p_data['energy_savings_percent'])
            edp_freqs.append(config['optimizations']['edp']['frequency'])
            ed2p_freqs.append(config['optimizations']['ed2p']['frequency'])
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('EDP Optimization Summary - Energy Savings vs Baseline', fontsize=16, fontweight='bold')
        
        x = np.arange(len(configs))
        width = 0.35
        
        # Plot 1: Energy Savings Comparison
        bars1 = ax1.bar(x - width/2, edp_savings, width, label='EDP Optimal', color=self.colors['edp'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, ed2p_savings, width, label='ED²P Optimal', color=self.colors['ed2p'], alpha=0.8)
        
        ax1.set_xlabel('GPU + Application', fontweight='bold')
        ax1.set_ylabel('Energy Savings (%)', fontweight='bold')
        ax1.set_title('Energy Savings vs Maximum Frequency', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Optimal Frequencies
        bars3 = ax2.bar(x - width/2, edp_freqs, width, label='EDP Optimal', color=self.colors['edp'], alpha=0.8)
        bars4 = ax2.bar(x + width/2, ed2p_freqs, width, label='ED²P Optimal', color=self.colors['ed2p'], alpha=0.8)
        
        ax2.set_xlabel('GPU + Application', fontweight='bold')
        ax2.set_ylabel('Optimal Frequency (MHz)', fontweight='bold')
        ax2.set_title('Optimal Frequencies', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(configs)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add frequency labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        filename = "edp_optimization_summary.png"
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved summary plot: {save_path / filename}")
        
        plt.show()
        
        return fig
    
    def create_all_plots(self, save_dir: str = "plots"):
        """Generate all visualization plots."""
        
        logger.info("Creating EDP optimization visualizations...")
        
        # Get unique GPU+application combinations
        combinations = self.df[['gpu', 'application']].drop_duplicates()
        
        # Create individual EDP curve plots
        for _, row in combinations.iterrows():
            self.plot_edp_curves(row['gpu'], row['application'], save_dir)
        
        # Create summary plot
        if self.optimal_results:
            self.plot_energy_savings_summary(save_dir)
        
        logger.info(f"All plots saved to {save_dir}/")


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(
        description="Visualize EDP optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create all plots
    python plot_edp_optimization.py --data ../data_aggregation/complete_aggregation.csv --optimal optimal_frequencies.json
    
    # Plot specific configuration
    python plot_edp_optimization.py --data ../data_aggregation/complete_aggregation.csv --gpu V100 --app LLAMA
        """
    )
    
    parser.add_argument("--data", type=str, required=True,
                       help="Path to aggregated profiling data CSV file")
    parser.add_argument("--optimal", type=str,
                       help="Path to optimal frequencies JSON file")
    parser.add_argument("--gpu", type=str, choices=["V100", "A100", "H100"],
                       help="Plot specific GPU (plots all if not specified)")
    parser.add_argument("--app", type=str, choices=["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER"],
                       help="Plot specific application (plots all if not specified)")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots (default: plots)")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = EDPVisualizer()
    
    logger.info("🚀 Creating EDP optimization visualizations")
    
    try:
        # Load data
        visualizer.load_data(args.data, args.optimal)
        
        if args.gpu and args.app:
            # Plot specific configuration
            visualizer.plot_edp_curves(args.gpu, args.app, args.output_dir)
        else:
            # Create all plots
            visualizer.create_all_plots(args.output_dir)
        
        logger.info("✅ Visualization completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
