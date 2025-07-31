#!/usr/bin/env python3
"""
Time-Series Visualization Demo for AI Inference Energy Profiling

This script demonstrates the time-series visualization capabilities for profiling metrics,
showing how to visualize any metric against normalized time for different frequencies
and applications.

Features demonstrated:
- Single metric vs normalized time plots
- Multi-metric dashboards
- Temporal pattern analysis
- Real profiling data loading
- Interactive visualization options

Usage:
    python time_series_visualization_demo.py [--data-dir results/] [--synthetic] [--save-plots]
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Import plotting libraries at the top
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("matplotlib/seaborn not available. Some visualizations will be limited.")

# Import visualization modules
try:
    from edp_analysis.visualization.performance_plots import PerformancePlotter
    from edp_analysis.visualization.data_preprocessor import (
        ProfilingDataPreprocessor, 
        create_synthetic_profiling_data
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False


def load_real_profiling_data(data_dir: str) -> Optional[pd.DataFrame]:
    """Load real profiling data from result directories."""
    try:
        data_path = Path(data_dir)
        
        # Look for result directories
        result_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('results_')]
        
        if not result_dirs:
            logger.warning(f"No result directories found in {data_dir}")
            return None
        
        # Use the first available result directory
        result_dir = result_dirs[0]
        logger.info(f"Loading data from {result_dir}")
        
        # Extract application name from directory
        app_name = "Unknown"
        if "vit" in result_dir.name.lower():
            app_name = "Vision Transformer"
        elif "llama" in result_dir.name.lower():
            app_name = "LLaMA"
        elif "stablediffusion" in result_dir.name.lower():
            app_name = "Stable Diffusion"
        elif "whisper" in result_dir.name.lower():
            app_name = "Whisper"
        elif "lstm" in result_dir.name.lower():
            app_name = "LSTM"
        
        # Load all profiling CSV files from the directory
        df = ProfilingDataPreprocessor.load_result_directory(
            str(result_dir),
            pattern="*_profile.csv",
            app_name=app_name
        )
        
        if df.empty:
            logger.warning("No profiling data found")
            return None
        
        # Prepare for time-series analysis
        df = ProfilingDataPreprocessor.prepare_for_time_series(df)
        
        logger.info(f"Loaded real profiling data: {len(df)} samples, {len(df['frequency'].unique())} frequencies")
        return df
        
    except Exception as e:
        logger.error(f"Error loading real data: {e}")
        return None


def demonstrate_single_metric_visualization(df: pd.DataFrame, plotter: PerformancePlotter, output_dir: Optional[Path] = None):
    """Demonstrate single metric vs time visualization."""
    logger.info("🔍 Demonstrating single metric visualization...")
    
    app_name = df['app_name'].iloc[0] if 'app_name' in df.columns else "Application"
    
    # Available metrics in the dataset
    metric_columns = [col for col in df.columns if col not in ['timestamp', 'frequency', 'app_name', 'run_id', 'experiment']]
    
    # Prioritize interesting metrics
    priority_metrics = ['power_draw', 'gpu_utilization', 'gpu_temperature', 'memory_utilization']
    available_priority = [m for m in priority_metrics if m in metric_columns]
    
    if not available_priority:
        available_priority = metric_columns[:4]  # Use first 4 available metrics
    
    for i, metric in enumerate(available_priority[:3]):  # Show top 3 metrics
        try:
            # Single frequency plot
            single_freq = df['frequency'].unique()[0]
            save_path = str(output_dir / f"single_metric_{metric}_freq_{single_freq}.png") if output_dir else None
            
            fig1 = plotter.plot_metric_vs_normalized_time(
                df,
                metric_col=metric,
                frequency_filter=single_freq,
                app_name=app_name,
                title=f"{metric.replace('_', ' ').title()} vs Time @ {single_freq} MHz",
                save_path=save_path
            )
            
            # Multi-frequency comparison
            frequencies = sorted(df['frequency'].unique())[:3]  # Use up to 3 frequencies
            save_path = str(output_dir / f"multi_freq_{metric}.png") if output_dir else None
            
            fig2 = plotter.plot_metric_vs_normalized_time(
                df,
                metric_col=metric,
                frequency_filter=frequencies,
                app_name=app_name,
                title=f"{metric.replace('_', ' ').title()} - Multi-Frequency Comparison",
                save_path=save_path
            )
            
            logger.info(f"✅ Created {metric} visualizations")
            
        except Exception as e:
            logger.warning(f"Could not create {metric} visualization: {e}")


def demonstrate_multi_metric_dashboard(df: pd.DataFrame, plotter: PerformancePlotter, output_dir: Optional[Path] = None):
    """Demonstrate multi-metric dashboard visualization."""
    logger.info("📊 Demonstrating multi-metric dashboard...")
    
    app_name = df['app_name'].iloc[0] if 'app_name' in df.columns else "Application"
    
    # Select metrics for dashboard
    dashboard_metrics = []
    priority_metrics = [
        'power_draw', 'gpu_utilization', 'gpu_temperature', 
        'memory_utilization', 'sm_active', 'power_per_mhz'
    ]
    
    for metric in priority_metrics:
        if metric in df.columns:
            dashboard_metrics.append(metric)
        if len(dashboard_metrics) >= 6:  # Limit to 6 metrics for clean display
            break
    
    if len(dashboard_metrics) < 2:
        logger.warning("Not enough metrics available for dashboard")
        return
    
    try:
        # Dashboard with multiple frequencies
        frequencies = sorted(df['frequency'].unique())[:2]  # Use up to 2 frequencies for clarity
        save_path = str(output_dir / "multi_metric_dashboard.png") if output_dir else None
        
        fig = plotter.plot_multi_metric_dashboard(
            df,
            metrics=dashboard_metrics,
            frequency_filter=frequencies,
            app_name=app_name,
            title=f"Performance Dashboard - {app_name}",
            cols=3,
            save_path=save_path
        )
        
        logger.info(f"✅ Created multi-metric dashboard with {len(dashboard_metrics)} metrics")
        
    except Exception as e:
        logger.error(f"Could not create dashboard: {e}")


def demonstrate_temporal_analysis(df: pd.DataFrame, plotter: PerformancePlotter):
    """Demonstrate temporal pattern analysis."""
    logger.info("⏱️  Demonstrating temporal pattern analysis...")
    
    # Analyze patterns for different metrics
    metrics_to_analyze = ['power_draw', 'gpu_utilization', 'gpu_temperature']
    available_metrics = [m for m in metrics_to_analyze if m in df.columns]
    
    for metric in available_metrics:
        try:
            patterns = plotter.analyze_temporal_patterns(
                df,
                metric_col=metric,
                window_size="1s"
            )
            
            logger.info(f"\n📈 Temporal Analysis for {metric}:")
            for freq, stats in patterns.items():
                logger.info(f"  {freq} MHz:")
                logger.info(f"    Mean: {stats['mean']:.2f}")
                logger.info(f"    Std Dev: {stats['std']:.2f}")
                logger.info(f"    Stability: {stats['stability']:.2f}")
                logger.info(f"    Anomalies: {stats['anomaly_count']}")
                logger.info(f"    Trend: {stats['trend']:.6f}")
                
        except Exception as e:
            logger.warning(f"Could not analyze {metric}: {e}")


def demonstrate_advanced_visualizations(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Demonstrate advanced visualization techniques."""
    logger.info("🚀 Demonstrating advanced visualizations...")
    
    if not HAS_PLOTTING:
        logger.warning("Matplotlib not available. Skipping advanced visualizations.")
        return
    
    try:
        # Correlation heatmap (without seaborn)
        if output_dir:
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_cols = [col for col in numeric_cols if col not in ['timestamp', 'relative_time_seconds']]
            
            if len(correlation_cols) > 3:
                corr_matrix = df[correlation_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                
                # Add correlation values as text
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                     ha="center", va="center", color="black")
                
                # Set ticks and labels
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr_matrix.columns)
                
                plt.colorbar(im, ax=ax)
                plt.title("Profiling Metrics Correlation Matrix")
                plt.tight_layout()
                plt.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info("✅ Created correlation heatmap")
        
        # Frequency vs metric scatter plots
        if 'power_draw' in df.columns and output_dir:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Power vs frequency
            for freq in df['frequency'].unique():
                freq_data = df[df['frequency'] == freq]
                axes[0].scatter(freq_data['frequency'], freq_data['power_draw'], 
                              alpha=0.6, label=f"{freq} MHz")
            axes[0].set_xlabel("Frequency (MHz)")
            axes[0].set_ylabel("Power Draw (W)")
            axes[0].set_title("Power Draw vs Frequency")
            axes[0].grid(True, alpha=0.3)
            
            # Utilization vs frequency  
            if 'gpu_utilization' in df.columns:
                for freq in df['frequency'].unique():
                    freq_data = df[df['frequency'] == freq]
                    axes[1].scatter(freq_data['frequency'], freq_data['gpu_utilization'], 
                                  alpha=0.6, label=f"{freq} MHz")
                axes[1].set_xlabel("Frequency (MHz)")
                axes[1].set_ylabel("GPU Utilization (%)")
                axes[1].set_title("GPU Utilization vs Frequency")
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "frequency_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Created frequency analysis plots")
            
    except Exception as e:
        logger.warning(f"Could not create advanced visualizations: {e}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Time-Series Visualization Demo")
    parser.add_argument("--data-dir", type=str, default="sample-collection-scripts/", 
                       help="Directory containing result folders")
    parser.add_argument("--synthetic", action="store_true", 
                       help="Use synthetic data instead of real data")
    parser.add_argument("--save-plots", action="store_true", 
                       help="Save plots to files")
    parser.add_argument("--output-dir", type=str, default="plots/", 
                       help="Output directory for saved plots")
    
    args = parser.parse_args()
    
    if not VISUALIZATION_AVAILABLE:
        logger.error("Visualization modules not available. Please install required dependencies.")
        return 1
    
    # Setup output directory
    output_dir = None
    if args.save_plots:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Plots will be saved to {output_dir}")
    
    logger.info("🚀 Starting Time-Series Visualization Demo")
    
    # Load data
    if args.synthetic:
        logger.info("📊 Creating synthetic profiling data...")
        df = create_synthetic_profiling_data(
            frequencies=[900, 1200, 1410],
            duration_seconds=30,
            app_name="Demo Application"
        )
    else:
        logger.info(f"📁 Loading real profiling data from {args.data_dir}...")
        df = load_real_profiling_data(args.data_dir)
        
        if df is None:
            logger.info("📊 Falling back to synthetic data...")
            df = create_synthetic_profiling_data(
                frequencies=[900, 1200, 1410],
                duration_seconds=30,
                app_name="Demo Application"
            )
    
    if df is None or df.empty:
        logger.error("No data available for visualization")
        return 1
    
    logger.info(f"📈 Data loaded: {len(df)} samples, {len(df['frequency'].unique())} frequencies")
    logger.info(f"Available metrics: {[col for col in df.columns if col not in ['timestamp', 'frequency', 'app_name', 'run_id']]}")
    
    # Initialize plotter
    plotter = PerformancePlotter(figsize=(12, 8))
    
    # Run demonstrations
    try:
        demonstrate_single_metric_visualization(df, plotter, output_dir)
        demonstrate_multi_metric_dashboard(df, plotter, output_dir)
        demonstrate_temporal_analysis(df, plotter)
        demonstrate_advanced_visualizations(df, output_dir)
        
        logger.info("🎉 Demo completed successfully!")
        
        if args.save_plots:
            logger.info(f"📁 All plots saved to {output_dir}")
        else:
            logger.info("💡 Use --save-plots to save visualizations to files")
            
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
