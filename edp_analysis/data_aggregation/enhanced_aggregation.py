#!/usr/bin/env python3
"""
Enhanced Aggregation Strategy with First-Run Filtering and Multi-Run Averaging

This script implements best practices for handling first-run outliers and
provides statistically robust aggregation of profiling data.

Author: Mert Side
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
import sys
from typing import Dict, List, Optional

# Import the existing aggregator
from aggregate_profiling_data import ProfilingDataAggregator

logger = logging.getLogger(__name__)

class EnhancedProfilingAggregator(ProfilingDataAggregator):
    """Enhanced aggregator with cold start handling and multi-run statistics."""
    
    def __init__(self, data_dir: str = "../../sample-collection-scripts"):
        super().__init__(data_dir)
        self.outlier_detection_enabled = True
        self.min_runs_required = 2  # Need at least 2 runs for statistics
        
    def aggregate_multi_run_data(self, 
                                gpu: str = None, 
                                app: str = None,
                                exclude_first_run: bool = True,
                                runs_to_include: List[int] = None,
                                aggregation_method: str = "mean") -> pd.DataFrame:
        """
        Aggregate data across multiple runs with cold start handling.
        
        Args:
            gpu: GPU type filter
            app: Application filter  
            exclude_first_run: Whether to exclude run 1 (cold start)
            runs_to_include: Specific runs to include (e.g., [2, 3])
            aggregation_method: 'mean', 'median', 'best' (fastest), or 'conservative' (slowest excluding outliers)
        """
        
        if runs_to_include is None:
            if exclude_first_run:
                runs_to_include = [2, 3]  # Default: use runs 2 and 3
            else:
                runs_to_include = [1, 2, 3]  # Include all runs
        
        logger.info(f"Aggregating runs: {runs_to_include}")
        logger.info(f"Aggregation method: {aggregation_method}")
        
        all_run_data = []
        
        # Collect data from each run
        for run_num in runs_to_include:
            try:
                run_data = self.aggregate_all_data(gpu=gpu, app=app, run_number=run_num)
                if not run_data.empty:
                    run_data['source_run'] = run_num
                    all_run_data.append(run_data)
                    logger.info(f"Run {run_num}: {len(run_data)} configurations")
                else:
                    logger.warning(f"No data found for run {run_num}")
            except Exception as e:
                logger.warning(f"Failed to load run {run_num}: {e}")
        
        if not all_run_data:
            logger.error("No run data collected!")
            return pd.DataFrame()
        
        # Combine all run data
        combined_df = pd.concat(all_run_data, ignore_index=True)
        logger.info(f"Combined data: {len(combined_df)} total measurements")
        
        # Group by configuration and aggregate
        group_cols = ['gpu', 'application', 'frequency']
        aggregated_results = []
        
        for group_key, group_data in combined_df.groupby(group_cols):
            gpu_val, app_val, freq_val = group_key
            
            if len(group_data) < self.min_runs_required:
                logger.warning(f"Insufficient runs for {gpu_val}+{app_val}@{freq_val}MHz: {len(group_data)} runs")
                continue
            
            # Apply aggregation method
            agg_stats = self._aggregate_group_statistics(group_data, aggregation_method)
            agg_stats.update({
                'gpu': gpu_val,
                'application': app_val, 
                'frequency': freq_val,
                'runs_used': list(group_data['source_run']),
                'num_runs': len(group_data)
            })
            
            aggregated_results.append(agg_stats)
        
        if not aggregated_results:
            logger.error("No aggregated results generated!")
            return pd.DataFrame()
        
        # Create final DataFrame
        final_df = pd.DataFrame(aggregated_results)
        
        # Calculate derived metrics
        final_df['edp'] = final_df['execution_time'] * final_df['avg_power']
        final_df['ed2p'] = final_df['edp'] * final_df['avg_power']
        
        logger.info(f"Final aggregated data: {len(final_df)} configurations")
        return final_df
    
    def _aggregate_group_statistics(self, group_data: pd.DataFrame, method: str) -> Dict:
        """Calculate statistics for a group of runs at the same configuration."""
        
        # Core metrics to aggregate
        metrics = ['execution_time', 'avg_power', 'energy', 'avg_gputl', 'avg_mcutl', 
                  'avg_temp', 'avg_smclk', 'avg_mmclk']
        
        stats = {}
        
        for metric in metrics:
            if metric in group_data.columns:
                values = group_data[metric].dropna()
                
                if len(values) == 0:
                    stats[metric] = np.nan
                    continue
                
                if method == "mean":
                    stats[metric] = values.mean()
                elif method == "median":
                    stats[metric] = values.median()
                elif method == "best":
                    # For time-based metrics, choose minimum (fastest)
                    # For power/efficiency metrics, choose based on context
                    if 'time' in metric.lower():
                        stats[metric] = values.min()
                    else:
                        stats[metric] = values.mean()  # Use mean for other metrics
                elif method == "conservative":
                    # Remove outliers, then take mean
                    if len(values) >= 3:
                        # Remove extreme outliers (beyond 2 standard deviations)
                        mean_val = values.mean()
                        std_val = values.std()
                        filtered_values = values[abs(values - mean_val) <= 2 * std_val]
                        stats[metric] = filtered_values.mean() if len(filtered_values) > 0 else mean_val
                    else:
                        stats[metric] = values.mean()
                else:
                    stats[metric] = values.mean()  # Default to mean
                
                # Add statistical measures
                if len(values) > 1:
                    stats[f'{metric}_std'] = values.std()
                    stats[f'{metric}_cv'] = values.std() / values.mean() if values.mean() != 0 else 0
                    stats[f'{metric}_min'] = values.min()
                    stats[f'{metric}_max'] = values.max()
        
        # Add sample count info
        stats['num_samples'] = group_data['num_samples'].sum() if 'num_samples' in group_data.columns else len(group_data)
        
        return stats
    
    def detect_and_flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag potential outliers in the data."""
        
        df_flagged = df.copy()
        df_flagged['outlier_flags'] = ''
        
        # Group by GPU+App and detect outliers within each group
        for (gpu, app), group in df.groupby(['gpu', 'application']):
            if len(group) < 3:
                continue
                
            # Check execution time outliers
            if 'execution_time' in group.columns:
                time_values = group['execution_time']
                time_mean = time_values.mean()
                time_std = time_values.std()
                
                # Flag values beyond 2.5 standard deviations
                outlier_threshold = 2.5
                time_outliers = abs(time_values - time_mean) > outlier_threshold * time_std
                
                if time_outliers.any():
                    outlier_indices = group[time_outliers].index
                    for idx in outlier_indices:
                        current_flags = df_flagged.loc[idx, 'outlier_flags']
                        df_flagged.loc[idx, 'outlier_flags'] = f"{current_flags},time_outlier" if current_flags else "time_outlier"
        
        return df_flagged
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> str:
        """Generate a comprehensive data quality report."""
        
        report_lines = []
        report_lines.append("🔍 Data Quality Assessment Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall statistics
        total_configs = len(df)
        unique_gpus = df['gpu'].nunique()
        unique_apps = df['application'].nunique()
        unique_freqs = df['frequency'].nunique()
        
        report_lines.append(f"📊 Dataset Overview:")
        report_lines.append(f"   Total configurations: {total_configs}")
        report_lines.append(f"   GPUs: {unique_gpus} ({', '.join(df['gpu'].unique())})")
        report_lines.append(f"   Applications: {unique_apps} ({', '.join(df['application'].unique())})")
        report_lines.append(f"   Frequency points: {unique_freqs}")
        report_lines.append("")
        
        # Run coverage analysis
        if 'runs_used' in df.columns:
            report_lines.append(f"🏃 Run Coverage Analysis:")
            all_runs = []
            for runs_list in df['runs_used']:
                if isinstance(runs_list, list):
                    all_runs.extend(runs_list)
            
            from collections import Counter
            run_counts = Counter(all_runs)
            
            for run_num, count in sorted(run_counts.items()):
                coverage = (count / total_configs) * 100
                report_lines.append(f"   Run {run_num}: {count}/{total_configs} configs ({coverage:.1f}%)")
            report_lines.append("")
        
        # Statistical quality metrics
        if 'execution_time_cv' in df.columns:
            report_lines.append(f"📈 Statistical Quality Metrics:")
            cv_values = df['execution_time_cv'].dropna()
            if len(cv_values) > 0:
                mean_cv = cv_values.mean() * 100
                max_cv = cv_values.max() * 100
                high_variance_count = (cv_values > 0.1).sum()  # >10% CV
                
                report_lines.append(f"   Mean coefficient of variation: {mean_cv:.2f}%")
                report_lines.append(f"   Maximum coefficient of variation: {max_cv:.2f}%")
                report_lines.append(f"   High variance configurations: {high_variance_count}/{len(cv_values)}")
                
                if mean_cv > 10:
                    report_lines.append("   ⚠️  High variability detected - consider more runs")
                elif mean_cv < 5:
                    report_lines.append("   ✅ Good measurement consistency")
                else:
                    report_lines.append("   📝 Moderate variability - acceptable for analysis")
            report_lines.append("")
        
        # Outlier analysis
        if 'outlier_flags' in df.columns:
            flagged_outliers = df[df['outlier_flags'] != ''].copy()
            if len(flagged_outliers) > 0:
                report_lines.append(f"🚨 Outlier Detection:")
                report_lines.append(f"   Configurations with outlier flags: {len(flagged_outliers)}/{total_configs}")
                
                flag_counts = {}
                for flags in flagged_outliers['outlier_flags']:
                    for flag in flags.split(','):
                        flag = flag.strip()
                        if flag:
                            flag_counts[flag] = flag_counts.get(flag, 0) + 1
                
                for flag, count in flag_counts.items():
                    report_lines.append(f"   {flag}: {count} configurations")
                report_lines.append("")
        
        # Recommendations
        report_lines.append(f"💡 Recommendations:")
        
        if 'runs_used' in df.columns:
            # Check run coverage
            if any(len(runs) < 2 for runs in df['runs_used'] if isinstance(runs, list)):
                report_lines.append("   • Increase number of runs per configuration (minimum 3 recommended)")
        
        if 'execution_time_cv' in df.columns:
            cv_values = df['execution_time_cv'].dropna()
            if len(cv_values) > 0 and cv_values.mean() > 0.1:
                report_lines.append("   • High variability detected - consider extending warm-up periods")
        
        report_lines.append("   • Always exclude first run to avoid cold start effects")
        report_lines.append("   • Use median aggregation for robust statistics")
        report_lines.append("   • Monitor for thermal throttling in long experiments")
        
        return "\n".join(report_lines)


def main():
    """Main execution with enhanced aggregation options."""
    parser = argparse.ArgumentParser(description="Enhanced profiling data aggregation with cold start handling")
    
    parser.add_argument("--input-dir", type=str, default="../../sample-collection-scripts",
                       help="Directory containing result folders")
    parser.add_argument("--output", type=str, default="enhanced_aggregation.csv",
                       help="Output CSV file path")
    parser.add_argument("--gpu", type=str, choices=["V100", "A100", "H100"],
                       help="Filter by GPU type")
    parser.add_argument("--app", type=str, choices=["LLAMA", "VIT", "STABLEDIFFUSION", "WHISPER"],
                       help="Filter by application")
    parser.add_argument("--exclude-first-run", action="store_true", default=True,
                       help="Exclude run 1 to avoid cold start effects (default: True)")
    parser.add_argument("--runs", type=int, nargs='+', default=[2, 3],
                       help="Specific runs to include (default: [2, 3])")
    parser.add_argument("--method", type=str, choices=["mean", "median", "best", "conservative"], 
                       default="median",
                       help="Aggregation method (default: median)")
    parser.add_argument("--quality-report", action="store_true",
                       help="Generate data quality report")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize enhanced aggregator
        aggregator = EnhancedProfilingAggregator(args.input_dir)
        
        logger.info(f"🚀 Starting enhanced data aggregation from {args.input_dir}")
        logger.info(f"Cold start handling: exclude_first_run={args.exclude_first_run}")
        logger.info(f"Runs to include: {args.runs}")
        logger.info(f"Aggregation method: {args.method}")
        
        # Aggregate with enhanced strategy
        df = aggregator.aggregate_multi_run_data(
            gpu=args.gpu,
            app=args.app,
            exclude_first_run=args.exclude_first_run,
            runs_to_include=args.runs,
            aggregation_method=args.method
        )
        
        if df.empty:
            logger.error("No data was aggregated!")
            return 1
        
        # Detect outliers
        df = aggregator.detect_and_flag_outliers(df)
        
        # Generate quality report
        if args.quality_report:
            quality_report = aggregator.generate_data_quality_report(df)
            quality_file = args.output.replace('.csv', '_quality_report.txt')
            with open(quality_file, 'w') as f:
                f.write(quality_report)
            print(quality_report)
            logger.info(f"Quality report saved to {quality_file}")
        
        # Save results
        df.to_csv(args.output, index=False)
        logger.info(f"Enhanced aggregated data saved to {args.output}")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Summary statistics
        logger.info("📊 Summary Statistics:")
        for gpu in df['gpu'].unique():
            for app in df['application'].unique():
                subset = df[(df['gpu'] == gpu) & (df['application'] == app)]
                if len(subset) > 0:
                    freq_range = f"{subset['frequency'].min()}-{subset['frequency'].max()}"
                    logger.info(f"   {gpu}+{app}: {len(subset)} frequencies ({freq_range} MHz)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
