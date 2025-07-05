"""
Performance Profiling Module

This module provides comprehensive performance measurement and analysis capabilities
for GPU applications, focusing on execution time, throughput, and frequency scaling.

Key features:
- Execution time measurement and analysis
- Performance scaling with frequency
- Statistical analysis of performance metrics
- Baseline performance calculation
- Integration with FGCS methodology
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Comprehensive performance profiling for GPU applications.

    Provides tools for measuring, processing, and analyzing execution time
    and performance characteristics across different GPU frequencies.
    """

    def __init__(self, time_units: str = "seconds"):
        """
        Initialize performance profiler.

        Args:
            time_units: Units for time measurements ('seconds', 'milliseconds')
        """
        self.time_units = time_units
        self.time_scale = 1.0 if time_units == "seconds" else 0.001

        logger.info(f"Performance profiler initialized: {time_units}")

    def calculate_performance_metrics(
        self,
        execution_times: Union[List[float], np.ndarray, pd.Series],
        frequencies: Optional[Union[List[int], np.ndarray, pd.Series]] = None,
        baseline_time: Optional[float] = None,
        baseline_frequency: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            execution_times: Execution time measurements
            frequencies: Corresponding frequencies (if available)
            baseline_time: Baseline execution time for comparison
            baseline_frequency: Baseline frequency for comparison

        Returns:
            Dictionary with performance metrics
        """
        times = np.array(execution_times) * self.time_scale

        # Basic statistics
        metrics = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "median_time": np.median(times),
            "cv_time": np.std(times) / np.mean(times),
            "num_measurements": len(times),
        }

        # Calculate throughput (inverse of time)
        metrics["mean_throughput"] = 1.0 / metrics["mean_time"]
        metrics["throughput_range"] = (1.0 / metrics["min_time"]) - (1.0 / metrics["max_time"])

        # Performance relative to baseline
        if baseline_time is not None:
            baseline_time_scaled = baseline_time * self.time_scale
            metrics["speedup"] = baseline_time_scaled / metrics["mean_time"]
            metrics["performance_improvement"] = (1 - metrics["mean_time"] / baseline_time_scaled) * 100

        # Frequency scaling analysis
        if frequencies is not None:
            freqs = np.array(frequencies)
            if len(freqs) == len(times):
                metrics["frequency_performance_correlation"] = np.corrcoef(freqs, 1 / times)[0, 1]

                # Calculate ideal scaling (if baseline provided)
                if baseline_frequency is not None and baseline_time is not None:
                    ideal_times = baseline_time_scaled * (baseline_frequency / freqs)
                    metrics["scaling_efficiency"] = np.mean(ideal_times / times)

        return metrics

    def analyze_frequency_scaling(
        self,
        performance_data: pd.DataFrame,
        frequency_col: str = "frequency",
        time_col: str = "execution_time",
        baseline_frequency: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze performance scaling across frequencies.

        Args:
            performance_data: DataFrame with frequency and time data
            frequency_col: Column name for frequency values
            time_col: Column name for execution time
            baseline_frequency: Reference frequency for scaling analysis

        Returns:
            Dictionary with scaling analysis results
        """
        if performance_data.empty:
            raise ValueError("Performance data is empty")

        required_cols = [frequency_col, time_col]
        missing_cols = [col for col in required_cols if col not in performance_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = performance_data.copy()
        freqs = df[frequency_col].values
        times = df[time_col].values * self.time_scale

        # Calculate throughput
        df["throughput"] = 1.0 / (df[time_col] * self.time_scale)

        # Determine baseline
        if baseline_frequency is not None:
            baseline_mask = df[frequency_col] == baseline_frequency
            if not baseline_mask.any():
                logger.warning(f"Baseline frequency {baseline_frequency} not found, using maximum frequency")
                baseline_frequency = df[frequency_col].max()
                baseline_mask = df[frequency_col] == baseline_frequency
        else:
            baseline_frequency = df[frequency_col].max()
            baseline_mask = df[frequency_col] == baseline_frequency

        baseline_time = df.loc[baseline_mask, time_col].iloc[0] * self.time_scale
        baseline_throughput = df.loc[baseline_mask, "throughput"].iloc[0]

        # Calculate scaling metrics
        df["speedup"] = baseline_time / (df[time_col] * self.time_scale)
        df["throughput_scaling"] = df["throughput"] / baseline_throughput
        df["frequency_scaling"] = df[frequency_col] / baseline_frequency

        # Ideal linear scaling
        df["ideal_speedup"] = df["frequency_scaling"]
        df["scaling_efficiency"] = df["speedup"] / df["ideal_speedup"]

        # Calculate correlations
        freq_time_corr = np.corrcoef(freqs, times)[0, 1]
        freq_throughput_corr = np.corrcoef(freqs, df["throughput"])[0, 1]

        # Find optimal points
        max_throughput_idx = df["throughput"].idxmax()
        min_time_idx = df[time_col].idxmin()
        best_efficiency_idx = df["scaling_efficiency"].idxmax()

        scaling_results = {
            "baseline_frequency": baseline_frequency,
            "baseline_time": baseline_time,
            "baseline_throughput": baseline_throughput,
            "frequency_range": (df[frequency_col].min(), df[frequency_col].max()),
            "time_range": (df[time_col].min() * self.time_scale, df[time_col].max() * self.time_scale),
            "throughput_range": (df["throughput"].min(), df["throughput"].max()),
            "freq_time_correlation": freq_time_corr,
            "freq_throughput_correlation": freq_throughput_corr,
            "mean_scaling_efficiency": df["scaling_efficiency"].mean(),
            "max_speedup": df["speedup"].max(),
            "max_throughput_config": {
                "frequency": df.loc[max_throughput_idx, frequency_col],
                "time": df.loc[max_throughput_idx, time_col] * self.time_scale,
                "throughput": df.loc[max_throughput_idx, "throughput"],
            },
            "min_time_config": {
                "frequency": df.loc[min_time_idx, frequency_col],
                "time": df.loc[min_time_idx, time_col] * self.time_scale,
                "speedup": df.loc[min_time_idx, "speedup"],
            },
            "best_efficiency_config": {
                "frequency": df.loc[best_efficiency_idx, frequency_col],
                "efficiency": df.loc[best_efficiency_idx, "scaling_efficiency"],
                "speedup": df.loc[best_efficiency_idx, "speedup"],
            },
            "processed_data": df,
        }

        logger.info(f"Frequency scaling analysis complete: {len(df)} configurations analyzed")
        return scaling_results

    def calculate_baseline_performance(
        self,
        profiling_data: pd.DataFrame,
        app_name: str,
        frequency_col: str = "frequency",
        time_col: str = "execution_time",
        baseline_strategy: str = "max_frequency",
    ) -> Dict[str, Any]:
        """
        Calculate baseline performance metrics for an application.

        Args:
            profiling_data: DataFrame with profiling results
            app_name: Application name
            frequency_col: Column name for frequency values
            time_col: Column name for execution time
            baseline_strategy: Strategy for baseline selection
                - 'max_frequency': Use highest frequency
                - 'min_time': Use configuration with minimum execution time
                - 'median_frequency': Use median frequency

        Returns:
            Dictionary with baseline performance metrics
        """
        if profiling_data.empty:
            raise ValueError("Profiling data is empty")

        df = profiling_data.copy()

        # Select baseline configuration
        if baseline_strategy == "max_frequency":
            baseline_idx = df[frequency_col].idxmax()
        elif baseline_strategy == "min_time":
            baseline_idx = df[time_col].idxmin()
        elif baseline_strategy == "median_frequency":
            median_freq = df[frequency_col].median()
            baseline_idx = (df[frequency_col] - median_freq).abs().idxmin()
        else:
            raise ValueError(f"Unknown baseline strategy: {baseline_strategy}")

        baseline_row = df.loc[baseline_idx]
        baseline_freq = baseline_row[frequency_col]
        baseline_time = baseline_row[time_col] * self.time_scale

        # Calculate statistics for this baseline
        baseline_metrics = {
            "app_name": app_name,
            "baseline_strategy": baseline_strategy,
            "baseline_frequency": baseline_freq,
            "baseline_time": baseline_time,
            "baseline_throughput": 1.0 / baseline_time,
            "total_configurations": len(df),
            "frequency_range": (df[frequency_col].min(), df[frequency_col].max()),
            "time_range": (df[time_col].min() * self.time_scale, df[time_col].max() * self.time_scale),
            "best_performance": {
                "min_time": df[time_col].min() * self.time_scale,
                "max_throughput": (1.0 / df[time_col].min()) / self.time_scale,
                "optimal_frequency": df.loc[df[time_col].idxmin(), frequency_col],
            },
        }

        # Calculate potential improvements
        min_time = df[time_col].min() * self.time_scale
        baseline_metrics["max_speedup_potential"] = baseline_time / min_time
        baseline_metrics["max_time_reduction"] = (1 - min_time / baseline_time) * 100

        logger.info(f"Baseline calculated for {app_name}: {baseline_freq} MHz, {baseline_time:.4f}s")
        return baseline_metrics

    def aggregate_performance_runs(
        self, run_data: List[pd.DataFrame], group_by_cols: List[str] = ["frequency"], time_col: str = "execution_time"
    ) -> pd.DataFrame:
        """
        Aggregate performance measurements across multiple runs.

        Args:
            run_data: List of DataFrames from multiple profiling runs
            group_by_cols: Columns to group by for aggregation
            time_col: Column name for execution time

        Returns:
            DataFrame with aggregated performance statistics
        """
        if not run_data:
            raise ValueError("No run data provided")

        # Combine all runs
        combined_df = pd.concat(run_data, ignore_index=True)

        # Apply time scaling
        combined_df[f"{time_col}_scaled"] = combined_df[time_col] * self.time_scale

        # Calculate statistics for each group
        stats_df = (
            combined_df.groupby(group_by_cols)[f"{time_col}_scaled"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        )

        # Rename columns for clarity
        stats_df.rename(
            columns={
                "mean": f"{time_col}_mean",
                "std": f"{time_col}_std",
                "min": f"{time_col}_min",
                "max": f"{time_col}_max",
                "count": "num_runs",
            },
            inplace=True,
        )

        # Calculate additional metrics
        stats_df["throughput_mean"] = 1.0 / stats_df[f"{time_col}_mean"]
        stats_df["throughput_std"] = stats_df[f"{time_col}_std"] / (stats_df[f"{time_col}_mean"] ** 2)
        stats_df[f"{time_col}_cv"] = stats_df[f"{time_col}_std"] / stats_df[f"{time_col}_mean"]
        stats_df[f"{time_col}_ci_95"] = 1.96 * stats_df[f"{time_col}_std"] / np.sqrt(stats_df["num_runs"])

        logger.info(f"Aggregated performance data for {len(stats_df)} configurations")
        return stats_df

    def detect_performance_anomalies(
        self,
        performance_data: pd.DataFrame,
        time_col: str = "execution_time",
        frequency_col: str = "frequency",
        outlier_threshold: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Detect anomalies in performance measurements.

        Args:
            performance_data: DataFrame with performance data
            time_col: Column name for execution time
            frequency_col: Column name for frequency
            outlier_threshold: Z-score threshold for outlier detection

        Returns:
            Dictionary with anomaly detection results
        """
        if performance_data.empty:
            raise ValueError("Performance data is empty")

        df = performance_data.copy()
        times = df[time_col] * self.time_scale

        # Calculate Z-scores
        mean_time = times.mean()
        std_time = times.std()
        z_scores = np.abs((times - mean_time) / std_time)

        # Identify outliers
        outlier_mask = z_scores > outlier_threshold
        outliers = df[outlier_mask]

        # Check for unexpected frequency-performance relationships
        if frequency_col in df.columns:
            # Sort by frequency and check for inversions
            sorted_df = df.sort_values(frequency_col)
            time_diffs = sorted_df[time_col].diff()
            unexpected_speedups = sorted_df[
                time_diffs < -0.1 * sorted_df[time_col].std()
            ]  # Significant unexpected improvements
            unexpected_slowdowns = sorted_df[
                time_diffs > 0.1 * sorted_df[time_col].std()
            ]  # Significant unexpected degradations
        else:
            unexpected_speedups = pd.DataFrame()
            unexpected_slowdowns = pd.DataFrame()

        anomaly_results = {
            "total_measurements": len(df),
            "outliers_detected": len(outliers),
            "outlier_percentage": (len(outliers) / len(df)) * 100,
            "outlier_data": outliers,
            "z_score_threshold": outlier_threshold,
            "unexpected_speedups": len(unexpected_speedups),
            "unexpected_slowdowns": len(unexpected_slowdowns),
            "speedup_anomalies": unexpected_speedups,
            "slowdown_anomalies": unexpected_slowdowns,
            "recommendations": [],
        }

        # Generate recommendations
        if len(outliers) > 0:
            anomaly_results["recommendations"].append(
                f"Found {len(outliers)} performance outliers. Review measurement conditions."
            )

        if len(unexpected_speedups) > 0:
            anomaly_results["recommendations"].append(
                f"Found {len(unexpected_speedups)} unexpected performance improvements. Verify measurements."
            )

        if len(unexpected_slowdowns) > 0:
            anomaly_results["recommendations"].append(
                f"Found {len(unexpected_slowdowns)} unexpected performance degradations. Check for thermal throttling."
            )

        if not anomaly_results["recommendations"]:
            anomaly_results["recommendations"].append("Performance measurements appear consistent.")

        logger.info(f"Anomaly detection complete: {len(outliers)} outliers found")
        return anomaly_results

    def predict_performance_scaling(
        self,
        df: pd.DataFrame,
        baseline_frequency: int,
        target_frequencies: List[int],
        time_col: str = "execution_time",
        frequency_col: str = "frequency",
    ) -> pd.DataFrame:
        """
        Predict performance scaling across frequencies using FGCS methodology.

        Args:
            df: DataFrame with baseline performance data
            baseline_frequency: Baseline frequency for scaling
            target_frequencies: Target frequencies to predict
            time_col: Execution time column name
            frequency_col: Frequency column name

        Returns:
            DataFrame with predicted performance metrics
        """
        logger.info(f"Predicting performance scaling from {baseline_frequency} MHz to {len(target_frequencies)} frequencies")

        # Get baseline performance
        baseline_data = df[df[frequency_col] == baseline_frequency]
        if baseline_data.empty:
            logger.warning(f"No baseline data found for frequency {baseline_frequency}")
            baseline_time = df[time_col].median()  # Use median as fallback
        else:
            baseline_time = baseline_data[time_col].mean()

        # Create predictions DataFrame
        predictions = []
        for target_freq in target_frequencies:
            # Simple frequency scaling model (can be enhanced with more sophisticated models)
            # Higher frequency -> lower execution time (inverse relationship)
            scaling_factor = baseline_frequency / target_freq
            predicted_time = baseline_time * scaling_factor

            # FGCS-compatible transformations
            prediction = {
                "frequency": target_freq,
                "sm_app_clock": target_freq,
                "n_sm_app_clock": np.log1p(target_freq),
                "predicted_execution_time": predicted_time,
                "predicted_n_run_time": np.log1p(predicted_time),
                "predicted_n_to_r_run_time": predicted_time,
                "scaling_factor": scaling_factor,
                "baseline_frequency": baseline_frequency,
                "baseline_time": baseline_time,
            }

            predictions.append(prediction)

        predictions_df = pd.DataFrame(predictions)
        logger.info(f"Generated performance predictions for {len(target_frequencies)} frequencies")
        return predictions_df

    def analyze_fgcs_performance_model(
        self,
        df: pd.DataFrame,
        fp_activity: float = 0.3,
        baseline_time: Optional[float] = None,
        time_col: str = "execution_time",
        frequency_col: str = "frequency",
    ) -> Dict[str, Any]:
        """
        Analyze performance using FGCS polynomial runtime model.

        Args:
            df: DataFrame with performance data
            fp_activity: FP operations activity (0.0 to 1.0)
            baseline_time: Baseline execution time (if None, calculated from data)
            time_col: Execution time column name
            frequency_col: Frequency column name

        Returns:
            FGCS performance analysis results
        """
        logger.info("Analyzing performance using FGCS polynomial model")

        # FGCS runtime model coefficients (from the paper)
        runtime_coeffs = [1.43847511, -0.16736726, -0.90400864, 0.48241361, 0.78898516]
        B0 = 0  # Baseline coefficient

        # Calculate baseline time if not provided
        if baseline_time is None:
            if not df.empty and time_col in df.columns:
                baseline_time = df[time_col].median()
            else:
                logger.warning("Cannot determine baseline time, using default")
                baseline_time = 1.0

        T_fmax = np.log1p(baseline_time)

        # Apply FGCS model to data
        df_analysis = df.copy()

        # Ensure we have frequency data
        if frequency_col in df_analysis.columns:
            df_analysis["sm_app_clock"] = df_analysis[frequency_col]
            df_analysis["n_sm_app_clock"] = np.log1p(df_analysis[frequency_col])
        elif "sm_clock" in df_analysis.columns:
            df_analysis[frequency_col] = df_analysis["sm_clock"]
            df_analysis["n_sm_app_clock"] = np.log1p(df_analysis["sm_clock"])

        if "n_sm_app_clock" in df_analysis.columns:
            # Apply FGCS polynomial runtime model
            df_analysis["fgcs_predicted_n_run_time"] = (
                T_fmax
                + B0
                + runtime_coeffs[0] * fp_activity
                + runtime_coeffs[1] * (7.230563 - df_analysis["n_sm_app_clock"])
                + runtime_coeffs[2] * (fp_activity**2)
                + runtime_coeffs[3] * fp_activity * (7.230563 - df_analysis["n_sm_app_clock"])
                + runtime_coeffs[4] * ((7.230563 - df_analysis["n_sm_app_clock"]) ** 2)
            )

            # Convert back to real values
            df_analysis["fgcs_predicted_time"] = np.expm1(df_analysis["fgcs_predicted_n_run_time"])

            # Calculate prediction accuracy if actual times are available
            accuracy_metrics = {}
            if time_col in df_analysis.columns:
                actual_times = df_analysis[time_col]
                predicted_times = df_analysis["fgcs_predicted_time"]

                # Remove any infinite or NaN values for accuracy calculation
                valid_mask = np.isfinite(actual_times) & np.isfinite(predicted_times)
                if valid_mask.sum() > 0:
                    actual_valid = actual_times[valid_mask]
                    predicted_valid = predicted_times[valid_mask]

                    mae = np.mean(np.abs(actual_valid - predicted_valid))
                    mape = np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100
                    rmse = np.sqrt(np.mean((actual_valid - predicted_valid) ** 2))

                    accuracy_metrics = {
                        "mae": mae,
                        "mape": mape,
                        "rmse": rmse,
                        "valid_predictions": valid_mask.sum(),
                        "total_predictions": len(df_analysis),
                    }

                    logger.info(f"FGCS model accuracy: MAE={mae:.3f}s, MAPE={mape:.1f}%, RMSE={rmse:.3f}s")

            # Performance analysis results
            analysis_results = {
                "model_parameters": {
                    "fp_activity": fp_activity,
                    "baseline_time": baseline_time,
                    "runtime_coefficients": runtime_coeffs,
                    "baseline_coefficient": B0,
                },
                "predictions": df_analysis[[frequency_col, "fgcs_predicted_time", "fgcs_predicted_n_run_time"]].to_dict(
                    "records"
                ),
                "accuracy_metrics": accuracy_metrics,
                "performance_insights": {},
                "recommendations": [],
            }

            # Generate performance insights
            if len(df_analysis) > 1:
                min_time_idx = df_analysis["fgcs_predicted_time"].idxmin()
                max_time_idx = df_analysis["fgcs_predicted_time"].idxmax()

                analysis_results["performance_insights"] = {
                    "fastest_frequency": df_analysis.loc[min_time_idx, frequency_col],
                    "fastest_time": df_analysis.loc[min_time_idx, "fgcs_predicted_time"],
                    "slowest_frequency": df_analysis.loc[max_time_idx, frequency_col],
                    "slowest_time": df_analysis.loc[max_time_idx, "fgcs_predicted_time"],
                    "performance_range": (df_analysis["fgcs_predicted_time"].min(), df_analysis["fgcs_predicted_time"].max()),
                    "performance_improvement_potential": (
                        (df_analysis["fgcs_predicted_time"].max() - df_analysis["fgcs_predicted_time"].min())
                        / df_analysis["fgcs_predicted_time"].max()
                        * 100
                    ),
                }

            # Generate recommendations
            if accuracy_metrics and accuracy_metrics.get("mape", 0) > 20:
                analysis_results["recommendations"].append(
                    f"High prediction error (MAPE: {accuracy_metrics['mape']:.1f}%) - consider collecting more training data"
                )

            if "performance_improvement_potential" in analysis_results["performance_insights"]:
                improvement = analysis_results["performance_insights"]["performance_improvement_potential"]
                if improvement > 20:
                    analysis_results["recommendations"].append(
                        f"Significant performance variation ({improvement:.1f}%) - frequency optimization recommended"
                    )

            return analysis_results

        else:
            logger.error("No frequency data available for FGCS analysis")
            return {"error": "No frequency data available"}

    def calculate_throughput_metrics(
        self,
        df: pd.DataFrame,
        time_col: str = "execution_time",
        frequency_col: str = "frequency",
        workload_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive throughput metrics for performance analysis.

        Args:
            df: DataFrame with performance data
            time_col: Execution time column name
            frequency_col: Frequency column name
            workload_size: Size of workload (operations, samples, etc.)

        Returns:
            Throughput analysis results
        """
        logger.info("Calculating comprehensive throughput metrics")

        df_throughput = df.copy()

        # Basic throughput calculations
        if time_col in df_throughput.columns:
            df_throughput["throughput_per_second"] = 1.0 / df_throughput[time_col]

            if workload_size is not None:
                df_throughput["operations_per_second"] = workload_size / df_throughput[time_col]
                df_throughput["time_per_operation"] = df_throughput[time_col] / workload_size

        # Frequency-normalized throughput
        if frequency_col in df_throughput.columns:
            df_throughput["throughput_per_mhz"] = df_throughput["throughput_per_second"] / df_throughput[frequency_col]
            df_throughput["efficiency_score"] = df_throughput["throughput_per_mhz"] * 1000  # Scale for readability

        # Statistical analysis
        throughput_metrics = {
            "basic_metrics": {},
            "frequency_analysis": {},
            "efficiency_analysis": {},
            "optimization_insights": [],
        }

        if "throughput_per_second" in df_throughput.columns:
            throughput_metrics["basic_metrics"] = {
                "mean_throughput": df_throughput["throughput_per_second"].mean(),
                "std_throughput": df_throughput["throughput_per_second"].std(),
                "min_throughput": df_throughput["throughput_per_second"].min(),
                "max_throughput": df_throughput["throughput_per_second"].max(),
                "throughput_range": df_throughput["throughput_per_second"].max()
                - df_throughput["throughput_per_second"].min(),
            }

            # Coefficient of variation
            cv = df_throughput["throughput_per_second"].std() / df_throughput["throughput_per_second"].mean()
            throughput_metrics["basic_metrics"]["coefficient_of_variation"] = cv

        # Frequency-based analysis
        if frequency_col in df_throughput.columns and len(df_throughput[frequency_col].unique()) > 1:
            freq_groups = df_throughput.groupby(frequency_col)

            frequency_analysis = {}
            for freq, group in freq_groups:
                freq_metrics = {
                    "frequency": freq,
                    "sample_count": len(group),
                    "avg_throughput": (
                        group["throughput_per_second"].mean() if "throughput_per_second" in group.columns else None
                    ),
                    "avg_execution_time": group[time_col].mean(),
                    "throughput_stability": (
                        1 / (group["throughput_per_second"].std() / group["throughput_per_second"].mean())
                        if "throughput_per_second" in group.columns and group["throughput_per_second"].std() > 0
                        else float("inf")
                    ),
                }

                if "throughput_per_mhz" in group.columns:
                    freq_metrics["normalized_efficiency"] = group["throughput_per_mhz"].mean()

                frequency_analysis[freq] = freq_metrics

            throughput_metrics["frequency_analysis"] = frequency_analysis

            # Find optimal frequencies
            if frequency_analysis:
                max_throughput_freq = max(
                    frequency_analysis.keys(), key=lambda f: frequency_analysis[f]["avg_throughput"] or 0
                )
                min_time_freq = min(frequency_analysis.keys(), key=lambda f: frequency_analysis[f]["avg_execution_time"])

                throughput_metrics["optimization_insights"] = [
                    f"Maximum throughput at {max_throughput_freq} MHz: {frequency_analysis[max_throughput_freq]['avg_throughput']:.2f} ops/sec",
                    f"Minimum execution time at {min_time_freq} MHz: {frequency_analysis[min_time_freq]['avg_execution_time']:.3f} sec",
                ]

                if "normalized_efficiency" in frequency_analysis[list(frequency_analysis.keys())[0]]:
                    max_efficiency_freq = max(
                        frequency_analysis.keys(), key=lambda f: frequency_analysis[f]["normalized_efficiency"]
                    )
                    throughput_metrics["optimization_insights"].append(
                        f"Maximum efficiency at {max_efficiency_freq} MHz: {frequency_analysis[max_efficiency_freq]['normalized_efficiency']:.6f} ops/sec/MHz"
                    )

        logger.info("Throughput metrics calculation completed")
        return throughput_metrics


class FGCSPerformanceCalculator:
    """
    Performance metrics calculator compatible with FGCS 2023 methodology.

    Provides specific calculations for FP activity, DRAM activity, and runtime
    prediction as used in the FGCS paper implementation.
    """

    @staticmethod
    def calculate_fgcs_metrics(profiling_file: str, n_runs: int = 3) -> Tuple[float, float]:
        """
        Calculate FGCS-compatible FP and DRAM activity metrics.

        Args:
            profiling_file: Path to profiling data file
            n_runs: Number of runs to average over

        Returns:
            Tuple of (fp_activity, dram_activity)
        """
        logger.info(f"Computing FGCS metrics for {n_runs} runs from {profiling_file}")

        fp64_values = []
        fp32_values = []
        dram_values = []

        # Read and process data (compatible with older pandas versions)
        try:
            df = pd.read_csv(profiling_file, sep=r"\s+", on_bad_lines="skip")
        except TypeError:
            try:
                df = pd.read_csv(profiling_file, sep=r"\s+", error_bad_lines=False)
            except TypeError:
                df = pd.read_csv(profiling_file, sep=r"\s+")

        # Clean data
        columns_to_remove = ["Entity", "#"]
        for col in columns_to_remove:
            if col in df.columns:
                del df[col]

        # Remove header rows
        if "POWER" in df.columns:
            df = df[df["POWER"] != "POWER"]

        df = df.dropna(axis=0)

        # Convert to numeric
        numeric_columns = ["FP32A", "FP64A", "DRAMA"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calculate metrics
        if "FP64A" in df.columns:
            fp64_data = df[df["FP64A"] > 0]
            if not fp64_data.empty:
                fp64_values.append(fp64_data["FP64A"].mean())

        if "FP32A" in df.columns:
            fp32_data = df[df["FP32A"] > 0]
            if not fp32_data.empty:
                fp32_values.append(fp32_data["FP32A"].mean())

        if "DRAMA" in df.columns:
            dram_data = df[df["DRAMA"] > 0]
            if not dram_data.empty:
                dram_values.append(dram_data["DRAMA"].mean())

        # Calculate averages
        fp64_avg = np.mean(fp64_values) if fp64_values else np.nan
        fp32_avg = np.mean(fp32_values) if fp32_values else np.nan
        dram_avg = np.mean(dram_values) if dram_values else 0

        # Combine FP metrics according to FGCS methodology
        if not np.isnan(fp32_avg) and not np.isnan(fp64_avg):
            fp_avg = fp32_avg / 2 + fp64_avg
        elif np.isnan(fp64_avg):
            fp_avg = fp32_avg / 2
        elif np.isnan(fp32_avg):
            fp_avg = fp64_avg
        else:
            fp_avg = 0

        logger.info(f"FGCS metrics calculated - FP: {fp_avg:.4f}, DRAM: {dram_avg:.4f}")
        return fp_avg, dram_avg

    @staticmethod
    def get_baseline_runtime(performance_file: str, app_name: str) -> float:
        """
        Get baseline runtime for an application from performance file.

        Args:
            performance_file: Path to performance data file
            app_name: Application name to look for

        Returns:
            Baseline runtime in seconds
        """
        try:
            perf_df = pd.read_csv(performance_file)

            if "app_name" in perf_df.columns and "runtime" in perf_df.columns:
                app_data = perf_df[perf_df["app_name"] == app_name]
                if not app_data.empty:
                    return app_data["runtime"].iloc[0]

            # Fallback: use mean runtime
            if "runtime" in perf_df.columns:
                return perf_df["runtime"].mean()
            elif "execution_time" in perf_df.columns:
                return perf_df["execution_time"].mean()

        except Exception as e:
            logger.warning(f"Could not load baseline runtime from {performance_file}: {e}")

        # Default baseline
        logger.warning(f"Using default baseline runtime for {app_name}")
        return 1.0


def quick_performance_analysis(
    profiling_data: pd.DataFrame, frequency_col: str = "frequency", time_col: str = "execution_time"
) -> Dict[str, Any]:
    """
    Quick performance analysis for profiling data.

    Args:
        profiling_data: DataFrame with profiling results
        frequency_col: Column name for frequency values
        time_col: Column name for execution time

    Returns:
        Dictionary with performance summary
    """
    profiler = PerformanceProfiler()

    # Basic performance metrics
    times = profiling_data[time_col].values
    metrics = profiler.calculate_performance_metrics(times)

    # Frequency scaling analysis if frequencies available
    if frequency_col in profiling_data.columns:
        scaling_results = profiler.analyze_frequency_scaling(profiling_data, frequency_col, time_col)
        metrics.update(scaling_results)

    return metrics
