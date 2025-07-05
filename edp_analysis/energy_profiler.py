"""
Energy Consumption Measurement Module

This module provides comprehensive energy profiling capabilities for GPU applications,
inspired by the gpupowermodel framework and modernized for the EDP analysis pipeline.

Key features:
- Power-to-energy conversion (Power * Time)
- Multi-run energy averaging
- Statistical analysis of energy measurements
- Energy normalization and validation
- Integration with profiling data formats
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EnergyProfiler:
    """
    Comprehensive energy profiling for GPU applications.

    Provides tools for measuring, processing, and analyzing energy consumption
    from power profiling data with statistical validation.
    """

    def __init__(self, power_units: str = "watts", time_units: str = "seconds"):
        """
        Initialize energy profiler.

        Args:
            power_units: Units for power measurements ('watts', 'milliwatts')
            time_units: Units for time measurements ('seconds', 'milliseconds')
        """
        self.power_units = power_units
        self.time_units = time_units
        self.power_scale = 1.0 if power_units == "watts" else 0.001
        self.time_scale = 1.0 if time_units == "seconds" else 0.001

        logger.info(f"Energy profiler initialized: {power_units}, {time_units}")

    def calculate_energy_from_power_time(
        self, power: Union[float, np.ndarray, pd.Series], time: Union[float, np.ndarray, pd.Series]
    ) -> Union[float, np.ndarray]:
        """
        Calculate energy consumption from power and time measurements.

        Energy (J) = Power (W) × Time (s)

        Args:
            power: Power measurements in configured units
            time: Time measurements in configured units

        Returns:
            Energy consumption in Joules
        """
        # Apply unit scaling
        power_watts = np.array(power) * self.power_scale
        time_seconds = np.array(time) * self.time_scale

        # Calculate energy
        energy_joules = power_watts * time_seconds

        return energy_joules

    def process_profiling_run(
        self,
        profiling_data: pd.DataFrame,
        power_col: str = "power",
        time_col: str = "execution_time",
        frequency_col: str = "frequency",
    ) -> pd.DataFrame:
        """
        Process single profiling run to calculate energy metrics.

        Args:
            profiling_data: DataFrame with profiling measurements
            power_col: Column name for power measurements
            time_col: Column name for execution time
            frequency_col: Column name for frequency settings

        Returns:
            DataFrame with added energy calculations
        """
        if profiling_data.empty:
            raise ValueError("Profiling data is empty")

        required_cols = [power_col, time_col]
        missing_cols = [col for col in required_cols if col not in profiling_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        result_df = profiling_data.copy()

        # Calculate energy for each measurement
        result_df["energy_joules"] = self.calculate_energy_from_power_time(result_df[power_col], result_df[time_col])

        # Calculate derived metrics
        result_df["power_efficiency"] = result_df["energy_joules"] / result_df[time_col]  # J/s = W

        if frequency_col in result_df.columns:
            result_df["energy_per_mhz"] = result_df["energy_joules"] / result_df[frequency_col]

        logger.info(f"Processed {len(result_df)} energy measurements")
        return result_df

    def aggregate_multi_run_energy(
        self, run_data: List[pd.DataFrame], group_by_cols: List[str] = ["frequency"], energy_col: str = "energy_joules"
    ) -> pd.DataFrame:
        """
        Aggregate energy measurements across multiple runs.

        Args:
            run_data: List of DataFrames from multiple profiling runs
            group_by_cols: Columns to group by for aggregation
            energy_col: Column name for energy measurements

        Returns:
            DataFrame with aggregated energy statistics
        """
        if not run_data:
            raise ValueError("No run data provided")

        # Combine all runs
        combined_df = pd.concat(run_data, ignore_index=True)

        # Calculate statistics for each group
        stats_df = combined_df.groupby(group_by_cols)[energy_col].agg(["mean", "std", "min", "max", "count"]).reset_index()

        # Rename columns for clarity
        stats_df.rename(
            columns={
                "mean": f"{energy_col}_mean",
                "std": f"{energy_col}_std",
                "min": f"{energy_col}_min",
                "max": f"{energy_col}_max",
                "count": "num_runs",
            },
            inplace=True,
        )

        # Calculate confidence intervals (95%)
        stats_df[f"{energy_col}_ci_95"] = 1.96 * stats_df[f"{energy_col}_std"] / np.sqrt(stats_df["num_runs"])

        # Calculate coefficient of variation (CV)
        stats_df[f"{energy_col}_cv"] = stats_df[f"{energy_col}_std"] / stats_df[f"{energy_col}_mean"]

        logger.info(f"Aggregated energy data for {len(stats_df)} configurations")
        return stats_df

    def load_and_process_profiling_files(
        self,
        file_paths: List[str],
        power_col: str = "power",
        time_col: str = "execution_time",
        frequency_col: str = "frequency",
    ) -> Dict[str, Any]:
        """
        Load and process multiple profiling files.

        Args:
            file_paths: List of profiling data file paths
            power_col: Column name for power measurements
            time_col: Column name for execution time
            frequency_col: Column name for frequency settings

        Returns:
            Dictionary with processed data and summary statistics
        """
        run_data = []
        failed_files = []

        for i, file_path in enumerate(file_paths):
            try:
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")

                # Try different delimiters and error handling for older pandas versions
                try:
                    df = pd.read_csv(file_path, sep=r"\s+", on_bad_lines="skip")
                except TypeError:
                    try:
                        df = pd.read_csv(file_path, sep=r"\s+", error_bad_lines=False)
                    except TypeError:
                        df = pd.read_csv(file_path, sep=r"\s+")

                # Process the profiling run
                processed_df = self.process_profiling_run(df, power_col, time_col, frequency_col)
                processed_df["run_id"] = i
                processed_df["source_file"] = Path(file_path).name

                run_data.append(processed_df)

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                failed_files.append((file_path, str(e)))

        if not run_data:
            raise ValueError("No files were successfully processed")

        # Aggregate across runs
        aggregated_data = self.aggregate_multi_run_energy(run_data)

        # Calculate overall statistics
        all_energy = pd.concat([df["energy_joules"] for df in run_data])
        overall_stats = {
            "total_measurements": len(all_energy),
            "mean_energy": all_energy.mean(),
            "std_energy": all_energy.std(),
            "min_energy": all_energy.min(),
            "max_energy": all_energy.max(),
            "energy_range": all_energy.max() - all_energy.min(),
            "cv_energy": all_energy.std() / all_energy.mean(),
        }

        results = {
            "raw_data": run_data,
            "aggregated_data": aggregated_data,
            "overall_stats": overall_stats,
            "failed_files": failed_files,
            "successfully_processed": len(run_data),
            "total_files": len(file_paths),
        }

        logger.info(f"Energy profiling complete: {len(run_data)}/{len(file_paths)} files processed")

        return results

    def normalize_energy_measurements(
        self,
        energy_data: pd.DataFrame,
        energy_col: str = "energy_joules_mean",
        baseline_frequency: Optional[int] = None,
        baseline_value: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Normalize energy measurements relative to a baseline.

        Args:
            energy_data: DataFrame with energy measurements
            energy_col: Column name for energy values
            baseline_frequency: Frequency to use as baseline (if available)
            baseline_value: Explicit baseline value to use

        Returns:
            DataFrame with normalized energy values
        """
        if energy_col not in energy_data.columns:
            raise ValueError(f"Energy column '{energy_col}' not found in data")

        result_df = energy_data.copy()

        # Determine baseline
        if baseline_value is not None:
            baseline = baseline_value
            logger.info(f"Using explicit baseline: {baseline:.4f} J")
        elif baseline_frequency is not None and "frequency" in energy_data.columns:
            baseline_row = energy_data[energy_data["frequency"] == baseline_frequency]
            if baseline_row.empty:
                raise ValueError(f"Baseline frequency {baseline_frequency} not found in data")
            baseline = baseline_row[energy_col].iloc[0]
            logger.info(f"Using baseline from {baseline_frequency} MHz: {baseline:.4f} J")
        else:
            baseline = energy_data[energy_col].max()  # Use maximum as baseline
            logger.info(f"Using maximum energy as baseline: {baseline:.4f} J")

        # Calculate normalized values
        result_df["energy_normalized"] = result_df[energy_col] / baseline
        result_df["energy_improvement_percent"] = (1 - result_df["energy_normalized"]) * 100

        return result_df

    def validate_energy_measurements(
        self,
        energy_data: pd.DataFrame,
        energy_col: str = "energy_joules_mean",
        cv_threshold: float = 0.2,
        outlier_threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Validate energy measurements for quality and consistency.

        Args:
            energy_data: DataFrame with energy measurements
            energy_col: Column name for energy values
            cv_threshold: Maximum acceptable coefficient of variation
            outlier_threshold: Z-score threshold for outlier detection

        Returns:
            Dictionary with validation results and recommendations
        """
        if energy_col not in energy_data.columns:
            raise ValueError(f"Energy column '{energy_col}' not found in data")

        energy_values = energy_data[energy_col].dropna()

        if len(energy_values) == 0:
            raise ValueError("No valid energy measurements found")

        # Calculate statistics
        mean_energy = energy_values.mean()
        std_energy = energy_values.std()
        cv = std_energy / mean_energy

        # Detect outliers using Z-score
        z_scores = np.abs((energy_values - mean_energy) / std_energy)
        outliers = energy_values[z_scores > outlier_threshold]

        # Check for missing standard deviations if available
        std_col = f'{energy_col.replace("_mean", "_std")}'
        high_variance_measurements = []
        if std_col in energy_data.columns:
            high_cv_mask = (energy_data[std_col] / energy_data[energy_col]) > cv_threshold
            high_variance_measurements = energy_data[high_cv_mask]

        # Generate validation results
        validation_results = {
            "total_measurements": len(energy_values),
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "coefficient_of_variation": cv,
            "cv_acceptable": cv <= cv_threshold,
            "outliers_detected": len(outliers),
            "outlier_values": outliers.tolist(),
            "high_variance_measurements": len(high_variance_measurements),
            "data_quality_score": self._calculate_quality_score(cv, len(outliers), len(energy_values)),
            "recommendations": [],
        }

        # Generate recommendations
        if cv > cv_threshold:
            validation_results["recommendations"].append(
                f"High measurement variability (CV={cv:.3f}). Consider more runs or investigate measurement conditions."
            )

        if len(outliers) > 0:
            validation_results["recommendations"].append(
                f"Detected {len(outliers)} outlier measurements. Review measurement conditions and consider removal."
            )

        if len(high_variance_measurements) > 0:
            validation_results["recommendations"].append(
                f"{len(high_variance_measurements)} measurements have high individual variance. Increase runs per configuration."
            )

        if not validation_results["recommendations"]:
            validation_results["recommendations"].append("Energy measurements appear to be of good quality.")

        logger.info(f"Energy validation complete: Quality score = {validation_results['data_quality_score']:.2f}")

        return validation_results

    def _calculate_quality_score(self, cv: float, num_outliers: int, total_measurements: int) -> float:
        """Calculate a quality score (0-1) for energy measurements."""
        # Base score from coefficient of variation
        cv_score = max(0, 1 - cv)  # Lower CV is better

        # Penalty for outliers
        outlier_penalty = (num_outliers / total_measurements) * 0.5

        # Final score
        quality_score = max(0, cv_score - outlier_penalty)

        return quality_score

    def calculate_fgcs_compatible_energy(
        self, df: pd.DataFrame, power_col: str = "power", time_col: str = "execution_time", frequency_col: str = "frequency"
    ) -> pd.DataFrame:
        """
        Calculate energy metrics compatible with FGCS methodology.

        Args:
            df: DataFrame with profiling data
            power_col: Power column name
            time_col: Execution time column name
            frequency_col: Frequency column name

        Returns:
            DataFrame with FGCS-compatible energy calculations
        """
        logger.info("Calculating FGCS-compatible energy metrics")

        df_energy = df.copy()

        # Basic energy calculation
        if power_col in df_energy.columns and time_col in df_energy.columns:
            df_energy["energy"] = (df_energy[power_col] * self.power_scale) * (df_energy[time_col] * self.time_scale)

            # FGCS-style logarithmic transformations
            df_energy["n_power_usage"] = np.log1p(df_energy[power_col])
            df_energy["n_run_time"] = np.log1p(df_energy[time_col])
            df_energy["n_energy"] = np.log1p(df_energy["energy"])

            # Real-space predictions (for compatibility with FGCS pipeline)
            df_energy["predicted_n_to_r_power_usage"] = df_energy[power_col]
            df_energy["predicted_n_to_r_run_time"] = df_energy[time_col]
            df_energy["predicted_n_to_r_energy"] = df_energy["energy"]

            logger.info("Added FGCS-compatible energy transformations")

        # Frequency transformations
        if frequency_col in df_energy.columns:
            df_energy["sm_app_clock"] = df_energy[frequency_col]
            df_energy["n_sm_app_clock"] = np.log1p(df_energy[frequency_col])
        elif "sm_clock" in df_energy.columns:
            df_energy["sm_app_clock"] = df_energy["sm_clock"]
            df_energy["n_sm_app_clock"] = np.log1p(df_energy["sm_clock"])

        return df_energy

    def analyze_energy_efficiency_across_frequencies(
        self, df: pd.DataFrame, power_col: str = "power", time_col: str = "execution_time", frequency_col: str = "frequency"
    ) -> Dict[str, Any]:
        """
        Analyze energy efficiency patterns across different frequencies.

        Args:
            df: DataFrame with profiling data across frequencies
            power_col: Power column name
            time_col: Execution time column name
            frequency_col: Frequency column name

        Returns:
            Energy efficiency analysis results
        """
        logger.info("Analyzing energy efficiency across frequencies")

        # Calculate FGCS-compatible energy metrics
        df_energy = self.calculate_fgcs_compatible_energy(df, power_col, time_col, frequency_col)

        # Group by frequency for analysis
        if frequency_col in df_energy.columns:
            freq_groups = df_energy.groupby(frequency_col)

            efficiency_analysis = {
                "frequency_analysis": {},
                "efficiency_metrics": {},
                "optimization_insights": {},
                "recommendations": [],
            }

            # Analyze each frequency
            for freq, group in freq_groups:
                freq_metrics = {
                    "frequency": freq,
                    "sample_count": len(group),
                    "avg_power": group[power_col].mean(),
                    "std_power": group[power_col].std(),
                    "avg_time": group[time_col].mean(),
                    "std_time": group[time_col].std(),
                    "avg_energy": group["energy"].mean(),
                    "std_energy": group["energy"].std(),
                    "power_efficiency": group[power_col].mean() / freq if freq > 0 else 0,  # Power per MHz
                    "energy_efficiency": (
                        group["energy"].mean() / group[time_col].mean() if group[time_col].mean() > 0 else 0
                    ),  # Energy per time
                }

                efficiency_analysis["frequency_analysis"][freq] = freq_metrics

            # Calculate overall efficiency metrics
            all_frequencies = list(efficiency_analysis["frequency_analysis"].keys())
            all_energies = [efficiency_analysis["frequency_analysis"][f]["avg_energy"] for f in all_frequencies]
            all_times = [efficiency_analysis["frequency_analysis"][f]["avg_time"] for f in all_frequencies]
            all_powers = [efficiency_analysis["frequency_analysis"][f]["avg_power"] for f in all_frequencies]

            # Find efficiency sweet spots
            min_energy_idx = np.argmin(all_energies)
            min_time_idx = np.argmin(all_times)
            min_power_idx = np.argmin(all_powers)

            efficiency_analysis["efficiency_metrics"] = {
                "min_energy_frequency": all_frequencies[min_energy_idx],
                "min_energy_value": all_energies[min_energy_idx],
                "min_time_frequency": all_frequencies[min_time_idx],
                "min_time_value": all_times[min_time_idx],
                "min_power_frequency": all_frequencies[min_power_idx],
                "min_power_value": all_powers[min_power_idx],
                "energy_range": (min(all_energies), max(all_energies)),
                "time_range": (min(all_times), max(all_times)),
                "power_range": (min(all_powers), max(all_powers)),
            }

            # Generate optimization insights
            energy_savings = (max(all_energies) - min(all_energies)) / max(all_energies) * 100
            time_penalty = (max(all_times) - min(all_times)) / min(all_times) * 100

            efficiency_analysis["optimization_insights"] = {
                "max_energy_savings_percent": energy_savings,
                "max_time_penalty_percent": time_penalty,
                "energy_vs_performance_tradeoff": energy_savings / time_penalty if time_penalty > 0 else float("inf"),
                "frequency_sweep_effectiveness": (
                    len(all_frequencies) / (max(all_frequencies) - min(all_frequencies)) * 100
                    if len(all_frequencies) > 1
                    else 0
                ),
            }

            # Generate recommendations
            if energy_savings > 20:
                efficiency_analysis["recommendations"].append(
                    f"Significant energy savings ({energy_savings:.1f}%) possible with frequency optimization"
                )

            if time_penalty < 10:
                efficiency_analysis["recommendations"].append(
                    f"Low performance penalty ({time_penalty:.1f}%) - good candidate for energy optimization"
                )

            if len(all_frequencies) < 5:
                efficiency_analysis["recommendations"].append(
                    "Consider testing more frequency points for comprehensive analysis"
                )

            logger.info(f"Energy efficiency analysis complete: {len(all_frequencies)} frequencies analyzed")
            return efficiency_analysis

        else:
            logger.warning(f"Frequency column '{frequency_col}' not found")
            return {"error": f"Frequency column '{frequency_col}' not found"}

    def validate_energy_measurements(
        self, df: pd.DataFrame, power_col: str = "power", time_col: str = "execution_time", tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        Validate energy measurement quality and consistency.

        Args:
            df: DataFrame with energy measurements
            power_col: Power column name
            time_col: Time column name
            tolerance: Acceptable variation tolerance (0.1 = 10%)

        Returns:
            Validation results
        """
        logger.info("Validating energy measurement quality")

        validation_results = {"data_quality": {}, "measurement_consistency": {}, "outlier_analysis": {}, "recommendations": []}

        # Data quality checks
        total_samples = len(df)
        missing_power = df[power_col].isnull().sum() if power_col in df.columns else total_samples
        missing_time = df[time_col].isnull().sum() if time_col in df.columns else total_samples

        validation_results["data_quality"] = {
            "total_samples": total_samples,
            "missing_power_measurements": missing_power,
            "missing_time_measurements": missing_time,
            "data_completeness": 1 - max(missing_power, missing_time) / total_samples,
            "power_range": (df[power_col].min(), df[power_col].max()) if power_col in df.columns else None,
            "time_range": (df[time_col].min(), df[time_col].max()) if time_col in df.columns else None,
        }

        # Measurement consistency analysis
        if power_col in df.columns and time_col in df.columns:
            # Calculate energy and analyze consistency
            energy_values = df[power_col] * df[time_col]

            # Check for repeated measurements at same frequency
            if "frequency" in df.columns or "sm_clock" in df.columns:
                freq_col = "frequency" if "frequency" in df.columns else "sm_clock"
                freq_groups = df.groupby(freq_col)

                consistency_metrics = {}
                for freq, group in freq_groups:
                    if len(group) > 1:
                        group_energy = group[power_col] * group[time_col]
                        cv = group_energy.std() / group_energy.mean() if group_energy.mean() > 0 else float("inf")
                        consistency_metrics[freq] = {
                            "sample_count": len(group),
                            "energy_cv": cv,
                            "is_consistent": cv < tolerance,
                        }

                validation_results["measurement_consistency"] = consistency_metrics

                # Overall consistency score
                consistent_freqs = sum(1 for metrics in consistency_metrics.values() if metrics["is_consistent"])
                total_freqs = len(consistency_metrics)
                validation_results["overall_consistency_score"] = consistent_freqs / total_freqs if total_freqs > 0 else 0

        # Outlier detection
        if power_col in df.columns:
            power_q1, power_q3 = df[power_col].quantile([0.25, 0.75])
            power_iqr = power_q3 - power_q1
            power_outliers = df[(df[power_col] < power_q1 - 1.5 * power_iqr) | (df[power_col] > power_q3 + 1.5 * power_iqr)]

            validation_results["outlier_analysis"]["power_outliers"] = {
                "count": len(power_outliers),
                "percentage": len(power_outliers) / total_samples * 100,
                "outlier_indices": power_outliers.index.tolist(),
            }

        if time_col in df.columns:
            time_q1, time_q3 = df[time_col].quantile([0.25, 0.75])
            time_iqr = time_q3 - time_q1
            time_outliers = df[(df[time_col] < time_q1 - 1.5 * time_iqr) | (df[time_col] > time_q3 + 1.5 * time_iqr)]

            validation_results["outlier_analysis"]["time_outliers"] = {
                "count": len(time_outliers),
                "percentage": len(time_outliers) / total_samples * 100,
                "outlier_indices": time_outliers.index.tolist(),
            }

        # Generate recommendations
        if validation_results["data_quality"]["data_completeness"] < 0.9:
            validation_results["recommendations"].append(
                f"Data completeness is {validation_results['data_quality']['data_completeness']:.1%} - consider collecting more complete measurements"
            )

        if "overall_consistency_score" in validation_results and validation_results["overall_consistency_score"] < 0.8:
            validation_results["recommendations"].append(
                f"Measurement consistency is {validation_results['overall_consistency_score']:.1%} - consider increasing measurement repetitions"
            )

        if any(analysis["percentage"] > 5 for analysis in validation_results["outlier_analysis"].values()):
            validation_results["recommendations"].append(
                "High percentage of outliers detected - review measurement conditions"
            )

        logger.info("Energy measurement validation completed")
        return validation_results


def calculate_energy_from_power_time(
    df: pd.DataFrame, power_col: str = "power", time_col: str = "execution_time", energy_col: str = "energy"
) -> pd.DataFrame:
    """
    Convenience function to calculate energy from power and time.

    Args:
        df: DataFrame with power and time data
        power_col: Column name for power values (Watts)
        time_col: Column name for execution time (seconds)
        energy_col: Name for the new energy column

    Returns:
        DataFrame with added energy column
    """
    profiler = EnergyProfiler()
    df = df.copy()
    df[energy_col] = profiler.calculate_energy_from_power_time(df[power_col], df[time_col])
    return df


def validate_energy_data_quality(energy_data: pd.DataFrame, energy_col: str = "energy", cv_threshold: float = 0.2) -> bool:
    """
    Quick validation of energy data quality.

    Args:
        energy_data: DataFrame with energy measurements
        energy_col: Column name for energy values
        cv_threshold: Maximum acceptable coefficient of variation

    Returns:
        True if data quality is acceptable
    """
    profiler = EnergyProfiler()
    validation_results = profiler.validate_energy_measurements(energy_data, energy_col, cv_threshold)

    return validation_results["cv_acceptable"] and validation_results["outliers_detected"] == 0
