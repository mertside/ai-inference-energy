"""
Optimization Analyzer Module

This module handles multi-objective optimization, configuration search, and recommendation
generation for energy-efficient GPU frequency management, implementing the optimization
strategies from the FGCS 2023 paper.

Key Features:
- Multi-objective optimization (energy vs. performance)
- Pareto frontier analysis
- DVFS optimization pipeline
- Configuration recommendations
- Improvement calculations
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Data class for optimization results."""

    frequency: int
    energy: float
    execution_time: float
    power: float
    score: float
    metric_type: str
    energy_improvement: Optional[float] = None
    time_improvement: Optional[float] = None


@dataclass
class OptimizationRecommendation:
    """Data class for optimization recommendations."""

    frequency: int
    reason: str
    use_case: str
    expected_energy_savings: Optional[str] = None
    expected_performance_impact: Optional[str] = None


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for energy-performance trade-offs.

    Implements various optimization strategies including Pareto frontier analysis,
    weighted scoring, and constraint-based optimization.
    """

    def __init__(self, energy_weight: float = 0.5, performance_weight: float = 0.5):
        """
        Initialize multi-objective optimizer.

        Args:
            energy_weight: Weight for energy objective (0.0 to 1.0)
            performance_weight: Weight for performance objective (0.0 to 1.0)
        """
        if not (0.0 <= energy_weight <= 1.0 and 0.0 <= performance_weight <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")

        # Normalize weights to sum to 1.0
        total_weight = energy_weight + performance_weight
        self.energy_weight = energy_weight / total_weight
        self.performance_weight = performance_weight / total_weight

        logger.info(
            f"Multi-objective optimizer initialized: energy_weight={self.energy_weight:.3f}, "
            f"performance_weight={self.performance_weight:.3f}"
        )

    def find_pareto_frontier(
        self,
        df: pd.DataFrame,
        energy_col: str = "energy",
        time_col: str = "execution_time",
        minimize_both: bool = True,
    ) -> pd.DataFrame:
        """
        Find Pareto-optimal solutions for energy-performance trade-offs.

        Args:
            df: DataFrame with optimization data
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            minimize_both: Whether to minimize both objectives (True) or minimize energy, maximize performance

        Returns:
            DataFrame containing Pareto-optimal solutions
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        pareto_solutions = []
        df_sorted = df.sort_values([energy_col, time_col])

        for i, row in df_sorted.iterrows():
            is_dominated = False

            for j, other_row in df_sorted.iterrows():
                if i == j:
                    continue

                if minimize_both:
                    # Both objectives should be minimized
                    if (
                        other_row[energy_col] <= row[energy_col]
                        and other_row[time_col] <= row[time_col]
                        and (
                            other_row[energy_col] < row[energy_col]
                            or other_row[time_col] < row[time_col]
                        )
                    ):
                        is_dominated = True
                        break
                else:
                    # Minimize energy, maximize performance (minimize time)
                    if (
                        other_row[energy_col] <= row[energy_col]
                        and other_row[time_col] <= row[time_col]
                        and (
                            other_row[energy_col] < row[energy_col]
                            or other_row[time_col] < row[time_col]
                        )
                    ):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_solutions.append(i)

        pareto_df = df.loc[pareto_solutions].copy()
        logger.info(
            f"Found {len(pareto_df)} Pareto-optimal solutions out of {len(df)} configurations"
        )

        return pareto_df

    def weighted_sum_optimization(
        self,
        df: pd.DataFrame,
        energy_col: str = "energy",
        time_col: str = "execution_time",
        normalize: bool = True,
    ) -> OptimizationResult:
        """
        Perform weighted sum optimization.

        Args:
            df: DataFrame with optimization data
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            normalize: Whether to normalize objectives before weighting

        Returns:
            OptimizationResult with optimal configuration
        """
        df_work = df.copy()

        if normalize:
            # Min-max normalization
            energy_min, energy_max = (
                df_work[energy_col].min(),
                df_work[energy_col].max(),
            )
            time_min, time_max = df_work[time_col].min(), df_work[time_col].max()

            if energy_max > energy_min:
                df_work["energy_norm"] = (df_work[energy_col] - energy_min) / (
                    energy_max - energy_min
                )
            else:
                df_work["energy_norm"] = 0.0

            if time_max > time_min:
                df_work["time_norm"] = (df_work[time_col] - time_min) / (
                    time_max - time_min
                )
            else:
                df_work["time_norm"] = 0.0

            df_work["weighted_score"] = (
                self.energy_weight * df_work["energy_norm"]
                + self.performance_weight * df_work["time_norm"]
            )
        else:
            df_work["weighted_score"] = (
                self.energy_weight * df_work[energy_col]
                + self.performance_weight * df_work[time_col]
            )

        # Find optimal solution (minimum weighted score)
        optimal_idx = df_work["weighted_score"].idxmin()
        optimal_row = df_work.loc[optimal_idx]

        return OptimizationResult(
            frequency=optimal_row.get("frequency", optimal_row.get("sm_app_clock", 0)),
            energy=optimal_row[energy_col],
            execution_time=optimal_row[time_col],
            power=optimal_row.get(
                "power", optimal_row.get("predicted_n_to_r_power_usage", 0)
            ),
            score=optimal_row["weighted_score"],
            metric_type="weighted_sum",
        )

    def constraint_based_optimization(
        self,
        df: pd.DataFrame,
        energy_col: str = "energy",
        time_col: str = "execution_time",
        max_energy: Optional[float] = None,
        max_time: Optional[float] = None,
        objective: str = "energy",
    ) -> Optional[OptimizationResult]:
        """
        Perform constraint-based optimization.

        Args:
            df: DataFrame with optimization data
            energy_col: Column name for energy values
            time_col: Column name for execution time values
            max_energy: Maximum allowed energy constraint
            max_time: Maximum allowed execution time constraint
            objective: Primary objective to optimize ('energy' or 'time')

        Returns:
            OptimizationResult if feasible solution found, None otherwise
        """
        feasible_df = df.copy()

        # Apply constraints
        if max_energy is not None:
            feasible_df = feasible_df[feasible_df[energy_col] <= max_energy]

        if max_time is not None:
            feasible_df = feasible_df[feasible_df[time_col] <= max_time]

        if feasible_df.empty:
            logger.warning("No feasible solutions found with given constraints")
            return None

        # Optimize primary objective
        if objective == "energy":
            optimal_idx = feasible_df[energy_col].idxmin()
        elif objective == "time":
            optimal_idx = feasible_df[time_col].idxmin()
        else:
            raise ValueError(f"Unknown objective: {objective}. Use 'energy' or 'time'")

        optimal_row = feasible_df.loc[optimal_idx]

        return OptimizationResult(
            frequency=optimal_row.get("frequency", optimal_row.get("sm_app_clock", 0)),
            energy=optimal_row[energy_col],
            execution_time=optimal_row[time_col],
            power=optimal_row.get(
                "power", optimal_row.get("predicted_n_to_r_power_usage", 0)
            ),
            score=(
                optimal_row[energy_col]
                if objective == "energy"
                else optimal_row[time_col]
            ),
            metric_type=f"constraint_{objective}",
        )


class FGCSOptimizer:
    """
    EDP optimizer implementing the exact methodology from FGCS 2023 paper.

    This class encapsulates the specific optimization algorithms and metrics
    used in the research paper.
    """

    @staticmethod
    def edp_optimal(
        df: pd.DataFrame,
        energy_col: str = "predicted_n_to_r_energy",
        time_col: str = "predicted_n_to_r_run_time",
        energy_weight: float = 0.5,
        time_weight: float = 0.5,
    ) -> OptimizationResult:
        """
        Find EDP optimal configuration using FGCS 2023 methodology.

        EDP = Energy × Delay (with optional weights)

        Args:
            df: DataFrame with frequency, energy, and time data
            energy_col: Column name for energy values
            time_col: Column name for time values
            energy_weight: Weight for energy component
            time_weight: Weight for time component

        Returns:
            OptimizationResult with optimal configuration
        """
        logger.info("Finding EDP optimal configuration using FGCS methodology")

        df_work = df.copy()
        df_work["edp_score"] = (time_weight * df_work[time_col]) * (
            energy_weight * df_work[energy_col]
        )

        optimal_idx = df_work["edp_score"].idxmin()
        optimal_row = df_work.loc[optimal_idx]

        result = OptimizationResult(
            frequency=int(
                optimal_row.get("sm_app_clock", optimal_row.get("frequency", 0))
            ),
            energy=round(optimal_row[energy_col], 2),
            execution_time=round(optimal_row[time_col], 2),
            power=round(
                optimal_row.get(
                    "predicted_n_to_r_power_usage", optimal_row.get("power", 0)
                ),
                2,
            ),
            score=optimal_row["edp_score"],
            metric_type="edp",
        )

        logger.info(
            f"EDP Optimal: f={result.frequency}MHz, t={result.execution_time}s, "
            f"p={result.power}W, e={result.energy}J"
        )

        return result

    @staticmethod
    def ed2p_optimal(
        df: pd.DataFrame,
        energy_col: str = "predicted_n_to_r_energy",
        time_col: str = "predicted_n_to_r_run_time",
        energy_weight: float = 0.5,
        time_weight: float = 0.5,
    ) -> OptimizationResult:
        """
        Find ED²P optimal configuration using FGCS 2023 methodology.

        ED²P = Energy × Delay² (with optional weights)

        Args:
            df: DataFrame with frequency, energy, and time data
            energy_col: Column name for energy values
            time_col: Column name for time values
            energy_weight: Weight for energy component
            time_weight: Weight for time component

        Returns:
            OptimizationResult with optimal configuration
        """
        logger.info("Finding ED²P optimal configuration using FGCS methodology")

        df_work = df.copy()
        df_work["ed2p_score"] = (time_weight * (df_work[time_col] ** 2)) * (
            energy_weight * df_work[energy_col]
        )

        optimal_idx = df_work["ed2p_score"].idxmin()
        optimal_row = df_work.loc[optimal_idx]

        result = OptimizationResult(
            frequency=int(
                optimal_row.get("sm_app_clock", optimal_row.get("frequency", 0))
            ),
            energy=round(optimal_row[energy_col], 2),
            execution_time=round(optimal_row[time_col], 2),
            power=round(
                optimal_row.get(
                    "predicted_n_to_r_power_usage", optimal_row.get("power", 0)
                ),
                2,
            ),
            score=optimal_row["ed2p_score"],
            metric_type="ed2p",
        )

        logger.info(
            f"ED²P Optimal: f={result.frequency}MHz, t={result.execution_time}s, "
            f"p={result.power}W, e={result.energy}J"
        )

        return result


class OptimizationAnalyzer:
    """
    Main optimization analyzer that coordinates different optimization strategies
    and provides comprehensive analysis and recommendations.
    """

    def __init__(self, energy_weight: float = 0.5, performance_weight: float = 0.5):
        """
        Initialize optimization analyzer.

        Args:
            energy_weight: Weight for energy objective
            performance_weight: Weight for performance objective
        """
        self.multi_objective = MultiObjectiveOptimizer(
            energy_weight, performance_weight
        )
        self.fgcs_optimizer = FGCSOptimizer()
        logger.info("Optimization analyzer initialized")

    def comprehensive_analysis(
        self,
        df: pd.DataFrame,
        app_name: str,
        energy_col: str = "predicted_n_to_r_energy",
        time_col: str = "predicted_n_to_r_run_time",
    ) -> Dict:
        """
        Perform comprehensive optimization analysis.

        Args:
            df: DataFrame with profiling results
            app_name: Application name for logging
            energy_col: Column name for energy values
            time_col: Column name for time values

        Returns:
            Dictionary with complete optimization analysis
        """
        logger.info(f"Performing comprehensive optimization analysis for {app_name}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        # EDP and ED²P optimization
        edp_result = self.fgcs_optimizer.edp_optimal(df, energy_col, time_col)
        ed2p_result = self.fgcs_optimizer.ed2p_optimal(df, energy_col, time_col)

        # Multi-objective optimization
        weighted_result = self.multi_objective.weighted_sum_optimization(
            df, energy_col, time_col
        )

        # Pareto frontier analysis
        pareto_frontier = self.multi_objective.find_pareto_frontier(
            df, energy_col, time_col
        )

        # Extreme points analysis
        min_energy_idx = df[energy_col].idxmin()
        min_energy_row = df.loc[min_energy_idx]

        min_time_idx = df[time_col].idxmin()
        min_time_row = df.loc[min_time_idx]

        # Calculate improvements
        baseline_energy = df[energy_col].max()
        baseline_time = df[time_col].max()

        edp_result.energy_improvement = self._calculate_improvement(
            baseline_energy, edp_result.energy
        )
        edp_result.time_improvement = self._calculate_improvement(
            baseline_time, edp_result.execution_time
        )

        results = {
            "application": app_name,
            "edp_optimal": edp_result,
            "ed2p_optimal": ed2p_result,
            "weighted_optimal": weighted_result,
            "pareto_frontier": pareto_frontier,
            "min_energy_config": {
                "frequency": int(
                    min_energy_row.get(
                        "sm_app_clock", min_energy_row.get("frequency", 0)
                    )
                ),
                "energy": min_energy_row[energy_col],
                "time": min_energy_row[time_col],
            },
            "min_time_config": {
                "frequency": int(
                    min_time_row.get("sm_app_clock", min_time_row.get("frequency", 0))
                ),
                "energy": min_time_row[energy_col],
                "time": min_time_row[time_col],
            },
            "statistics": {
                "total_configurations": len(df),
                "pareto_points": len(pareto_frontier),
                "energy_range": df[energy_col].max() - df[energy_col].min(),
                "time_range": df[time_col].max() - df[time_col].min(),
                "frequency_range": df.get(
                    "sm_app_clock", df.get("frequency", pd.Series([0]))
                ).max()
                - df.get("sm_app_clock", df.get("frequency", pd.Series([0]))).min(),
            },
        }

        logger.info(f"Comprehensive analysis complete for {app_name}")
        return results

    def generate_recommendations(
        self, optimization_results: Dict
    ) -> Dict[str, OptimizationRecommendation]:
        """
        Generate practical recommendations based on optimization results.

        Args:
            optimization_results: Results from comprehensive_analysis

        Returns:
            Dictionary of recommendations for different use cases
        """
        edp_optimal = optimization_results["edp_optimal"]
        ed2p_optimal = optimization_results["ed2p_optimal"]
        min_energy = optimization_results["min_energy_config"]
        min_time = optimization_results["min_time_config"]

        recommendations = {
            "primary": OptimizationRecommendation(
                frequency=edp_optimal.frequency,
                reason="EDP optimal - best energy-delay trade-off",
                use_case="General purpose workloads requiring balanced energy-performance trade-off",
                expected_energy_savings=(
                    f"{edp_optimal.energy_improvement:.1f}%"
                    if edp_optimal.energy_improvement
                    else "N/A"
                ),
                expected_performance_impact=(
                    f"{edp_optimal.time_improvement:.1f}%"
                    if edp_optimal.time_improvement
                    else "N/A"
                ),
            ),
            "performance_priority": OptimizationRecommendation(
                frequency=ed2p_optimal.frequency,
                reason="ED²P optimal - prioritizes performance over energy",
                use_case="Latency-sensitive applications where response time is critical",
            ),
            "energy_conservative": OptimizationRecommendation(
                frequency=min_energy["frequency"],
                reason="Minimum energy consumption",
                use_case="Battery-powered devices or energy-constrained environments",
            ),
            "maximum_performance": OptimizationRecommendation(
                frequency=min_time["frequency"],
                reason="Maximum performance configuration",
                use_case="Compute-intensive workloads where execution time is paramount",
            ),
        }

        return recommendations

    def _calculate_improvement(self, baseline: float, optimized: float) -> float:
        """Calculate percentage improvement from baseline to optimized value."""
        if baseline > 0 and not np.isnan(baseline) and not np.isnan(optimized):
            return (baseline - optimized) / baseline * 100
        return 0.0


class DVFSOptimizationPipeline:
    """
    Complete DVFS optimization pipeline integrating power modeling and EDP analysis.

    This pipeline coordinates the entire optimization process from power prediction
    to final recommendations.
    """

    def __init__(self, power_model, runtime_model=None):
        """
        Initialize optimization pipeline.

        Args:
            power_model: Trained power prediction model
            runtime_model: Optional runtime prediction model
        """
        self.power_model = power_model
        self.runtime_model = runtime_model
        self.optimizer = OptimizationAnalyzer()
        logger.info("DVFS optimization pipeline initialized")

    def optimize_application(
        self,
        fp_activity: float,
        dram_activity: float,
        baseline_runtime: float,
        frequencies: List[int],
        app_name: str = "Application",
    ) -> Dict:
        """
        Complete optimization pipeline for an application.

        Args:
            fp_activity: FP operations activity metric
            dram_activity: DRAM activity metric
            baseline_runtime: Baseline execution time
            frequencies: List of frequencies to evaluate
            app_name: Application name for results

        Returns:
            Dictionary with optimization results and recommendations
        """
        logger.info(f"Starting optimization pipeline for {app_name}")

        # Step 1: Power prediction across frequencies
        power_df = self._predict_power_across_frequencies(
            fp_activity, dram_activity, frequencies
        )

        # Step 2: Runtime prediction or scaling
        result_df = self._predict_runtime(
            power_df, baseline_runtime, fp_activity, frequencies
        )

        # Step 3: Comprehensive optimization analysis
        optimization_results = self.optimizer.comprehensive_analysis(
            result_df, app_name, "predicted_n_to_r_energy", "predicted_n_to_r_run_time"
        )

        # Step 4: Generate recommendations
        recommendations = self.optimizer.generate_recommendations(optimization_results)

        final_results = {
            "optimization_results": optimization_results,
            "recommendations": recommendations,
            "frequency_sweep_data": result_df,
            "input_parameters": {
                "fp_activity": fp_activity,
                "dram_activity": dram_activity,
                "baseline_runtime": baseline_runtime,
                "frequencies_evaluated": frequencies,
            },
            "pipeline_metadata": {
                "has_runtime_model": self.runtime_model is not None,
                "power_model_type": type(self.power_model).__name__,
            },
        }

        logger.info(f"Optimization pipeline complete for {app_name}")
        return final_results

    def _predict_power_across_frequencies(
        self, fp_activity: float, dram_activity: float, frequencies: List[int]
    ) -> pd.DataFrame:
        """Predict power consumption across frequency range."""
        if hasattr(self.power_model, "predict_power"):
            # FGCS model with built-in frequency prediction
            return self.power_model.predict_power(
                fp_activity, dram_activity, frequencies
            )
        else:
            # Generic model - create features and predict
            features = pd.DataFrame(
                {
                    "fp_activity": [fp_activity] * len(frequencies),
                    "dram_activity": [dram_activity] * len(frequencies),
                    "sm_clock": frequencies,
                }
            )
            power_predictions = self.power_model.predict(features.values)
            return pd.DataFrame(
                {"sm_app_clock": frequencies, "predicted_power": power_predictions}
            )

    def _predict_runtime(
        self,
        power_df: pd.DataFrame,
        baseline_runtime: float,
        fp_activity: float,
        frequencies: List[int],
    ) -> pd.DataFrame:
        """Predict runtime across frequencies."""
        if self.runtime_model and hasattr(self.power_model, "predict_runtime"):
            return self.power_model.predict_runtime(
                power_df, baseline_runtime, fp_activity
            )
        else:
            # Simple frequency scaling assumption
            result_df = power_df.copy()
            max_freq = max(frequencies)
            result_df["predicted_n_to_r_run_time"] = baseline_runtime * (
                max_freq / result_df["sm_app_clock"]
            )
            result_df["predicted_n_to_r_power_usage"] = result_df.get(
                "predicted_power", result_df.get("predicted_n_to_r_power_usage", 0)
            )
            result_df["predicted_n_to_r_energy"] = (
                result_df["predicted_n_to_r_run_time"]
                * result_df["predicted_n_to_r_power_usage"]
            )
            return result_df
