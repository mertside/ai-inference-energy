#!/usr/bin/env python3

"""
EDP & ED²P Results Data Visualization Script

Creates scatter plots using actual experimental data from sample collection scripts
with EDP and ED²P optimization points clearly annotated.

Author: Mert Side
Version: 2.0 - Enhanced with experimental data integration
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style for clean plots
plt.style.use("default")


def detect_outliers(data: List[float], method: str = "iqr", threshold: float = 2.0) -> List[bool]:
    """Detect outliers in data using specified method

    Args:
        data: List of numerical values
        method: "iqr" for Interquartile Range or "zscore" for Z-score
        threshold: Threshold for outlier detection (IQR multiplier or Z-score threshold)

    Returns:
        List of booleans indicating which values are outliers (True = outlier)
    """
    if len(data) < 3:
        return [False] * len(data)  # Can't detect outliers with too few points

    data_array = np.array(data)

    if method == "iqr":
        # Interquartile Range method
        Q1 = np.percentile(data_array, 25)
        Q3 = np.percentile(data_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (data_array < lower_bound) | (data_array > upper_bound)

    elif method == "zscore":
        # Z-score method
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:  # All values are the same
            return [False] * len(data)
        z_scores = np.abs((data_array - mean) / std)
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


class DataVisualizer:
    """Visualizer for EDP/ED²P optimization results using experimental data"""

    def __init__(self, results_file: str, sample_scripts_dir: str = None, output_dir: str = "edp-plots"):
        """Initialize data visualizer

        Args:
            results_file: JSON file with EDP/ED²P optimization results
            sample_scripts_dir: Path to sample-collection-scripts directory
            output_dir: Output directory for plots
        """
        self.results_file = Path(results_file)
        self.sample_scripts_dir = (
            Path(sample_scripts_dir)
            if sample_scripts_dir
            else Path("../../../sample-collection-scripts")
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results data
        with open(self.results_file, "r") as f:
            self.results = json.load(f)

        # Color scheme
        self.colors = {
            "frequency_points": "#95a5a6",
            "max_freq": "#cc0000",  # '#e74c3c',
            "edp_optimal": "#00cc00",  # '#2ecc71',
            "ed2p_optimal": "#0000cc",  # '#3498db'
        }

    def find_result_directory(self, gpu: str, workload: str) -> Optional[Path]:
        """Find the experimental results directory for a GPU-workload combination"""
        # Define base directory for experimental results
        base_dir = Path("../../../sample-collection-scripts")

        if not base_dir.exists():
            print(f"Warning: Sample collection scripts directory not found: {base_dir}")
            return None

        # Build search pattern for result directories
        gpu_lower = gpu.lower()
        workload_lower = workload.lower()
        pattern = f"results_{gpu_lower}_{workload_lower}_job_*"

        print(f"  Searched for pattern: {pattern}")
        print(f"  In directory: {base_dir}")

        # Search for matching directories
        matching_dirs = list(base_dir.glob(pattern))

        if not matching_dirs:
            print(f"Warning: No results directory found for {gpu}-{workload}")
            return None

        # Return the first (and hopefully only) matching directory
        result_dir = matching_dirs[0]
        print(f"  Found: {result_dir.name}")
        return result_dir

    def load_timing_data(self, result_dir: Path) -> Dict[int, float]:
        """Load timing data from timing_summary.log file, averaging warm runs (excluding first run)

        Returns:
            Dict mapping frequency (MHz) to averaged execution time (seconds)
        """
        timing_data = {}

        # Load from timing_summary.log
        timing_file = result_dir / "timing_summary.log"
        if timing_file.exists():
            try:
                # Collect all runs by frequency
                frequency_runs = {}

                with open(timing_file, "r") as f:
                    for line in f:
                        # Skip comments and empty lines
                        if line.startswith("#") or not line.strip():
                            continue

                        # Parse format: run_id,frequency_mhz,duration_seconds,exit_code,status
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            try:
                                run_id = parts[0]
                                frequency = int(parts[1])
                                duration = float(parts[2])
                                exit_code = int(parts[3])

                                # Only include successful runs
                                if exit_code == 0:
                                    if frequency not in frequency_runs:
                                        frequency_runs[frequency] = []
                                    frequency_runs[frequency].append((run_id, duration))
                            except (ValueError, IndexError):
                                continue

                # Average runs for each frequency, excluding the first (cold) run and outliers
                for frequency, runs in frequency_runs.items():
                    if len(runs) > 1:
                        # Sort by run_id to ensure consistent ordering
                        runs.sort(key=lambda x: x[0])
                        # Exclude first run (cold run) and get warm runs
                        warm_runs = [duration for _, duration in runs[1:]]

                        if len(warm_runs) > 2:  # Need at least 3 points for outlier detection
                            # Detect outliers using IQR method with threshold of 1.5
                            outlier_mask = detect_outliers(warm_runs, method="iqr", threshold=1.5)

                            # Filter out outliers
                            filtered_runs = [duration for i, duration in enumerate(warm_runs) if not outlier_mask[i]]

                            # Report outliers if any were found
                            outliers = [duration for i, duration in enumerate(warm_runs) if outlier_mask[i]]
                            if outliers:
                                print(f"    Frequency {frequency} MHz: excluded {len(outliers)} outlier(s) {outliers}")

                            # Use filtered data if we still have at least 2 points, otherwise use all warm runs
                            final_runs = filtered_runs if len(filtered_runs) >= 2 else warm_runs
                        else:
                            final_runs = warm_runs

                        if final_runs:
                            avg_duration = sum(final_runs) / len(final_runs)
                            timing_data[frequency] = avg_duration
                            print(f"    Frequency {frequency} MHz: averaged {len(final_runs)} warm runs (excluded cold run)")
                    elif len(runs) == 1:
                        # If only one run, use it (better than no data)
                        timing_data[frequency] = runs[0][1]
                        print(f"    Frequency {frequency} MHz: only 1 run available (no warm runs to average)")

            except Exception as e:
                print(f"    Error reading timing file: {e}")

        return timing_data

    def load_power_data(self, result_dir: Path) -> Dict[int, Tuple[float, float]]:
        """Load power data from DCGMI profile CSV files, averaging warm runs (excluding first run)

        Returns:
            Dict mapping frequency (MHz) to (average_power, total_energy) tuple
        """
        power_data = {}

        # Look for profile CSV files
        profile_files = list(result_dir.glob("run_*_profile.csv"))

        if not profile_files:
            print(f"    Warning: No profile CSV files found in {result_dir}")
            return power_data

        print(f"    Found {len(profile_files)} profile files")

        # Group files by frequency
        frequency_files = {}
        for profile_file in profile_files:
            try:
                # Extract frequency and run number from filename
                # Format: run_{run_id}_{run_number}_freq_{frequency}_profile.csv
                freq_match = re.search(r"freq_(\d+)", profile_file.name)
                run_match = re.search(r"run_\d+_(\d+)_freq_", profile_file.name)

                if not freq_match or not run_match:
                    continue

                frequency = int(freq_match.group(1))
                run_number = int(run_match.group(1))

                if frequency not in frequency_files:
                    frequency_files[frequency] = []
                frequency_files[frequency].append((run_number, profile_file))

            except (ValueError, IndexError):
                continue

        # Process each frequency
        for frequency, files in frequency_files.items():
            try:
                # Sort by run number to ensure consistent ordering
                files.sort(key=lambda x: x[0])

                # Exclude first run (run number 01) and process warm runs
                warm_run_files = [f for run_num, f in files if run_num > 1]

                if not warm_run_files:
                    # If no warm runs, use all available runs
                    warm_run_files = [f for _, f in files]
                    print(f"    Frequency {frequency} MHz: no warm runs found, using all {len(warm_run_files)} runs")
                else:
                    print(f"    Frequency {frequency} MHz: averaging {len(warm_run_files)} warm runs (excluded cold run)")

                # Process each warm run file and collect power data
                all_power_values = []
                all_energies = []

                for profile_file in warm_run_files:
                    power_values = self._extract_power_from_file(profile_file)

                    if power_values is not None and len(power_values) > 0:
                        # Calculate energy for this run
                        sampling_rate = 0.05  # 50ms sampling rate
                        avg_power = power_values.mean()
                        execution_time = len(power_values) * sampling_rate
                        total_energy = avg_power * execution_time

                        all_power_values.append(avg_power)
                        all_energies.append(total_energy)

                # Average across warm runs, excluding outliers
                if all_power_values and len(all_power_values) > 2:
                    # Detect outliers in both power and energy measurements
                    power_outliers = detect_outliers(all_power_values, method="iqr", threshold=1.5)
                    energy_outliers = detect_outliers(all_energies, method="iqr", threshold=1.5)

                    # Combine outlier masks (exclude if outlier in either power or energy)
                    combined_outliers = [p_out or e_out for p_out, e_out in zip(power_outliers, energy_outliers)]

                    # Filter out outliers
                    filtered_power = [power for i, power in enumerate(all_power_values) if not combined_outliers[i]]
                    filtered_energy = [energy for i, energy in enumerate(all_energies) if not combined_outliers[i]]

                    # Report outliers if any were found
                    outlier_indices = [i for i, is_outlier in enumerate(combined_outliers) if is_outlier]
                    if outlier_indices:
                        outlier_powers = [all_power_values[i] for i in outlier_indices]
                        outlier_energies = [all_energies[i] for i in outlier_indices]
                        print(f"    Frequency {frequency} MHz: excluded {len(outlier_indices)} power/energy outlier(s)")
                        print(f"      Power outliers: {[f'{p:.1f}W' for p in outlier_powers]}")
                        print(f"      Energy outliers: {[f'{e:.1f}J' for e in outlier_energies]}")

                    # Use filtered data if we still have at least 2 points, otherwise use all data
                    if len(filtered_power) >= 2:
                        avg_power = sum(filtered_power) / len(filtered_power)
                        avg_energy = sum(filtered_energy) / len(filtered_energy)
                    else:
                        avg_power = sum(all_power_values) / len(all_power_values)
                        avg_energy = sum(all_energies) / len(all_energies)

                    power_data[frequency] = (avg_power, avg_energy)

                elif all_power_values:
                    # Not enough data for outlier detection, use all values
                    avg_power = sum(all_power_values) / len(all_power_values)
                    avg_energy = sum(all_energies) / len(all_energies)
                    power_data[frequency] = (avg_power, avg_energy)
                else:
                    print(f"    Warning: No valid power data found for frequency {frequency} MHz")

            except Exception as e:
                print(f"    Warning: Error processing frequency {frequency} MHz: {e}")
                continue

        return power_data

    def _extract_power_from_file(self, profile_file: Path) -> Optional[pd.Series]:
        """Extract power values from a single DCGMI profile CSV file"""
        try:
            # Load CSV data - handle DCGMI format with fixed-width columns
            try:
                # First, read the file and identify the structure
                with open(profile_file, "r") as f:
                    lines = f.readlines()

                # Find the header line (starts with #Entity) and parse column positions
                header_line = None
                power_col_start = None
                power_col_end = None
                data_start_idx = 0

                for i, line in enumerate(lines):
                    # Handle both "#Entity" and "# Entity" formats
                    if line.startswith("#") and "Entity" in line:
                        header_line = line.strip()
                        # Remove the # symbol and any leading spaces
                        if header_line.startswith("# "):
                            header_line = header_line[2:].strip()
                        elif header_line.startswith("#"):
                            header_line = header_line[1:].strip()

                        data_start_idx = i + 2  # Skip header and units line

                        # Find POWER column position
                        power_start = line.find("POWER")
                        if power_start > 0:
                            # For "# Entity" format, we need to account for the extra space
                            if line.startswith("# "):
                                power_col_start = power_start - 2  # Account for removed "# "
                            else:
                                power_col_start = power_start - 1  # Account for removed "#"

                            # Find the end of the POWER column by looking for the next column
                            remaining = line[power_start + 5 :]  # After "POWER"
                            next_col_match = re.search(r"[A-Z]", remaining)
                            if next_col_match:
                                power_col_end = power_start + 5 + next_col_match.start() - 1
                            else:
                                power_col_end = power_start + 20  # Default width
                        break

                if header_line is not None and power_col_start is not None:
                    # Extract power values using fixed-width positions
                    power_values = []
                    for line in lines[data_start_idx:]:
                        if len(line) > power_col_start:
                            power_str = line[power_col_start:power_col_end].strip()
                            try:
                                power_val = float(power_str)
                                # Check if it's in reasonable power range (10-500W)
                                if 10 <= power_val <= 500:
                                    power_values.append(power_val)
                            except ValueError:
                                continue

                    if len(power_values) > 0:
                        return pd.Series(power_values)
                    else:
                        return None
                else:
                    # Fallback to pandas parsing
                    df = pd.read_csv(profile_file, sep=r"\s+", comment="#", on_bad_lines="skip")
                    if "POWER" in df.columns:
                        return pd.to_numeric(df["POWER"], errors="coerce").dropna()
                    else:
                        return None
            except:
                try:
                    # Final fallback - try comma-delimited
                    df = pd.read_csv(profile_file, comment="#", on_bad_lines="skip")
                    if "POWER" in df.columns:
                        return pd.to_numeric(df["POWER"], errors="coerce").dropna()
                    else:
                        return None
                except:
                    return None

        except Exception as e:
            print(f"    Warning: Error processing {profile_file.name}: {e}")
            return None

    def load_experimental_data(self, gpu: str, workload: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load experimental data for a GPU-workload combination

        Returns:
            Tuple of (frequencies, execution_times, energies) or None if no data found
        """
        # Find experimental data directory
        result_dir = self.find_result_directory(gpu, workload)
        if not result_dir:
            return None

        print(f"    Loading data from: {result_dir.name}")

        # Load timing and power data
        timing_data = self.load_timing_data(result_dir)
        power_data = self.load_power_data(result_dir)

        if not timing_data and not power_data:
            print(f"    Warning: No experimental data found for {gpu}-{workload}")
            return None

        # Combine timing and energy data
        frequencies = []
        execution_times = []
        energies = []

        # Use frequencies from both timing and power data
        all_frequencies = set(timing_data.keys()) | set(power_data.keys())

        for freq in sorted(all_frequencies):
            # Get execution time (prefer timing_data over power_data timing)
            if freq in timing_data:
                exec_time = timing_data[freq]
            elif freq in power_data:
                # Estimate from power sampling count
                avg_power, total_energy = power_data[freq]
                exec_time = total_energy / avg_power if avg_power > 0 else 1.0
            else:
                continue

            # Get energy data
            if freq in power_data:
                avg_power, total_energy = power_data[freq]
                energy = total_energy
            else:
                # Estimate energy if only timing available
                # Use a frequency-dependent power model
                estimated_power = 100 + (freq - 500) * 0.2  # Rough linear model
                energy = estimated_power * exec_time

            frequencies.append(freq)
            execution_times.append(exec_time)
            energies.append(energy)

        if len(frequencies) < 3:
            print(f"    Warning: Insufficient data points ({len(frequencies)}) for {gpu}-{workload}")
            return None

        print(f"    Loaded {len(frequencies)} data points: {sorted(frequencies)}")

        return np.array(frequencies), np.array(execution_times), np.array(energies)

    def create_scatter_plot(self, gpu: str, workload: str):
        """Create scatter plot for a specific GPU-workload combination"""

        # Find configuration data
        config = None
        for result in self.results:
            if result["gpu"] == gpu and result["workload"] == workload:
                config = result
                break

        if not config:
            print(f"Warning: Configuration {gpu}-{workload} not found")
            return None

        # Try to load experimental data first
        real_data = self.load_experimental_data(gpu, workload)

        if real_data is not None:
            frequencies, timings, energies = real_data
            data_source = "Experimental Data"
            print(f"    Using experimental data with {len(frequencies)} frequency points")
        else:
            # Fallback to synthetic data based on optimization results
            frequencies, timings, energies = self.generate_synthetic_fallback(config)
            data_source = "Synthetic Data (Real data not available)"
            print(f"    Using synthetic fallback data with {len(frequencies)} points")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot all frequency points as scatter
        scatter = ax.scatter(
            timings,
            energies,
            c=frequencies,
            cmap="plasma",
            s=80,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            label="Frequency Points",
        )

        # Add colorbar for frequency
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("GPU Frequency (MHz)", rotation=270, labelpad=20, fontsize=16)

        # Find and mark special points
        max_freq_idx = np.argmax(frequencies)

        # Find the points closest to our optimal frequencies
        edp_optimal_freq = config["optimal_frequency_edp_mhz"]
        ed2p_optimal_freq = config["optimal_frequency_ed2p_mhz"]

        edp_idx = np.argmin(np.abs(frequencies - edp_optimal_freq))
        ed2p_idx = np.argmin(np.abs(frequencies - ed2p_optimal_freq))

        # Mark special points
        ax.scatter(
            timings[max_freq_idx],
            energies[max_freq_idx],
            color=self.colors["max_freq"],
            s=300,
            marker="*",
            linewidth=3,
            label="Max Frequency",
            zorder=4,
        )

        ax.scatter(
            timings[edp_idx],
            energies[edp_idx],
            color=self.colors["edp_optimal"],
            s=300,
            marker="1",
            linewidth=3,
            label="EDP Optimal",
            zorder=5,
        )

        ax.scatter(
            timings[ed2p_idx],
            energies[ed2p_idx],
            color=self.colors["ed2p_optimal"],
            s=300,
            marker="2",
            linewidth=3,
            label="ED²P Optimal",
            zorder=6,
        )

        # Add annotations with detailed information
        # ax.annotate(f'Max Frequency',
        #            xy=(timings[max_freq_idx], energies[max_freq_idx]),
        #            xytext=(15, 15), textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['max_freq'], alpha=0.8),
        #            fontsize=10, ha='left', color='white', weight='bold',
        #            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # ax.annotate(f'EDP Optimal\\n',
        #            xy=(timings[edp_idx], energies[edp_idx]),
        #            xytext=(-20, -50), textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['edp_optimal'], alpha=0.8),
        #            fontsize=10, ha='center', color='white', weight='bold',
        #            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # ax.annotate(f'ED²P Optimal',
        #            xy=(timings[ed2p_idx], energies[ed2p_idx]),
        #            xytext=(20, -50), textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['ed2p_optimal'], alpha=0.8),
        #            fontsize=10, ha='center', color='white', weight='bold',
        #            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # ax.annotate(f'Max Frequency\\n{frequencies[max_freq_idx]:.0f} MHz\\n'
        #            f'Time: {timings[max_freq_idx]:.1f}s\\n'
        #            f'Energy: {energies[max_freq_idx]:.0f}J',
        #            xy=(timings[max_freq_idx], energies[max_freq_idx]),
        #            xytext=(15, 15), textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['max_freq'], alpha=0.8),
        #            fontsize=10, ha='left', color='white', weight='bold',
        #            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # ax.annotate(f'EDP Optimal\\n{frequencies[edp_idx]:.0f} MHz\\n'
        #            f'Energy Savings: {config["energy_savings_edp_percent"]:.1f}%\\n'
        #            f'Performance: {config["performance_vs_max_edp_percent"]:+.1f}%',
        #            xy=(timings[edp_idx], energies[edp_idx]),
        #            xytext=(-20, -50), textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['edp_optimal'], alpha=0.8),
        #            fontsize=10, ha='center', color='white', weight='bold',
        #            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # ax.annotate(f'ED²P Optimal\\n{frequencies[ed2p_idx]:.0f} MHz\\n'
        #            f'Energy Savings: {config["energy_savings_ed2p_percent"]:.1f}%\\n'
        #            f'Performance: {config["performance_vs_max_ed2p_percent"]:+.1f}%',
        #            xy=(timings[ed2p_idx], energies[ed2p_idx]),
        #            xytext=(20, -50), textcoords='offset points',
        #            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['ed2p_optimal'], alpha=0.8),
        #            fontsize=10, ha='center', color='white', weight='bold',
        #            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        # Customize plot appearance
        ax.set_xlabel("Execution Time (seconds)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Energy Consumption (Joules)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.set_title(
            f"{gpu} GPU - {workload.title()} Workload\n" f"Energy vs Performance Trade-off Analysis\n" f"({data_source})",
            fontsize=20,
            fontweight="bold",
            pad=20,
        )

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add legend - positioned outside and below the figure in one row
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=4, fontsize=18, edgecolor="black", fancybox=True)
        # ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)

        # Add summary statistics box
        edp_improvement = config["edp_improvement_percent"]
        ed2p_improvement = config["ed2p_improvement_percent"]

        stats_text = f"""Optimization Results Summary:

EDP Strategy:
• Frequency: {edp_optimal_freq} MHz ({(1-edp_optimal_freq/config['max_frequency_mhz'])*100:.1f}% reduction)
• Energy Savings: {config['energy_savings_edp_percent']:.1f}%
• EDP Improvement: {edp_improvement:.1f}%
• Performance Impact: {config['performance_vs_max_edp_percent']:+.1f}%

ED²P Strategy:
• Frequency: {ed2p_optimal_freq} MHz ({(1-ed2p_optimal_freq/config['max_frequency_mhz'])*100:.1f}% reduction)
• Energy Savings: {config['energy_savings_ed2p_percent']:.1f}%
• ED²P Improvement: {ed2p_improvement:.1f}%
• Performance Impact: {config['performance_vs_max_ed2p_percent']:+.1f}%

Data Source: {data_source}
Data Points: {len(frequencies)} frequency measurements"""

        # ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        #        verticalalignment='bottom', horizontalalignment='right',
        #        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, edgecolor='gray'),
        #        fontsize=10, family='monospace')

        # Set reasonable axis limits
        time_margin = (max(timings) - min(timings)) * 0.1
        energy_margin = (max(energies) - min(energies)) * 0.1
        ax.set_xlim(min(timings) - time_margin, max(timings) + time_margin)
        ax.set_ylim(min(energies) - energy_margin, max(energies) + energy_margin)

        plt.tight_layout()
        return fig

    def generate_synthetic_fallback(self, config):
        """Generate synthetic data when experimental data is not available"""
        max_freq = config["max_frequency_mhz"]
        min_freq = max(500, max_freq - 900)

        num_points = 25
        frequencies = np.linspace(min_freq, max_freq, num_points)

        edp_freq = config["optimal_frequency_edp_mhz"]
        edp_timing = config["optimal_timing_edp_seconds"]
        edp_energy = config["optimal_energy_edp_joules"]

        timings = []
        energies = []

        for freq in frequencies:
            timing_scale = np.sqrt(edp_freq / freq)
            timing = edp_timing * timing_scale

            if freq <= edp_freq:
                energy_scale = freq / edp_freq
                energy = edp_energy * (0.7 + 0.3 * energy_scale)
            else:
                energy_scale = (freq / edp_freq) ** 1.8
                energy = edp_energy * energy_scale

            timing *= 1 + np.random.normal(0, 0.03)
            energy *= 1 + np.random.normal(0, 0.03)

            timing = max(timing, 0.1)
            energy = max(energy, 1.0)

            timings.append(timing)
            energies.append(energy)

        return np.array(frequencies), np.array(timings), np.array(energies)

    def create_all_plots(self):
        """Create scatter plots for all GPU-workload combinations"""

        # Get unique combinations
        combinations = []
        for result in self.results:
            gpu = result["gpu"]
            workload = result["workload"]
            combinations.append((gpu, workload))

        print(f"🎨 Creating {len(combinations)} scatter plots...")

        for gpu, workload in combinations:
            print(f"  📊 Processing {gpu} - {workload}...")

            try:
                fig = self.create_scatter_plot(gpu, workload)
                if fig:
                    # Save plot
                    filename = f"{gpu}_{workload}_energy_performance_scatter.png"
                    filepath = self.output_dir / filename
                    fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
                    plt.close(fig)
                    print(f"    ✅ Saved: {filepath}")

            except Exception as e:
                print(f"    ❌ Error creating plot for {gpu}-{workload}: {e}")

        print(f"\n🎯 All plots saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate EDP/ED²P optimization visualizations")
    parser.add_argument(
        "--input",
        "-i",
        default="../results/edp_optimization_results.json",
        help="Input JSON file with optimization results",
    )
    parser.add_argument("--output-dir", "-o", default="edp-plots", help="Output directory for plot files")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Error: Input file {args.input} not found!")
        return 1

    # Create visualizer
    visualizer = DataVisualizer(args.input, args.output_dir)

    # Generate all visualizations
    visualizer.create_all_plots()

    return 0


if __name__ == "__main__":
    exit(main())
