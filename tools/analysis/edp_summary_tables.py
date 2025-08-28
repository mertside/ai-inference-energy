#!/usr/bin/env python3

"""
EDP & ED²P Results Summary Table Generator

This script reads the JSON output from edp_optimizer.py and generates
nicely formatted summary tables for easy interpretation and CSV export.
Supports both EDP and ED²P optimization results.
"""

import argparse
import csv
import json
from pathlib import Path


def create_summary_table(results_file: str, export_csv: bool = False):
    """Create formatted summary tables from EDP & ED²P optimization results"""

    with open(results_file, "r") as f:
        results = json.load(f)

    # Group by GPU
    gpu_groups = {}
    for result in results:
        gpu = result["gpu"]
        if gpu not in gpu_groups:
            gpu_groups[gpu] = []
        gpu_groups[gpu].append(result)

    print("📊 EDP & ED²P OPTIMIZATION RESULTS SUMMARY")
    print("=" * 90)

    # Overall summary
    total_results = len(results)
    avg_energy_savings_edp = sum(r["energy_savings_edp_percent"] for r in results) / total_results
    avg_energy_savings_ed2p = sum(r["energy_savings_ed2p_percent"] for r in results) / total_results
    avg_edp_improvement = sum(r["edp_improvement_percent"] for r in results) / total_results
    avg_ed2p_improvement = sum(r["ed2p_improvement_percent"] for r in results) / total_results
    faster_than_max_edp = sum(1 for r in results if r["performance_vs_max_edp_percent"] < 0)
    faster_than_max_ed2p = sum(1 for r in results if r["performance_vs_max_ed2p_percent"] < 0)

    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   • Total configurations: {total_results}")
    print(f"   📊 EDP Results:")
    print(f"     • Average energy savings: {avg_energy_savings_edp:.1f}%")
    print(f"     • Average EDP improvement: {avg_edp_improvement:.1f}%")
    print(f"     • Faster than max frequency: {faster_than_max_edp}/{total_results} ({100*faster_than_max_edp/total_results:.1f}%)")
    print(f"   📈 ED²P Results:")
    print(f"     • Average energy savings: {avg_energy_savings_ed2p:.1f}%")
    print(f"     • Average ED²P improvement: {avg_ed2p_improvement:.1f}%")
    print(f"     • Faster than max frequency: {faster_than_max_ed2p}/{total_results} ({100*faster_than_max_ed2p/total_results:.1f}%)")

    # Per-GPU detailed tables
    for gpu in sorted(gpu_groups.keys()):
        gpu_results = gpu_groups[gpu]

        print(f"\n🔧 {gpu} GPU DETAILED RESULTS:")
        print("-" * 90)
        print(f"{'Workload':<15} {'EDP':<25} {'ED²P':<25} {'Reference':<20}")
        print(f"{'':15} {'Freq Energy Perf':<25} {'Freq Energy Perf':<25} {'Max/Fast MHz':<20}")
        print("-" * 90)

        total_energy_edp = 0
        total_energy_ed2p = 0
        faster_count_edp = 0
        faster_count_ed2p = 0

        for result in sorted(gpu_results, key=lambda x: x["workload"]):
            workload = result["workload"]

            # EDP data
            optimal_freq_edp = result["optimal_frequency_edp_mhz"]
            energy_savings_edp = result["energy_savings_edp_percent"]
            perf_vs_max_edp = result["performance_vs_max_edp_percent"]

            # ED²P data
            optimal_freq_ed2p = result["optimal_frequency_ed2p_mhz"]
            energy_savings_ed2p = result["energy_savings_ed2p_percent"]
            perf_vs_max_ed2p = result["performance_vs_max_ed2p_percent"]

            # Reference data
            max_freq = result["max_frequency_mhz"]
            fastest_freq = result["fastest_frequency_mhz"]

            # Format performance changes
            perf_str_edp = f"{abs(perf_vs_max_edp):.1f}{'↑' if perf_vs_max_edp < 0 else '↓'}"
            perf_str_ed2p = f"{abs(perf_vs_max_ed2p):.1f}{'↑' if perf_vs_max_ed2p < 0 else '↓'}"

            edp_str = f"{optimal_freq_edp:4d} {energy_savings_edp:5.1f}% {perf_str_edp:>6}"
            ed2p_str = f"{optimal_freq_ed2p:4d} {energy_savings_ed2p:5.1f}% {perf_str_ed2p:>6}"
            ref_str = f"{max_freq}/{fastest_freq}"

            print(f"{workload:<15} {edp_str:<25} {ed2p_str:<25} {ref_str:<20}")

            total_energy_edp += energy_savings_edp
            total_energy_ed2p += energy_savings_ed2p
            if perf_vs_max_edp < 0:
                faster_count_edp += 1
            if perf_vs_max_ed2p < 0:
                faster_count_ed2p += 1

        avg_energy_edp = total_energy_edp / len(gpu_results)
        avg_energy_ed2p = total_energy_ed2p / len(gpu_results)
        print("-" * 90)
        print(f"{'GPU Average:':<15} {'':>25} {'':>25} {'':>20}")
        print(f"{'EDP:':<15} {avg_energy_edp:5.1f}% energy, {faster_count_edp}/{len(gpu_results)} faster")
        print(f"{'ED²P:':<15} {avg_energy_ed2p:5.1f}% energy, {faster_count_ed2p}/{len(gpu_results)} faster")

    # Cross-workload analysis for EDP
    print(f"\n🔬 CROSS-GPU WORKLOAD ANALYSIS (EDP):")
    print("-" * 80)
    print(f"{'Workload':<15} {'A100':<12} {'H100':<12} {'V100':<12} {'Best GPU':<10}")
    print(f"{'':15} {'Savings':<12} {'Savings':<12} {'Savings':<12} {'(Savings)':<10}")
    print("-" * 80)

    # Group by workload
    workload_groups = {}
    for result in results:
        workload = result["workload"]
        if workload not in workload_groups:
            workload_groups[workload] = {}
        workload_groups[workload][result["gpu"]] = result

    for workload in sorted(workload_groups.keys()):
        workload_data = workload_groups[workload]

        # Get EDP savings for each GPU
        a100_savings = workload_data.get("A100", {}).get("energy_savings_edp_percent", 0)
        h100_savings = workload_data.get("H100", {}).get("energy_savings_edp_percent", 0)
        v100_savings = workload_data.get("V100", {}).get("energy_savings_edp_percent", 0)

        # Find best GPU for EDP
        best_gpu_edp = max([("A100", a100_savings), ("H100", h100_savings), ("V100", v100_savings)], key=lambda x: x[1])

        print(f"{workload:<15} {a100_savings:<11.1f}% {h100_savings:<11.1f}% {v100_savings:<11.1f}% {best_gpu_edp[0]:<10}")

    # Cross-workload analysis for ED²P
    print(f"\n🔬 CROSS-GPU WORKLOAD ANALYSIS (ED²P):")
    print("-" * 80)
    print(f"{'Workload':<15} {'A100':<12} {'H100':<12} {'V100':<12} {'Best GPU':<10}")
    print(f"{'':15} {'Savings':<12} {'Savings':<12} {'Savings':<12} {'(Savings)':<10}")
    print("-" * 80)

    for workload in sorted(workload_groups.keys()):
        workload_data = workload_groups[workload]

        # Get ED²P savings for each GPU
        a100_savings = workload_data.get("A100", {}).get("energy_savings_ed2p_percent", 0)
        h100_savings = workload_data.get("H100", {}).get("energy_savings_ed2p_percent", 0)
        v100_savings = workload_data.get("V100", {}).get("energy_savings_ed2p_percent", 0)

        # Find best GPU for ED²P
        best_gpu_ed2p = max([("A100", a100_savings), ("H100", h100_savings), ("V100", v100_savings)], key=lambda x: x[1])

        print(f"{workload:<15} {a100_savings:<11.1f}% {h100_savings:<11.1f}% {v100_savings:<11.1f}% {best_gpu_ed2p[0]:<10}")

    # Frequency reduction analysis
    print(f"\n⚡ FREQUENCY REDUCTION ANALYSIS:")
    print("-" * 95)
    print(f"{'GPU-Workload':<20} {'Max':<6} {'EDP':<15} {'ED²P':<15} {'Fastest':<8}")
    print(f"{'':20} {'Freq':<6} {'Opt  Reduc%':<15} {'Opt  Reduc%':<15} {'Freq':<8}")
    print("-" * 95)

    for result in sorted(results, key=lambda x: (x["gpu"], x["workload"])):
        config = f"{result['gpu']} {result['workload']}"
        max_freq = result["max_frequency_mhz"]
        optimal_freq_edp = result["optimal_frequency_edp_mhz"]
        optimal_freq_ed2p = result["optimal_frequency_ed2p_mhz"]
        fastest_freq = result["fastest_frequency_mhz"]

        reduction_edp = max_freq - optimal_freq_edp
        reduction_pct_edp = (reduction_edp / max_freq) * 100
        reduction_ed2p = max_freq - optimal_freq_ed2p
        reduction_pct_ed2p = (reduction_ed2p / max_freq) * 100

        edp_str = f"{optimal_freq_edp:4d} {reduction_pct_edp:5.1f}%"
        ed2p_str = f"{optimal_freq_ed2p:4d} {reduction_pct_ed2p:5.1f}%"

        print(f"{config:<20} {max_freq:<6} {edp_str:<15} {ed2p_str:<15} {fastest_freq:<8}")

    # Export to CSV if requested
    if export_csv:
        export_to_csv(results, results_file)


def export_to_csv(results: list, input_file: str):
    """Export results to multiple CSV files for detailed analysis including both EDP and ED²P"""

    # Generate base filename from input file and ensure results directory exists
    base_name = Path(input_file).stem
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 1. Main results CSV with both EDP and ED²P data
    main_csv = results_dir / f"{base_name}_summary.csv"
    with open(main_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "GPU",
                "Workload",
                "Max_Freq_MHz",
                "Fastest_Freq_MHz",
                "EDP_Optimal_Freq_MHz",
                "EDP_Energy_Savings_Percent",
                "EDP_Performance_vs_Max_Percent",
                "EDP_Improvement_Percent",
                "ED2P_Optimal_Freq_MHz",
                "ED2P_Energy_Savings_Percent",
                "ED2P_Performance_vs_Max_Percent",
                "ED2P_Improvement_Percent",
                "Is_Max_Freq_Fastest",
                "Runs_Averaged",
            ]
        )

        # Data rows
        for result in sorted(results, key=lambda x: (x["gpu"], x["workload"])):
            writer.writerow(
                [
                    result["gpu"],
                    result["workload"],
                    result["max_frequency_mhz"],
                    result["fastest_frequency_mhz"],
                    result["optimal_frequency_edp_mhz"],
                    round(result["energy_savings_edp_percent"], 2),
                    round(result["performance_vs_max_edp_percent"], 2),
                    round(result["edp_improvement_percent"], 2),
                    result["optimal_frequency_ed2p_mhz"],
                    round(result["energy_savings_ed2p_percent"], 2),
                    round(result["performance_vs_max_ed2p_percent"], 2),
                    round(result["ed2p_improvement_percent"], 2),
                    result["is_max_frequency_fastest"],
                    result["runs_averaged"],
                ]
            )

    # 2. EDP Cross-workload comparison CSV
    workload_edp_csv = results_dir / f"{base_name}_workload_comparison_edp.csv"
    with open(workload_edp_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "Workload",
                "A100_EDP_Energy_Savings",
                "H100_EDP_Energy_Savings",
                "V100_EDP_Energy_Savings",
                "Best_GPU_EDP",
                "Best_EDP_Savings",
            ]
        )

        # Group by workload
        workload_groups = {}
        for result in results:
            workload = result["workload"]
            if workload not in workload_groups:
                workload_groups[workload] = {}
            workload_groups[workload][result["gpu"]] = result

        # Data rows
        for workload in sorted(workload_groups.keys()):
            workload_data = workload_groups[workload]

            a100_savings = workload_data.get("A100", {}).get("energy_savings_edp_percent", 0)
            h100_savings = workload_data.get("H100", {}).get("energy_savings_edp_percent", 0)
            v100_savings = workload_data.get("V100", {}).get("energy_savings_edp_percent", 0)

            # Find best GPU for EDP
            best_gpu_data = max([("A100", a100_savings), ("H100", h100_savings), ("V100", v100_savings)], key=lambda x: x[1])

            writer.writerow(
                [
                    workload,
                    round(a100_savings, 2),
                    round(h100_savings, 2),
                    round(v100_savings, 2),
                    best_gpu_data[0],
                    round(best_gpu_data[1], 2),
                ]
            )

    # 3. ED²P Cross-workload comparison CSV
    workload_ed2p_csv = results_dir / f"{base_name}_workload_comparison_ed2p.csv"
    with open(workload_ed2p_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "Workload",
                "A100_ED2P_Energy_Savings",
                "H100_ED2P_Energy_Savings",
                "V100_ED2P_Energy_Savings",
                "Best_GPU_ED2P",
                "Best_ED2P_Savings",
            ]
        )

        # Data rows
        for workload in sorted(workload_groups.keys()):
            workload_data = workload_groups[workload]

            a100_savings = workload_data.get("A100", {}).get("energy_savings_ed2p_percent", 0)
            h100_savings = workload_data.get("H100", {}).get("energy_savings_ed2p_percent", 0)
            v100_savings = workload_data.get("V100", {}).get("energy_savings_ed2p_percent", 0)

            # Find best GPU for ED²P
            best_gpu_data = max([("A100", a100_savings), ("H100", h100_savings), ("V100", v100_savings)], key=lambda x: x[1])

            writer.writerow(
                [
                    workload,
                    round(a100_savings, 2),
                    round(h100_savings, 2),
                    round(v100_savings, 2),
                    best_gpu_data[0],
                    round(best_gpu_data[1], 2),
                ]
            )

    # 4. Frequency reduction analysis CSV
    freq_csv = results_dir / f"{base_name}_frequency_analysis.csv"
    with open(freq_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            [
                "GPU",
                "Workload",
                "Max_Freq_MHz",
                "EDP_Optimal_Freq_MHz",
                "EDP_Reduction_MHz",
                "EDP_Reduction_Percent",
                "ED2P_Optimal_Freq_MHz",
                "ED2P_Reduction_MHz",
                "ED2P_Reduction_Percent",
                "Fastest_Freq_MHz",
            ]
        )

        # Data rows
        for result in sorted(results, key=lambda x: (x["gpu"], x["workload"])):
            max_freq = result["max_frequency_mhz"]
            edp_optimal_freq = result["optimal_frequency_edp_mhz"]
            ed2p_optimal_freq = result["optimal_frequency_ed2p_mhz"]
            fastest_freq = result["fastest_frequency_mhz"]

            edp_reduction = max_freq - edp_optimal_freq
            edp_reduction_pct = (edp_reduction / max_freq) * 100
            ed2p_reduction = max_freq - ed2p_optimal_freq
            ed2p_reduction_pct = (ed2p_reduction / max_freq) * 100

            writer.writerow(
                [
                    result["gpu"],
                    result["workload"],
                    max_freq,
                    edp_optimal_freq,
                    edp_reduction,
                    round(edp_reduction_pct, 2),
                    ed2p_optimal_freq,
                    ed2p_reduction,
                    round(ed2p_reduction_pct, 2),
                    fastest_freq,
                ]
            )

    # 5. GPU summary CSV for EDP
    gpu_edp_csv = results_dir / f"{base_name}_gpu_summary_edp.csv"
    with open(gpu_edp_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            ["GPU", "EDP_Avg_Energy_Savings", "EDP_Configurations_Faster_Than_Max", "Total_Configurations", "EDP_Faster_Percentage"]
        )

        # Group by GPU and calculate EDP averages
        gpu_groups = {}
        for result in results:
            gpu = result["gpu"]
            if gpu not in gpu_groups:
                gpu_groups[gpu] = []
            gpu_groups[gpu].append(result)

        # Data rows
        for gpu in sorted(gpu_groups.keys()):
            gpu_results = gpu_groups[gpu]
            avg_energy = sum(r["energy_savings_edp_percent"] for r in gpu_results) / len(gpu_results)
            faster_count = sum(1 for r in gpu_results if r["performance_vs_max_edp_percent"] < 0)
            total_configs = len(gpu_results)
            faster_pct = (faster_count / total_configs) * 100

            writer.writerow([gpu, round(avg_energy, 2), faster_count, total_configs, round(faster_pct, 1)])

    # 6. GPU summary CSV for ED²P
    gpu_ed2p_csv = results_dir / f"{base_name}_gpu_summary_ed2p.csv"
    with open(gpu_ed2p_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(
            ["GPU", "ED2P_Avg_Energy_Savings", "ED2P_Configurations_Faster_Than_Max", "Total_Configurations", "ED2P_Faster_Percentage"]
        )

        # Data rows
        for gpu in sorted(gpu_groups.keys()):
            gpu_results = gpu_groups[gpu]
            avg_energy = sum(r["energy_savings_ed2p_percent"] for r in gpu_results) / len(gpu_results)
            faster_count = sum(1 for r in gpu_results if r["performance_vs_max_ed2p_percent"] < 0)
            total_configs = len(gpu_results)
            faster_pct = (faster_count / total_configs) * 100

            writer.writerow([gpu, round(avg_energy, 2), faster_count, total_configs, round(faster_pct, 1)])

    print(f"\n📁 CSV FILES EXPORTED:")
    print(f"   • Main results: {main_csv}")
    print(f"   • EDP workload comparison: {workload_edp_csv}")
    print(f"   • ED²P workload comparison: {workload_ed2p_csv}")
    print(f"   • Frequency analysis: {freq_csv}")
    print(f"   • EDP GPU summary: {gpu_edp_csv}")
    print(f"   • ED²P GPU summary: {gpu_ed2p_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables from EDP optimization results")
    parser.add_argument("--input", "-i", default="results/edp_optimization_results.json", help="Input JSON file with EDP results")
    parser.add_argument("--csv", "-c", action="store_true", help="Export results to CSV files")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        return 1

    create_summary_table(args.input, export_csv=args.csv)
    return 0


if __name__ == "__main__":
    exit(main())
