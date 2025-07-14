#!/usr/bin/env python3
"""
GPU Power Profiling Utility for AI Inference Workloads.

This module provides power and performance profiling capabilities for AI inference
tasks using NVIDIA's DCGMI (Data Center GPU Manager Interface). It measures power
consumption, utilization, and other GPU metrics during application execution.

The profiler runs in parallel with the target application, collecting metrics
at regular intervals and saving them to CSV files for analysis.

Requirements:
    - NVIDIA GPU with DCGMI support
    - DCGMI tools installed and accessible
    - Appropriate permissions to run GPU monitoring tools

Author: Mert Side
"""

import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import profiling_config
    from utils import (
        get_timestamp,
        run_command,
        setup_logging,
        validate_dcgmi_available,
    )
except ImportError:
    # Fallback configuration if imports fail
    class ProfilingConfig:
        # DCGMI_FIELDS = [
        #     1001,
        #     1002,
        #     1003,
        #     1004,
        #     1005,
        #     1006,
        #     1007,
        #     1008,
        #     1009,
        #     1010,
        #     203,
        #     204,
        #     210,
        #     211,
        #     155,
        #     156,
        #     110,
        # ]
        DCGMI_FIELDS = [
            # "timestamp", # 0 ─ dcgmi dmon prints host timestamp automatically
            52,          # 1 ─ DCGM_FI_DEV_NVML_INDEX
            50,          # 2 ─ DCGM_FI_DEV_NAME
            155,         # 3 ─ DCGM_FI_DEV_POWER_USAGE
            160,         # 4 ─ DCGM_FI_DEV_POWER_MGMT_LIMIT
            150,         # 5 ─ DCGM_FI_DEV_GPU_TEMP
            203,         # 6 ─ DCGM_FI_DEV_GPU_UTIL          (coarse)
            204,         # 7 ─ DCGM_FI_DEV_MEM_COPY_UTIL     (≈ util.mem)
            250,         # 8 ─ DCGM_FI_DEV_FB_TOTAL
            251,         # 9 ─ DCGM_FI_DEV_FB_FREE
            252,         # 10 ─ DCGM_FI_DEV_FB_USED
            100,         # 11 ─ DCGM_FI_DEV_SM_CLOCK
            101,         # 12 ─ DCGM_FI_DEV_MEM_CLOCK
            100,         # 13 ─ (proxy for graphics clock)
            110,         # 14 ─ DCGM_FI_DEV_APP_SM_CLOCK
            111,         # 15 ─ DCGM_FI_DEV_APP_MEM_CLOCK
            190,         # 16 ─ DCGM_FI_DEV_PSTATE
            140,         # memory (HBM) temperature
            156,         # total energy consumption (mJ)
            1001,        # graphics active 
            1002,        # SM active
            1003,        # SM occupancy
            1004,        # tensor pipe active
            1005,        # DRAM active
            1006,        # FP64 active
            1007,        # FP32 active
            1008         # FP16 active
        ]
        # One-liner: dcgmi dmon -d 100 -e 52,50,155,160,150,203,204,250,251,252,100,101,100,110,111,190,140,156,1001,1002,1003,1004,1005,1006,1007,1008 -c 1
        DEFAULT_INTERVAL_MS = 50
        TEMP_OUTPUT_FILE = "changeme"

    profiling_config = ProfilingConfig()

    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)

    def validate_dcgmi_available():
        try:
            subprocess.run(
                ["dcgmi", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_timestamp():
        return time.strftime("%Y-%m-%d_%H-%M-%S")


class GPUProfiler:
    """
    GPU profiler for monitoring power and performance metrics during application execution.

    This class uses DCGMI to collect GPU metrics in parallel with the target application,
    providing detailed power and performance profiling capabilities.
    """

    def __init__(
        self,
        output_file: str = None,
        interval_ms: int = None,
        gpu_id: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the GPU profiler.

        Args:
            output_file: Output file for profiling data
            interval_ms: Sampling interval in milliseconds
            gpu_id: GPU device ID to monitor
            logger: Optional logger instance
        """
        self.output_file = output_file or profiling_config.TEMP_OUTPUT_FILE
        self.interval_ms = interval_ms or profiling_config.DEFAULT_INTERVAL_MS
        self.gpu_id = gpu_id
        self.logger = logger or setup_logging()
        self.monitoring_process = None
        self.start_time = None

        # Validate DCGMI availability
        if not validate_dcgmi_available():
            raise RuntimeError(
                "DCGMI not available. Please install NVIDIA DCGMI tools."
            )

    def _build_dcgmi_command(self) -> List[str]:
        """
        Build the DCGMI monitoring command.

        Returns:
            List of command arguments for DCGMI
        """
        fields_str = ",".join(str(field) for field in profiling_config.DCGMI_FIELDS)

        command = [
            "dcgmi",
            "dmon",
            "-i",
            str(self.gpu_id),
            "-e",
            fields_str,
            "-d",
            str(self.interval_ms),
        ]

        return command

    def start_monitoring(self) -> None:
        """Start GPU monitoring in a separate process."""
        try:
            command = self._build_dcgmi_command()
            self.logger.info(
                f"Starting GPU monitoring with command: {' '.join(command)}"
            )
            self.logger.info(f"Output file: {self.output_file}")
            self.logger.info(f"Sampling interval: {self.interval_ms}ms")

            # Start monitoring process with output redirection
            with open(self.output_file, "w") as output:
                self.monitoring_process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,  # Create new process group
                )

            self.start_time = time.time()
            self.logger.info(
                f"GPU monitoring started (PID: {self.monitoring_process.pid})"
            )

        except Exception as e:
            self.logger.error(f"Failed to start GPU monitoring: {e}")
            raise

    def stop_monitoring(self) -> float:
        """
        Stop GPU monitoring and return the monitoring duration.

        Returns:
            Monitoring duration in seconds
        """
        if self.monitoring_process is None:
            self.logger.warning("No monitoring process to stop")
            return 0.0

        try:
            # Terminate the monitoring process group
            os.killpg(os.getpgid(self.monitoring_process.pid), signal.SIGTERM)

            # Wait for process to terminate
            self.monitoring_process.wait(timeout=5)

            duration = time.time() - self.start_time if self.start_time else 0.0
            self.logger.info(f"GPU monitoring stopped. Duration: {duration:.2f}s")

            return duration

        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Monitoring process did not terminate gracefully, forcing kill"
            )
            os.killpg(os.getpgid(self.monitoring_process.pid), signal.SIGKILL)
            return time.time() - self.start_time if self.start_time else 0.0
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {e}")
            return 0.0
        finally:
            self.monitoring_process = None
            self.start_time = None

    def profile_command(self, command) -> Dict[str, Any]:
        """
        Profile a command execution with GPU monitoring.

        Args:
            command: Command string or list of arguments to execute and profile

        Returns:
            Dictionary containing profiling results
        """
        # Handle both string and list inputs
        if isinstance(command, str):
            if not command.strip():
                self.logger.warning("Empty command provided, only monitoring GPU")
                time.sleep(2)  # Monitor for 2 seconds
                return {"duration": 2.0, "command": "", "exit_code": 0}
            command_display = command
            use_shell = True
        else:
            # List of arguments
            if not command or not command[0]:
                self.logger.warning("Empty command provided, only monitoring GPU")
                time.sleep(2)  # Monitor for 2 seconds
                return {"duration": 2.0, "command": "", "exit_code": 0}
            command_display = " ".join(command)
            use_shell = False

        self.logger.info(f"Profiling command: {command_display}")

        # Start GPU monitoring
        self.start_monitoring()

        try:
            # Execute the target command
            app_start_time = time.time()

            result = subprocess.run(
                command,
                shell=use_shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # Python 3.6 compatible (text=True in 3.7+)
            )

            app_duration = time.time() - app_start_time

            # Stop monitoring
            monitoring_duration = self.stop_monitoring()

            # Log results
            self.logger.info(f"Command completed in {app_duration:.2f}s")
            self.logger.info(f"Exit code: {result.returncode}")

            if result.stdout:
                self.logger.info(f"Command stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Command stderr: {result.stderr}")

            return {
                "duration": app_duration,
                "monitoring_duration": monitoring_duration,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            self.stop_monitoring()
            raise

    def cleanup(self) -> None:
        """Clean up any remaining monitoring processes."""
        if self.monitoring_process and self.monitoring_process.poll() is None:
            self.stop_monitoring()


def profile_application(
    command, output_file: str = None, interval_ms: int = None, gpu_id: int = 0
) -> Dict[str, Any]:
    """
    Convenience function to profile an application with GPU monitoring.

    Args:
        command: Command to execute and profile (string or list of arguments)
        output_file: Output file for profiling data
        interval_ms: Sampling interval in milliseconds
        gpu_id: GPU device ID to monitor

    Returns:
        Dictionary containing profiling results
    """
    profiler = GPUProfiler(
        output_file=output_file, interval_ms=interval_ms, gpu_id=gpu_id
    )

    try:
        return profiler.profile_command(command)
    finally:
        profiler.cleanup()


def main():
    """Main function for standalone execution."""
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="GPU power profiling utility for AI inference workloads"
    )
    parser.add_argument(
        "command", nargs="*", help="Command to profile (if empty, just monitors GPU)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=profiling_config.TEMP_OUTPUT_FILE,
        help=f"Output file for profiling data (default: {profiling_config.TEMP_OUTPUT_FILE})",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=profiling_config.DEFAULT_INTERVAL_MS,
        help=f"Sampling interval in milliseconds (default: {profiling_config.DEFAULT_INTERVAL_MS})",
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="GPU device ID to monitor (default: 0)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)

    # Join command arguments
    command = " ".join(args.command) if args.command else ""

    try:
        # Profile the command
        result = profile_application(
            command=command,
            output_file=args.output,
            interval_ms=args.interval,
            gpu_id=args.gpu,
        )

        # Display results
        logger.info("Profiling completed successfully")
        logger.info(f"Results saved to: {args.output}")

        if command:
            logger.info(f"Application duration: {result['duration']:.2f}s")
            logger.info(f"Exit code: {result['exit_code']}")

    except KeyboardInterrupt:
        logger.info("Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
