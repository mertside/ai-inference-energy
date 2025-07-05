#!/usr/bin/env python3
"""
GPU Power Profiling Utility using nvidia-smi (Alternative to DCGMI).

This module provides power and performance profiling capabilities for AI inference
tasks using NVIDIA's nvidia-smi tool as an alternative to DCGMI. It measures power
consumption, utilization, and other GPU metrics during application execution.

The profiler runs in parallel with the target application, collecting metrics
at regular intervals and saving them to CSV files for analysis.

Requirements:
    - NVIDIA GPU with nvidia-smi support
    - nvidia-smi tool (part of NVIDIA drivers)
    - Python 3.6+

Note:
    This is an alternative to the DCGMI-based profile.py script.
    nvidia-smi may have different metrics and sampling capabilities compared to DCGMI.

Author: Mert Side
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import profiling_config
    from utils import get_timestamp, setup_logging, validate_gpu_available
except ImportError:
    # Fallback configuration if imports fail
    class ProfilingConfig:
        DEFAULT_INTERVAL_MS = 50
        TEMP_OUTPUT_FILE = "changeme"

    profiling_config = ProfilingConfig()

    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)

    def validate_gpu_available():
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_timestamp():
        return time.strftime("%Y-%m-%d_%H-%M-%S")


class NvidiaSmiProfiler:
    """
    GPU profiler for monitoring power and performance metrics using nvidia-smi.

    This class uses nvidia-smi to collect GPU metrics in parallel with the target
    application, providing power and performance profiling capabilities as an
    alternative to DCGMI.
    """

    def __init__(
        self, output_file: str = None, interval_ms: int = None, gpu_id: int = 0, logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the nvidia-smi GPU profiler.

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
        self.stop_monitoring_flag = threading.Event()

        # Validate nvidia-smi availability
        if not validate_gpu_available():
            raise RuntimeError("nvidia-smi not available. Please install NVIDIA drivers.")

    def _build_nvidia_smi_command(self) -> List[str]:
        """
        Build the nvidia-smi monitoring command.

        Returns:
            List of command arguments for nvidia-smi
        """
        # nvidia-smi query fields for comprehensive monitoring
        query_fields = [
            "timestamp",
            "index",
            "name",
            "power.draw",
            "power.limit",
            "temperature.gpu",
            "utilization.gpu",
            "utilization.memory",
            "memory.total",
            "memory.free",
            "memory.used",
            "clocks.sm",
            "clocks.mem",
            "clocks.gr",
            "pstate",
        ]

        # Convert interval from ms to seconds for nvidia-smi
        interval_sec = max(1, self.interval_ms // 1000)  # nvidia-smi uses seconds, minimum 1s

        command = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(query_fields),
            "--format=csv,noheader,nounits",
            f"--loop={interval_sec}",
            "-i",
            str(self.gpu_id),
        ]

        return command

    def _monitoring_worker(self) -> None:
        """Worker function to run nvidia-smi monitoring in a separate thread."""
        try:
            command = self._build_nvidia_smi_command()
            self.logger.debug(f"nvidia-smi command: {' '.join(command)}")

            with open(self.output_file, "w") as output:
                # Write CSV header
                header_fields = [
                    "timestamp",
                    "index",
                    "name",
                    "power.draw",
                    "power.limit",
                    "temperature.gpu",
                    "utilization.gpu",
                    "utilization.memory",
                    "memory.total",
                    "memory.free",
                    "memory.used",
                    "clocks.sm",
                    "clocks.mem",
                    "clocks.gr",
                    "pstate",
                ]
                output.write(",".join(header_fields) + "\n")
                output.flush()

                # Start nvidia-smi process
                self.monitoring_process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,  # Python 3.6 compatible (text=True in 3.7+)
                )

                # Wait for stop signal or process completion
                while not self.stop_monitoring_flag.is_set():
                    if self.monitoring_process.poll() is not None:
                        break
                    time.sleep(0.1)

                # Terminate the process if still running
                if self.monitoring_process.poll() is None:
                    self.monitoring_process.terminate()
                    try:
                        self.monitoring_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.monitoring_process.kill()

        except Exception as e:
            self.logger.error(f"Monitoring worker error: {e}")

    def start_monitoring(self) -> None:
        """Start GPU monitoring in a separate thread."""
        try:
            self.logger.info(f"Starting nvidia-smi GPU monitoring")
            self.logger.info(f"Output file: {self.output_file}")
            self.logger.info(f"Sampling interval: {self.interval_ms}ms (nvidia-smi uses 1s minimum)")
            self.logger.info(f"GPU ID: {self.gpu_id}")

            # Reset stop flag
            self.stop_monitoring_flag.clear()

            # Start monitoring in a separate thread
            monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            monitoring_thread.start()

            # Give the thread time to start
            time.sleep(1)

            self.start_time = time.time()
            self.logger.info("nvidia-smi GPU monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to start nvidia-smi GPU monitoring: {e}")
            raise

    def stop_monitoring(self) -> float:
        """
        Stop GPU monitoring and return the monitoring duration.

        Returns:
            Monitoring duration in seconds
        """
        try:
            # Signal the monitoring thread to stop
            self.stop_monitoring_flag.set()

            # Wait a moment for graceful shutdown
            time.sleep(0.5)

            # Force terminate if process is still running
            if self.monitoring_process and self.monitoring_process.poll() is None:
                self.monitoring_process.terminate()
                try:
                    self.monitoring_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.monitoring_process.kill()
                    self.logger.warning("Had to force kill nvidia-smi process")

            duration = time.time() - self.start_time if self.start_time else 0.0
            self.logger.info(f"nvidia-smi GPU monitoring stopped. Duration: {duration:.2f}s")

            return duration

        except Exception as e:
            self.logger.error(f"Error stopping nvidia-smi monitoring: {e}")
            return 0.0
        finally:
            self.monitoring_process = None
            self.start_time = None

    def profile_command(self, command: str) -> Dict[str, Any]:
        """
        Profile a command execution with nvidia-smi GPU monitoring.

        Args:
            command: Command string to execute and profile

        Returns:
            Dictionary containing profiling results
        """
        if not command.strip():
            self.logger.warning("Empty command provided, only monitoring GPU")
            time.sleep(2)  # Monitor for 2 seconds
            return {"duration": 2.0, "command": "", "exit_code": 0}

        self.logger.info(f"Profiling command with nvidia-smi: {command}")

        # Start GPU monitoring
        self.start_monitoring()

        try:
            # Execute the target command
            app_start_time = time.time()

            result = subprocess.run(
                command,
                shell=True,
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
                self.logger.debug(f"Command stdout: {result.stdout}")
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


def profile_application(command: str, output_file: str = None, interval_ms: int = None, gpu_id: int = 0) -> Dict[str, Any]:
    """
    Convenience function to profile an application with nvidia-smi GPU monitoring.

    Args:
        command: Command to execute and profile
        output_file: Output file for profiling data
        interval_ms: Sampling interval in milliseconds
        gpu_id: GPU device ID to monitor

    Returns:
        Dictionary containing profiling results
    """
    profiler = NvidiaSmiProfiler(output_file=output_file, interval_ms=interval_ms, gpu_id=gpu_id)

    try:
        return profiler.profile_command(command)
    finally:
        profiler.cleanup()


def main():
    """Main function for standalone execution."""
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="GPU power profiling utility using nvidia-smi for AI inference workloads")
    parser.add_argument("command", nargs="*", help="Command to profile (if empty, just monitors GPU)")
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
        help=f"Sampling interval in milliseconds (default: {profiling_config.DEFAULT_INTERVAL_MS}, nvidia-smi uses 1s minimum)",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device ID to monitor (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)

    # Join command arguments
    command = " ".join(args.command) if args.command else ""

    try:
        # Profile the command
        result = profile_application(command=command, output_file=args.output, interval_ms=args.interval, gpu_id=args.gpu)

        # Display results
        logger.info("Profiling completed successfully using nvidia-smi")
        logger.info(f"Results saved to: {args.output}")

        if command:
            logger.info(f"Application duration: {result['duration']:.2f}s")
            logger.info(f"Exit code: {result['exit_code']}")

    except KeyboardInterrupt:
        logger.info("Profiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"nvidia-smi profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
