#!/usr/bin/env python3
"""
Integration and System Test Suite

Tests the integration between different components of the power modeling framework
and system-level functionality. This includes file I/O, data processing pipelines,
and cross-component interactions.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataProcessingIntegration(unittest.TestCase):
    """Test cases for data processing and file I/O integration."""

    def setUp(self):
        """Set up test data for integration tests."""
        np.random.seed(42)
        self.sample_size = 50

        # Create sample profiling data
        self.sample_profiling_data = pd.DataFrame(
            {
                "app_name": ["TestApp"] * self.sample_size,
                "timestamp": pd.date_range(
                    "2024-01-01", periods=self.sample_size, freq="1s"
                ),
                "fp_activity": np.random.uniform(0.1, 0.8, self.sample_size),
                "dram_activity": np.random.uniform(0.05, 0.4, self.sample_size),
                "sm_clock": np.random.choice(range(800, 1401, 50), self.sample_size),
                "power": np.random.uniform(150, 300, self.sample_size),
                "temperature": np.random.uniform(40, 80, self.sample_size),
            }
        )

    def test_csv_file_processing(self):
        """Test CSV file reading and processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write test data to CSV
            test_file = os.path.join(temp_dir, "test_profiling.csv")
            self.sample_profiling_data.to_csv(test_file, index=False)

            # Test file exists and is readable
            self.assertTrue(os.path.exists(test_file))

            # Test data can be read back correctly
            loaded_data = pd.read_csv(test_file)
            self.assertEqual(len(loaded_data), len(self.sample_profiling_data))
            self.assertEqual(
                list(loaded_data.columns), list(self.sample_profiling_data.columns)
            )

            # Test data integrity
            pd.testing.assert_frame_equal(
                loaded_data.drop("timestamp", axis=1),
                self.sample_profiling_data.drop("timestamp", axis=1),
                check_dtype=False,  # Allow for minor type differences after CSV round-trip
            )

    def test_framework_file_integration(self):
        """Test framework integration with file input."""
        try:
            from power_modeling import analyze_application

            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test profiling file
                test_file = os.path.join(temp_dir, "integration_test.csv")
                self.sample_profiling_data.to_csv(test_file, index=False)

                # Test analyze_application with file input
                results = analyze_application(
                    profiling_file=test_file,
                    app_name="IntegrationTestApp",
                    gpu_type="V100",
                )

                # Verify basic structure
                self.assertIsInstance(results, dict)
                self.assertIn("summary", results)
                self.assertIn("optimization_results", results)

        except ImportError:
            self.skipTest("analyze_application function not available")

    def test_data_validation_pipeline(self):
        """Test data validation and preprocessing pipeline."""
        # Test with valid data
        valid_data = self.sample_profiling_data.copy()

        # Basic validation checks
        self.assertFalse(valid_data.empty)
        self.assertTrue(
            all(
                col in valid_data.columns
                for col in ["fp_activity", "dram_activity", "sm_clock", "power"]
            )
        )

        # Test data ranges are reasonable
        self.assertTrue(all(valid_data["fp_activity"].between(0, 1)))
        self.assertTrue(all(valid_data["dram_activity"].between(0, 1)))
        self.assertTrue(all(valid_data["sm_clock"] > 0))
        self.assertTrue(all(valid_data["power"] > 0))

        # Test with problematic data
        problematic_data = valid_data.copy()
        problematic_data.loc[0, "fp_activity"] = -0.1  # Invalid value
        problematic_data.loc[1, "power"] = np.nan  # Missing value

        # Count issues
        invalid_fp = (problematic_data["fp_activity"] < 0) | (
            problematic_data["fp_activity"] > 1
        )
        missing_power = problematic_data["power"].isna()

        self.assertTrue(any(invalid_fp))
        self.assertTrue(any(missing_power))


class TestModelIntegration(unittest.TestCase):
    """Test cases for model integration and pipeline functionality."""

    def setUp(self):
        """Set up models for integration testing."""
        self.test_data_size = 100
        np.random.seed(42)

        # Create test training data
        self.X_train = np.random.rand(self.test_data_size, 3)
        self.y_train = np.random.rand(self.test_data_size) * 200 + 50

        self.X_test = np.random.rand(20, 3)
        self.y_test = np.random.rand(20) * 200 + 50

    def test_model_pipeline_integration(self):
        """Test integration between different models in the pipeline."""
        try:
            from power_modeling.models.model_factory import ModelPipeline

            # Create and test model pipeline
            pipeline = ModelPipeline(model_types=["polynomial_deg2"])

            # Test training
            results = pipeline.train_models(
                self.X_train, self.y_train, self.X_test, self.y_test
            )

            # Verify pipeline results
            self.assertIn("models", results)
            self.assertIn("best_model", results)
            self.assertIn("evaluations", results)  # Changed from 'evaluation_results'

            # Test that models can make predictions
            best_model = results["best_model"]
            if best_model is not None:
                predictions = best_model.predict(self.X_test)
                self.assertEqual(len(predictions), len(self.X_test))

        except ImportError:
            self.skipTest("ModelPipeline not available")

    def test_framework_model_integration(self):
        """Test integration between framework and individual models."""
        try:
            from power_modeling import FGCSPowerModelingFramework

            # Create lightweight framework to avoid heavy model training
            framework = FGCSPowerModelingFramework(
                model_types=["polynomial_deg2"]
            )

            # Create test data in expected format
            training_data = pd.DataFrame(
                {
                    "fp_activity": self.X_train[:, 0],
                    "dram_activity": self.X_train[:, 1],
                    "sm_clock": (self.X_train[:, 2] * 600 + 800).astype(
                        int
                    ),  # Scale to reasonable frequency range
                    "power": self.y_train,
                }
            )

            # Test model training through framework
            results = framework.train_models(training_data, target_column="power")

            self.assertIn("models", results)
            self.assertIn("best_model", results)

            # Test power prediction integration
            if results["best_model"] is not None:
                power_sweep = framework.predict_power_sweep(
                    fp_activity=0.3, dram_activity=0.15, frequencies=[1000, 1100, 1200]
                )

                self.assertIsInstance(power_sweep, pd.DataFrame)
                self.assertEqual(len(power_sweep), 3)

        except ImportError:
            self.skipTest("FGCSPowerModelingFramework not available")


class TestSystemIntegration(unittest.TestCase):
    """Test cases for system-level integration and compatibility."""

    def test_path_and_import_integration(self):
        """Test that all modules can be imported from expected paths."""
        # Test main module imports
        modules_to_test = [
            "power_modeling",
            "power_modeling.models",
            "power_modeling.models.fgcs_models",
            "power_modeling.models.ensemble_models",
            "power_modeling.models.model_factory",
        ]

        for module_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")

    def test_logging_integration(self):
        """Test logging system integration."""
        import logging

        # Test that logging can be configured without errors
        logger = logging.getLogger("test_power_modeling")
        logger.setLevel(logging.INFO)

        # Test logging doesn't break
        try:
            logger.info("Test log message")
            logger.warning("Test warning message")
            logger.error("Test error message")
        except Exception as e:
            self.fail(f"Logging failed: {e}")

    def test_temporary_directory_handling(self):
        """Test proper handling of temporary directories and cleanup."""
        import shutil
        import tempfile

        # Test temporary directory creation and cleanup
        temp_dirs = []

        try:
            # Create multiple temporary directories
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f"power_modeling_test_{i}_")
                temp_dirs.append(temp_dir)

                # Verify directory exists
                self.assertTrue(os.path.exists(temp_dir))
                self.assertTrue(os.path.isdir(temp_dir))

                # Test writing to temporary directory
                test_file = os.path.join(temp_dir, "test.txt")
                with open(test_file, "w") as f:
                    f.write(f"Test content {i}")

                self.assertTrue(os.path.exists(test_file))

        finally:
            # Cleanup
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    self.assertFalse(os.path.exists(temp_dir))

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility features."""
        import platform

        # Test platform detection
        system = platform.system()
        self.assertIn(system, ["Windows", "Linux", "Darwin"])  # macOS is Darwin

        # Test path handling
        test_path = os.path.join("test", "path", "structure")
        normalized_path = os.path.normpath(test_path)

        # Should work on all platforms
        self.assertIsInstance(normalized_path, str)
        self.assertIn("path", normalized_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
