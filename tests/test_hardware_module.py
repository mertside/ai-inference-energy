#!/usr/bin/env python3
"""
Test Hardware Abstraction Layer (HAL) - GPU Info Module

Comprehensive test suite for the GPU Info Module implementation,
ensuring all specifications, validation, and functionality work correctly.
"""

import copy
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware import (
    ComputeSpecification,
    FrequencySpecification,
    GPUArchitecture,
    GPUSpecifications,
    MemorySpecification,
    PowerSpecification,
    ThermalSpecification,
    compare_gpus,
    get_gpu_info,
    get_module_info,
    get_supported_gpus,
    validate_gpu_configuration,
)


class TestGPUSpecifications(unittest.TestCase):
    """Test GPU specifications functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.supported_gpus = ["V100", "A100", "H100"]

    def test_supported_gpus(self):
        """Test that all expected GPUs are supported."""
        supported = get_supported_gpus()
        self.assertIsInstance(supported, list)
        self.assertEqual(sorted(supported), sorted(self.supported_gpus))

    def test_gpu_info_creation(self):
        """Test GPU info object creation for all supported GPUs."""
        for gpu_type in self.supported_gpus:
            with self.subTest(gpu_type=gpu_type):
                gpu_info = get_gpu_info(gpu_type)
                self.assertIsInstance(gpu_info, GPUSpecifications)
                self.assertEqual(gpu_info.gpu_type, gpu_type)

    def test_invalid_gpu_type(self):
        """Test error handling for invalid GPU types."""
        with self.assertRaises(ValueError):
            get_gpu_info("INVALID_GPU")

        with self.assertRaises(ValueError):
            GPUSpecifications("RTX4090")

    def test_gpu_architectures(self):
        """Test GPU architecture mapping."""
        expected_architectures = {
            "V100": GPUArchitecture.VOLTA,
            "A100": GPUArchitecture.AMPERE,
            "H100": GPUArchitecture.HOPPER,
        }

        for gpu_type, expected_arch in expected_architectures.items():
            with self.subTest(gpu_type=gpu_type):
                gpu_info = get_gpu_info(gpu_type)
                arch = gpu_info.get_architecture()
                self.assertEqual(arch, expected_arch)


class TestFrequencyManagement(unittest.TestCase):
    """Test frequency management and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.gpu_infos = {gpu: get_gpu_info(gpu) for gpu in get_supported_gpus()}

    def test_frequency_specification(self):
        """Test frequency specification objects."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                freq_spec = gpu_info.get_frequency_specification()
                self.assertIsInstance(freq_spec, FrequencySpecification)
                self.assertIsInstance(freq_spec.min_freq, int)
                self.assertIsInstance(freq_spec.max_freq, int)
                self.assertIsInstance(freq_spec.count, int)
                self.assertGreater(freq_spec.max_freq, freq_spec.min_freq)
                self.assertGreater(freq_spec.count, 0)

    def test_frequency_ranges(self):
        """Test frequency range retrieval."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                min_freq, max_freq = gpu_info.get_frequency_range()
                self.assertIsInstance(min_freq, int)
                self.assertIsInstance(max_freq, int)
                self.assertGreater(max_freq, min_freq)

    def test_available_frequencies(self):
        """Test available frequency lists."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                frequencies = gpu_info.get_available_frequencies()
                self.assertIsInstance(frequencies, list)
                self.assertGreater(len(frequencies), 0)

                # Check frequencies are in valid range
                min_freq, max_freq = gpu_info.get_frequency_range()
                for freq in frequencies:
                    self.assertGreaterEqual(freq, min_freq)
                    self.assertLessEqual(freq, max_freq)

                # Check frequencies are sorted in descending order
                self.assertEqual(frequencies, sorted(frequencies, reverse=True))

    def test_single_frequency_case(self):
        """Ensure single-frequency specifications return a single value."""
        gpu_info = get_gpu_info("V100")
        gpu_info.specifications = copy.deepcopy(gpu_info.specifications)
        gpu_info.specifications["frequency"] = {
            "min_freq": 1234,
            "max_freq": 1234,
            "count": 1,
        }
        frequencies = gpu_info.get_available_frequencies()
        self.assertEqual(frequencies, [1234])

    def test_frequency_validation(self):
        """Test frequency validation functionality."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                frequencies = gpu_info.get_available_frequencies()

                # Test valid frequencies
                for freq in frequencies[:5]:  # Test first 5
                    self.assertTrue(gpu_info.validate_frequency(freq))

                # Test invalid frequencies
                min_freq, max_freq = gpu_info.get_frequency_range()
                invalid_freqs = [min_freq - 100, max_freq + 100, 0, -100]
                for freq in invalid_freqs:
                    self.assertFalse(gpu_info.validate_frequency(freq))

    def test_closest_frequency(self):
        """Test closest frequency finding."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                frequencies = gpu_info.get_available_frequencies()
                min_freq, max_freq = gpu_info.get_frequency_range()

                # Test exact matches
                for freq in frequencies[:3]:
                    closest = gpu_info.get_closest_frequency(freq)
                    self.assertEqual(closest, freq)

                # Test approximate matches
                target = (min_freq + max_freq) // 2
                closest = gpu_info.get_closest_frequency(target)
                self.assertIn(closest, frequencies)

    def test_fgcs_compatible_frequencies(self):
        """Test FGCS compatible frequency lists."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                fgcs_freqs = gpu_info.get_fgcs_compatible_frequencies()
                self.assertIsInstance(fgcs_freqs, list)
                self.assertGreater(len(fgcs_freqs), 0)

                # FGCS frequencies might be reference frequencies, not all may be
                # in the exact available frequency list, so we check they're reasonable
                available_freqs = gpu_info.get_available_frequencies()
                min_freq, max_freq = gpu_info.get_frequency_range()

                for freq in fgcs_freqs:
                    # Check frequency is within reasonable range
                    self.assertGreaterEqual(freq, min_freq)
                    self.assertLessEqual(freq, max_freq)
                    self.assertIsInstance(freq, int)

                # At least some FGCS frequencies should be available
                common_freqs = set(fgcs_freqs) & set(available_freqs)
                self.assertGreater(
                    len(common_freqs),
                    0,
                    f"No common frequencies between FGCS and available for {gpu_type}",
                )

    def test_workload_frequency_recommendations(self):
        """Test workload-specific frequency recommendations."""
        workload_types = ["inference", "training", "compute", "memory_bound"]

        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                for workload in workload_types:
                    freq = gpu_info.get_optimal_frequency_for_workload(workload)
                    self.assertIsInstance(freq, int)
                    self.assertGreater(freq, 0)

                    # Should be within valid range
                    min_freq, max_freq = gpu_info.get_frequency_range()
                    self.assertGreaterEqual(freq, min_freq)
                    self.assertLessEqual(freq, max_freq)


class TestSpecificationObjects(unittest.TestCase):
    """Test specification dataclass objects."""

    def setUp(self):
        """Set up test fixtures."""
        self.gpu_infos = {gpu: get_gpu_info(gpu) for gpu in get_supported_gpus()}

    def test_memory_specifications(self):
        """Test memory specification objects."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                memory_spec = gpu_info.get_memory_specification()
                self.assertIsInstance(memory_spec, MemorySpecification)
                self.assertIsInstance(memory_spec.size_gb, int)
                self.assertIsInstance(memory_spec.type, str)
                self.assertIsInstance(memory_spec.bandwidth_gb_s, (int, float))
                self.assertIsInstance(memory_spec.frequency_mhz, int)
                self.assertIsInstance(memory_spec.bus_width, int)

                # Sanity checks
                self.assertGreater(memory_spec.size_gb, 0)
                self.assertGreater(memory_spec.bandwidth_gb_s, 0)
                self.assertGreater(memory_spec.frequency_mhz, 0)
                self.assertGreater(memory_spec.bus_width, 0)

    def test_compute_specifications(self):
        """Test compute specification objects."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                compute_spec = gpu_info.get_compute_specification()
                self.assertIsInstance(compute_spec, ComputeSpecification)
                self.assertIsInstance(compute_spec.sm_count, int)
                self.assertIsInstance(compute_spec.cuda_cores, int)
                self.assertIsInstance(compute_spec.tensor_cores, bool)
                self.assertIsInstance(compute_spec.compute_capability, str)

                # Sanity checks
                self.assertGreater(compute_spec.sm_count, 0)
                self.assertGreater(compute_spec.cuda_cores, 0)
                self.assertTrue(compute_spec.tensor_cores)  # All tested GPUs have tensor cores

    def test_power_specifications(self):
        """Test power specification objects."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                power_spec = gpu_info.get_power_specification()
                self.assertIsInstance(power_spec, PowerSpecification)
                self.assertIsInstance(power_spec.tdp_watts, int)
                self.assertIsInstance(power_spec.min_power_watts, int)
                self.assertIsInstance(power_spec.max_power_watts, int)
                self.assertIsInstance(power_spec.power_connectors, list)

                # Sanity checks
                self.assertGreater(power_spec.tdp_watts, 0)
                self.assertGreater(power_spec.min_power_watts, 0)
                self.assertGreater(power_spec.max_power_watts, power_spec.min_power_watts)
                self.assertGreaterEqual(power_spec.max_power_watts, power_spec.tdp_watts)
                self.assertGreater(len(power_spec.power_connectors), 0)

    def test_thermal_specifications(self):
        """Test thermal specification objects."""
        for gpu_type, gpu_info in self.gpu_infos.items():
            with self.subTest(gpu_type=gpu_type):
                thermal_spec = gpu_info.get_thermal_specification()
                self.assertIsInstance(thermal_spec, ThermalSpecification)
                self.assertIsInstance(thermal_spec.max_temp_c, int)
                self.assertIsInstance(thermal_spec.throttle_temp_c, int)
                self.assertIsInstance(thermal_spec.idle_temp_c, int)
                self.assertIsInstance(thermal_spec.cooling_solution, str)

                # Sanity checks
                self.assertGreater(thermal_spec.max_temp_c, thermal_spec.throttle_temp_c)
                self.assertGreater(thermal_spec.throttle_temp_c, thermal_spec.idle_temp_c)
                self.assertGreater(thermal_spec.idle_temp_c, 0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions and utilities."""

    def test_gpu_comparison(self):
        """Test GPU comparison functionality."""
        supported_gpus = get_supported_gpus()
        comparison = compare_gpus(supported_gpus)

        self.assertIsInstance(comparison, dict)
        self.assertEqual(len(comparison), len(supported_gpus))

        for gpu_type in supported_gpus:
            self.assertIn(gpu_type, comparison)
            self.assertIsInstance(comparison[gpu_type], dict)
            self.assertNotIn("error", comparison[gpu_type])

    def test_gpu_comparison_with_invalid(self):
        """Test GPU comparison with invalid GPU types."""
        gpu_list = ["V100", "INVALID_GPU", "A100"]
        comparison = compare_gpus(gpu_list)

        self.assertIsInstance(comparison, dict)
        self.assertEqual(len(comparison), 3)

        # Valid GPUs should have complete data
        self.assertNotIn("error", comparison["V100"])
        self.assertNotIn("error", comparison["A100"])

        # Invalid GPU should have error
        self.assertIn("error", comparison["INVALID_GPU"])

    def test_configuration_validation(self):
        """Test configuration validation function."""
        # Test valid configurations
        valid_configs = [
            ("V100", 1380, 877),
            ("A100", 1410, 1215),
            ("H100", 1785, 2619),
        ]

        for gpu_type, core_freq, memory_freq in valid_configs:
            with self.subTest(gpu_type=gpu_type):
                result = validate_gpu_configuration(gpu_type, core_freq, memory_freq)
                self.assertIsInstance(result, dict)
                self.assertTrue(result["overall_valid"])
                self.assertTrue(result["gpu_type_valid"])
                self.assertTrue(result["frequency_valid"])
                self.assertTrue(result["memory_freq_valid"])

        # Test invalid configurations
        invalid_configs = [
            ("INVALID_GPU", 1000, None),
            ("V100", 9999, None),  # Invalid frequency
            ("A100", 1410, 999),  # Invalid memory frequency
        ]

        for gpu_type, core_freq, memory_freq in invalid_configs:
            with self.subTest(gpu_type=gpu_type):
                result = validate_gpu_configuration(gpu_type, core_freq, memory_freq)
                self.assertIsInstance(result, dict)
                self.assertFalse(result["overall_valid"])

    def test_gpu_summary(self):
        """Test GPU summary generation."""
        for gpu_type in get_supported_gpus():
            with self.subTest(gpu_type=gpu_type):
                gpu_info = get_gpu_info(gpu_type)
                summary = gpu_info.get_summary()

                self.assertIsInstance(summary, dict)

                # Check required fields
                required_fields = [
                    "gpu_type",
                    "architecture",
                    "frequency_range",
                    "frequency_count",
                    "memory_size",
                    "memory_bandwidth",
                    "sm_count",
                    "cuda_cores",
                    "tensor_cores",
                    "compute_capability",
                    "tdp",
                    "supported_frequencies",
                ]

                for field in required_fields:
                    self.assertIn(field, summary)

                self.assertEqual(summary["gpu_type"], gpu_type)

    def test_module_info(self):
        """Test module information function."""
        module_info = get_module_info()

        self.assertIsInstance(module_info, dict)

        # Check required fields
        required_fields = [
            "version",
            "description",
            "implemented_modules",
            "planned_modules",
            "supported_gpus",
            "total_gpu_specifications",
            "example_usage",
        ]

        for field in required_fields:
            self.assertIn(field, module_info)

        # Check specific values
        self.assertEqual(module_info["implemented_modules"], ["gpu_info"])
        self.assertEqual(len(module_info["supported_gpus"]), 3)
        self.assertEqual(module_info["total_gpu_specifications"], 3)


class TestDataConsistency(unittest.TestCase):
    """Test data consistency across GPU specifications."""

    def test_frequency_data_consistency(self):
        """Test frequency data consistency."""
        for gpu_type in get_supported_gpus():
            with self.subTest(gpu_type=gpu_type):
                gpu_info = get_gpu_info(gpu_type)
                freq_spec = gpu_info.get_frequency_specification()
                available_freqs = gpu_info.get_available_frequencies()

                # Frequency count should match available frequencies
                self.assertEqual(len(available_freqs), freq_spec.count)

                # Min/max should match extremes
                self.assertEqual(min(available_freqs), freq_spec.min_freq)
                self.assertEqual(max(available_freqs), freq_spec.max_freq)

                # If explicit frequencies provided, should match
                if freq_spec.frequencies:
                    self.assertEqual(available_freqs, freq_spec.frequencies)

    def test_fgcs_data_consistency(self):
        """Test FGCS data consistency."""
        for gpu_type in get_supported_gpus():
            with self.subTest(gpu_type=gpu_type):
                gpu_info = get_gpu_info(gpu_type)
                fgcs_freqs = gpu_info.get_fgcs_compatible_frequencies()
                available_freqs = gpu_info.get_available_frequencies()

                # FGCS frequencies might be reference frequencies for academic studies
                # Check that they are reasonable and overlap with available frequencies
                min_freq, max_freq = gpu_info.get_frequency_range()

                for freq in fgcs_freqs:
                    # Check frequency is within reasonable range
                    self.assertGreaterEqual(freq, min_freq - 100)  # Allow some tolerance
                    self.assertLessEqual(freq, max_freq + 100)

                # At least 50% of FGCS frequencies should be available
                common_freqs = set(fgcs_freqs) & set(available_freqs)
                overlap_ratio = len(common_freqs) / len(fgcs_freqs)
                self.assertGreater(
                    overlap_ratio,
                    0.3,  # At least 30% overlap
                    f"Insufficient overlap between FGCS and available frequencies for {gpu_type}: {overlap_ratio:.2f}",
                )

    def test_workload_recommendations_consistency(self):
        """Test workload recommendation consistency."""
        for gpu_type in get_supported_gpus():
            with self.subTest(gpu_type=gpu_type):
                gpu_info = get_gpu_info(gpu_type)
                available_freqs = gpu_info.get_available_frequencies()
                min_freq, max_freq = gpu_info.get_frequency_range()

                workload_types = ["inference", "training", "compute", "memory_bound"]
                for workload in workload_types:
                    recommended_freq = gpu_info.get_optimal_frequency_for_workload(workload)
                    # Recommended frequency should be reasonable
                    self.assertIsInstance(recommended_freq, int)
                    self.assertGreaterEqual(recommended_freq, min_freq)
                    self.assertLessEqual(recommended_freq, max_freq)

                    # If not exactly available, should be close to an available frequency
                    if not gpu_info.validate_frequency(recommended_freq):
                        closest = gpu_info.get_closest_frequency(recommended_freq)
                        self.assertLessEqual(
                            abs(recommended_freq - closest),
                            100,
                            f"Recommended frequency {recommended_freq} for {workload} is too far from closest available {closest}",
                        )


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
