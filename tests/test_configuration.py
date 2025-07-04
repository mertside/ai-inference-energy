#!/usr/bin/env python3
"""
Configuration and Environment Test Suite

Tests the configuration loading and environment setup functionality.
This includes GPU configurations, profiling parameters, and system settings.
"""

import sys
import os
import unittest

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration loading."""
    
    def test_config_module_import(self):
        """Test that config module can be imported successfully."""
        try:
            from config import profiling_config, gpu_config, model_config, system_config
            
            # Verify basic structure
            self.assertTrue(hasattr(profiling_config, 'DCGMI_FIELDS'))
            self.assertTrue(hasattr(gpu_config, 'A100_CORE_FREQUENCIES'))
            self.assertTrue(hasattr(profiling_config, 'DEFAULT_INTERVAL_MS'))
            self.assertTrue(hasattr(model_config, 'LLAMA_MODEL_NAME'))
            
            # Verify data types and reasonable values
            self.assertIsInstance(profiling_config.DCGMI_FIELDS, (list, tuple))
            self.assertIsInstance(gpu_config.A100_CORE_FREQUENCIES, (list, tuple))
            self.assertIsInstance(profiling_config.DEFAULT_INTERVAL_MS, (int, float))
            self.assertIsInstance(model_config.LLAMA_MODEL_NAME, str)
            
            # Verify non-empty collections
            self.assertGreater(len(profiling_config.DCGMI_FIELDS), 0)
            self.assertGreater(len(gpu_config.A100_CORE_FREQUENCIES), 0)
            
        except ImportError as e:
            self.fail(f"Config module import failed: {e}")
        except AttributeError as e:
            self.fail(f"Config module missing expected attributes: {e}")
    
    def test_gpu_configurations(self):
        """Test GPU-specific configurations."""
        try:
            from config import gpu_config
            
            # Test that GPU configurations exist for major GPU types
            expected_gpus = ['A100', 'V100', 'H100']
            
            for gpu in expected_gpus:
                freq_attr = f"{gpu}_CORE_FREQUENCIES"
                if hasattr(gpu_config, freq_attr):
                    frequencies = getattr(gpu_config, freq_attr)
                    self.assertIsInstance(frequencies, (list, tuple))
                    self.assertGreater(len(frequencies), 0)
                    
                    # Verify frequencies are reasonable (between 100 and 2000 MHz)
                    for freq in frequencies[:5]:  # Check first 5 frequencies
                        self.assertGreater(freq, 100)
                        self.assertLess(freq, 2000)
                        
        except ImportError:
            self.skipTest("GPU config not available")
    
    def test_profiling_parameters(self):
        """Test profiling configuration parameters."""
        try:
            from config import profiling_config
            
            # Test default interval is reasonable (between 1ms and 10s)
            self.assertGreater(profiling_config.DEFAULT_INTERVAL_MS, 0)
            self.assertLess(profiling_config.DEFAULT_INTERVAL_MS, 10000)
            
            # Test DCGMI fields are properly defined
            if hasattr(profiling_config, 'DCGMI_FIELDS'):
                for field in profiling_config.DCGMI_FIELDS[:3]:  # Check first 3 fields
                    # DCGMI fields can be strings or integers (field IDs)
                    self.assertIsInstance(field, (str, int))
                    if isinstance(field, str):
                        self.assertGreater(len(field), 0)
                    
        except ImportError:
            self.skipTest("Profiling config not available")


class TestEnvironmentSetup(unittest.TestCase):
    """Test cases for environment setup and Python compatibility."""
    
    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        # Test minimum Python version (3.6+)
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3)
        self.assertGreaterEqual(version_info.minor, 6)
    
    def test_required_imports(self):
        """Test that required external packages can be imported."""
        required_packages = [
            'numpy',
            'pandas',
            'sklearn',
        ]
        
        for package in required_packages:
            with self.subTest(package=package):
                try:
                    __import__(package)
                except ImportError:
                    self.fail(f"Required package '{package}' not available")
    
    def test_optional_imports(self):
        """Test optional packages and skip gracefully if not available."""
        optional_packages = [
            'xgboost',
            'matplotlib',
            'seaborn',
        ]
        
        for package in optional_packages:
            with self.subTest(package=package):
                try:
                    __import__(package)
                except ImportError:
                    pass  # Optional packages are allowed to be missing
    
    def test_subprocess_compatibility(self):
        """Test subprocess functionality for Python 3.6+ compatibility."""
        import subprocess
        
        # Test basic subprocess functionality
        try:
            # Use a simple command that works on all platforms
            if sys.platform.startswith('win'):
                result = subprocess.run(['echo', 'test'], capture_output=True, text=True, shell=True)
            else:
                result = subprocess.run(['echo', 'test'], capture_output=True, text=True)
            
            self.assertEqual(result.returncode, 0)
            self.assertIn('test', result.stdout)
            
        except Exception as e:
            self.fail(f"Subprocess functionality failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
