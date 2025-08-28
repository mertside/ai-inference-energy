#!/usr/bin/env python3
"""CLI tests for the profile.py script."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestProfileCLI(unittest.TestCase):
    """Tests for sample-collection-scripts/profile.py command-line interface."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.script = cls.project_root / "sample-collection-scripts" / "profile.py"
        assert cls.script.exists(), f"Profile script not found: {cls.script}"

    def setUp(self):
        # Create a temporary directory with a mock dcgmi executable
        self.temp_dir = tempfile.TemporaryDirectory()
        dcgmi = Path(self.temp_dir.name) / "dcgmi"
        dcgmi.write_text("#!/bin/sh\necho 'dcgmi mock'\n")
        dcgmi.chmod(0o755)

        # Environment with project root in PYTHONPATH and mock PATH
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(self.project_root)
        self.env["PATH"] = f"{self.temp_dir.name}{os.pathsep}" + self.env.get("PATH", "")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_help_option(self):
        """profile.py --help displays help text."""
        result = subprocess.run(
            [sys.executable, str(self.script), "--help"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("GPU power profiling utility", result.stdout)

    def test_invalid_argument(self):
        """profile.py fails gracefully on invalid arguments."""
        result = subprocess.run(
            [sys.executable, str(self.script), "--badflag"],
            capture_output=True,
            text=True,
            env=self.env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage", result.stderr.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
