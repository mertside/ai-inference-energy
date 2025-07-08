#!/usr/bin/env python3
"""Unit tests for utils helper functions."""

import logging
import os
import sys
import tempfile
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch
import subprocess

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402


class TestUtilsLogging(unittest.TestCase):
    """Tests for the setup_logging function."""

    def test_setup_logging_basic(self):
        logger = utils.setup_logging(log_level="DEBUG")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertGreater(len(logger.handlers), 0)

    def test_setup_logging_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "test.log")
            logger = utils.setup_logging(log_level="INFO", log_file=log_path)
            logger.info("hello")
            logger.handlers[0].flush()
            with open(log_path, "r") as fh:
                content = fh.read()
            self.assertIn("hello", content)


class TestUtilsParsing(unittest.TestCase):
    """Tests for parsing and formatting helpers."""

    def test_parse_csv_line(self):
        line = "a, b, c"
        self.assertEqual(utils.parse_csv_line(line), ["a", "b", "c"])
        line_semicolon = "1;2;3"
        self.assertEqual(utils.parse_csv_line(line_semicolon, delimiter=";"), ["1", "2", "3"])

    def test_clean_filename(self):
        dirty = "inva:lid/fi*le?name.txt"
        cleaned = utils.clean_filename(dirty)
        self.assertNotIn(":", cleaned)
        self.assertNotIn("/", cleaned)
        self.assertTrue(cleaned.startswith("inva_lid_fi_le_name"))

    def test_format_duration(self):
        self.assertEqual(utils.format_duration(42.3), "42.30s")
        self.assertEqual(utils.format_duration(90), "1m 30.00s")
        self.assertEqual(utils.format_duration(3661), "1h 1m 1.00s")


class TestUtilsValidation(unittest.TestCase):
    """Tests for GPU validation helpers with mocked subprocess calls."""

    @patch("utils.run_command")
    def test_validate_gpu_available_true(self, mock_run):
        mock_run.return_value = CompletedProcess([], 0, stdout="GPU0\n", stderr="")
        self.assertTrue(utils.validate_gpu_available())

    @patch("utils.run_command")
    def test_validate_gpu_available_false(self, mock_run):
        mock_run.return_value = CompletedProcess([], 0, stdout="", stderr="")
        self.assertFalse(utils.validate_gpu_available())
        mock_run.side_effect = FileNotFoundError
        self.assertFalse(utils.validate_gpu_available())

    @patch("utils.run_command")
    def test_validate_dcgmi_available(self, mock_run):
        mock_run.return_value = CompletedProcess([], 0, stdout="", stderr="")
        self.assertTrue(utils.validate_dcgmi_available())
        mock_run.return_value = CompletedProcess([], 1, stdout="", stderr="err")
        self.assertFalse(utils.validate_dcgmi_available())
        mock_run.side_effect = subprocess.CalledProcessError(1, "dcgmi")
        self.assertFalse(utils.validate_dcgmi_available())


if __name__ == "__main__":
    unittest.main()
