#!/usr/bin/env python3
"""
Test Runner for Power Modeling Framework

This script runs all consolidated tests for the power modeling framework.
It provides a unified entry point for running the complete test suite.
"""

import argparse
import os
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set PYTHONPATH for subprocesses
os.environ["PYTHONPATH"] = str(project_root)


def discover_tests(test_dir="tests", pattern="test_*.py"):
    """Discover and return all test cases."""
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__))
    suite = loader.discover(start_dir, pattern=pattern)
    return suite


def run_specific_test(test_name):
    """Run a specific test module."""
    loader = unittest.TestLoader()

    try:
        if test_name == "all":
            # Run all tests by discovering them
            return discover_tests()
        elif test_name in ["configuration", "config"]:
            from tests import test_configuration

            return loader.loadTestsFromModule(test_configuration)
        elif test_name in ["framework", "power"]:
            from tests import test_power_modeling_framework

            return loader.loadTestsFromModule(test_power_modeling_framework)
        elif test_name in ["integration", "int"]:
            from tests import test_integration

            return loader.loadTestsFromModule(test_integration)
        else:
            # Try to load by exact name
            if test_name.startswith("test_"):
                module_name = test_name
            else:
                module_name = f"test_{test_name}"

            suite = loader.loadTestsFromName(module_name)
            return suite
    except (ImportError, AttributeError) as e:
        print(f"Error loading test '{test_name}': {e}")
        return None


def run_tests(suite, verbosity=2):
    """Run the test suite and return results."""
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    return result


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run power modeling framework tests")
    parser.add_argument(
        "--test", "-t", help="Run specific test module (e.g., power_modeling_framework)"
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=2, help="Increase verbosity"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Reduce verbosity")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available tests"
    )
    parser.add_argument(
        "--pattern", "-p", default="test_*.py", help="Test file pattern"
    )

    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        verbosity = 0
    else:
        verbosity = min(args.verbose, 2)

    # List available tests
    if args.list:
        print("Available test modules:")
        test_files = []
        tests_dir = os.path.dirname(__file__)
        for file in os.listdir(tests_dir):
            if (
                file.startswith("test_")
                and file.endswith(".py")
                and file != "__init__.py"
            ):
                test_files.append(file[:-3])  # Remove .py extension

        for test_file in sorted(test_files):
            print(f"  - {test_file}")
        return 0

    # Run specific test or all tests
    if args.test:
        suite = run_specific_test(args.test)
        if suite is None:
            return 1
    else:
        suite = discover_tests(pattern=args.pattern)

    print("Running Power Modeling Framework Test Suite")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    print("=" * 50)

    # Run tests
    result = run_tests(suite, verbosity)

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")

    if result.skipped:
        print(f"\nSKIPPED ({len(result.skipped)}):")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")

    # Return exit code
    if result.failures or result.errors:
        print("\n❌ Some tests failed!")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
