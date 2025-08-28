#!/usr/bin/env python3
"""
Whisper Application Test Suite for AI Inference Energy Profiling.

This script tests the Whisper implementation to ensure it works correctly
with the energy profiling framework.

Author: Mert Side
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path to import WhisperViaHF
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import torch
    from WhisperViaHF import WHISPER_MODELS, WhisperEnergyProfiler
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


def test_basic_functionality():
    """Test basic Whisper functionality."""
    print("Testing basic functionality...")

    # Test with tiny model for faster testing
    profiler = WhisperEnergyProfiler(model_name="tiny", device="auto", torch_dtype="float16")

    try:
        # Load model
        profiler.load_model()
        print("✓ Model loaded successfully")

        # Generate synthetic audio
        audio = profiler.generate_audio_sample(text="Test audio", duration=5.0)
        print(f"✓ Audio generated: {len(audio)} samples")

        # Transcribe
        result = profiler.transcribe_audio(audio)
        print(f"✓ Transcription completed: {result['inference_time']:.2f}s")
        print(f"✓ RTF: {result['rtf']:.3f}")

        # Cleanup
        profiler.cleanup()
        print("✓ Cleanup completed")

        return True

    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        return False


def test_benchmark_mode():
    """Test benchmark mode."""
    print("\nTesting benchmark mode...")

    profiler = WhisperEnergyProfiler(model_name="tiny", device="auto", torch_dtype="float16")

    try:
        # Run small benchmark
        results = profiler.run_benchmark(num_samples=2, use_dataset=False, language="en")

        print(f"✓ Benchmark completed")
        print(f"  Samples processed: {results['total_samples']}")
        print(f"  Total inference time: {results['total_inference_time']:.2f}s")
        print(f"  Average RTF: {results['average_rtf']:.3f}")

        # Cleanup
        profiler.cleanup()

        return True

    except Exception as e:
        print(f"✗ Error in benchmark test: {e}")
        return False


def test_model_loading():
    """Test different model loading."""
    print("\nTesting model loading...")

    # Test tiny model (should be fast)
    try:
        profiler = WhisperEnergyProfiler(model_name="tiny", device="auto", torch_dtype="float16")
        profiler.load_model()
        print(f"✓ Tiny model loaded: {profiler.metrics['model_parameters']:,} parameters")
        profiler.cleanup()

    except Exception as e:
        print(f"✗ Error loading tiny model: {e}")
        return False

    return True


def test_audio_processing():
    """Test audio processing capabilities."""
    print("\nTesting audio processing...")

    profiler = WhisperEnergyProfiler(model_name="tiny", device="auto", torch_dtype="float16")

    try:
        # Test different audio durations
        durations = [2.0, 5.0, 10.0]

        for duration in durations:
            audio = profiler.generate_audio_sample(text=f"Test audio {duration}s", duration=duration)

            expected_length = int(16000 * duration)  # 16kHz sample rate
            actual_length = len(audio)

            if abs(actual_length - expected_length) < 100:  # Allow small tolerance
                print(f"✓ Audio generation {duration}s: {actual_length} samples")
            else:
                print(f"✗ Audio generation {duration}s: expected ~{expected_length}, got {actual_length}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error in audio processing test: {e}")
        return False


def test_command_line_interface():
    """Test command line interface."""
    print("\nTesting command line interface...")

    try:
        # Test help
        exit_code = os.system("python ../WhisperViaHF.py --help > /dev/null 2>&1")
        if exit_code == 0:
            print("✓ Help command works")
        else:
            print("✗ Help command failed")
            return False

        # Test basic execution with synthetic audio
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_results.json")

            cmd = f"python ../WhisperViaHF.py --model tiny --generate-audio --audio-duration 3 --output-file {output_file} --quiet"
            exit_code = os.system(cmd)

            if exit_code == 0:
                print("✓ Basic CLI execution works")

                # Check if output file was created
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        results = json.load(f)
                    print(f"✓ Output file created with {len(results)} result fields")
                else:
                    print("✗ Output file not created")
                    return False
            else:
                print("✗ Basic CLI execution failed")
                return False

        return True

    except Exception as e:
        print(f"✗ Error in CLI test: {e}")
        return False


def test_gpu_functionality():
    """Test GPU-specific functionality."""
    print("\nTesting GPU functionality...")

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping GPU tests")
        return True

    try:
        # Test CUDA device
        profiler = WhisperEnergyProfiler(model_name="tiny", device="cuda", torch_dtype="float16")

        profiler.load_model()

        # Generate and process audio
        audio = profiler.generate_audio_sample(duration=3.0)
        result = profiler.transcribe_audio(audio)

        print(f"✓ GPU execution completed")
        print(f"  GPU memory used: {profiler.metrics['gpu_memory_used']:.2f} GB")
        print(f"  RTF: {result['rtf']:.3f}")

        profiler.cleanup()

        return True

    except Exception as e:
        print(f"✗ Error in GPU test: {e}")
        return False


def main():
    """Run all tests."""
    print("Whisper Application Test Suite")
    print("=" * 50)

    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Benchmark Mode", test_benchmark_mode),
        ("Model Loading", test_model_loading),
        ("Audio Processing", test_audio_processing),
        ("Command Line Interface", test_command_line_interface),
        ("GPU Functionality", test_gpu_functionality),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))

        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
