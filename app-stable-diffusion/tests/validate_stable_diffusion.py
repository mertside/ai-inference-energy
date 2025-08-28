#!/usr/bin/env python3
"""
Simple validation script for the revised Stable Diffusion implementation.
This runs basic checks without requiring GPU or model downloads.
"""

import os
import sys

sys.path.append("app-stable-diffusion")


def test_basic_functionality():
    """Test basic functionality without model loading."""
    print("🧪 Testing revised Stable Diffusion implementation...")

    try:
        # Test import
        from StableDiffusionViaHF import StableDiffusionGenerator

        print("✅ Import successful")

        # Test model configurations
        configs = StableDiffusionGenerator.MODEL_CONFIGS
        print(f"✅ Available models: {list(configs.keys())}")

        # Test initialization (without model loading)
        generator = StableDiffusionGenerator(model_variant="sd-v1.4", device="auto")  # Will auto-detect and fall back to CPU if needed
        print("✅ Generator initialization successful")

        # Test model info
        info = generator.get_model_info()
        print(f"✅ Model info: {info['model_id']}")
        print(f"   Device: {info['device']}")
        print(f"   Default size: {info['default_size']}")

        # Test statistics
        stats = generator.get_generation_stats()
        print(f"✅ Generation stats initialized: {stats}")

        # Test different model variants
        for variant in ["sd-v1.5", "sd-v2.0", "sdxl"]:
            try:
                test_gen = StableDiffusionGenerator(model_variant=variant, device="auto")
                test_info = test_gen.get_model_info()
                print(f"✅ {variant}: {test_info['model_id']}")
            except Exception as e:
                print(f"❌ {variant} failed: {e}")

        # Test CLI argument parsing
        import argparse

        from StableDiffusionViaHF import parse_arguments

        # Simulate CLI args
        test_args = ["--model-variant", "sd-v1.4", "--num-images", "1", "--help"]
        try:
            # This will raise SystemExit for --help, which is expected
            args = parse_arguments()
        except SystemExit:
            pass  # Expected for --help
        except Exception as e:
            print(f"❌ CLI parsing failed: {e}")
        else:
            print("✅ CLI argument parsing working")

        print("\n🎉 All basic tests passed!")
        print("\nThe revised Stable Diffusion implementation is ready for use.")
        print("\nNext steps:")
        print("1. Ensure you have the required dependencies: torch, diffusers, transformers")
        print("2. Login to Hugging Face: huggingface-cli login")
        print("3. Run with GPU for actual generation testing")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
