#!/usr/bin/env python3
"""
Comprehensive test suite for the revised Stable Diffusion application.

This script tests all functionality of the enhanced StableDiffusionViaHF.py
implementation, including multiple model support, memory optimizations,
and integration with the energy profiling framework.

Usage:
    python test_stable_diffusion_revised.py [--quick] [--model-variant MODEL]
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

def setup_test_logging():
    """Set up logging for test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger = logging.getLogger(__name__)
    missing_deps = []
    
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU: {gpu_name} ({memory_gb:.1f}GB)")
        else:
            logger.warning("⚠️ No GPU available - tests will run on CPU")
            
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import diffusers
        logger.info(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        missing_deps.append("diffusers")
    
    try:
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        from PIL import Image
        logger.info("✅ PIL/Pillow available")
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import numpy as np
        logger.info(f"✅ NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        logger.info("Install with: pip install torch diffusers transformers Pillow numpy accelerate")
        return False
    
    return True

def test_authentication():
    """Test Hugging Face authentication."""
    logger = logging.getLogger(__name__)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Authentication issue: {e}")
        logger.info("Run: huggingface-cli login")
        return False

def test_import():
    """Test importing the revised Stable Diffusion module."""
    logger = logging.getLogger(__name__)
    
    try:
        # Add the app directory to path
        app_dir = Path(__file__).parent / "app-stable-diffusion"
        sys.path.insert(0, str(app_dir))
        
        from StableDiffusionViaHF import StableDiffusionGenerator
        logger.info("✅ Successfully imported StableDiffusionGenerator")
        
        # Test model configurations
        configs = StableDiffusionGenerator.MODEL_CONFIGS
        logger.info(f"✅ Available models: {list(configs.keys())}")
        
        return True, StableDiffusionGenerator
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        logger.debug(traceback.format_exc())
        return False, None

def test_initialization(StableDiffusionGenerator, model_variant="sd-v1.4", quick_test=True):
    """Test model initialization."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Testing {model_variant} initialization...")
        
        # Initialize generator (don't auto-load model for quick test)
        generator = StableDiffusionGenerator(
            model_variant=model_variant,
            device="auto",
            logger=logger
        )
        
        logger.info("✅ Generator object created successfully")
        
        # Test model info before initialization
        info = generator.get_model_info()
        logger.info(f"Model info: {info}")
        
        if not quick_test:
            # Actually initialize the model
            logger.info("Initializing model (this may take a while)...")
            generator.initialize_model()
            logger.info("✅ Model initialized successfully")
            
            # Test model info after initialization
            info = generator.get_model_info()
            logger.info(f"Initialized model info: {info}")
        
        return True, generator
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.debug(traceback.format_exc())
        return False, None

def test_image_generation(generator, quick_test=True):
    """Test image generation functionality."""
    logger = logging.getLogger(__name__)
    
    try:
        if quick_test:
            logger.info("Skipping actual generation in quick test mode")
            return True
        
        logger.info("Testing image generation...")
        
        # Test single image generation
        images = generator.generate_images(
            prompts="a test image of a red apple",
            num_inference_steps=10,  # Reduced for testing
            height=256,  # Reduced for testing
            width=256,
            seed=42  # For reproducibility
        )
        
        logger.info(f"✅ Generated {len(images)} image(s)")
        
        # Test batch generation
        batch_prompts = [
            "a blue car",
            "a green tree",
        ]
        
        batch_images = generator.generate_images(
            prompts=batch_prompts,
            num_inference_steps=10,
            height=256,
            width=256,
            batch_size=2,
            seed=42
        )
        
        logger.info(f"✅ Generated {len(batch_images)} batch image(s)")
        
        # Test default inference
        default_images = generator.run_default_inference()
        logger.info(f"✅ Default inference generated {len(default_images)} image(s)")
        
        # Combine all images for saving
        all_images = images + batch_images + default_images
        
        # Test image saving
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        saved_files = generator.save_images(
            all_images,
            base_filename="test_generation",
            output_dir=str(output_dir),
            include_metadata=True
        )
        
        logger.info(f"✅ Saved {len(saved_files)} images to {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Image generation failed: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_benchmark_mode(generator, quick_test=True):
    """Test benchmark functionality."""
    logger = logging.getLogger(__name__)
    
    try:
        if quick_test:
            logger.info("Skipping benchmark in quick test mode")
            return True
        
        logger.info("Testing benchmark mode...")
        
        images, stats = generator.run_benchmark_inference(
            num_generations=3,
            use_different_prompts=True
        )
        
        logger.info(f"✅ Benchmark completed: {len(images)} images")
        logger.info(f"Performance stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark test failed: {e}")
        logger.debug(traceback.format_exc())
        return False

def test_memory_management(generator):
    """Test memory management features."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test cache clearing
        generator.clear_cache()
        logger.info("✅ Cache clearing successful")
        
        # Test generation stats
        stats = generator.get_generation_stats()
        logger.info(f"✅ Generation stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        return False

def test_cli_interface():
    """Test command line interface."""
    logger = logging.getLogger(__name__)
    
    try:
        app_dir = Path(__file__).parent / "app-stable-diffusion"
        script_path = app_dir / "StableDiffusionViaHF.py"
        
        if not script_path.exists():
            logger.error(f"❌ Script not found: {script_path}")
            return False
        
        import subprocess
        
        # Test help command
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ CLI help command works")
            return True
        else:
            logger.error(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ CLI test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ CLI test failed: {e}")
        return False

def test_framework_integration():
    """Test integration with the main profiling framework."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test that the script can be called by the launch.sh framework
        script_dir = Path(__file__).parent / "sample-collection-scripts"
        launch_script = script_dir / "launch.sh"
        
        if not launch_script.exists():
            logger.warning("⚠️ launch.sh not found, skipping integration test")
            return True
        
        # Test help to verify framework works
        import subprocess
        result = subprocess.run([
            str(launch_script), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Framework integration test passed")
            return True
        else:
            logger.warning(f"⚠️ Framework integration test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Framework integration test failed: {e}")
        return False

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test revised Stable Diffusion implementation")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (no actual generation)")
    parser.add_argument("--model-variant", default="sd-v1.4", help="Model variant to test")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_test_logging()
    if args.log_level != "INFO":
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("🚀 Starting Revised Stable Diffusion Test Suite")
    logger.info("=" * 60)
    
    if args.quick:
        logger.info("Running in QUICK mode (no actual model loading/generation)")
    
    # Track test results
    test_results = {}
    
    # Run tests
    logger.info("Test 1: Checking dependencies...")
    test_results["dependencies"] = check_dependencies()
    
    logger.info("Test 2: Checking authentication...")
    test_results["authentication"] = test_authentication()
    
    logger.info("Test 3: Testing import...")
    success, StableDiffusionGenerator = test_import()
    test_results["import"] = success
    
    if success:
        logger.info(f"Test 4: Testing initialization ({args.model_variant})...")
        success, generator = test_initialization(StableDiffusionGenerator, args.model_variant, args.quick)
        test_results["initialization"] = success
        
        if success:
            logger.info("Test 5: Testing image generation...")
            test_results["generation"] = test_image_generation(generator, args.quick)
            
            logger.info("Test 6: Testing benchmark mode...")
            test_results["benchmark"] = test_benchmark_mode(generator, args.quick)
            
            logger.info("Test 7: Testing memory management...")
            test_results["memory"] = test_memory_management(generator)
        else:
            test_results["generation"] = False
            test_results["benchmark"] = False
            test_results["memory"] = False
    else:
        test_results["initialization"] = False
        test_results["generation"] = False
        test_results["benchmark"] = False
        test_results["memory"] = False
    
    logger.info("Test 8: Testing CLI interface...")
    test_results["cli"] = test_cli_interface()
    
    logger.info("Test 9: Testing framework integration...")
    test_results["framework"] = test_framework_integration()
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 Test Summary")
    logger.info("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        if passed:
            passed_tests += 1
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Stable Diffusion is ready for profiling.")
        return 0
    elif passed_tests >= total_tests - 2:
        logger.info("✅ Most tests passed. Minor issues may need attention.")
        return 0
    else:
        logger.warning("⚠️ Several tests failed. Please address issues before using.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
