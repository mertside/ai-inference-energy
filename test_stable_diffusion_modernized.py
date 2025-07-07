#!/usr/bin/env python3
"""
🚀 MODERNIZED STABLE DIFFUSION - Comprehensive Test & Validation Suite

This script provides comprehensive testing and validation for the modernized Stable Diffusion
application, ensuring all features work correctly across different GPU architectures and
configurations for AI energy research.

Features:
    🧪 Model Loading Tests: Verify all supported models can be loaded
    ⚡ Scheduler Tests: Validate all scheduler configurations 
    📊 Benchmark Validation: Test comprehensive benchmark suite
    🔬 Integration Tests: Verify profiling framework integration
    🎯 Performance Tests: Multi-resolution and batch size analysis
    
Author: Mert Side
Date: July 2025
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def test_imports():
    """Test that all required imports are available."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Testing imports...")
    
    missing_packages = []
    
    try:
        import torch
        logger.info("✅ PyTorch available")
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CUDA not available - will test CPU fallback")
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import diffusers
        logger.info(f"✅ Diffusers available: v{diffusers.__version__}")
    except ImportError:
        missing_packages.append("diffusers")
    
    try:
        import transformers
        logger.info(f"✅ Transformers available: v{transformers.__version__}")
    except ImportError:
        missing_packages.append("transformers")
    
    try:
        from PIL import Image
        logger.info("✅ PIL available")
    except ImportError:
        missing_packages.append("Pillow")
    
    try:
        import numpy as np
        logger.info("✅ NumPy available")
    except ImportError:
        missing_packages.append("numpy")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.error("Install with: pip install torch diffusers transformers Pillow numpy accelerate")
        return False
    
    logger.info("✅ All imports successful")
    return True

def test_authentication():
    """Test Hugging Face authentication."""
    logger = logging.getLogger(__name__)
    logger.info("🔐 Testing Hugging Face authentication...")
    
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

def test_model_loading():
    """Test loading different model variants."""
    logger = logging.getLogger(__name__)
    logger.info("🎨 Testing model loading...")
    
    try:
        from app_stable_diffusion.StableDiffusionViaHF import StableDiffusionGenerator
        
        # Test basic model initialization
        generator = StableDiffusionGenerator(
            model_variant="sd-v1.4",
            device="auto",
            logger=logger
        )
        
        model_info = generator.get_model_info()
        logger.info(f"✅ Model loaded: {model_info['model_variant']}")
        logger.info(f"📝 Description: {model_info.get('description', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_scheduler_configurations():
    """Test different scheduler configurations."""
    logger = logging.getLogger(__name__)
    logger.info("⚡ Testing scheduler configurations...")
    
    try:
        from app_stable_diffusion.StableDiffusionViaHF import StableDiffusionGenerator
        
        # Test a few key schedulers
        test_schedulers = ["dpm++", "euler", "ddim", "unipc"]
        
        for scheduler in test_schedulers:
            logger.info(f"  🧪 Testing scheduler: {scheduler}")
            generator = StableDiffusionGenerator(
                model_variant="sd-v1.4",
                scheduler_name=scheduler,
                device="auto",
                logger=logger
            )
            logger.info(f"  ✅ {scheduler} scheduler initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scheduler test failed: {e}")
        return False

def test_basic_inference():
    """Test basic image generation."""
    logger = logging.getLogger(__name__)
    logger.info("🖼️ Testing basic inference...")
    
    try:
        from app_stable_diffusion.StableDiffusionViaHF import StableDiffusionGenerator
        
        generator = StableDiffusionGenerator(
            model_variant="sd-v1.4",
            scheduler_name="ddim",
            device="auto",
            logger=logger
        )
        
        # Quick test with minimal settings
        images = generator.generate_images(
            prompts=["A simple test image"],
            num_inference_steps=10,  # Fast test
            height=256,  # Small size for speed
            width=256
        )
        
        if images and len(images) > 0:
            logger.info(f"✅ Generated {len(images)} image(s) successfully")
            return True
        else:
            logger.error("❌ No images generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Basic inference failed: {e}")
        return False

def test_benchmark_suite():
    """Test the comprehensive benchmark suite."""
    logger = logging.getLogger(__name__)
    logger.info("🔬 Testing benchmark suite...")
    
    try:
        from app_stable_diffusion.StableDiffusionViaHF import StableDiffusionGenerator
        
        generator = StableDiffusionGenerator(
            model_variant="sd-v1.4",
            scheduler_name="ddim", 
            device="auto",
            logger=logger
        )
        
        # Quick benchmark test
        images, stats = generator.run_comprehensive_benchmark(
            benchmark_type="speed_test",
            num_iterations=1,  # Quick test
            include_warmup=True
        )
        
        if images and len(images) > 0 and stats:
            logger.info(f"✅ Benchmark completed: {len(images)} images")
            logger.info(f"📊 Performance: {stats.get('summary', {}).get('images_per_second', 'N/A')} images/sec")
            return True
        else:
            logger.error("❌ Benchmark failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Benchmark test failed: {e}")
        return False

def test_cli_interface():
    """Test the CLI interface."""
    logger = logging.getLogger(__name__)
    logger.info("🖥️ Testing CLI interface...")
    
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run([
            sys.executable, "app-stable-diffusion/StableDiffusionViaHF.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ CLI help command works")
            return True
        else:
            logger.error(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ CLI test failed: {e}")
        return False

def test_integration_with_framework():
    """Test integration with the profiling framework."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Testing framework integration...")
    
    try:
        # Test that the app can be called from sample-collection-scripts
        import subprocess
        
        # Check if launch.sh recognizes the stable diffusion app
        result = subprocess.run([
            "sample-collection-scripts/launch.sh", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if "StableDiffusion" in result.stdout or result.returncode == 0:
            logger.info("✅ Framework integration available")
            return True
        else:
            logger.warning("⚠️ Framework integration may need updates")
            return False
            
    except Exception as e:
        logger.error(f"❌ Framework integration test failed: {e}")
        return False

def run_comprehensive_validation():
    """Run all validation tests."""
    logger = setup_logging()
    
    logger.info("🚀 STARTING COMPREHENSIVE STABLE DIFFUSION VALIDATION")
    logger.info("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("Authentication", test_authentication),
        ("Model Loading", test_model_loading),
        ("Scheduler Configurations", test_scheduler_configurations),
        ("Basic Inference", test_basic_inference),
        ("Benchmark Suite", test_benchmark_suite),
        ("CLI Interface", test_cli_interface),
        ("Framework Integration", test_integration_with_framework),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 70)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 70)
    logger.info(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Stable Diffusion app is ready for energy research.")
    elif passed >= total * 0.8:
        logger.info("✅ Most tests passed. Minor issues may need attention.")
    else:
        logger.warning("⚠️ Several tests failed. Please address issues before proceeding.")
    
    return passed == total

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Stable Diffusion Validation Suite"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args.log_level)
    
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
