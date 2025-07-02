#!/usr/bin/env python3
"""
Example Usage Script for AI Inference Energy Profiling Framework.

This script demonstrates how to use the various components of the framework
for energy profiling of AI inference workloads.

Usage:
    python example_usage.py [--demo-mode]

Arguments:
    --demo-mode    Run in demonstration mode (shorter experiments)

Author: AI Inference Energy Research Team
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import gpu_config, model_config, profiling_config
    from utils import setup_logging, validate_gpu_available, validate_dcgmi_available
    from app_llama_collection.LlamaViaHF import LlamaTextGenerator
    from app_stable_diffusion_collection.StableDiffusionViaHF import StableDiffusionGenerator
    from sample_collection_scripts.profile import GPUProfiler
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def run_llama_demo(logger, profiler=None):
    """Run LLaMA text generation demo."""
    logger.info("=== LLaMA Text Generation Demo ===")
    
    try:
        # Initialize LLaMA generator
        generator = LlamaTextGenerator(logger=logger)
        
        # Run inference with profiling if available
        if profiler:
            logger.info("Running LLaMA inference with GPU profiling...")
            result = profiler.profile_command("python -c \"from app_llama_collection.LlamaViaHF import LlamaTextGenerator; gen = LlamaTextGenerator(); gen.run_default_inference()\"")
            logger.info(f"LLaMA inference completed in {result['duration']:.2f}s")
        else:
            logger.info("Running LLaMA inference without profiling...")
            texts = generator.run_default_inference()
            
            # Display results
            for i, text in enumerate(texts):
                logger.info(f"Generated text {i+1}:")
                print(text[:200] + "..." if len(text) > 200 else text)
                print("-" * 50)
        
        logger.info("LLaMA demo completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"LLaMA demo failed: {e}")
        return False


def run_stable_diffusion_demo(logger, profiler=None):
    """Run Stable Diffusion image generation demo."""
    logger.info("=== Stable Diffusion Image Generation Demo ===")
    
    try:
        # Initialize Stable Diffusion generator
        generator = StableDiffusionGenerator(logger=logger)
        
        # Run inference with profiling if available
        if profiler:
            logger.info("Running Stable Diffusion inference with GPU profiling...")
            result = profiler.profile_command("python -c \"from app_stable_diffusion_collection.StableDiffusionViaHF import StableDiffusionGenerator; gen = StableDiffusionGenerator(); gen.run_default_inference()\"")
            logger.info(f"Stable Diffusion inference completed in {result['duration']:.2f}s")
        else:
            logger.info("Running Stable Diffusion inference without profiling...")
            images = generator.run_default_inference()
            
            # Save images
            saved_files = generator.save_images(images, base_filename="demo_output")
            logger.info(f"Generated and saved {len(saved_files)} images")
            for file in saved_files:
                logger.info(f"  - {file}")
        
        logger.info("Stable Diffusion demo completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Stable Diffusion demo failed: {e}")
        return False


def run_profiling_demo(logger):
    """Run GPU profiling demo."""
    logger.info("=== GPU Profiling Demo ===")
    
    try:
        # Initialize profiler
        profiler = GPUProfiler(
            output_file="demo_profile.csv",
            interval_ms=100,  # Faster sampling for demo
            logger=logger
        )
        
        # Profile a simple command
        logger.info("Profiling a simple GPU operation...")
        result = profiler.profile_command("python -c \"import torch; x = torch.randn(1000, 1000).cuda(); y = torch.mm(x, x); print('GPU operation completed')\"")
        
        logger.info(f"Profiling completed in {result['duration']:.2f}s")
        logger.info(f"Command exit code: {result['exit_code']}")
        
        # Check if profile file was created
        if os.path.exists("demo_profile.csv"):
            logger.info("Profile data saved to: demo_profile.csv")
            
            # Show first few lines
            with open("demo_profile.csv", 'r') as f:
                lines = f.readlines()[:5]  # First 5 lines
            
            logger.info("Sample profiling data:")
            for line in lines:
                print(f"  {line.strip()}")
        
        logger.info("GPU profiling demo completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"GPU profiling demo failed: {e}")
        return False


def check_prerequisites(logger):
    """Check if all prerequisites are available."""
    logger.info("Checking prerequisites...")
    
    # Check GPU availability
    if not validate_gpu_available():
        logger.error("NVIDIA GPU not available")
        return False
    
    logger.info("✓ NVIDIA GPU available")
    
    # Check DCGMI availability
    if not validate_dcgmi_available():
        logger.warning("⚠ DCGMI not available - profiling features will be limited")
        dcgmi_available = False
    else:
        logger.info("✓ DCGMI available")
        dcgmi_available = True
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ PyTorch CUDA available (devices: {torch.cuda.device_count()})")
        else:
            logger.error("PyTorch CUDA not available")
            return False
    except ImportError:
        logger.error("PyTorch not installed")
        return False
    
    return dcgmi_available


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Example usage of AI Inference Energy Profiling Framework"
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Run in demonstration mode (shorter experiments)"
    )
    parser.add_argument(
        "--llama-only",
        action="store_true",
        help="Run only LLaMA demo"
    )
    parser.add_argument(
        "--sd-only",
        action="store_true",
        help="Run only Stable Diffusion demo"
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Run only profiling demo"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting AI Inference Energy Profiling Framework Demo")
    logger.info("=" * 60)
    
    # Check prerequisites
    dcgmi_available = check_prerequisites(logger)
    
    # Initialize profiler if available
    profiler = None
    if dcgmi_available and not (args.llama_only or args.sd_only):
        try:
            profiler = GPUProfiler(
                output_file="framework_demo.csv",
                interval_ms=profiling_config.DEFAULT_INTERVAL_MS,
                logger=logger
            )
            logger.info("GPU profiler initialized")
        except Exception as e:
            logger.warning(f"Could not initialize profiler: {e}")
    
    # Run demos based on arguments
    success_count = 0
    total_demos = 0
    
    if args.profile_only:
        total_demos = 1
        if dcgmi_available and run_profiling_demo(logger):
            success_count += 1
    elif args.llama_only:
        total_demos = 1
        if run_llama_demo(logger, profiler):
            success_count += 1
    elif args.sd_only:
        total_demos = 1
        if run_stable_diffusion_demo(logger, profiler):
            success_count += 1
    else:
        # Run all demos
        demos = [
            ("LLaMA", lambda: run_llama_demo(logger, profiler)),
            ("Stable Diffusion", lambda: run_stable_diffusion_demo(logger, profiler)),
        ]
        
        if dcgmi_available:
            demos.append(("GPU Profiling", lambda: run_profiling_demo(logger)))
        
        total_demos = len(demos)
        
        for demo_name, demo_func in demos:
            logger.info(f"\nStarting {demo_name} demo...")
            if demo_func():
                success_count += 1
            else:
                logger.error(f"{demo_name} demo failed")
            
            # Small delay between demos
            time.sleep(2)
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Demo Summary: {success_count}/{total_demos} demos completed successfully")
    
    if success_count == total_demos:
        logger.info("🎉 All demos completed successfully!")
        logger.info("The AI Inference Energy Profiling Framework is ready to use.")
    else:
        logger.warning(f"⚠ {total_demos - success_count} demo(s) failed")
        logger.info("Please check the error messages above and ensure all dependencies are properly installed.")
    
    # Cleanup
    temp_files = ["demo_profile.csv", "framework_demo.csv"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
    
    logger.info("Demo completed. Thank you for using the AI Inference Energy Profiling Framework!")


if __name__ == "__main__":
    main()
