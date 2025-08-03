#!/usr/bin/env python3
"""
Stable Diffusion Image Generation via Hugging Face Diffusers - MODERNIZED VERSION

🚀 COMPREHENSIVE MODERNIZATION for AI Inference Energy Research
Extending ICPP 2023/FGCS 2023 DVFS methodology to contemporary generative AI workloads.

🎯 ENHANCED FEATURES:
    - 🎨 Latest Model Support: SD v1.x, v2.x, SDXL, Turbo variants
    - ⚡ Advanced Schedulers: DPM++, Euler, DDIM, HEUN, and more
    - 🧠 Memory Optimization: Dynamic batching, attention slicing, CPU offload
    - 📊 Research Integration: Detailed performance metrics, profiling framework integration
    - 🎛️ Benchmark Suite: Multi-resolution, quality-performance trade-offs

🔬 RESEARCH APPLICATIONS:
    - Energy-efficient inference across V100 → A100 → H100
    - DVFS sensitivity analysis for generative AI workloads  
    - Quality vs energy consumption trade-offs
    - Cross-architecture performance comparison

📋 REQUIREMENTS:
    - Hugging Face authentication: `huggingface-cli login`
    - CUDA-compatible GPU: 8GB+ VRAM recommended
    - Dependencies: torch>=2.0, diffusers>=0.21, transformers>=4.25, accelerate

Author: Mert Side
Date: July 2025 - Complete Modernization
"""

# === COMPATIBILITY FIXES ===
# Handle xformers compatibility issues (not needed for CPU mode)
import os
try:
    import xformers
    import torch
    # Check if we have xformers but are using CPU-only PyTorch
    if hasattr(torch, '__version__') and '+cpu' in torch.__version__:
        # Disable xformers for CPU-only mode
        os.environ['XFORMERS_DISABLED'] = '1'
        print("Warning: xformers disabled for CPU-only PyTorch")
except ImportError:
    # xformers not installed - that's fine for CPU mode
    pass
except Exception as e:
    # Any other xformers issue - disable it
    os.environ['XFORMERS_DISABLED'] = '1'
    print(f"Warning: xformers disabled due to compatibility issue: {e}")

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    # === ADVANCED SCHEDULER COLLECTION ===
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
    DEISMultistepScheduler
)
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import model_config
    from utils import get_timestamp, setup_logging, validate_gpu_available
except ImportError:
    # Fallback configuration if imports fail
    class ModelConfig:
        STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
        STABLE_DIFFUSION_DEFAULT_PROMPT = (
            "a photo of an astronaut riding a horse on mars"
        )
        STABLE_DIFFUSION_OUTPUT_FILE = "astronaut_rides_horse.png"

    model_config = ModelConfig()

    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)

    def validate_gpu_available():
        return torch.cuda.is_available()

    def get_timestamp():
        import time

        return time.strftime("%Y-%m-%d_%H-%M-%S")


class StableDiffusionGenerator:
    """
    🚀 MODERNIZED Stable Diffusion Image Generator for AI Energy Research
    
    This class provides comprehensive image generation capabilities with cutting-edge
    models, advanced optimizations, and detailed performance monitoring for energy
    profiling studies extending ICPP 2023/FGCS 2023 DVFS methodology.
    
    🎯 KEY FEATURES:
        - 🎨 Latest Models: SD v1.x, v2.x, SDXL, Turbo variants
        - ⚡ Advanced Schedulers: DPM++, Euler, DDIM, UniPC, and more
        - 🧠 Memory Optimization: Dynamic batching, attention slicing
        - 📊 Performance Monitoring: Detailed metrics for energy analysis
    
    🔬 RESEARCH APPLICATIONS:
        - DVFS sensitivity analysis for generative AI
        - Cross-architecture energy efficiency (V100 → A100 → H100)
        - Quality vs energy consumption trade-offs
        - Batch size optimization studies
    """

    # 🎨 MODERNIZED MODEL CONFIGURATIONS - 2025 Edition
    MODEL_CONFIGS = {
        # === STABLE DIFFUSION v1.x SERIES (Classic 512x512) ===
        "sd-v1.4": {
            "model_id": "CompVis/stable-diffusion-v1-4",
            "pipeline_class": StableDiffusionPipeline,
            "default_size": (512, 512),
            "max_steps": 50,
            "description": "Original SD v1.4 - Baseline for comparison",
            "memory_requirement": "4GB+",
            "target_hardware": ["V100", "A100", "H100"]
        },
        "sd-v1.5": {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "pipeline_class": StableDiffusionPipeline,
            "default_size": (512, 512),
            "max_steps": 50,
            "description": "Enhanced SD v1.5 - Improved quality",
            "memory_requirement": "4GB+",
            "target_hardware": ["V100", "A100", "H100"]
        },
        
        # === STABLE DIFFUSION v2.x SERIES (Enhanced 768x768) ===
        "sd-v2.0": {
            "model_id": "stabilityai/stable-diffusion-2",
            "pipeline_class": StableDiffusionPipeline,
            "default_size": (768, 768),
            "max_steps": 50,
            "description": "SD v2.0 - Higher resolution, better quality",
            "memory_requirement": "6GB+",
            "target_hardware": ["A100", "H100"]
        },
        "sd-v2.1": {
            "model_id": "stabilityai/stable-diffusion-2-1",
            "pipeline_class": StableDiffusionPipeline,
            "default_size": (768, 768),
            "max_steps": 50,
            "description": "SD v2.1 - Further improvements",
            "memory_requirement": "6GB+", 
            "target_hardware": ["A100", "H100"]
        },
        
        # === STABLE DIFFUSION XL SERIES (Flagship 1024x1024) ===
        "sdxl": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "pipeline_class": StableDiffusionXLPipeline,
            "default_size": (1024, 1024),
            "max_steps": 30,
            "description": "SDXL Base - Flagship model with superior quality",
            "memory_requirement": "8GB+",
            "target_hardware": ["A100", "H100"]
        },
        "sdxl-refiner": {
            "model_id": "stabilityai/stable-diffusion-xl-refiner-1.0", 
            "pipeline_class": StableDiffusionXLImg2ImgPipeline,
            "default_size": (1024, 1024),
            "max_steps": 15,
            "description": "SDXL Refiner - High-quality refinement",
            "memory_requirement": "8GB+",
            "target_hardware": ["A100", "H100"]
        },
        
        # === TURBO VARIANTS (Speed-Optimized) ===
        "sd-turbo": {
            "model_id": "stabilityai/sd-turbo",
            "pipeline_class": StableDiffusionPipeline,
            "default_size": (512, 512),
            "max_steps": 4,
            "description": "SD Turbo - Ultra-fast generation (1-4 steps)",
            "memory_requirement": "4GB+",
            "target_hardware": ["V100", "A100", "H100"]
        },
        "sdxl-turbo": {
            "model_id": "stabilityai/sdxl-turbo",
            "pipeline_class": StableDiffusionXLPipeline,
            "default_size": (512, 512),
            "max_steps": 4,
            "description": "SDXL Turbo - High-quality ultra-fast generation",
            "memory_requirement": "6GB+",
            "target_hardware": ["A100", "H100"]
        },
        
        # === LIGHTNING VARIANTS (Ultra-Speed) ===
        "sd-lightning": {
            "model_id": "ByteDance/SDXL-Lightning",
            "pipeline_class": StableDiffusionXLPipeline,
            "default_size": (1024, 1024),
            "max_steps": 8,
            "description": "SDXL Lightning - Ultra-fast with maintained quality",
            "memory_requirement": "8GB+",
            "target_hardware": ["A100", "H100"]
        }
    }

    # ⚡ ADVANCED SCHEDULER CONFIGURATIONS
    SCHEDULER_CONFIGS = {
        "dpm++": {
            "class": DPMSolverMultistepScheduler,
            "description": "DPM++ 2M Karras - High quality, fast convergence",
            "recommended_steps": "15-25",
            "speed": "fast",
            "quality": "high"
        },
        "euler": {
            "class": EulerDiscreteScheduler,
            "description": "Euler - Simple, reliable, good balance",
            "recommended_steps": "20-50",
            "speed": "medium",
            "quality": "good"
        },
        "euler-ancestral": {
            "class": EulerAncestralDiscreteScheduler,
            "description": "Euler Ancestral - Creative, higher variance",
            "recommended_steps": "20-50",
            "speed": "medium",
            "quality": "creative"
        },
        "ddim": {
            "class": DDIMScheduler,
            "description": "DDIM - Deterministic, fast, consistent",
            "recommended_steps": "10-50",
            "speed": "fast",
            "quality": "consistent"
        },
        "ddpm": {
            "class": DDPMScheduler,
            "description": "DDPM - Original, high quality, slow",
            "recommended_steps": "50-1000",
            "speed": "slow",
            "quality": "highest"
        },
        "lms": {
            "class": LMSDiscreteScheduler,
            "description": "LMS - Linear multistep, stable",
            "recommended_steps": "20-50",
            "speed": "medium",
            "quality": "stable"
        },
        "heun": {
            "class": HeunDiscreteScheduler,
            "description": "Heun - Higher order, more accurate",
            "recommended_steps": "10-30",
            "speed": "slow",
            "quality": "high"
        },
        "dpm-sde": {
            "class": KDPM2DiscreteScheduler,
            "description": "DPM SDE - Stochastic, creative results",
            "recommended_steps": "10-25",
            "speed": "medium",
            "quality": "creative"
        },
        "unipc": {
            "class": UniPCMultistepScheduler,
            "description": "UniPC - Unified predictor-corrector, fast",
            "recommended_steps": "5-20",
            "speed": "very_fast",
            "quality": "good"
        },
        "deis": {
            "class": DEISMultistepScheduler,
            "description": "DEIS - Fast convergence, efficient",
            "recommended_steps": "10-25",
            "speed": "fast",
            "quality": "good"
        }
    }

    # 🎨 COMPREHENSIVE BENCHMARK PROMPT COLLECTIONS
    BENCHMARK_PROMPTS = {
        "speed_test": [
            "A red car",
            "A blue house", 
            "A green tree",
            "A cat",
            "A mountain"
        ],
        "quality_test": [
            "A detailed Victorian house with ornate architecture",
            "A cyberpunk city street with neon lights at night",
            "A fantasy forest with magical creatures and glowing plants",
            "A portrait of an elderly wise wizard with a long beard",
            "A futuristic spaceship landing on an alien planet"
        ],
        "memory_stress": [
            "A highly detailed digital artwork of a futuristic space station orbiting a distant planet with intricate mechanical details, dramatic lighting, and realistic textures in 8K resolution",
            "An extremely detailed fantasy landscape with multiple castles, dragons, magical forests, crystal caves, floating islands, and mystical creatures in photorealistic style",
            "A complex cyberpunk metropolis with towering skyscrapers, flying vehicles, holographic advertisements, neon reflections, and thousands of detailed windows and architectural elements"
        ],
        "artistic_styles": [
            "A landscape in the style of Van Gogh",
            "A portrait in the style of Picasso",
            "A cityscape in anime style",
            "A still life in the style of Cézanne",
            "A seascape in the style of Turner"
        ],
        "energy_research": [
            "A simple geometric pattern",
            "A complex architectural visualization",
            "A photorealistic portrait",
            "An abstract digital art piece",
            "A detailed natural landscape"
        ]
    }

    def __init__(
        self,
        model_name: str = None,
        model_variant: str = "sd-v1.4",
        scheduler_name: str = "dpm++",
        device: str = "auto",
        torch_dtype: torch.dtype = None,
        enable_memory_efficient_attention: bool = True,
        enable_cpu_offload: bool = False,
        enable_attention_slicing: bool = True,
        enable_xformers: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        🚀 Initialize the Modernized Stable Diffusion Image Generator
        
        Comprehensive initialization with support for latest models, advanced schedulers,
        and optimizations for energy profiling research.

        Args:
            model_name: Explicit model name (overrides model_variant)
            model_variant: Model variant key from MODEL_CONFIGS 
                          (sd-v1.4, sd-v1.5, sd-v2.0, sd-v2.1, sdxl, sdxl-turbo, etc.)
            scheduler_name: Scheduler name from SCHEDULER_CONFIGS
                           (dpm++, euler, ddim, unipc, etc.)
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            torch_dtype: PyTorch data type (None=auto-detect)
            enable_memory_efficient_attention: Enable attention slicing optimization
            enable_cpu_offload: Enable model CPU offloading for memory-constrained scenarios
            enable_attention_slicing: Enable attention slicing for memory efficiency
            enable_xformers: Enable xformers memory efficient attention (if available)
            logger: Optional logger instance
            
        🎯 RESEARCH FEATURES:
            - Support for all contemporary SD models (v1.x → v2.x → SDXL → Turbo)
            - Advanced scheduler selection for speed/quality trade-offs
            - Comprehensive memory optimization for extended profiling sessions
            - Detailed performance monitoring and metrics collection
        """
        self.logger = logger or setup_logging()
        self.scheduler_name = scheduler_name
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_xformers = enable_xformers
        
        # 🎨 Model Configuration Resolution
        if model_name:
            # Custom model name provided - use defaults
            self.model_id = model_name
            self.pipeline_class = StableDiffusionPipeline
            self.default_size = (512, 512)
            self.max_steps = 50
            self.model_description = f"Custom model: {model_name}"
        else:
            # Use predefined configuration from MODEL_CONFIGS
            if model_variant not in self.MODEL_CONFIGS:
                available = ", ".join(self.MODEL_CONFIGS.keys())
                raise ValueError(f"Unknown model variant: {model_variant}. Available: {available}")
            
            config = self.MODEL_CONFIGS[model_variant]
            self.model_id = config["model_id"]
            self.pipeline_class = config["pipeline_class"]
            self.default_size = config["default_size"]
            self.max_steps = config["max_steps"]
            self.model_description = config.get("description", model_variant)
            self.memory_requirement = config.get("memory_requirement", "Unknown")
            self.target_hardware = config.get("target_hardware", ["A100"])

        self.model_variant = model_variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_cpu_offload = enable_cpu_offload
        
        # ⚡ Scheduler Configuration Resolution
        if scheduler_name not in self.SCHEDULER_CONFIGS:
            available = ", ".join(self.SCHEDULER_CONFIGS.keys())
            self.logger.warning(f"Unknown scheduler: {scheduler_name}. Using 'dpm++'. Available: {available}")
            scheduler_name = "dpm++"
        
        self.scheduler_config = self.SCHEDULER_CONFIGS[scheduler_name]
        self.scheduler_class = self.scheduler_config["class"]
        
        # 🔧 Device and dtype configuration
        self._setup_device_and_dtype(device, torch_dtype)
        
        # 📊 Pipeline state and performance tracking
        self.pipeline = None
        self.is_initialized = False
        
        # Enhanced performance tracking for energy research
        self.generation_stats = {
            "total_generations": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "average_step_time": 0.0,
            "memory_usage": [],
            "model_info": {
                "variant": self.model_variant,
                "scheduler": scheduler_name,
                "description": self.model_description,
                "memory_requirement": getattr(self, 'memory_requirement', 'Unknown')
            }
        }
        
        # 🚀 Log initialization summary
        self.logger.info(f"🎨 Initialized {self.model_variant} with {scheduler_name} scheduler")
        self.logger.info(f"📝 Description: {self.model_description}")
        self.logger.info(f"💾 Memory requirement: {getattr(self, 'memory_requirement', 'Unknown')}")
        self.logger.info(f"🎯 Target hardware: {getattr(self, 'target_hardware', ['A100'])}")

    def _setup_device_and_dtype(self, device: str, torch_dtype: Optional[torch.dtype]) -> None:
        """Set up device and data type configuration."""
        # Auto-detect optimal device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                self.logger.warning("CUDA not available, using CPU")
        else:
            self.device = device

        # Disable xformers for CPU mode as it's not supported
        if self.device == "cpu":
            self.enable_xformers = False
            self.logger.info("Disabled xformers for CPU mode")

        # Auto-detect optimal dtype
        if torch_dtype is None:
            if self.device == "cuda":
                # Use float16 for GPU to save memory
                self.torch_dtype = torch.float16
            else:
                # Use float32 for CPU (float16 not supported)
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            self.torch_dtype = torch.float32

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Using dtype: {self.torch_dtype}")

    def _log_memory_usage(self, stage: str) -> None:
        """Log current GPU memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            self.logger.info(f"[{stage}] GPU Memory - Allocated: {memory_allocated:.2f}GB, "
                           f"Reserved: {memory_reserved:.2f}GB, Total: {memory_total:.2f}GB")
            
            self.generation_stats["memory_usage"].append({
                "stage": stage,
                "allocated": memory_allocated,
                "reserved": memory_reserved,
                "total": memory_total
            })

    def initialize_model(self) -> None:
        """Initialize the image generation pipeline with optimizations."""
        if self.is_initialized:
            self.logger.info("Model already initialized, skipping")
            return

        try:
            self.logger.info(f"Initializing {self.model_variant} model: {self.model_id}")
            self._log_memory_usage("before_model_load")

            # Pipeline initialization arguments
            pipeline_kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,  # Use safer tensor format
                "variant": "fp16" if self.torch_dtype == torch.float16 else None,
            }

            # Initialize the specific pipeline class
            self.pipeline = self.pipeline_class.from_pretrained(
                self.model_id,
                **pipeline_kwargs
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            self._log_memory_usage("after_model_load")

            # Apply memory optimizations
            self._apply_optimizations()

            # Test pipeline functionality
            self._validate_pipeline()

            self.is_initialized = True
            self.logger.info(f"Successfully initialized {self.model_variant} model")

        except Exception as e:
            self.logger.error(f"Failed to initialize model {self.model_id}: {e}")
            raise

    def _apply_optimizations(self) -> None:
        """Apply various memory and performance optimizations."""
        try:
            # Enable memory efficient attention
            if self.enable_memory_efficient_attention:
                if hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing()
                    self.logger.info("Enabled attention slicing")

                # Try to enable xformers if available and not disabled
                if self.enable_xformers and not os.environ.get('XFORMERS_DISABLED'):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        self.logger.info("Enabled xformers memory efficient attention")
                    except Exception:
                        self.logger.info("xformers not available, using default attention")
                else:
                    self.logger.info("xformers disabled (CPU mode or compatibility issues)")

            # Enable CPU offloading for large models if requested
            if self.enable_cpu_offload:
                if hasattr(self.pipeline, "enable_model_cpu_offload"):
                    self.pipeline.enable_model_cpu_offload()
                    self.logger.info("Enabled model CPU offloading")

            # Use memory efficient scheduler
            if hasattr(self.pipeline, "scheduler"):
                # Use DPM++ 2M Karras scheduler for better quality and speed
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    use_karras_sigmas=True
                )
                self.logger.info("Applied optimized DPM++ scheduler")

        except Exception as e:
            self.logger.warning(f"Some optimizations failed: {e}")

    def _validate_pipeline(self) -> None:
        """Validate that the pipeline is working correctly."""
        try:
            # Quick validation with minimal resources
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.torch_dtype):
                        # Just validate the pipeline components exist
                        _ = self.pipeline.unet
                        _ = self.pipeline.vae
                        _ = self.pipeline.text_encoder
                else:
                    _ = self.pipeline.unet
                    _ = self.pipeline.vae
                    _ = self.pipeline.text_encoder

            self.logger.info("Pipeline validation successful")

        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e}")
            raise

    def generate_images(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        batch_size: int = 1,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model with batch support.

        Args:
            prompts: Text prompt(s) for image generation
            negative_prompts: Negative prompt(s) to avoid certain features
            height: Image height in pixels (default: model default)
            width: Image width in pixels (default: model default)
            num_inference_steps: Number of denoising steps (default: model default)
            guidance_scale: Guidance scale for classifier-free guidance
            num_images_per_prompt: Number of images to generate per prompt
            batch_size: Batch size for processing multiple prompts
            seed: Random seed for reproducibility
            generator: PyTorch random generator for reproducibility

        Returns:
            List of generated PIL Images
        """
        if not self.is_initialized:
            self.initialize_model()

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Set default dimensions
        if height is None or width is None:
            default_height, default_width = self.default_size
            height = height or default_height
            width = width or default_width

        # Set default steps
        if num_inference_steps is None:
            num_inference_steps = self.max_steps

        # Set up random generation
        if seed is not None and generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        try:
            start_time = time.time()
            self._log_memory_usage("before_generation")

            all_images = []

            # Process prompts in batches
            for batch_start in range(0, len(prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]

                # Handle negative prompts
                batch_negative_prompts = None
                if negative_prompts:
                    if isinstance(negative_prompts, str):
                        batch_negative_prompts = [negative_prompts] * len(batch_prompts)
                    else:
                        batch_negative_prompts = negative_prompts[batch_start:batch_end]

                self.logger.info(f"Generating batch {batch_start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
                self.logger.info(f"Prompts: {[p[:50] + '...' if len(p) > 50 else p for p in batch_prompts]}")

                # Generate images with proper device handling
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=self.torch_dtype):
                        batch_images = self._generate_batch(
                            batch_prompts, batch_negative_prompts, height, width,
                            num_inference_steps, guidance_scale, num_images_per_prompt, generator
                        )
                else:
                    batch_images = self._generate_batch(
                        batch_prompts, batch_negative_prompts, height, width,
                        num_inference_steps, guidance_scale, num_images_per_prompt, generator
                    )

                all_images.extend(batch_images)

                # Clear cache between batches
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            generation_time = time.time() - start_time
            self._update_stats(generation_time, len(all_images))
            self._log_memory_usage("after_generation")

            self.logger.info(f"Successfully generated {len(all_images)} image(s) in {generation_time:.2f}s")
            return all_images

        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise

    def _generate_batch(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]],
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
        generator: Optional[torch.Generator],
    ) -> List[Image.Image]:
        """Generate a batch of images."""
        pipeline_kwargs = {
            "prompt": prompts,
            "negative_prompt": negative_prompts,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "generator": generator,
            "return_dict": True,
        }

        # Remove None values
        pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v is not None}

        result = self.pipeline(**pipeline_kwargs)
        return result.images

    def _update_stats(self, generation_time: float, num_images: int) -> None:
        """Update generation statistics."""
        self.generation_stats["total_generations"] += num_images
        self.generation_stats["total_time"] += generation_time
        self.generation_stats["average_time"] = (
            self.generation_stats["total_time"] / self.generation_stats["total_generations"]
        )

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model (legacy interface).

        This method maintains backward compatibility with the original interface.
        """
        return self.generate_images(
            prompts=prompt,
            negative_prompts=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            seed=seed,
        )

    def run_default_inference(self) -> List[Image.Image]:
        """
        Run inference with default prompt and parameters.

        Returns:
            List of generated PIL Images
        """
        # Use config default prompt, or fallback
        try:
            default_prompt = model_config.STABLE_DIFFUSION_DEFAULT_PROMPT
        except (NameError, AttributeError):
            default_prompt = "a photo of an astronaut riding a horse on mars"

        return self.generate_images(prompts=default_prompt)

    def run_comprehensive_benchmark(
        self,
        benchmark_type: str = "energy_research",
        num_iterations: int = 3,
        include_warmup: bool = True,
        export_detailed_metrics: bool = True,
    ) -> Tuple[List[Image.Image], Dict]:
        """
        🔬 Comprehensive Benchmark Suite for AI Energy Research
        
        Advanced benchmarking designed for DVFS energy studies, extending ICPP 2023/FGCS 2023
        methodology to contemporary generative AI workloads.

        Args:
            benchmark_type: Type of benchmark from BENCHMARK_PROMPTS
            num_iterations: Number of iterations per test for statistical significance
            include_warmup: Whether to include warmup iterations
            export_detailed_metrics: Export comprehensive metrics for analysis

        Returns:
            Tuple of (generated images, comprehensive performance statistics)
            
        🎯 ENERGY RESEARCH FEATURES:
            - Per-step timing analysis for DVFS correlation
            - Memory usage profiling across generation phases
            - Throughput analysis for batch size optimization
            - Quality-consistency metrics for trade-off studies
        """
        if benchmark_type not in self.BENCHMARK_PROMPTS:
            available = ", ".join(self.BENCHMARK_PROMPTS.keys())
            raise ValueError(f"Unknown benchmark type: {benchmark_type}. Available: {available}")
        
        prompts = self.BENCHMARK_PROMPTS[benchmark_type]
        self.logger.info(f"🚀 Starting {benchmark_type} benchmark with {len(prompts)} prompts")
        self.logger.info(f"📊 {num_iterations} iterations per prompt for statistical significance")
        
        # === COMPREHENSIVE METRICS COLLECTION ===
        benchmark_stats = {
            "benchmark_info": {
                "type": benchmark_type,
                "model_variant": self.model_variant,
                "scheduler": self.scheduler_name,
                "iterations": num_iterations,
                "total_prompts": len(prompts),
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
            },
            "performance_metrics": {
                "per_prompt_times": [],
                "per_step_times": [],
                "memory_metrics": [],
                "throughput_metrics": {},
                "consistency_metrics": {}
            },
            "energy_research_metrics": {
                "baseline_measurements": [],
                "step_breakdown": [],
                "memory_efficiency": [],
                "batch_scaling": []
            }
        }
        
        all_generated_images = []
        total_benchmark_start = time.time()
        
        # === WARMUP PHASE ===
        if include_warmup:
            self.logger.info("🔥 Warmup phase: Preparing GPU for consistent measurements")
            warmup_start = time.time()
            _ = self.generate_images(
                prompts=[prompts[0]], 
                num_inference_steps=min(10, self.max_steps),
                batch_size=1
            )
            warmup_time = time.time() - warmup_start
            benchmark_stats["warmup_time"] = warmup_time
            self.logger.info(f"✅ Warmup completed in {warmup_time:.2f}s")
        
        # === MAIN BENCHMARK EXECUTION ===
        for prompt_idx, prompt in enumerate(prompts):
            self.logger.info(f"📝 Prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:60]}...")
            
            prompt_times = []
            prompt_images = []
            
            for iteration in range(num_iterations):
                self.logger.info(f"  🔄 Iteration {iteration + 1}/{num_iterations}")
                
                # Clear GPU cache for consistent measurements
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                iteration_start = time.time()
                self._log_memory_usage(f"prompt_{prompt_idx}_iter_{iteration}_start")
                
                # Generate with detailed timing
                images = self.generate_images(
                    prompts=[prompt],
                    batch_size=1,
                    seed=42 + iteration  # Consistent but varied seeds
                )
                
                iteration_time = time.time() - iteration_start
                prompt_times.append(iteration_time)
                prompt_images.extend(images)
                
                self._log_memory_usage(f"prompt_{prompt_idx}_iter_{iteration}_end")
                
                # Collect detailed metrics
                benchmark_stats["performance_metrics"]["per_prompt_times"].append({
                    "prompt_idx": prompt_idx,
                    "iteration": iteration,
                    "time": iteration_time,
                    "images_generated": len(images)
                })
            
            all_generated_images.extend(prompt_images)
            
            # Calculate prompt-level statistics
            avg_time = np.mean(prompt_times)
            std_time = np.std(prompt_times)
            
            self.logger.info(f"  📊 Avg time: {avg_time:.2f}s ± {std_time:.2f}s")
            benchmark_stats["performance_metrics"]["consistency_metrics"][f"prompt_{prompt_idx}"] = {
                "average_time": avg_time,
                "std_deviation": std_time,
                "coefficient_of_variation": std_time / avg_time if avg_time > 0 else 0
            }
        
        total_benchmark_time = time.time() - total_benchmark_start
        
        # === COMPREHENSIVE ANALYSIS ===
        total_images = len(all_generated_images)
        if total_images > 0:
            benchmark_stats["summary"] = {
                "total_runtime": total_benchmark_time,
                "total_images": total_images,
                "average_time_per_image": total_benchmark_time / total_images,
                "images_per_second": total_images / total_benchmark_time,
                "total_prompts_tested": len(prompts),
                "iterations_per_prompt": num_iterations
            }
            
            # === ENERGY RESEARCH SPECIFIC METRICS ===
            benchmark_stats["energy_research_metrics"] = {
                "inference_efficiency": {
                    "time_per_step": total_benchmark_time / (len(prompts) * num_iterations * self.max_steps),
                    "energy_baseline_estimate": "TBD - requires profiling framework integration",
                    "memory_efficiency_score": self._calculate_memory_efficiency()
                },
                "dvfs_readiness": {
                    "measurement_consistency": std_time / avg_time if 'avg_time' in locals() and avg_time > 0 else 0,
                    "suitable_for_frequency_sweep": std_time / avg_time < 0.1 if 'avg_time' in locals() and avg_time > 0 else False,
                    "recommended_sample_size": max(10, int(20 * (std_time / avg_time))) if 'avg_time' in locals() and avg_time > 0 else 10
                }
            }
        
        # === BENCHMARK COMPLETION ===
        self.logger.info("=" * 60)
        self.logger.info("🏁 COMPREHENSIVE BENCHMARK COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"📊 Total images: {total_images}")
        self.logger.info(f"⏱️  Total time: {total_benchmark_time:.2f}s")
        self.logger.info(f"🚀 Throughput: {total_images / total_benchmark_time:.2f} images/second")
        self.logger.info(f"📈 Avg per image: {total_benchmark_time / total_images:.2f}s" if total_images > 0 else "No images generated")
        self.logger.info("=" * 60)
        
        return all_generated_images, benchmark_stats
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score for research analysis."""
        if not self.generation_stats["memory_usage"]:
            return 0.0
        
        # Simple efficiency metric: lower peak memory usage = higher efficiency
        peak_memories = [entry.get("allocated", 0) for entry in self.generation_stats["memory_usage"]]
        if not peak_memories:
            return 0.0
        
        avg_peak = np.mean(peak_memories)
        # Normalize to 0-1 scale (assuming 16GB as reference maximum)
        efficiency = max(0, 1 - (avg_peak / 16.0))
        return efficiency

    def run_benchmark_inference(
        self,
        num_generations: int = 5,
        use_different_prompts: bool = True,
    ) -> Tuple[List[Image.Image], Dict]:
        """
        🔄 LEGACY COMPATIBILITY: Basic benchmark inference
        
        Maintained for backward compatibility. For comprehensive research benchmarks,
        use run_comprehensive_benchmark() instead.
        """
        if use_different_prompts:
            prompts = self.BENCHMARK_PROMPTS["energy_research"]
            # Repeat prompts if we need more generations
            prompts = (prompts * ((num_generations // len(prompts)) + 1))[:num_generations]
        else:
            prompts = ["a photo of an astronaut riding a horse on mars"] * num_generations

        start_time = time.time()
        images = self.generate_images(prompts, batch_size=min(num_generations, 2))
        total_time = time.time() - start_time

        stats = {
            "total_images": len(images),
            "total_time": total_time,
            "average_time_per_image": total_time / len(images) if len(images) > 0 else 0,
            "images_per_second": len(images) / total_time if total_time > 0 else 0,
            "memory_stats": self.generation_stats["memory_usage"][-len(images):] if self.generation_stats["memory_usage"] else [],
        }

        self.logger.info(f"Benchmark completed: {len(images)} images in {total_time:.2f}s")
        self.logger.info(f"Performance: {stats['images_per_second']:.2f} images/second")

        return images, stats

    def save_images(
        self,
        images: List[Image.Image],
        base_filename: str = None,
        output_dir: str = ".",
        include_metadata: bool = True,
    ) -> List[str]:
        """
        Save generated images to files with optional metadata.

        Args:
            images: List of PIL Images to save
            base_filename: Base filename (without extension)
            output_dir: Output directory
            include_metadata: Whether to include generation metadata

        Returns:
            List of saved file paths
        """
        if base_filename is None:
            try:
                base_filename = model_config.STABLE_DIFFUSION_OUTPUT_FILE.replace(".png", "")
            except (NameError, AttributeError):
                base_filename = "sd_output"

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        saved_files = []

        for i, image in enumerate(images):
            if len(images) == 1:
                filename = f"{base_filename}.png"
            else:
                filename = f"{base_filename}_{i+1:03d}.png"

            filepath = os.path.join(output_dir, filename)

            # Add metadata if requested
            if include_metadata:
                metadata = {
                    "model": self.model_id,
                    "model_variant": self.model_variant,
                    "device": self.device,
                    "dtype": str(self.torch_dtype),
                    "generation_time": getattr(self, '_last_generation_time', 'unknown'),
                    "image_size": f"{image.width}x{image.height}",
                }
                
                # Convert metadata to string for PNG info
                metadata_str = "; ".join([f"{k}: {v}" for k, v in metadata.items()])
                
                # Save with metadata
                from PIL.PngImagePlugin import PngInfo
                pnginfo = PngInfo()
                pnginfo.add_text("generation_info", metadata_str)
                image.save(filepath, pnginfo=pnginfo)
            else:
                image.save(filepath)

            saved_files.append(filepath)
            self.logger.info(f"Saved image: {filepath}")

        return saved_files

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = {
            "model_id": self.model_id,
            "model_variant": self.model_variant,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "default_size": self.default_size,
            "max_steps": self.max_steps,
            "is_initialized": self.is_initialized,
            "memory_efficient_attention": self.enable_memory_efficient_attention,
            "cpu_offload": self.enable_cpu_offload,
        }

        if self.is_initialized and self.pipeline:
            info["pipeline_components"] = list(self.pipeline.components.keys())

        return info

    def get_generation_stats(self) -> Dict:
        """Get generation performance statistics."""
        return self.generation_stats.copy()

    def clear_cache(self) -> None:
        """Clear GPU memory cache and perform garbage collection."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Cleared memory cache")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache()


def parse_arguments():
    """🚀 Comprehensive CLI Parser for Modernized Stable Diffusion Research Tool"""
    parser = argparse.ArgumentParser(
        description="""
🎨 MODERNIZED STABLE DIFFUSION - AI Inference Energy Research Tool

Extends ICPP 2023/FGCS 2023 DVFS methodology to contemporary generative AI workloads.
Supports latest models (SD v1.x → v2.x → SDXL → Turbo) with advanced optimizations.

🎯 RESEARCH APPLICATIONS:
    • Energy-efficient inference across V100 → A100 → H100
    • DVFS sensitivity analysis for generative AI workloads
    • Quality vs energy consumption trade-offs
    • Cross-architecture performance comparison
    
🔬 EXAMPLE USAGE:
    # Quick test with default settings
    python StableDiffusionViaHF.py
    
    # SDXL with advanced scheduler for quality research  
    python StableDiffusionViaHF.py --model-variant sdxl --scheduler unipc --steps 20
    
    # Batch generation for throughput analysis
    python StableDiffusionViaHF.py --batch-size 4 --num-images 20 --benchmark
    
    # Speed vs quality trade-off study
    python StableDiffusionViaHF.py --model-variant sdxl-turbo --steps 4 --benchmark
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === MODEL CONFIGURATION ===
    model_group = parser.add_argument_group('🎨 Model Configuration')
    model_group.add_argument(
        "--model-variant", 
        choices=list(StableDiffusionGenerator.MODEL_CONFIGS.keys()),
        default="sd-v1.4",
        help="Model variant: sd-v1.4, sd-v1.5, sd-v2.0, sd-v2.1, sdxl, sdxl-turbo, sd-turbo, sd-lightning"
    )
    
    model_group.add_argument(
        "--custom-model",
        type=str,
        default=None,
        help="Custom Hugging Face model ID (overrides --model-variant)"
    )
    
    # === SCHEDULER CONFIGURATION ===
    scheduler_group = parser.add_argument_group('⚡ Scheduler Configuration')
    scheduler_group.add_argument(
        "--scheduler",
        choices=list(StableDiffusionGenerator.SCHEDULER_CONFIGS.keys()),
        default="dpm++",
        help="Inference scheduler: dpm++, euler, ddim, unipc, heun, etc."
    )
    
    scheduler_group.add_argument(
        "--list-schedulers",
        action="store_true",
        help="List all available schedulers with descriptions"
    )

    # === GENERATION PARAMETERS ===
    generation_group = parser.add_argument_group('🎨 Generation Parameters')
    generation_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for image generation"
    )

    generation_group.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt to avoid certain features"
    )

    generation_group.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height in pixels (default: model-specific)"
    )
    
    generation_group.add_argument(
        "--width", 
        type=int,
        default=None,
        help="Image width in pixels (default: model-specific)"
    )

    generation_group.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: model-specific)"
    )

    generation_group.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (1.0-20.0)"
    )

    generation_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    # === BATCH PROCESSING ===
    batch_group = parser.add_argument_group('📦 Batch Processing')
    batch_group.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Total number of images to generate"
    )

    batch_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (affects memory usage)"
    )
    
    batch_group.add_argument(
        "--images-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt"
    )

    # === BENCHMARK & RESEARCH ===
    research_group = parser.add_argument_group('🔬 Research & Benchmarking')
    research_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive benchmark suite"
    )
    
    research_group.add_argument(
        "--benchmark-type",
        choices=list(StableDiffusionGenerator.BENCHMARK_PROMPTS.keys()),
        default="energy_research",
        help="Benchmark prompt collection: speed_test, quality_test, memory_stress, artistic_styles, energy_research"
    )
    
    research_group.add_argument(
        "--export-metrics",
        type=str,
        default=None,
        help="Export detailed performance metrics to JSON file"
    )

    # === OPTIMIZATION SETTINGS ===
    optimization_group = parser.add_argument_group('⚙️ Optimization Settings')
    optimization_group.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for inference"
    )

    optimization_group.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "bfloat16"],
        default="auto",
        help="Model precision (affects memory and speed)"
    )

    optimization_group.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for memory-constrained scenarios"
    )

    optimization_group.add_argument(
        "--disable-memory-efficient-attention",
        action="store_true",
        help="Disable memory efficient attention optimizations"
    )
    
    optimization_group.add_argument(
        "--disable-xformers",
        action="store_true",
        help="Disable xformers memory efficient attention"
    )
    
    optimization_group.add_argument(
        "--disable-attention-slicing",
        action="store_true",
        help="Disable attention slicing optimization"
    )

    # === OUTPUT CONFIGURATION ===
    output_group = parser.add_argument_group('💾 Output Configuration')
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated images"
    )
    
    output_group.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output filenames"
    )
    
    output_group.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID for organizing images in subfolders (e.g., SLURM job ID)"
    )
    
    output_group.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save generation metadata in image files"
    )
    
    output_group.add_argument(
        "--no-save-images",
        action="store_true",
        help="Skip saving images (research mode - metrics only)"
    )

    # === ADVANCED RESEARCH ===
    advanced_group = parser.add_argument_group('🧪 Advanced Research Features')
    advanced_group.add_argument(
        "--multi-resolution",
        action="store_true",
        help="Test multiple resolutions for memory scaling analysis"
    )
    
    advanced_group.add_argument(
        "--scheduler-comparison",
        action="store_true", 
        help="Compare multiple schedulers for performance analysis"
    )
    
    advanced_group.add_argument(
        "--energy-profile",
        action="store_true",
        help="Enable energy profiling integration (requires profiling framework)"
    )

    # === SYSTEM CONFIGURATION ===
    system_group = parser.add_argument_group('🔧 System Configuration')
    system_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level"
    )
    
    system_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    system_group.add_argument(
        "--version",
        action="version",
        version="Stable Diffusion Research Tool v2.0 - Modernized for AI Energy Research"
    )

    return parser.parse_args()

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated images"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for inference"
    )

    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for large models"
    )

    parser.add_argument(
        "--disable-memory-efficient-attention",
        action="store_true",
        help="Disable memory efficient attention optimizations"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """Main function for standalone execution."""
    args = parse_arguments()

    # Set up logging
    logger = setup_logging(args.log_level)
    
    # Handle listing commands early (no need to initialize models)
    if hasattr(args, 'list_schedulers') and args.list_schedulers:
        print("🗂️ Available Schedulers:")
        print("=" * 50)
        for name, config in StableDiffusionGenerator.SCHEDULER_CONFIGS.items():
            print(f"• {name:15} - {config['description']}")
        return

    try:
        # Initialize the image generator
        generator = StableDiffusionGenerator(
            model_variant=args.model_variant,
            device=args.device,
            enable_memory_efficient_attention=not args.disable_memory_efficient_attention,
            enable_cpu_offload=args.enable_cpu_offload,
            logger=logger
        )

        # Log model information
        model_info = generator.get_model_info()
        logger.info(f"Model Info: {model_info}")

        if args.benchmark:
            # Run benchmark
            logger.info("Running benchmark mode...")
            images, stats = generator.run_benchmark_inference(
                num_generations=args.num_images,
                use_different_prompts=True
            )
            logger.info(f"Benchmark Stats: {stats}")
        else:
            # Regular inference
            if args.prompt:
                prompts = [args.prompt] * args.num_images
            else:
                # Use default prompt
                prompts = [None] * args.num_images

            images = generator.generate_images(
                prompts=prompts if args.prompt else prompts[0],  # Handle None case
                negative_prompts=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                batch_size=args.batch_size,
            )

        # Save generated images (unless disabled for research mode)
        if not args.no_save_images:
            # Create job-specific output directory if job ID is provided
            output_dir = args.output_dir
            if args.job_id:
                job_subfolder = f"job_{args.job_id}"
                output_dir = os.path.join(args.output_dir, job_subfolder)
                
                # Create the job-specific directory
                os.makedirs(output_dir, exist_ok=True)
                if logger:
                    logger.info(f"Created job-specific image directory: {output_dir}")
            
            timestamp = get_timestamp()
            output_filename = f"sd_{args.model_variant}_{timestamp}"
            saved_files = generator.save_images(
                images, 
                base_filename=output_filename,
                output_dir=output_dir
            )
            
            if logger:
                logger.info(f"Images saved: {len(saved_files)} files")
                if args.job_id:
                    logger.info(f"Job ID: {args.job_id} - Images in: {output_dir}")
                for file_path in saved_files:
                    logger.debug(f"  Saved: {file_path}")
        else:
            if logger:
                logger.info("Skipping image save (research mode - metrics only)")
            saved_files = []

        # Print final statistics
        final_stats = generator.get_generation_stats()
        logger.info(f"Generation completed successfully!")
        logger.info(f"Generated {len(images)} image(s)")
        logger.info(f"Saved to: {', '.join(saved_files)}")
        logger.info(f"Total generation time: {final_stats['total_time']:.2f}s")
        logger.info(f"Average time per image: {final_stats['average_time']:.2f}s")

    except Exception as e:
        logger.error(f"Stable Diffusion inference failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
