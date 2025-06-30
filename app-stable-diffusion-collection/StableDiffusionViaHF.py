#!/usr/bin/env python3
"""
Stable Diffusion Image Generation via Hugging Face Diffusers.

This script demonstrates image generation using the Stable Diffusion v1.4 model
through Hugging Face's diffusers library. It's designed for energy profiling
studies on GPU inference workloads.

Requirements:
    - Hugging Face account with access to Stable Diffusion models
    - Login via: huggingface-cli login
    - CUDA-compatible GPU

Author: AI Inference Energy Research Team
"""

import sys
import os
import logging
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from typing import Optional, Union, List
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import model_config
    from utils import setup_logging, validate_gpu_available, get_timestamp
except ImportError:
    # Fallback configuration if imports fail
    class ModelConfig:
        STABLE_DIFFUSION_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
        STABLE_DIFFUSION_DEFAULT_PROMPT = "a photo of an astronaut riding a horse on mars"
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
    Image generator using Stable Diffusion model via Hugging Face diffusers.
    
    This class encapsulates the Stable Diffusion model initialization and image
    generation functionality for consistent usage across energy profiling experiments.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Stable Diffusion image generator.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on (cuda/cpu)
            torch_dtype: PyTorch data type for model weights
            logger: Optional logger instance
        """
        self.model_name = model_name or model_config.STABLE_DIFFUSION_MODEL_NAME
        self.device = device
        self.torch_dtype = torch_dtype
        self.logger = logger or setup_logging()
        self.pipeline = None
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the image generation pipeline."""
        try:
            # Validate GPU availability if using CUDA
            if self.device == "cuda" and not validate_gpu_available():
                self.logger.warning("CUDA GPU not available, falling back to CPU")
                self.device = "cpu"
                self.torch_dtype = torch.float32  # CPU doesn't support float16
            
            self.logger.info(f"Initializing Stable Diffusion model: {self.model_name}")
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"Using dtype: {self.torch_dtype}")
            
            # Initialize the diffusion pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                use_auth_token=True,
                safety_checker=None,  # Disable safety checker for performance
                requires_safety_checker=False
            ).to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
            
            # Enable xformers memory efficient attention if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.logger.info("Enabled xformers memory efficient attention")
            except Exception:
                self.logger.info("xformers not available, using default attention")
            
            self.logger.info("Stable Diffusion model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Stable Diffusion model: {e}")
            raise
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images using the Stable Diffusion model.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to avoid certain features
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for reproducibility
        
        Returns:
            List of generated PIL Images
        """
        if self.pipeline is None:
            raise RuntimeError("Model not initialized")
        
        try:
            self.logger.info(f"Generating image for prompt: '{prompt[:50]}...'")
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Generate images with autocast for mixed precision
            with autocast(self.device):
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images_per_prompt
                )
            
            images = result.images
            self.logger.info(f"Successfully generated {len(images)} image(s)")
            
            return images
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise
    
    def run_default_inference(self) -> List[Image.Image]:
        """
        Run inference with default prompt and parameters.
        
        Returns:
            List of generated PIL Images
        """
        return self.generate_image(model_config.STABLE_DIFFUSION_DEFAULT_PROMPT)
    
    def save_images(
        self,
        images: List[Image.Image],
        base_filename: str = None,
        output_dir: str = "."
    ) -> List[str]:
        """
        Save generated images to files.
        
        Args:
            images: List of PIL Images to save
            base_filename: Base filename (without extension)
            output_dir: Output directory
        
        Returns:
            List of saved file paths
        """
        if base_filename is None:
            base_filename = model_config.STABLE_DIFFUSION_OUTPUT_FILE.replace(".png", "")
        
        saved_files = []
        
        for i, image in enumerate(images):
            if len(images) == 1:
                filename = f"{base_filename}.png"
            else:
                filename = f"{base_filename}_{i+1}.png"
            
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved_files.append(filepath)
            self.logger.info(f"Saved image: {filepath}")
        
        return saved_files


def main():
    """Main function for standalone execution."""
    # Set up logging
    logger = setup_logging()
    
    try:
        # Initialize the image generator
        generator = StableDiffusionGenerator(logger=logger)
        
        # Run default inference
        images = generator.run_default_inference()
        
        # Save generated images
        output_filename = f"sd_output_{get_timestamp()}"
        saved_files = generator.save_images(images, base_filename=output_filename)
        
        logger.info(f"Stable Diffusion inference completed successfully")
        logger.info(f"Generated images saved to: {', '.join(saved_files)}")
        
    except Exception as e:
        logger.error(f"Stable Diffusion inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()