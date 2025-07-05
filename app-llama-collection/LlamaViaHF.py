#!/usr/bin/env python3
"""
LLaMA Text Generation via Hugging Face Transformers.

This script demonstrates text generation using the LLaMA-7B model
through Hugging Face's transformers library. It's designed for
energy profiling studies on GPU inference workloads.

Requirements:
    - Hugging Face account with access to LLaMA models
    - Login via: huggingface-cli login
    - CUDA-compatible GPU

Author: Mert Side
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import pipeline

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import model_config
    from utils import setup_logging, validate_gpu_available
except ImportError:
    # Fallback configuration if imports fail
    class ModelConfig:
        LLAMA_MODEL_NAME = "huggyllama/llama-7b"
        LLAMA_TORCH_DTYPE = "float16"
        LLAMA_DEFAULT_PROMPT = "Plants create energy through a process known as"

    model_config = ModelConfig()

    def setup_logging(level="INFO"):
        logging.basicConfig(level=getattr(logging, level))
        return logging.getLogger(__name__)

    def validate_gpu_available():
        return torch.cuda.is_available()


class LlamaTextGenerator:
    """
    Text generator using LLaMA model via Hugging Face transformers.

    This class encapsulates the LLaMA model initialization and text generation
    functionality for consistent usage across energy profiling experiments.
    """

    def __init__(
        self,
        model_name: str = None,
        device: int = 0,
        torch_dtype: str = "float16",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the LLaMA text generator.

        Args:
            model_name: Hugging Face model identifier
            device: GPU device ID (default: 0)
            torch_dtype: PyTorch data type for model weights
            logger: Optional logger instance
        """
        self.model_name = model_name or model_config.LLAMA_MODEL_NAME
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.logger = logger or setup_logging()
        self.pipeline = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the text generation pipeline."""
        try:
            # Validate GPU availability
            if not validate_gpu_available():
                raise RuntimeError("CUDA GPU not available")

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA not available")

            self.logger.info(f"Initializing LLaMA model: {self.model_name}")
            self.logger.info(f"Using device: cuda:{self.device}")
            self.logger.info(f"Using dtype: {self.torch_dtype}")

            # Initialize the text generation pipeline
            self.pipeline = pipeline(
                task="text-generation",
                model=self.model_name,
                torch_dtype=self.torch_dtype,
                device=self.device,
                trust_remote_code=True,  # Added for better compatibility
            )

            self.logger.info("LLaMA model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize LLaMA model: {e}")
            raise

    def generate_text(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> List[str]:
        """
        Generate text using the LLaMA model.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            List of generated text sequences
        """
        if self.pipeline is None:
            raise RuntimeError("Model not initialized")

        try:
            self.logger.info(f"Generating text for prompt: '{prompt[:50]}...'")

            # Generate text
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            # Extract generated text
            generated_texts = [output["generated_text"] for output in outputs]

            self.logger.info(
                f"Successfully generated {len(generated_texts)} text sequence(s)"
            )

            return generated_texts

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise

    def run_default_inference(self) -> List[str]:
        """
        Run inference with default prompt and parameters.

        Returns:
            List of generated text sequences
        """
        return self.generate_text(model_config.LLAMA_DEFAULT_PROMPT)


def main():
    """Main function for standalone execution."""
    # Set up logging
    logger = setup_logging()

    try:
        # Initialize the text generator
        generator = LlamaTextGenerator(logger=logger)

        # Run default inference
        results = generator.run_default_inference()

        # Display results
        for i, text in enumerate(results):
            logger.info(f"Generated text {i+1}:")
            print(text)
            print("-" * 50)

        logger.info("LLaMA inference completed successfully")

    except Exception as e:
        logger.error(f"LLaMA inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
