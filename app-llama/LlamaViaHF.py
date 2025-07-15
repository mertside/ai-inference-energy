#!/usr/bin/env python3
"""
LLaMA Text Generation via Hugging Face Transformers.

This script provides comprehensive text generation using various LLaMA model variants
through Hugging Face's transformers library. It's designed for energy profiling 
studies on GPU inference workloads with configurable parameters and robust error handling.

Supported Models:
    - LLaMA-7B (huggyllama/llama-7b)
    - LLaMA-13B (huggyllama/llama-13b) 
    - LLaMA-30B (huggyllama/llama-30b)
    - LLaMA-65B (huggyllama/llama-65b)
    - Llama-2-7B (meta-llama/Llama-2-7b-hf)
    - Llama-2-13B (meta-llama/Llama-2-13b-hf)
    - Custom models via --model parameter

Requirements:
    - Hugging Face account with access to LLaMA models
    - Login via: huggingface-cli login
    - CUDA-compatible GPU (8GB+ VRAM recommended)
    - PyTorch with CUDA support
    - transformers>=4.21.0

Usage Examples:
    # Basic usage with default 7B model
    python LlamaViaHF.py --prompt "The future of AI is"
    
    # Use specific model variant
    python LlamaViaHF.py --model "meta-llama/Llama-2-13b-hf" --max-tokens 100
    
    # Multiple generations for benchmarking
    python LlamaViaHF.py --prompt "Climate change" --num-generations 5 --max-tokens 50
    
    # Profiling mode with minimal output
    python LlamaViaHF.py --num-generations 3 --log-level WARNING

Author: Mert Side
Version: 2.0.1
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default model configurations
DEFAULT_MODELS = {
    "llama-7b": "huggyllama/llama-7b",
    "llama-13b": "huggyllama/llama-13b", 
    "llama-30b": "huggyllama/llama-30b",
    "llama-65b": "huggyllama/llama-65b",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-hf"
}

DEFAULT_PROMPTS = [
    "The future of artificial intelligence is",
    "Climate change is one of the most pressing issues",
    "In the field of renewable energy",
    "Machine learning algorithms have revolutionized",
    "The impact of social media on society",
    "Quantum computing represents a paradigm shift",
    "The exploration of space has always fascinated",
    "Sustainable development goals are essential"
]


class LlamaInferenceEngine:
    """
    Enhanced LLaMA inference engine with comprehensive configuration and profiling support.
    
    Features:
        - Multiple model variant support
        - Configurable generation parameters
        - Robust error handling and logging
        - GPU memory management
        - Performance metrics collection
        - Energy profiling compatibility
    """
    
    def __init__(self, 
                 model_name: str = "huggyllama/llama-7b",
                 device: Optional[str] = None,
                 precision: str = "float16",
                 max_memory_mb: Optional[int] = None):
        """
        Initialize the LLaMA inference engine.
        
        Args:
            model_name: Hugging Face model identifier or shorthand
            device: Target device ('cuda', 'cpu', or None for auto-detection)
            precision: Model precision ('float16', 'float32', 'int8')
            max_memory_mb: Maximum GPU memory to use in MB
        """
        self.model_name = self._resolve_model_name(model_name)
        self.device = self._setup_device(device)
        self.precision = precision
        self.max_memory_mb = max_memory_mb
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.generation_count = 0
        
        # Performance tracking
        self.metrics = {
            "model_load_time": 0.0,
            "total_inference_time": 0.0,
            "total_tokens_generated": 0,
            "average_tokens_per_second": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model name from shorthand to full Hugging Face identifier."""
        if model_name in DEFAULT_MODELS:
            resolved = DEFAULT_MODELS[model_name]
            self.logger.info(f"Resolved model shorthand '{model_name}' to '{resolved}'")
            return resolved
        return model_name
        
    def _setup_device(self, device: Optional[str]) -> str:
        """Setup and validate the target device."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        else:
            self.logger.info("Using CPU for inference")
            
        return device
        
    def load_model(self) -> None:
        """Load the model and tokenizer with optimized settings."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            
            # Configure torch dtype
            if self.precision == "float16":
                torch_dtype = torch.float16
            elif self.precision == "int8":
                torch_dtype = torch.int8  
            else:
                torch_dtype = torch.float32
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Configure model loading arguments
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                if self.max_memory_mb:
                    max_memory = {0: f"{self.max_memory_mb}MB"}
                    model_kwargs["max_memory"] = max_memory
                    
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                **model_kwargs
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch_dtype
            )
            
            load_time = time.time() - start_time
            self.metrics["model_load_time"] = load_time
            
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            # Print GPU memory usage if on CUDA
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 50,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     do_sample: bool = True,
                     repetition_penalty: float = 1.1) -> Dict[str, Any]:
        """
        Generate text from a prompt with configurable parameters.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            repetition_penalty: Penalty for repeated tokens
            
        Returns:
            Dictionary containing generated text and metadata
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        start_time = time.time()
        
        try:
            self.logger.debug(f"Generating text for prompt: {prompt[:50]}...")
            
            # Generate text
            result = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False  # Only return generated portion
            )
            
            inference_time = time.time() - start_time
            generated_text = result[0]['generated_text']
            
            # Count tokens in generated text
            generated_tokens = len(self.tokenizer.encode(generated_text))
            
            # Update metrics
            self.generation_count += 1
            self.metrics["total_inference_time"] += inference_time
            self.metrics["total_tokens_generated"] += generated_tokens
            
            if self.metrics["total_inference_time"] > 0:
                self.metrics["average_tokens_per_second"] = (
                    self.metrics["total_tokens_generated"] / 
                    self.metrics["total_inference_time"]
                )
            
            return {
                "prompt": prompt,
                "generated_text": generated_text,
                "full_text": prompt + generated_text,
                "inference_time": inference_time,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / inference_time if inference_time > 0 else 0,
                "generation_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "do_sample": do_sample,
                    "repetition_penalty": repetition_penalty
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise
            
    def run_benchmark(self, 
                     prompts: List[str],
                     num_generations: int = 1,
                     **generation_kwargs) -> List[Dict[str, Any]]:
        """
        Run benchmark with multiple prompts and generations.
        
        Args:
            prompts: List of input prompts
            num_generations: Number of generations per prompt
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generation results
        """
        results = []
        total_prompts = len(prompts) * num_generations
        
        self.logger.info(f"Starting benchmark: {total_prompts} total generations")
        
        for i, prompt in enumerate(prompts):
            for gen_num in range(num_generations):
                self.logger.info(f"Generation {len(results) + 1}/{total_prompts}: "
                               f"Prompt {i+1}/{len(prompts)}, Run {gen_num+1}/{num_generations}")
                
                result = self.generate_text(prompt, **generation_kwargs)
                result["prompt_index"] = i
                result["generation_number"] = gen_num + 1
                results.append(result)
                
        return results
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self.metrics,
            "generation_count": self.generation_count,
            "model_name": self.model_name,
            "device": self.device,
            "precision": self.precision
        }
        
    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cache cleared")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
        
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLaMA Text Generation for Energy Profiling Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "The future of AI is"
  %(prog)s --model llama2-13b --max-tokens 100 --num-generations 5
  %(prog)s --benchmark --num-generations 3 --log-level WARNING
  %(prog)s --model meta-llama/Llama-2-7b-hf --temperature 0.8 --top-p 0.95

Supported model shortcuts:
  llama-7b, llama-13b, llama-30b, llama-65b
  llama2-7b, llama2-13b, llama2-70b
        """
    )
    
    # Model configuration
    parser.add_argument("--model", "-m", 
                       default="llama-7b",
                       help="Model name or shorthand (default: llama-7b)")
    
    parser.add_argument("--device",
                       choices=["cuda", "cpu", "auto"],
                       default="auto", 
                       help="Target device (default: auto)")
    
    parser.add_argument("--precision",
                       choices=["float16", "float32", "int8"],
                       default="float16",
                       help="Model precision (default: float16)")
    
    parser.add_argument("--max-memory-mb", type=int,
                       help="Maximum GPU memory to use in MB")
    
    # Generation parameters
    parser.add_argument("--prompt", "-p",
                       help="Input text prompt")
    
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum new tokens to generate (default: 50)")
    
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling parameter (default: 0.9)")
    
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling parameter (default: 50)")
    
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Repetition penalty (default: 1.1)")
    
    parser.add_argument("--no-sampling", action="store_true",
                       help="Use greedy decoding instead of sampling")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark with default prompts")
    
    parser.add_argument("--num-generations", type=int, default=1,
                       help="Number of generations per prompt (default: 1)")
    
    # Output options
    parser.add_argument("--output-file", "-o",
                       help="Save results to JSON file")
    
    parser.add_argument("--log-level",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level (default: INFO)")
    
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress generated text output")
    
    parser.add_argument("--metrics", action="store_true",
                       help="Display performance metrics")
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize inference engine
        engine = LlamaInferenceEngine(
            model_name=args.model,
            device=args.device if args.device != "auto" else None,
            precision=args.precision,
            max_memory_mb=args.max_memory_mb
        )
        
        # Load model
        logger.info("Initializing LLaMA inference engine...")
        engine.load_model()
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": not args.no_sampling,
            "repetition_penalty": args.repetition_penalty
        }
        
        # Run generation
        if args.benchmark:
            logger.info("Running benchmark mode with default prompts")
            results = engine.run_benchmark(
                DEFAULT_PROMPTS, 
                num_generations=args.num_generations,
                **generation_kwargs
            )
        else:
            # Single prompt mode
            prompt = args.prompt if args.prompt else DEFAULT_PROMPTS[0]
            logger.info(f"Running single prompt mode: {args.num_generations} generations")
            
            results = []
            for i in range(args.num_generations):
                logger.info(f"Generation {i+1}/{args.num_generations}")
                result = engine.generate_text(prompt, **generation_kwargs)
                result["generation_number"] = i + 1
                results.append(result)
        
        # Display results
        if not args.quiet:
            print("\n" + "="*80)
            print("GENERATION RESULTS")
            print("="*80)
            
            for i, result in enumerate(results):
                print(f"\n--- Generation {i+1} ---")
                print(f"Prompt: {result['prompt']}")
                print(f"Generated: {result['generated_text']}")
                print(f"Tokens: {result['generated_tokens']} "
                      f"({result['tokens_per_second']:.1f} tok/s)")
                print(f"Time: {result['inference_time']:.2f}s")
        
        # Display metrics
        if args.metrics:
            metrics = engine.get_metrics()
            print("\n" + "="*80)
            print("PERFORMANCE METRICS")
            print("="*80)
            print(f"Model: {metrics['model_name']}")
            print(f"Device: {metrics['device']}")
            print(f"Precision: {metrics['precision']}")
            print(f"Model Load Time: {metrics['model_load_time']:.2f}s")
            print(f"Total Generations: {metrics['generation_count']}")
            print(f"Total Inference Time: {metrics['total_inference_time']:.2f}s")
            print(f"Total Tokens Generated: {metrics['total_tokens_generated']}")
            print(f"Average Tokens/Second: {metrics['average_tokens_per_second']:.1f}")
        
        # Save results if requested
        if args.output_file:
            output_data = {
                "args": vars(args),
                "metrics": engine.get_metrics(),
                "results": results
            }
            
            with open(args.output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output_file}")
        
        # Cleanup
        engine.cleanup()
        logger.info("Inference completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
