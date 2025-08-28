#!/usr/bin/env python3
"""
Vision Transformer (ViT) Implementation via Hugging Face.

This module provides Vision Transformer functionality for image classification
with comprehensive energy profiling capabilities across different GPU architectures.

Author: Mert Side
"""

import logging
import os
import sys
import time
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import GPUtil
import psutil
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import system_config
from utils import setup_experiment_directories

# Get temp images directory
TEMP_IMAGES_DIR = system_config.TEMP_IMAGES_DIR


class ViTViaHF:
    """Vision Transformer implementation using Hugging Face transformers."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[str] = None,
        precision: str = "float32",
        temperature: float = 1.0,
    ):
        """
        Initialize Vision Transformer model.

        Args:
            model_name: HuggingFace model identifier
            device: Target device (cuda/cpu)
            precision: Model precision (float32, float16, bfloat16)
            temperature: Temperature scaling for predictions
        """
        self.model_name = model_name

        # Handle device selection with explicit CUDA forcing
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("CUDA is available and will be used")
            else:
                logger.warning("CUDA requested but torch.cuda.is_available() returns False")
                logger.warning("This might be due to PyTorch/CUDA version mismatch")
                logger.warning("Attempting to force CUDA usage anyway...")
                # Force CUDA device even if PyTorch reports it as unavailable
                self.device = "cuda:0"
                logger.info("Forced CUDA device - this may work despite the warning")
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.precision = precision
        self.temperature = temperature

        # Initialize model components
        self.processor = None
        self.model = None
        self.image_cache = {}

        # Debug CUDA availability
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available - this is likely due to PyTorch/CUDA version mismatch")
            logger.warning("Your system has CUDA 11.0 but PyTorch may be compiled for a different CUDA version")
            logger.warning("To fix this issue:")
            logger.warning("1. Uninstall current PyTorch: pip uninstall torch torchvision torchaudio")
            logger.warning(
                "2. Install CUDA 11.0 compatible version: conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia"
            )
            logger.warning("3. Alternatively, use conda to create a fresh environment")
            if device == "cuda":
                logger.error("CUDA was explicitly requested but is not available. Exiting.")
                raise RuntimeError("CUDA requested but not available. Please fix PyTorch/CUDA compatibility first.")

        logger.info(f"Initializing ViT model: {model_name}")
        logger.info(f"Device: {self.device}, Precision: {precision}")

    def load_model(self) -> None:
        """Load the Vision Transformer model and processor."""
        try:
            logger.info(f"Loading ViT model: {self.model_name}")

            # Load processor and model
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)

            # Set precision
            if self.precision == "float16" and self.device == "cuda":
                self.model = self.model.half()
            elif self.precision == "bfloat16" and self.device == "cuda":
                self.model = self.model.to(torch.bfloat16)

            # Move to device
            try:
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded successfully on {self.device}")
            except Exception as e:
                if "cuda" in str(self.device).lower():
                    logger.error(f"Failed to move model to CUDA: {e}")
                    logger.info("Falling back to CPU")
                    self.device = "cpu"
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Model loaded successfully on {self.device} (fallback)")
                else:
                    raise

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
        }

        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info.update(
                        {
                            "gpu_name": gpu.name,
                            "gpu_memory_used_mb": gpu.memoryUsed,
                            "gpu_memory_total_mb": gpu.memoryTotal,
                            "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                            "gpu_temperature": gpu.temperature,
                            "gpu_uuid": gpu.uuid,
                        }
                    )
            except Exception as e:
                logger.warning(f"Could not get GPU info: {e}")

        return info

    def download_sample_image(self, url: str) -> Image.Image:
        """Download and cache sample image."""
        if url in self.image_cache:
            return self.image_cache[url]

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            self.image_cache[url] = image
            logger.info(f"Downloaded sample image: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            raise

    def load_local_image(self, image_path: str) -> Image.Image:
        """Load local image file."""
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Loaded local image: {image.size}")
            return image
        except Exception as e:
            logger.error(f"Failed to load local image: {e}")
            raise

    def classify_image(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify an image and return top-k predictions.

        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return

        Returns:
            List of (label, confidence) tuples
        """
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Set precision for inputs
            if self.precision == "float16" and self.device == "cuda":
                inputs = {k: v.half() if v.dtype.is_floating_point else v for k, v in inputs.items()}
            elif self.precision == "bfloat16" and self.device == "cuda":
                inputs = {k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Apply temperature scaling
                if self.temperature != 1.0:
                    logits = logits / self.temperature

                # Get probabilities
                probabilities = F.softmax(logits, dim=-1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities[0], k=top_k)

            # Convert to labels
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                label = self.model.config.id2label[idx]
                predictions.append((label, float(prob)))

            return predictions

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise

    def run_benchmark(self, num_images: int = 10, batch_size: int = 1) -> Dict[str, Any]:
        """
        Run comprehensive benchmarking on sample images.

        Args:
            num_images: Number of images to process
            batch_size: Batch size for processing

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Starting benchmark: {num_images} images, batch_size={batch_size}")

        # Sample image URLs for benchmarking
        sample_urls = [
            # Hugging Face hosted images (most reliable)
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
            # PyTorch/GitHub hosted samples
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            "https://raw.githubusercontent.com/pytorch/vision/main/test/assets/encode_jpeg/grace_hopper_517x606.jpg",
            # Direct image URLs (no User-Agent restrictions)
            "https://www.w3schools.com/css/img_5terre.jpg",
            "https://www.w3schools.com/css/img_forest.jpg",
            "https://www.w3schools.com/css/img_lights.jpg",
            "https://www.w3schools.com/css/img_mountains.jpg",
            "https://www.w3schools.com/howto/img_snow.jpg",
        ]

        results = {
            "model_name": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "temperature": self.temperature,
            "num_images": num_images,
            "batch_size": batch_size,
            "processing_times": [],
            "predictions": [],
            "system_info_before": self.get_system_info(),
            "system_info_after": None,
            "total_time": 0,
            "average_time": 0,
            "throughput_images_per_second": 0,
        }

        try:
            # Pre-load sample images
            images = []
            for i in range(min(num_images, len(sample_urls))):
                image = self.download_sample_image(sample_urls[i % len(sample_urls)])
                images.append(image)

            # Fill remaining with repeated images if needed
            while len(images) < num_images:
                images.append(images[len(images) % len(sample_urls)])

            # Warm up
            logger.info("Warming up model...")
            for _ in range(3):
                self.classify_image(images[0], top_k=1)

            # Run benchmark
            start_time = time.time()

            for i in range(0, num_images, batch_size):
                batch_start = time.time()
                batch_images = images[i : i + batch_size]

                batch_predictions = []
                for image in batch_images:
                    predictions = self.classify_image(image, top_k=5)
                    batch_predictions.append(predictions)

                batch_time = time.time() - batch_start
                results["processing_times"].append(batch_time)
                results["predictions"].extend(batch_predictions)

                logger.info(f"Processed batch {i//batch_size + 1}: {batch_time:.3f}s")

            total_time = time.time() - start_time
            results["total_time"] = total_time
            results["average_time"] = total_time / num_images
            results["throughput_images_per_second"] = num_images / total_time
            results["system_info_after"] = self.get_system_info()

            logger.info(
                f"Benchmark completed: {total_time:.2f}s total, "
                f"{results['average_time']:.3f}s avg, "
                f"{results['throughput_images_per_second']:.2f} images/sec"
            )

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            results["error"] = str(e)
            return results

    def interactive_demo(self) -> None:
        """Run interactive classification demo."""
        logger.info("Starting interactive ViT demo")
        print("\n=== Vision Transformer Interactive Demo ===")
        print("Enter image URLs or local paths (or 'quit' to exit)")

        sample_urls = [
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        ]

        print("\nSample URLs you can try:")
        for i, url in enumerate(sample_urls, 1):
            print(f"{i}. {url}")

        while True:
            user_input = input("\nEnter image URL/path (or 'quit'): ").strip()

            if user_input.lower() == "quit":
                break

            if not user_input:
                continue

            try:
                # Load image
                if user_input.startswith(("http://", "https://")):
                    image = self.download_sample_image(user_input)
                else:
                    image = self.load_local_image(user_input)

                # Classify
                start_time = time.time()
                predictions = self.classify_image(image, top_k=5)
                inference_time = time.time() - start_time

                # Display results
                print(f"\nClassification Results (inference: {inference_time:.3f}s):")
                print("-" * 60)
                for i, (label, confidence) in enumerate(predictions, 1):
                    print(f"{i}. {label:<40} {confidence:.4f} ({confidence*100:.2f}%)")

            except Exception as e:
                print(f"Error processing image: {e}")

        print("Demo ended.")


def main():
    """Main execution function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Vision Transformer Classification")
    parser.add_argument("--model", default="google/vit-base-patch16-224", help="Model name (default: google/vit-base-patch16-224)")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32", help="Model precision")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of interactive demo")
    parser.add_argument("--num-images", type=int, default=10, help="Number of images for benchmark")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark")

    args = parser.parse_args()

    # Create ViT instance
    vit = ViTViaHF(model_name=args.model, device=args.device, precision=args.precision, temperature=args.temperature)

    # Setup directories
    setup_experiment_directories()

    # Load model
    vit.load_model()

    if args.benchmark:
        # Run benchmark
        results = vit.run_benchmark(num_images=args.num_images, batch_size=args.batch_size)
        print(f"\nBenchmark Results:")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Average time: {results['average_time']:.3f}s")
        print(f"Throughput: {results['throughput_images_per_second']:.2f} images/sec")

        # Show sample predictions
        if results["predictions"]:
            print(f"\nSample Predictions:")
            for i, predictions in enumerate(results["predictions"][:3], 1):
                print(f"Image {i}:")
                for label, conf in predictions[:3]:
                    print(f"  {label}: {conf:.4f}")
    else:
        # Run interactive demo
        vit.interactive_demo()


if __name__ == "__main__":
    main()
