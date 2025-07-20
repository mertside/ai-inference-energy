#!/usr/bin/env python3
"""
Whisper Speech Recognition via Hugging Face for AI Inference Energy Profiling.

This script provides OpenAI Whisper integration for comprehensive speech-to-text
energy profiling experiments across different GPU architectures and frequencies.
It supports various Whisper model sizes and provides benchmarking capabilities
for energy consumption analysis.

Features:
- Multiple Whisper model sizes (tiny, base, small, medium, large)
- Batch processing for audio files
- Built-in audio sample generation
- Comprehensive benchmarking and metrics
- Integration with energy profiling framework
- Support for different audio formats and sampling rates

Author: Mert Side
"""

import argparse
import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torchaudio
    import numpy as np
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        pipeline
    )
    from datasets import load_dataset
    import librosa
    import soundfile as sf
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install missing packages:")
    print("pip install torch torchaudio transformers datasets librosa soundfile scipy")
    sys.exit(1)

# Configuration
WHISPER_MODELS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base", 
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3"
}

DEFAULT_MODEL = "base"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION = 30  # seconds
DEFAULT_LANGUAGE = "en"

# Benchmark audio samples (various lengths and complexities)
BENCHMARK_PROMPTS = [
    "Short test audio for energy profiling",
    "This is a medium-length audio sample for testing Whisper speech recognition energy consumption across different GPU frequencies and architectures",
    "This is a comprehensive audio sample designed specifically for energy profiling experiments with OpenAI Whisper models, including various speech patterns and technical terminology",
    "Quick benchmark test for real-time factor measurement",
    "Energy profiling analysis for AI inference workloads using Whisper automatic speech recognition technology"
]

class WhisperEnergyProfiler:
    """
    Whisper Speech Recognition Energy Profiling Application.
    
    This class provides comprehensive Whisper model benchmarking for energy
    consumption analysis across different GPU configurations.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        torch_dtype: str = "float16",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Whisper energy profiling application.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large, large-v3)
            device: Device to use (auto, cuda, cpu)
            torch_dtype: Torch data type (float16, float32, bfloat16)
            logger: Optional logger instance
        """
        # Initialize logger first
        self.logger = logger or self._setup_logging()
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.torch_dtype = self._get_torch_dtype(torch_dtype)
        
        # Model components
        self.processor = None
        self.model = None
        self.pipeline = None
        
        # Metrics
        self.metrics = {
            "total_inference_time": 0.0,
            "total_audio_duration": 0.0,
            "total_samples_processed": 0,
            "average_rtf": 0.0,  # Real-time factor
            "model_parameters": 0,
            "gpu_memory_used": 0.0,
            "transcriptions": []
        }
        
        self.logger.info(f"Whisper Energy Profiler initialized")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Data Type: {torch_dtype}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Set up the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                self.logger.info("CUDA not available, using CPU")
        
        return torch.device(device)
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_model(self) -> None:
        """Load the Whisper model and processor."""
        try:
            model_id = WHISPER_MODELS.get(self.model_name, self.model_name)
            
            self.logger.info(f"Loading Whisper model: {model_id}")
            
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(model_id)
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device.type == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device.type == "cuda" and not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
            
            # Create pipeline for easier inference
            # Don't pass device if model is loaded with accelerate
            pipeline_kwargs = {
                "model": self.model,
                "tokenizer": self.processor.tokenizer,
                "feature_extractor": self.processor.feature_extractor,
                "torch_dtype": self.torch_dtype
            }
            
            # Only add device if model is not loaded with accelerate
            if not hasattr(self.model, 'hf_device_map'):
                pipeline_kwargs["device"] = self.device
            
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                **pipeline_kwargs
            )
            
            # Count parameters
            self.metrics["model_parameters"] = sum(p.numel() for p in self.model.parameters())
            
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Parameters: {self.metrics['model_parameters']:,}")
            
            # Get GPU memory usage
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**3  # GB
                self.logger.info(f"GPU memory used: {self.metrics['gpu_memory_used']:.2f} GB")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_audio_sample(
        self,
        text: str,
        duration: float = DEFAULT_DURATION,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate synthetic audio sample for testing.
        
        Args:
            text: Text to convert to audio (for documentation)
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            output_path: Optional path to save audio file
            
        Returns:
            Audio array
        """
        # Generate synthetic audio with more realistic speech-like patterns
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create multiple frequency components to simulate formants
        fundamental = 150 + 50 * np.random.randn()  # Vary fundamental frequency
        
        # Generate multiple harmonics with different amplitudes
        audio = np.zeros_like(t)
        for harmonic in range(1, 6):
            freq = fundamental * harmonic
            amplitude = 0.5 / harmonic  # Decreasing amplitude for higher harmonics
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase
            audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Add formant-like resonances
        for formant_freq in [800, 1200, 2500]:  # Typical formant frequencies
            audio += 0.1 * np.sin(2 * np.pi * formant_freq * t) * np.exp(-t * 0.5)
        
        # Apply realistic amplitude envelope (speech-like)
        # Create pauses and varied intensity
        segments = int(duration * 2)  # 2 segments per second
        envelope = np.ones_like(t)
        
        for i in range(segments):
            start_idx = int(i * len(t) / segments)
            end_idx = int((i + 1) * len(t) / segments)
            
            # Random amplitude variation and occasional pauses
            if np.random.random() > 0.1:  # 90% chance of sound
                seg_amplitude = 0.3 + 0.7 * np.random.random()
                # Add gradual fade in/out
                seg_env = seg_amplitude * np.hanning(end_idx - start_idx)
                envelope[start_idx:end_idx] = seg_env
            else:  # 10% chance of silence (pause)
                envelope[start_idx:end_idx] = 0.0
        
        # Apply envelope
        audio = audio * envelope
        
        # Add some background noise for realism
        noise_amplitude = 0.05
        audio += noise_amplitude * np.random.randn(len(t))
        
        # Apply bandpass filter to simulate speech bandwidth (300-3400 Hz)
        try:
            from scipy import signal
            nyquist = sample_rate // 2
            low_freq = 300 / nyquist
            high_freq = 3400 / nyquist
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            audio = signal.filtfilt(b, a, audio)
        except ImportError:
            # If scipy is not available, skip filtering
            pass
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        if output_path:
            sf.write(output_path, audio, sample_rate)
            self.logger.info(f"Audio sample saved to: {output_path}")
        
        return audio
    
    def load_audio_file(self, file_path: str) -> np.ndarray:
        """
        Load audio file using librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio array
        """
        try:
            audio, sr = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)
            self.logger.info(f"Loaded audio file: {file_path}")
            self.logger.info(f"Duration: {len(audio) / sr:.2f} seconds")
            return audio
        except Exception as e:
            self.logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def load_sample_dataset(self, num_samples: int = 5) -> List[np.ndarray]:
        """
        Load sample audio from a public dataset.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            List of audio arrays
        """
        try:
            self.logger.info("Loading sample dataset...")
            
            # Load LibriSpeech dataset samples
            dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
            
            audio_samples = []
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                
                audio = sample["audio"]["array"]
                # Resample if needed
                if sample["audio"]["sampling_rate"] != DEFAULT_SAMPLE_RATE:
                    audio = librosa.resample(
                        audio, 
                        orig_sr=sample["audio"]["sampling_rate"], 
                        target_sr=DEFAULT_SAMPLE_RATE
                    )
                
                audio_samples.append(audio)
                self.logger.info(f"Loaded sample {i+1}/{num_samples}: {len(audio)/DEFAULT_SAMPLE_RATE:.2f}s")
            
            return audio_samples
            
        except Exception as e:
            self.logger.warning(f"Could not load dataset: {e}")
            self.logger.info("Falling back to synthetic audio generation")
            return self.generate_benchmark_audio()
    
    def generate_benchmark_audio(self) -> List[np.ndarray]:
        """Generate benchmark audio samples for profiling."""
        audio_samples = []
        
        for i, prompt in enumerate(BENCHMARK_PROMPTS):
            # Vary duration based on prompt length, but limit to 30 seconds to avoid long-form issues
            duration = min(5 + len(prompt) * 0.1, 30)  # 5-30 seconds (Whisper's limit)
            
            audio = self.generate_audio_sample(
                text=prompt,
                duration=duration,
                sample_rate=DEFAULT_SAMPLE_RATE
            )
            
            audio_samples.append(audio)
            self.logger.info(f"Generated benchmark audio {i+1}/{len(BENCHMARK_PROMPTS)}: {duration:.1f}s")
        
        return audio_samples
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        language: str = DEFAULT_LANGUAGE,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio array
            language: Language code (e.g., "en")
            return_timestamps: Whether to return timestamps
            
        Returns:
            Transcription result with timing information
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Measure inference time
            start_time = time.time()
            
            # Prepare audio
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # Convert to mono if stereo
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # Check if audio is longer than 30 seconds (Whisper's limit for standard mode)
            audio_duration = len(audio) / DEFAULT_SAMPLE_RATE
            if audio_duration > 30 and not return_timestamps:
                self.logger.warning(f"Audio duration ({audio_duration:.1f}s) > 30s, enabling timestamps for long-form generation")
                return_timestamps = True
            
            # Transcribe
            result = self.pipeline(
                audio,
                return_timestamps=return_timestamps,
                generate_kwargs={"language": language}
            )
            
            inference_time = time.time() - start_time
            rtf = inference_time / audio_duration  # Real-time factor
            
            # Update metrics
            self.metrics["total_inference_time"] += inference_time
            self.metrics["total_audio_duration"] += audio_duration
            self.metrics["total_samples_processed"] += 1
            
            # Store transcription
            transcription_data = {
                "text": result["text"],
                "inference_time": inference_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "timestamps": result.get("chunks", []) if return_timestamps else None
            }
            
            self.metrics["transcriptions"].append(transcription_data)
            
            self.logger.info(f"Transcription completed in {inference_time:.2f}s")
            self.logger.info(f"RTF: {rtf:.3f} (lower is better)")
            self.logger.info(f"Text: {result['text'][:100]}...")
            
            return transcription_data
            
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            raise
    
    def run_benchmark(
        self,
        num_samples: int = 3,
        use_dataset: bool = False,
        language: str = DEFAULT_LANGUAGE,
        return_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark for energy profiling.
        
        Args:
            num_samples: Number of audio samples to process
            use_dataset: Whether to use real dataset or synthetic audio
            language: Language code
            return_timestamps: Whether to return timestamps
            
        Returns:
            Benchmark results
        """
        self.logger.info("Starting Whisper benchmark...")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Get audio samples
        if use_dataset:
            audio_samples = self.load_sample_dataset(num_samples)
        else:
            audio_samples = self.generate_benchmark_audio()[:num_samples]
        
        # Process each sample
        benchmark_start = time.time()
        
        for i, audio in enumerate(audio_samples):
            self.logger.info(f"Processing sample {i+1}/{len(audio_samples)}")
            
            try:
                result = self.transcribe_audio(
                    audio,
                    language=language,
                    return_timestamps=return_timestamps
                )
                
                # Log sample results
                self.logger.info(f"Sample {i+1} Results:")
                self.logger.info(f"  Duration: {result['audio_duration']:.2f}s")
                self.logger.info(f"  Inference Time: {result['inference_time']:.2f}s")
                self.logger.info(f"  RTF: {result['rtf']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error processing sample {i+1}: {e}")
                continue
        
        benchmark_time = time.time() - benchmark_start
        
        # Calculate final metrics
        if self.metrics["total_samples_processed"] > 0:
            self.metrics["average_rtf"] = (
                self.metrics["total_inference_time"] / self.metrics["total_audio_duration"]
            )
        
        # Summary
        summary = {
            "model": self.model_name,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype),
            "total_samples": self.metrics["total_samples_processed"],
            "total_audio_duration": self.metrics["total_audio_duration"],
            "total_inference_time": self.metrics["total_inference_time"],
            "average_rtf": self.metrics["average_rtf"],
            "benchmark_time": benchmark_time,
            "model_parameters": self.metrics["model_parameters"],
            "gpu_memory_used": self.metrics["gpu_memory_used"],
            "transcriptions": self.metrics["transcriptions"]
        }
        
        self.logger.info("Benchmark completed!")
        self.logger.info(f"Total samples processed: {summary['total_samples']}")
        self.logger.info(f"Total audio duration: {summary['total_audio_duration']:.2f}s")
        self.logger.info(f"Total inference time: {summary['total_inference_time']:.2f}s")
        self.logger.info(f"Average RTF: {summary['average_rtf']:.3f}")
        
        return summary
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Cleanup completed")

def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Whisper Speech Recognition Energy Profiling Application"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        choices=list(WHISPER_MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)"
    )
    
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float16",
        help="Data type to use (default: float16)"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode for energy profiling"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to process in benchmark (default: 3)"
    )
    
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Use real dataset instead of synthetic audio"
    )
    
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})"
    )
    
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Return timestamps in transcription"
    )
    
    # Audio input
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to audio file to transcribe"
    )
    
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="Generate synthetic audio sample"
    )
    
    parser.add_argument(
        "--audio-duration",
        type=float,
        default=DEFAULT_DURATION,
        help=f"Duration for generated audio (default: {DEFAULT_DURATION})"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save results JSON file"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.quiet:
        log_level = "WARNING"
    else:
        log_level = args.log_level
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize profiler
        profiler = WhisperEnergyProfiler(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            logger=logger
        )
        
        # Load model
        profiler.load_model()
        
        results = {}
        
        if args.benchmark:
            # Run benchmark
            results = profiler.run_benchmark(
                num_samples=args.num_samples,
                use_dataset=args.use_dataset,
                language=args.language,
                return_timestamps=args.timestamps
            )
            
        elif args.audio_file:
            # Process specific audio file
            audio = profiler.load_audio_file(args.audio_file)
            results = profiler.transcribe_audio(
                audio,
                language=args.language,
                return_timestamps=args.timestamps
            )
            
        elif args.generate_audio:
            # Generate and process synthetic audio
            audio = profiler.generate_audio_sample(
                text="Sample audio for energy profiling",
                duration=args.audio_duration
            )
            results = profiler.transcribe_audio(
                audio,
                language=args.language,
                return_timestamps=args.timestamps
            )
            
        else:
            # Default: run small benchmark
            results = profiler.run_benchmark(
                num_samples=1,
                use_dataset=False,
                language=args.language,
                return_timestamps=args.timestamps
            )
        
        # Save results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*50)
            print("WHISPER ENERGY PROFILING SUMMARY")
            print("="*50)
            
            if args.benchmark:
                print(f"Model: {results['model']}")
                print(f"Device: {results['device']}")
                print(f"Samples Processed: {results['total_samples']}")
                print(f"Total Audio Duration: {results['total_audio_duration']:.2f}s")
                print(f"Total Inference Time: {results['total_inference_time']:.2f}s")
                print(f"Average RTF: {results['average_rtf']:.3f}")
                print(f"Model Parameters: {results['model_parameters']:,}")
                if results['gpu_memory_used'] > 0:
                    print(f"GPU Memory Used: {results['gpu_memory_used']:.2f} GB")
            else:
                print(f"Text: {results['text']}")
                print(f"Inference Time: {results['inference_time']:.2f}s")
                print(f"Audio Duration: {results['audio_duration']:.2f}s")
                print(f"RTF: {results['rtf']:.3f}")
            
            print("="*50)
        
        # Cleanup
        profiler.cleanup()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
