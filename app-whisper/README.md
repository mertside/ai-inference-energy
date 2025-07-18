# Whisper Speech Recognition for AI Inference Energy Profiling

This directory contains the Whisper speech recognition implementation for energy profiling experiments across different GPU architectures and frequencies.

## 🎯 Overview

OpenAI Whisper is a transformer-based automatic speech recognition (ASR) model that provides state-of-the-art accuracy across multiple languages. This implementation provides comprehensive energy profiling capabilities for Whisper models, making it perfect for analyzing the energy consumption patterns of audio processing workloads.

## ✨ Features

### 🎤 Multiple Whisper Models
- **tiny**: 39M parameters, fastest inference
- **base**: 74M parameters, balanced performance
- **small**: 244M parameters, good accuracy
- **medium**: 769M parameters, better accuracy
- **large**: 1550M parameters, best accuracy
- **large-v3**: Latest large model with improvements

### 🔧 Comprehensive Profiling
- **Real-time Factor (RTF)**: Measures inference speed vs audio duration
- **Memory Usage**: GPU memory consumption tracking
- **Model Parameters**: Automatic parameter counting
- **Batch Processing**: Support for multiple audio samples
- **Synthetic Audio**: Built-in audio generation for consistent testing

### 🎨 Audio Processing
- **Multiple Input Formats**: Support for various audio file formats
- **Sample Rate Conversion**: Automatic resampling to 16kHz
- **Dataset Integration**: LibriSpeech dataset support
- **Timestamp Support**: Optional word-level timestamps
- **Language Support**: Multi-language transcription

### 📊 Benchmarking
- **Integrated Metrics**: RTF, inference time, accuracy tracking
- **Reproducible Results**: Consistent synthetic audio generation
- **Flexible Configuration**: Model size, precision, device selection
- **Export Results**: JSON output for analysis

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings (base model, synthetic audio)
python WhisperViaHF.py

# Run benchmark mode for energy profiling
python WhisperViaHF.py --benchmark --num-samples 5

# Use specific model size
python WhisperViaHF.py --model large --benchmark

# Process real audio file
python WhisperViaHF.py --audio-file path/to/audio.wav

# Generate synthetic audio for testing
python WhisperViaHF.py --generate-audio --audio-duration 30
```

### Energy Profiling Integration

```bash
# Integration with profiling framework
python WhisperViaHF.py --benchmark --num-samples 3 --quiet --output-file results.json

# Use with different precisions for energy analysis
python WhisperViaHF.py --dtype float16 --benchmark  # Lower precision, less energy
python WhisperViaHF.py --dtype float32 --benchmark  # Higher precision, more energy
```

## 📋 Command Line Options

### Model Configuration
```bash
--model {tiny,base,small,medium,large,large-v3}    # Whisper model size
--device {auto,cuda,cpu}                           # Compute device
--dtype {float16,float32,bfloat16}                 # Data precision
```

### Benchmark Mode
```bash
--benchmark                                        # Enable benchmark mode
--num-samples N                                    # Number of samples to process
--use-dataset                                      # Use LibriSpeech dataset
--language LANG                                    # Language code (e.g., "en")
--timestamps                                       # Include word timestamps
```

### Audio Input
```bash
--audio-file PATH                                  # Process specific audio file
--generate-audio                                   # Generate synthetic audio
--audio-duration SECONDS                           # Duration for generated audio
```

### Output Control
```bash
--output-file PATH                                 # Save results to JSON
--quiet                                            # Reduce verbosity
--log-level {DEBUG,INFO,WARNING,ERROR}             # Logging level
```

## 🔬 Technical Details

### Model Architecture
- **Transformer-based**: Attention mechanism for sequence processing
- **Encoder-Decoder**: Speech-to-text transformation
- **Multi-head Attention**: Parallel processing of audio features
- **Positional Encoding**: Temporal information preservation

### Energy Profiling Metrics
- **RTF (Real-time Factor)**: `inference_time / audio_duration`
- **Throughput**: Audio duration processed per second
- **Memory Efficiency**: GPU memory usage per parameter
- **Computational Load**: Operations per second

### Benchmark Audio Samples
The implementation includes various audio samples for consistent testing:
- **Short samples**: 5-10 seconds, simple speech patterns
- **Medium samples**: 15-30 seconds, conversational speech
- **Long samples**: 45-60 seconds, complex vocabulary and patterns
- **Technical samples**: Domain-specific terminology

## 📊 Results Analysis

### Expected Outputs
```json
{
  "model": "base",
  "device": "cuda",
  "torch_dtype": "torch.float16",
  "total_samples": 3,
  "total_audio_duration": 75.5,
  "total_inference_time": 12.3,
  "average_rtf": 0.163,
  "model_parameters": 74000000,
  "gpu_memory_used": 1.2,
  "transcriptions": [...]
}
```

### Performance Expectations
| Model | Parameters | RTF (A100) | GPU Memory | Accuracy |
|-------|------------|------------|------------|----------|
| tiny  | 39M        | 0.05       | 0.3 GB     | Good     |
| base  | 74M        | 0.08       | 0.5 GB     | Better   |
| small | 244M       | 0.15       | 1.2 GB     | High     |
| medium| 769M       | 0.25       | 2.8 GB     | Higher   |
| large | 1550M      | 0.35       | 5.2 GB     | Best     |

## 🔄 Integration with Energy Framework

### SLURM Job Integration
```bash
# H100 Whisper profiling
LAUNCH_ARGS="--gpu-type H100 --profiling-mode baseline --app-name Whisper --app-executable ../app-whisper/WhisperViaHF.py --app-params '--benchmark --num-samples 3 --model base --quiet' --num-runs 3"

# Custom frequency analysis
LAUNCH_ARGS="--gpu-type A100 --profiling-mode custom --custom-frequencies '510,960,1410' --app-name Whisper --app-executable ../app-whisper/WhisperViaHF.py --app-params '--benchmark --model medium --num-samples 5' --num-runs 3"
```

### Expected Results Directory Structure
```
results_a100_whisper/
├── A100_freq_510_run_01_profile.csv
├── A100_freq_510_run_01_log.txt
├── A100_freq_960_run_01_profile.csv
├── A100_freq_960_run_01_log.txt
├── A100_freq_1410_run_01_profile.csv
├── A100_freq_1410_run_01_log.txt
└── experiment_summary.json
```

## 📈 Research Applications

### Energy Efficiency Analysis
- **Model Size vs Energy**: Compare energy consumption across model sizes
- **Precision Impact**: Analyze float16 vs float32 energy usage
- **Batch Size Effects**: Energy scaling with concurrent processing
- **Real-time Constraints**: RTF vs energy trade-offs

### Frequency Analysis
- **Low Frequency**: Energy-efficient processing for batch jobs
- **High Frequency**: Real-time processing requirements
- **Optimal Frequency**: Best energy-performance balance

### Comparative Studies
- **Cross-Architecture**: H100 vs A100 vs V100 performance
- **Workload Characterization**: Audio processing vs text/image generation
- **Memory Patterns**: Different memory access patterns than other AI workloads

## 🛠️ Installation

### Dependencies
```bash
pip install torch torchaudio transformers datasets librosa soundfile
```

### Optional Dependencies
```bash
pip install accelerate  # For distributed inference
pip install optimum     # For optimized inference
```

### Environment Setup
```bash
# Create conda environment
conda create -n whisper-gpu python=3.10
conda activate whisper-gpu

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Testing

### Validation Tests
```bash
# Test basic functionality
python WhisperViaHF.py --model tiny --generate-audio --audio-duration 5

# Test all models
for model in tiny base small medium large; do
    python WhisperViaHF.py --model $model --benchmark --num-samples 1
done

# Test different precisions
python WhisperViaHF.py --dtype float16 --benchmark
python WhisperViaHF.py --dtype float32 --benchmark
```

### Performance Benchmarks
```bash
# RTF benchmark
python WhisperViaHF.py --benchmark --num-samples 10 --output-file rtf_benchmark.json

# Memory usage test
python WhisperViaHF.py --model large --benchmark --num-samples 3
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Use smaller model
python WhisperViaHF.py --model small  # instead of large

# Use float16 precision
python WhisperViaHF.py --dtype float16

# Reduce batch size (processed sequentially)
python WhisperViaHF.py --num-samples 1
```

#### Audio Format Issues
```bash
# Check audio file format
python -c "import librosa; print(librosa.load('audio.wav', sr=None))"

# Convert format if needed
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

#### Model Loading Issues
```bash
# Check transformers version
pip install transformers>=4.21.0

# Clear cache if needed
python -c "from transformers import AutoModel; AutoModel.from_pretrained('openai/whisper-base', cache_dir='/tmp/cache')"
```

## 📚 References

- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356)
- [Hugging Face Whisper Documentation](https://huggingface.co/docs/transformers/model_doc/whisper)
- [LibriSpeech Dataset](https://www.openslr.org/12)
- [Whisper Model Cards](https://huggingface.co/openai)

## 🎯 Future Enhancements

- **Streaming Processing**: Real-time audio processing
- **Batch Optimization**: Multi-sample parallel processing
- **Quality Metrics**: WER (Word Error Rate) calculation
- **Language Detection**: Automatic language identification
- **Speaker Diarization**: Multi-speaker support
- **Quantization**: Int8/Int4 precision support

---

*This Whisper implementation provides comprehensive energy profiling capabilities for speech recognition workloads, enabling detailed analysis of transformer-based audio processing across different GPU architectures and configurations.*
