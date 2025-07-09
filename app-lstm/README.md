# LSTM Sentiment Analysis Benchmark

This directory contains the LSTM sentiment analysis benchmark application for the AI Inference Energy Profiling Framework.

## Overview

The LSTM application provides a standardized binary sentiment classification benchmark using TensorFlow/Keras. It's designed as a consistent, reproducible workload for energy profiling experiments across different GPU frequency settings.

## Files

- **`lstm.py`**: Main LSTM sentiment analysis benchmark application
- **`lstm_modern.py`**: Modern TensorFlow 2.x LSTM implementation with enhanced features
- **`requirements_h100.txt`**: Complete pip requirements for H100 environment
- **`requirements_h100_minimal.txt`**: Minimal requirements for easy setup
- **`lstm-h100-20250708.yml`**: Conda environment file for H100 TensorFlow setup
- **`tensorflow.yml`**: Alternative TensorFlow environment configuration
- **`setEnv.sh`**: Environment setup script

## Features

- 🧠 **LSTM Neural Network**: Binary sentiment classification using IMDB movie reviews
- 📊 **Consistent Workload**: Standardized parameters for reproducible profiling
- ⚡ **TensorFlow/Keras**: Optimized for GPU acceleration
- 🔧 **Configurable Parameters**: Adjustable epochs, batch size, and model architecture
- 📈 **Performance Metrics**: Accuracy and loss tracking
- ⏱️ **Timing**: Built-in execution time measurement

## Model Architecture

```python
Sequential([
    Embedding(5000, 15, input_length=300),  # Word embeddings
    LSTM(10),                               # LSTM layer
    Dense(1, activation='sigmoid')          # Binary classification
])
```

## Default Configuration

```python
# Dataset parameters
num_distinct_words = 5000        # Vocabulary size
max_sequence_length = 300        # Maximum review length

# Training parameters
number_of_epochs = 1             # Training epochs
batch_size = 1024               # Batch size
validation_split = 0.20         # Validation data percentage

# Model parameters
embedding_output_dims = 15       # Embedding dimensions
lstm_units = 10                 # LSTM hidden units
```

## Usage

### Quick Start (H100 Environment)

**Using the pre-configured conda environment:**
```bash
# Load conda and activate the H100 environment
conda activate tensorflow-lstm
cd app-lstm
python lstm_modern.py       # Run modern LSTM benchmark
```

**Creating the environment from scratch:**
```bash
# Option 1: From conda environment file (H100 optimized)
conda env create -f lstm-h100-20250708.yml

# Option 2: From alternative conda environment file
conda env create -f tensorflow.yml

# Option 3: From pip requirements (minimal)
conda create -n tensorflow-lstm python=3.10
conda activate tensorflow-lstm
pip install -r requirements_h100_minimal.txt

# Option 4: From pip requirements (complete)
pip install -r requirements_h100.txt

# Option 5: Using setup script
source setEnv.sh
```

### Standalone Usage
```bash
cd app-lstm

# Run original LSTM
python lstm.py

# Run modern LSTM (recommended)
python lstm_modern.py
```

### Profiling Usage
```bash
# From sample-collection-scripts directory
./launch.sh  # Uses LSTM by default

# Explicit LSTM specification
./launch.sh --app-name "LSTM" --app-executable "lstm"
```

### Custom Configuration
You can modify the parameters directly in `lstm.py`:
```python
# Increase epochs for longer workload
number_of_epochs = 5

# Adjust batch size for different memory usage
batch_size = 2048

# Modify model complexity
lstm_units = 20
embedding_output_dims = 30
```

## Dataset

- **Source**: IMDB Movie Reviews Dataset (via TensorFlow/Keras)
- **Task**: Binary sentiment classification (positive/negative)
- **Training samples**: ~20,000 reviews
- **Test samples**: ~5,000 reviews
- **Vocabulary**: Top 5,000 most frequent words

## Performance Characteristics

### H100 Performance (Tested July 8, 2025)
- **GPU**: NVIDIA H100 NVL (93GB memory)
- **1 epoch**: ~6-8 seconds
- **Memory usage**: ~2GB GPU memory
- **Accuracy**: ~64% (20 epochs, IMDB dataset)

### Typical Execution Times
- **1 epoch**: ~6-60 seconds (depending on GPU)
- **Memory usage**: ~2-4GB GPU memory
- **CPU usage**: Moderate (data preprocessing)

### Profiling Benefits
- **Consistent workload**: Same data and model every run
- **Predictable duration**: Reliable for timed experiments
- **GPU utilization**: Good balance of compute and memory operations
- **Scalable**: Easily adjustable for different experiment needs

## Requirements

### H100 Environment (Tested & Working)
- **Python**: 3.10.18
- **TensorFlow**: 2.19.0 (with CUDA 12.x support)
- **CUDA**: 12.2+ (bundled with TensorFlow)
- **cuDNN**: 9.3.0+ (bundled with TensorFlow)
- **NumPy**: 1.26.4
- **Pandas**: 2.3.1
- **Matplotlib**: 3.10.3
- **Scikit-learn**: 1.7.0

### General Requirements
- Python 3.10+
- TensorFlow 2.15+ (2.19.0 recommended for H100)
- CUDA-capable GPU (H100 tested and verified)
- 8GB+ GPU memory (for full dataset)

## Output

The application provides:
```
TensorFlow version and CUDA support status
GPU device information
Train/validation accuracy and loss
Test accuracy and loss
Total execution time
Model summary
```

Example output (lstm_modern.py on H100):
```
=== TensorFlow H100 Test ===
TensorFlow version: 2.19.0
Built with CUDA: True
GPU devices found: 1
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

Training data shape: (25000,)
Test data shape: (25000,)
Building model...
Starting training...
20/20 ━━━━━━━━━━━━━━━━━━━━ 3s 25ms/step - accuracy: 0.5528 - loss: 0.6912 - val_accuracy: 0.6278 - val_loss: 0.6834
Evaluating model...
Test results - Loss: 0.6834374070167542 - Accuracy: 63.98%
Total execution time: 6.29 seconds
```

## Troubleshooting

### Common Issues

1. **TensorFlow GPU Issues**
   ```bash
   # Check TensorFlow GPU support
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **H100 CUDA Issues**
   - Use TensorFlow 2.19.0 with bundled CUDA: `pip install tensorflow[and-cuda]==2.19.0`
   - Avoid mixing conda CUDA with pip TensorFlow
   - Use the provided environment files for tested configurations

3. **Memory Issues**
   - Reduce batch_size (e.g., from 1024 to 512)
   - Reduce max_sequence_length
   - Close other GPU applications

4. **Environment Setup Issues**
   ```bash
   # Reset environment
   conda deactivate
   conda env remove -n tensorflow-lstm
   conda env create -f lstm-h100-20250708.yml
   ```

5. **Dataset Download Issues**
   - Ensure internet connectivity
   - TensorFlow will automatically download IMDB dataset on first run

### H100 Specific Notes
- **Tested on**: Texas Tech REPACSS cluster (July 8, 2025)
- **Working config**: TensorFlow 2.19.0 with CUDA 12.2
- **Memory**: 93GB available on H100 NVL
- **Compute capability**: 9.0

## Integration

This application integrates with:
- **Main profiling framework** (`sample-collection-scripts/`)
- **Configuration system** (`config.py`)
- **SLURM job submission** scripts
- **Frequency sweeping** experiments

## Profiling Characteristics

The LSTM benchmark is ideal for profiling because it:
- **Consistent**: Same computation every run
- **Balanced**: Uses both compute and memory operations
- **Scalable**: Easy to adjust workload intensity
- **Fast**: Completes quickly for rapid experimentation
- **Reliable**: Stable across different GPU frequencies

For detailed usage examples, see [`documentation/USAGE_EXAMPLES.md`](../documentation/USAGE_EXAMPLES.md).
