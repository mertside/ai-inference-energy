# LSTM Sentiment Analysis Benchmark

This directory contains the LSTM sentiment analysis benchmark application for the AI Inference Energy Profiling Framework.

## Overview

The LSTM application provides a standardized binary sentiment classification benchmark using TensorFlow/Keras. It's designed as a consistent, reproducible workload for energy profiling experiments across different GPU frequency settings.

## Files

- **`lstm.py`**: Main LSTM sentiment analysis benchmark application

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

### Standalone Usage
```bash
cd app-lstm
python lstm.py
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

### Typical Execution Times
- **1 epoch**: ~30-60 seconds (depending on GPU)
- **Memory usage**: ~2-4GB GPU memory
- **CPU usage**: Moderate (data preprocessing)

### Profiling Benefits
- **Consistent workload**: Same data and model every run
- **Predictable duration**: Reliable for timed experiments
- **GPU utilization**: Good balance of compute and memory operations
- **Scalable**: Easily adjustable for different experiment needs

-## Requirements

- Python 3.6+
- TensorFlow 2.x
- Keras (included with TensorFlow)
- NumPy
- CUDA-capable GPU (recommended)

## Output

The application provides:
```
Train/validation accuracy and loss
Test accuracy and loss
Total execution time
Model summary
```

Example output:
```
Train on 16000 samples, validate on 4000 samples
Epoch 1/1
16000/16000 [==============================] - 45s 3ms/sample
Test results - Loss: 0.4234 - Accuracy: 78.45%
42.56 seconds
```

## Troubleshooting

### Common Issues

1. **TensorFlow GPU Issues**
   ```bash
   # Check TensorFlow GPU support
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

2. **Memory Issues**
   - Reduce batch_size (e.g., from 1024 to 512)
   - Reduce max_sequence_length
   - Close other GPU applications

3. **Dataset Download Issues**
   - Ensure internet connectivity
   - TensorFlow will automatically download IMDB dataset on first run

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
