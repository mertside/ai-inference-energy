import sys
import time

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model configuration
additional_metrics = ["accuracy"]
batch_size = 1024
embedding_output_dims = 15
loss_function = BinaryCrossentropy()
max_sequence_length = 300
num_distinct_words = 5000
number_of_epochs = 1
optimizer = Adam()
validation_split = 0.20
verbosity_mode = 1

# Check for GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    print("WARNING: No GPU detected. Running on CPU (may be slow).")
else:
    print(f"GPU devices found: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"Failed to set memory growth for {gpu}: {e}")

# Load dataset
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_distinct_words)
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Pad all sequences
print("Padding sequences...")
padded_inputs = pad_sequences(x_train, maxlen=max_sequence_length, value=0.0)
padded_inputs_test = pad_sequences(x_test, maxlen=max_sequence_length, value=0.0)

# Define the Keras model
print("Building model...")
model = Sequential()
model.add(Embedding(num_distinct_words, embedding_output_dims, input_length=max_sequence_length))
model.add(LSTM(10))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)

# Give a summary
model.summary()

print("Starting training...")
start_time = time.time()

# Train the model (TensorFlow will automatically use GPU if available)
history = model.fit(
    padded_inputs,
    y_train,
    batch_size=batch_size,
    epochs=number_of_epochs,
    verbose=verbosity_mode,
    validation_split=validation_split,
)

# Test the model after training
print("Evaluating model...")
test_results = model.evaluate(padded_inputs_test, y_test, verbose=False)
end_time = time.time()

print(f"Test results - Loss: {test_results[0]} - Accuracy: {100*test_results[1]:.2f}%")
print(f"Total execution time: {end_time - start_time:.2f} seconds")

# Additional GPU info if available
if gpus:
    print("\nGPU Memory Usage:")
    for gpu in gpus:
        gpu_details = tf.config.experimental.get_device_details(gpu)
        print(f"  {gpu.name}: {gpu_details}")
