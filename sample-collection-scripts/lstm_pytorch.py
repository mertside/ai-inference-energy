#!/usr/bin/env python3
"""
PyTorch LSTM for Sentiment Analysis - Energy Profiling Compatible
Compatible with AI Inference Energy Framework
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Minimal IMDB-style dataset simulation for consistent benchmarking
def create_synthetic_imdb_data(num_samples=1000, max_sequence_length=300, vocab_size=5000):
    """Create synthetic IMDB-like data for consistent energy profiling"""
    # Generate random sequences
    x_data = np.random.randint(1, vocab_size, size=(num_samples, max_sequence_length))
    y_data = np.random.randint(0, 2, size=(num_samples,))  # Binary sentiment
    
    return x_data, y_data

class LSTMSentimentClassifier(nn.Module):
    """PyTorch LSTM for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMSentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last output
        output = self.fc(lstm_out[:, -1, :])
        
        return self.sigmoid(output)

def main():
    """Main execution function for energy profiling"""
    print("🤖 Starting PyTorch LSTM Sentiment Analysis")
    
    # Configuration
    vocab_size = 5000
    embedding_dim = 15
    hidden_dim = 64
    output_dim = 1
    batch_size = 1024
    num_epochs = 1
    max_sequence_length = 300
    num_samples = 10000
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Generate synthetic data
    print("📊 Generating synthetic IMDB-like dataset...")
    x_data, y_data = create_synthetic_imdb_data(num_samples, max_sequence_length, vocab_size)
    
    # Convert to tensors
    x_tensor = torch.LongTensor(x_data)
    y_tensor = torch.FloatTensor(y_data).unsqueeze(1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"📈 Dataset size: {len(dataset)} samples")
    print(f"🎯 Batch size: {batch_size}")
    print(f"🔄 Number of batches: {len(dataloader)}")
    
    # Initialize model
    print("🧠 Initializing LSTM model...")
    model = LSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"📋 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("🚀 Starting training...")
    start_time = time.time()
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)
            
            if batch_idx % 2 == 0:  # Print every 2 batches
                print(f"⚡ Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        epoch_acc = total_correct / total_samples
        
        print(f"✅ Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"📊 Accuracy: {epoch_acc:.4f}, Average Loss: {total_loss/len(dataloader):.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("🧪 Final evaluation...")
    model.eval()
    with torch.no_grad():
        test_data, test_target = x_tensor[:1000].to(device), y_tensor[:1000].to(device)
        test_output = model(test_data)
        test_predicted = (test_output > 0.5).float()
        test_accuracy = (test_predicted == test_target).sum().item() / test_target.size(0)
    
    print(f"🎯 Final test accuracy: {test_accuracy:.4f}")
    print(f"⏱️  Total training time: {training_time:.2f} seconds")
    print(f"🔥 Training completed successfully!")
    
    # GPU memory info (if available)
    if torch.cuda.is_available():
        print(f"💾 GPU Memory Used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"💾 GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    return training_time, test_accuracy

if __name__ == "__main__":
    main()
