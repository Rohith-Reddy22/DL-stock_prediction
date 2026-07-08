import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==========================================
# 1. THE ARCHITECTURE (Late Fusion Model)
# ==========================================
class MultimodalStockPredictor(nn.Module):
    def __init__(self, num_features=5, lstm_hidden=64, text_dim=768, num_classes=2):
        super(MultimodalStockPredictor, self).__init__()
        
        # Numerical Branch (LSTM)
        self.lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        
        # Fusion & Classification Head
        fusion_dim = lstm_hidden + text_dim
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes) 

    def forward(self, price_seq, text_vector):
        # 1. Process numerical data
        lstm_out, (hn, cn) = self.lstm(price_seq)
        lstm_last_state = hn[-1] # Grab the final hidden state
        
        # 2. Fuse the modalities
        fused_vector = torch.cat((lstm_last_state, text_vector), dim=1)
        
        # 3. Classify
        x = self.fc1(fused_vector)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

# ==========================================
# 2. DUMMY DATA GENERATOR (For instant testing)
# ==========================================
def create_dummy_data(num_samples=1000):
    print(f"Generating {num_samples} samples of dummy market data...")
    
    # Simulate 30 days of OHLCV data for each sample (Shape: [1000, 30, 5])
    # Values between 0 and 1 (simulating Min-Max scaled prices)
    X_prices = np.random.rand(num_samples, 30, 5).astype(np.float32)
    
    # Simulate FinBERT 768-dim embeddings for each sample (Shape: [1000, 768])
    X_text = np.random.randn(num_samples, 768).astype(np.float32)
    
    # Simulate binary targets: 0 (Down) or 1 (Up) (Shape: [1000])
    y_targets = np.random.randint(0, 2, size=(num_samples)).astype(np.int64)
    
    # Convert to PyTorch Tensors
    return torch.tensor(X_prices), torch.tensor(X_text), torch.tensor(y_targets)

# ==========================================
# 3. THE EXECUTION & TRAINING LOOP
# ==========================================
def main():
    # 1. Setup Data Pipeline
    X_prices, X_text, y_targets = create_dummy_data(num_samples=1000)
    
    # Package into PyTorch Datasets and DataLoaders
    dataset = TensorDataset(X_prices, X_text, y_targets)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print("Torch version:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())
    # 2. Initialize Model, Loss, and Optimizer
    device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Training on device: {device}\n")
    
    model = MultimodalStockPredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    
    # 3. Training Loop
    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for price_batch, text_batch, labels in train_loader:
            # Move data to GPU if available
            price_batch = price_batch.to(device)
            text_batch = text_batch.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(price_batch, text_batch)
            
            # Calculate Loss
            loss = criterion(predictions, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = epoch_loss / len(train_loader)
        accuracy = (correct_predictions / total_samples) * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print("\nTraining Complete! Model successfully learned the fusion architecture.")

# Run the script
if __name__ == "__main__":
    main()