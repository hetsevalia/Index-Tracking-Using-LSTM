import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os

# Hyperparameters
SEQ_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 100
LR = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Loader
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_len):
            self.X.append(data[i:i+seq_len])
            self.y.append(data[i+seq_len])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 🧠 Only run training if file is executed directly
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    # Load Data 
    train_df = pd.read_csv("processed_data/train.csv", parse_dates=["Date"])
    test_df = pd.read_csv("processed_data/test.csv", parse_dates=["Date"])

    # Use only features (exclude Date & NIFTY50)
    feature_cols = [col for col in train_df.columns if col not in ["Date", "NIFTY50"]]
    target_col = "NIFTY50"

    train_targets = train_df[target_col].values

    train_dataset = TimeSeriesDataset(train_targets.reshape(-1, 1), SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model Setup
    model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training
    losses = []
    print("Training started...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # Save model
    os.makedirs("processed_data", exist_ok=True)
    torch.save(model.state_dict(), "processed_data/model.pth")
    print("Model saved to processed_data/model.pth")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Train Loss", color="blue")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("processed_data/loss_curve.png")
    plt.show()
