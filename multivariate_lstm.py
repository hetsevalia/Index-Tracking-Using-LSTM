import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# === Dataset Definition ===
class NiftyDataset(Dataset):
    def __init__(self, csv_file, window_size):
        df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        self.window_size = window_size

        # Features and target
        self.features = df.drop(columns=["NIFTY50"]).values
        self.targets = df["NIFTY50"].values

        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.features) - self.window_size):
            X.append(self.features[i : i + self.window_size])
            y.append(self.targets[i + self.window_size])
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# === Model Definition ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# === Training & Evaluation Pipeline ===
def main():
    # Paths
    input_dir = "processed_data"
    output_dir = os.path.join(input_dir, "final_multivariate")
    os.makedirs(output_dir, exist_ok=True)

    train_csv = os.path.join(input_dir, "train.csv")
    test_csv = os.path.join(input_dir, "test.csv")

    # Hyperparameters
    window_size = 30
    batch_size = 64
    lr = 1e-3
    epochs = 40

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_set = NiftyDataset(train_csv, window_size)
    test_set = NiftyDataset(test_csv, window_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Scaler for NIFTY50 inverse transform
    df_train = pd.read_csv(train_csv)
    nifty_scaler = MinMaxScaler()
    nifty_scaler.fit(df_train[["NIFTY50"]])

    # Save scaler
    scaler_path = os.path.join(output_dir, "nifty_scaler.save")
    joblib.dump(nifty_scaler, scaler_path)

    # Model, optimizer, loss
    input_size = train_set.X.shape[2]
    model = LSTMRegressor(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.6f}")

    # Save training loss plot
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "training_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Save model
    model_path = os.path.join(output_dir, "best_lstm_model.pth")
    torch.save(model.state_dict(), model_path)

    # Evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_true = np.array(all_true).reshape(-1, 1)
    all_preds = np.array(all_preds).reshape(-1, 1)
    true_denorm = nifty_scaler.inverse_transform(all_true)
    pred_denorm = nifty_scaler.inverse_transform(all_preds)

    mae = mean_absolute_error(true_denorm, pred_denorm)
    rmse = np.sqrt(mean_squared_error(true_denorm, pred_denorm))
    print(f"\nEvaluation on real scale -> MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(true_denorm, label='True')
    plt.plot(pred_denorm, label='Predicted')
    plt.title('NIFTY50 Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Index Value')
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "nifty50_prediction_plot.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    main()
