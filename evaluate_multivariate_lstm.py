import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# === Dataset for Testing Only ===
class NiftyDataset(Dataset):
    def __init__(self, csv_file, window_size):
        df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        self.window_size = window_size
        self.features = df.drop(columns=["NIFTY50"]).values
        self.targets = df["NIFTY50"].values
        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.features) - self.window_size):
            X.append(self.features[i:i+self.window_size])
            y.append(self.targets[i + self.window_size])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Model Definition ===
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def evaluate():
    input_dir = "processed_data"
    output_dir = os.path.join(input_dir, "final_multivariate")
    test_csv = os.path.join(input_dir, "test.csv")
    model_path = os.path.join(output_dir, "best_lstm_model.pth")
    scaler_path = os.path.join(output_dir, "nifty_scaler.save")

    window_size = 30
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and scaler
    test_set = NiftyDataset(test_csv, window_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    nifty_scaler = joblib.load(scaler_path)

    # Load model
    input_size = test_set.X.shape[2]
    model = LSTMRegressor(input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run inference
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_true = np.array(all_true).reshape(-1, 1)
    all_preds = np.array(all_preds).reshape(-1, 1)

    # Inverse scale
    true_denorm = nifty_scaler.inverse_transform(all_true)
    pred_denorm = nifty_scaler.inverse_transform(all_preds)

    # Metrics
    mse = mean_squared_error(true_denorm, pred_denorm)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_denorm, pred_denorm)

    print(f"\nEvaluation on real scale:")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")
    
    # Refined Real vs Predicted Line Plot
    plt.figure(figsize=(14, 6))
    plt.plot(true_denorm, label='Actual NIFTY50', linewidth=2)
    plt.plot(pred_denorm, label='Predicted NIFTY50', linewidth=2, linestyle='--')
    plt.title('Actual vs Predicted NIFTY50 (Test Set)', fontsize=14)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('NIFTY50 Index Value', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "nifty50_prediction_plot.png")
    plt.savefig(plot_path)
    plt.show()



if __name__ == "__main__":
    evaluate()
