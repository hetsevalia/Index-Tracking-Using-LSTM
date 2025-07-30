import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lstm_model import LSTMModel, TimeSeriesDataset
import matplotlib.pyplot as plt
import os

# Hyperparameters (must match training)
SEQ_LENGTH = 30
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Test Data
test_df = pd.read_csv("processed_data/test.csv", parse_dates=["Date"])
feature_cols = [col for col in test_df.columns if col not in ["Date", "NIFTY50"]]
target_col = "NIFTY50"
test_targets = test_df[target_col].values

# Use target column for time series prediction
test_dataset = TimeSeriesDataset(test_targets.reshape(-1, 1), SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load Model
model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE,
                  num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
model.load_state_dict(torch.load("processed_data/model.pth"))
model.eval()

# Predict
predictions = []
actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        output = model(x_batch)
        predictions.append(output.item())
        actuals.append(y_batch.item())

# Convert to NumPy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)

# Metrics
rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f"\n Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# Save plot to processed_data
plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual', color='black')
plt.plot(predictions, label='Predicted', color='red')
plt.title('NIFTY50: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs("processed_data", exist_ok=True)
plt.savefig("processed_data/prediction_plot.png")
plt.show()

print(" Prediction plot saved to: processed_data/prediction_plot.png")
