import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Directories
input_dir = "nifty50_lstm_data"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

# Read NIFTY50 index
nifty_path = os.path.join(input_dir, "NIFTY50.csv")
nifty_df = pd.read_csv(nifty_path, parse_dates=["Date"])
nifty_df.set_index("Date", inplace=True)

# Create empty DataFrame with NIFTY50 dates as base
combined_df = pd.DataFrame(index=nifty_df.index)
tickers = []

# Read all stock files except NIFTY50
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename != "NIFTY50.csv":
        ticker = filename.replace(".csv", "")
        tickers.append(ticker)

        df = pd.read_csv(os.path.join(input_dir, filename), parse_dates=["Date"])
        df.set_index("Date", inplace=True)

        # Keep relevant columns only
        df = df[["Adj_Close", "MA10", "MA30", "RSI", "MACD"]]
        df.columns = [f"{ticker}_{col}" for col in df.columns]

        # Merge with combined_df
        combined_df = combined_df.join(df, how="outer")

# Add NIFTY50 as target
combined_df["NIFTY50"] = nifty_df["NIFTY50"]

# Drop rows with missing values
combined_df.dropna(inplace=True)

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(combined_df),
                          columns=combined_df.columns,
                          index=combined_df.index)

# Train/test split (80% train, 20% test)
split_index = int(len(normalized) * 0.8)
train_df = normalized.iloc[:split_index]
test_df = normalized.iloc[split_index:]

# Save processed CSVs
train_df.to_csv(os.path.join(output_dir, "train.csv"))
test_df.to_csv(os.path.join(output_dir, "test.csv"))

print(" Preprocessing complete. Files saved in 'processed_data/'")
