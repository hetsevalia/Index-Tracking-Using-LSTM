import yfinance as yf
import pandas as pd
import os

ticker = "^NSEI"
start_date = "2020-01-01"
end_date = "2025-06-30"
output_dir = "nifty50_lstm_data"
os.makedirs(output_dir, exist_ok=True)

df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

print("Downloaded DataFrame:")
print(df.head())

# Handle MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    # Try to find a usable 'Close' or 'Adj Close'
    close_col = None
    for col in df.columns:
        if col[0] in ['Adj Close', 'Close', 'Price']:
            close_col = col
            break

    if close_col:
        df = df[[close_col]]
        df.columns = ['NIFTY50']
        df.index.name = 'Date'
        df.dropna(inplace=True)
        df.to_csv(os.path.join(output_dir, "NIFTY50.csv"))
        print(" NIFTY50 data saved.")
    else:
        print(" 'Close' or 'Adj Close' column not found.")
else:
    if 'Adj Close' in df.columns:
        df = df[['Adj Close']].rename(columns={'Adj Close': 'NIFTY50'})
    elif 'Close' in df.columns:
        df = df[['Close']].rename(columns={'Close': 'NIFTY50'})
    else:
        print(" No usable price column found.")
        exit()

    df.index.name = 'Date'
    df.dropna(inplace=True)
    df.to_csv(os.path.join(output_dir, "NIFTY50.csv"))
    print(" NIFTY50 data saved.")
