import yfinance as yf
import pandas as pd
import ta
import os

# Stable Nifty 50 stocks from Jan 2020 to June 2025
tickers = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
    'KOTAKBANK.NS', 'LT.NS', 'BHARTIARTL.NS', 'ITC.NS', 'ASIANPAINT.NS',
    'HINDUNILVR.NS', 'BAJFINANCE.NS', 'HCLTECH.NS', 'MARUTI.NS', 'TITAN.NS',
    'WIPRO.NS', 'SBIN.NS', 'NTPC.NS', 'POWERGRID.NS', 'TECHM.NS'
]

start_date = "2020-01-01"
end_date = "2025-06-30"
output_dir = "nifty50_lstm_data"
os.makedirs(output_dir, exist_ok=True)

def add_technical_indicators(df):
    df['MA10'] = df['Adj_Close'].rolling(window=10).mean()
    df['MA30'] = df['Adj_Close'].rolling(window=30).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Adj_Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Adj_Close'])
    df['MACD'] = macd.macd_diff()
    return df

for ticker in tickers:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if df.empty:
        print(f"[WARNING] No data for {ticker}. Skipping.")
        continue

    try:
        if isinstance(df.columns, pd.MultiIndex):
            adj_close = None
            for col in df.columns:
                if isinstance(col, tuple) and 'Adj Close' in col:
                    adj_close = df[col]
                    break
            if adj_close is None:
                raise KeyError("'Adj Close' not found in MultiIndex.")
            df = pd.DataFrame({'Adj_Close': adj_close})
        else:
            df = df[['Adj Close']].rename(columns={'Adj Close': 'Adj_Close'})

    except Exception as e:
        print(f"[ERROR] Could not process {ticker}: {e}")
        print("Columns present:", df.columns)
        continue

    df = add_technical_indicators(df)
    df.dropna(inplace=True)
    df.to_csv(f"{output_dir}/{ticker.replace('.NS','')}.csv")
    print(f"Saved {ticker}")

print("Downloading Nifty 50 index...")
nifty = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=False, progress=False)

if not nifty.empty and 'Adj Close' in nifty.columns:
    nifty = nifty[['Adj Close']].rename(columns={'Adj Close': 'NIFTY50'})
    nifty.dropna(inplace=True)
    nifty.index.name = 'Date'  # Sets the index column name for CSV
    nifty.to_csv(os.path.join(output_dir, "NIFTY50.csv"))
    print("Nifty 50 index saved in clean format.")
else:
    print("[ERROR] Failed to download Nifty 50 index.")
