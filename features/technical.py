import pandas as pd
import numpy as np

def add_technical_features(df):
    df["Return"] = df["Close"].pct_change()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Volatility"] = df["Return"].rolling(14).std()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    return df.dropna()
