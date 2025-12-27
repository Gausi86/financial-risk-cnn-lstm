import yfinance as yf
import pandas as pd

def fetch_nifty_data():
    nifty = yf.download("^NSEI", period="2y", interval="1d")
    nifty["NIFTY_Return"] = nifty["Close"].pct_change()
    nifty["NIFTY_Volatility"] = nifty["NIFTY_Return"].rolling(14).std()
    return nifty[["NIFTY_Return", "NIFTY_Volatility"]]

def market_is_bullish(index_df):
    """
    Determines if overall market trend is bullish using MA50 & MA200
    Works with both single-index and multi-index yfinance outputs
    """

    df = index_df.copy()

    # ---------------- HANDLE COLUMN FORMAT ----------------
    if isinstance(df.columns, tuple) or hasattr(df.columns, "levels"):
        # MultiIndex case (yfinance group_by)
        close_cols = [c for c in df.columns if "close" in str(c).lower()]
        if not close_cols:
            return True  # fail-safe
        close_col = close_cols[0]
        close = df[close_col]
    else:
        # Normal single-level columns
        if "Close" not in df.columns:
            return True  # fail-safe
        close = df["Close"]

    # ---------------- TREND LOGIC ----------------
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    if len(ma50.dropna()) == 0 or len(ma200.dropna()) == 0:
        return True  # not enough data → neutral bullish

    return ma50.iloc[-1] > ma200.iloc[-1]
