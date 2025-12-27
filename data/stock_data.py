import yfinance as yf

def fetch_stock_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

