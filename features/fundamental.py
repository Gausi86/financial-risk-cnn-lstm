import yfinance as yf

def fetch_fundamentals(symbol):
    info = yf.Ticker(symbol).info

    return {
        "PE_Ratio": info.get("trailingPE", 0),
        "PB_Ratio": info.get("priceToBook", 0),
        "ROE": info.get("returnOnEquity", 0),
        "DebtToEquity": info.get("debtToEquity", 0),
        "EPS": info.get("trailingEps", 0)
    }


def compute_fundamental_score(fundamentals):
    """
    Returns a normalized score [0,1]
    """

    score = 0.0

    pe = fundamentals.get("PE_Ratio")
    roe = fundamentals.get("ROE")
    debt = fundamentals.get("DebtToEquity")

    if pe and pe < 25:
        score += 0.4

    if roe and roe > 15:
        score += 0.3

    if debt and debt < 1:
        score += 0.3

    return min(score, 1.0)

