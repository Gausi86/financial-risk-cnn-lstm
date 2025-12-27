from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import os

nltk.download("vader_lexicon")

NEWS_API_KEY = "1a953371497c49b7a43ee8c003846107"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
csv_path = os.path.join(base_dir, "features", "securities_available_for_trading.csv")

df = pd.read_csv(csv_path)

# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]

# Rename columns to match expected names
df = df.rename(columns={
    "SYMBOL": "Symbol",
    "NAME OF COMPANY": "Company Name",
    "SERIES": "Series"  # if you want to keep existing logic
})

# Now filter equity
df_equity = df[df["Series"] == "EQ"].copy()

# Build SYMBOL_MAP
SYMBOL_MAP = dict(zip(df_equity["Symbol"] + ".NS", df_equity["Company Name"]))

def fetch_sentiment(symbol, return_news=False):

    company = SYMBOL_MAP.get(symbol, symbol.replace(".NS", ""))
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    articles = newsapi.get_everything(
        q=company,
        language="en",
        sort_by="publishedAt",
        page_size=30
    )

    sia = SentimentIntensityAnalyzer()

    scores = []
    headlines = []

    for article in articles["articles"]:
        text = f"{article.get('title','')} {article.get('description','')} {article.get('content','')}"
        compound = sia.polarity_scores(text)["compound"]

        # Ignore weak signals
        if abs(compound) > 0.15:
            scores.append(compound)
            headlines.append(article["title"])

    if not scores:
        return (0, []) if return_news else 0

    # Weighted sentiment (recent news matters more)
    weights = list(range(1, len(scores)+1))
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    if return_news:
        return weighted_score, headlines

    return weighted_score

