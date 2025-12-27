import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------- IMPORT PROJECT MODULES --------------------
from data.stock_data import fetch_stock_data
from data.index_data import fetch_nifty_data, market_is_bullish
from features.technical import add_technical_features
from features.fundamental import fetch_fundamentals, compute_fundamental_score
from features.sentiment import fetch_sentiment
from models.train_model import train_model
from evaluation.metrics import get_metrics

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Stock Investment Decision System",
    layout="wide"
)

st.title("📊 AI-Based Stock Investment Decision System (CNN + LSTM)")



# -------------------- DISCLAIMER --------------------
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

if not st.session_state.disclaimer_accepted:
    st.warning(
        """
        ⚠ **Academic Disclaimer**

        This dashboard is part of an M.Tech (Applied AI) academic project.

        - This system is **NOT financial advice**
        - Outputs are **experimental**
        - Do **NOT** use for real trading or investment decisions
        """
    )

    def accept_disclaimer():
        st.session_state.disclaimer_accepted = True

    st.button("✅ I Understand & Proceed", on_click=accept_disclaimer)
    st.stop()  # Stop further execution until accepted

# -------------------- LOAD SYMBOLS --------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "../features/securities_available_for_trading.csv")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Filter only equity series
    if 'Series' in df.columns:
        df_eq = df[df['Series'] == 'EQ'].copy()
    elif 'SYMBOL' in df.columns and 'NAME OF COMPANY' in df.columns:
        # If Series column is missing, include all for autocomplete
        df_eq = df.copy()
    else:
        st.error(f"CSV missing required columns. Found: {df.columns.tolist()}")
        st.stop()

    SYMBOL_MAP = dict(zip(df_eq["SYMBOL"] + ".NS", df_eq["NAME OF COMPANY"]))

except Exception as e:
    st.error(f"Error loading securities CSV: {e}")
    st.stop()

# -------------------- SYMBOL INPUT (AUTOCOMPLETE) --------------------
symbol_full = st.selectbox(
    "Select NSE Stock Symbol",
    options=list(SYMBOL_MAP.keys()),
    format_func=lambda x: f"{x} — {SYMBOL_MAP[x]}",
    index=None
)

symbol = symbol_full.split(" — ")[0] if symbol_full else None
if not symbol:
    st.warning("Please select a stock symbol")
    st.stop()
    
analyze_btn = st.button("🔍 Analyze Stock")

# -------------------- FINAL DECISION FUNCTION --------------------
def final_investment_decision(model_class, model_confidence, sentiment_score, fundamental_score, market_ok):
    sentiment_norm = (sentiment_score + 1) / 2
    if model_class == 2:
        model_score = model_confidence
    elif model_class == 1:
        model_score = 0.5
    else:
        model_score = 0.2

    final_score = 0.45*model_score + 0.30*sentiment_norm + 0.25*fundamental_score

    if not market_ok and final_score < 0.60:
        return "HOLD (Market Risk)", final_score
    if model_class == 2 and final_score > 0.55:
        return "GOOD TO INVEST", final_score
    elif model_class >= 1 and final_score > 0.40:
        return "HOLD / WATCH", final_score
    else:
        return "AVOID INVESTING", final_score

# -------------------- MAIN PIPELINE --------------------
if analyze_btn:
    
    for key in ["df", "nifty", "preds", "model"]:
        if key in st.session_state:
            del st.session_state[key]
    
    progress = st.progress(0)
    status = st.empty()

    # 1️⃣ Fetch stock & index data
    status.text("📥 Fetching stock & index data...")
    progress.progress(10)
    df = fetch_stock_data(symbol)
    nifty = fetch_nifty_data()

    # 2️⃣ Technical features
    status.text("⚙️ Computing technical indicators...")
    progress.progress(25)
    df = add_technical_features(df)
    df = df.join(nifty, how="inner")
    df.dropna(inplace=True)

    # 3️⃣ Sentiment
    status.text("📰 Analyzing market news sentiment...")
    progress.progress(40)
    sentiment_score, sentiment_news = fetch_sentiment(symbol, return_news=True)

    # 4️⃣ Fundamentals
    status.text("📊 Fetching fundamental data...")
    progress.progress(50)
    fundamentals = fetch_fundamentals(symbol)
    fundamental_score = compute_fundamental_score(fundamentals)

    # 5️⃣ Train model
    status.text("🧠 Training CNN-LSTM model...")
    progress.progress(70)
    model, X_test, y_test = train_model(df)
    preds = model.predict(X_test)
    latest_probs = preds[-1]
    model_class = int(np.argmax(latest_probs))
    model_confidence = float(np.max(latest_probs))
    class_map = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

    # 6️⃣ Market regime
    market_ok = market_is_bullish(nifty)

    # 7️⃣ Final decision
    status.text("📈 Generating investment decision...")
    progress.progress(90)
    decision, final_score = final_investment_decision(model_class, model_confidence, sentiment_score, fundamental_score, market_ok)
    progress.progress(100)
    status.success("✅ Analysis Complete")

    # -------------------- OUTPUT --------------------
    st.subheader("📌 Investment Recommendation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Regime", class_map[model_class])
    col2.metric("Model Confidence", round(model_confidence, 2))
    col3.metric("Sentiment Score", round(sentiment_score, 2))
    col4.metric("Final Score", round(final_score, 2))

    if "GOOD" in decision:
        st.success(f"✅ {decision}")
    elif "HOLD" in decision:
        st.warning(f"⚠️ {decision}")
    else:
        st.error(f"❌ {decision}")

    # -------------------- DECISION EXPLANATION --------------------
    st.subheader("🧾 How This Decision Was Made")
    softmax_probs = latest_probs.round(3)
    sentiment_norm_val = (sentiment_score + 1) / 2
    sentiment_label = "Positive" if sentiment_score > 0.2 else "Neutral" if sentiment_score > -0.2 else "Negative"

    st.markdown(f"""
    ### 1️⃣ CNN-LSTM Market Regime Detection
    • Predicted regime: **{class_map[model_class]}**  
    • Softmax probabilities: Bearish **{softmax_probs[0]}**, Neutral **{softmax_probs[1]}**, Bullish **{softmax_probs[2]}**  
    • Model confidence: **{model_confidence:.2f}**
    """)

    st.markdown(f"""
    ### 2️⃣ Market News Sentiment
    • Score: **{sentiment_score:.2f}**  
    • Normalized (0-1): **{sentiment_norm_val:.2f}**  
    • Interpretation: **{sentiment_label}**
    """)

    st.markdown(f"""
    ### 3️⃣ Fundamental Strength
    • Score: **{fundamental_score:.2f}**
    • Based on valuation, profitability, leverage (PE, ROE, Debt/Equity)
    """)

    st.markdown(f"""
    ### 4️⃣ Overall Market Trend
    • NIFTY: **{'Bullish' if market_ok else 'Cautious / Bearish'}**
    """)

    weighted_table = pd.DataFrame({
        "Component": ["CNN-LSTM Model", "Market Sentiment", "Fundamental Strength"],
        "Weight": [0.45, 0.30, 0.25],
        "Raw Score": [round(model_confidence,2), round(sentiment_norm_val,2), round(fundamental_score,2)],
        "Weighted Contribution": [round(0.45*model_confidence,2), round(0.30*sentiment_norm_val,2), round(0.25*fundamental_score,2)]
    })
    st.markdown("### 🔍 Score Contribution Breakdown")
    st.table(weighted_table)
    st.markdown(f"### 📌 Final Decision: **{decision}** (Final Score: {round(final_score,3)})")

    # -------------------- CHARTS --------------------
    st.markdown("---")
    st.subheader("📈 Price & Moving Averages")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["MA20"], label="MA20")
    ax.plot(df.index, df["MA50"], label="MA50")
    ax.legend(fontsize=10)
    st.pyplot(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 CNN-LSTM Confidence Over Time")
        conf_series = pd.Series(preds.max(axis=1), index=df.tail(len(preds)).index)
        st.line_chart(conf_series)
    with col2:
        st.subheader("📊 Volatility vs Model Confidence")
        scatter_df = pd.DataFrame({
            "Volatility": df["Volatility"].tail(len(preds)),
            "Confidence": preds.max(axis=1)
        })
        st.scatter_chart(scatter_df)

    st.subheader("🧠 Market Sentiment Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={
            "axis": {"range": [-1, 1]},
            "steps": [
                {"range": [-1, -0.3], "color": "red"},
                {"range": [-0.3, 0.3], "color": "lightgray"},
                {"range": [0.3, 1], "color": "green"},
            ],
            "bar": {"color": "darkblue"}
        }
    ))
    st.plotly_chart(gauge, use_container_width=True, height=250)

    st.subheader("📰 Key Market News")
    if sentiment_news:
        for news in sentiment_news[:5]:
            st.write("•", news)
    else:
        st.info("No strong sentiment news found.")

    st.subheader("📉 RSI Indicator")
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, linestyle="--", color="red")
    ax2.axhline(30, linestyle="--", color="green")
    ax2.legend(fontsize=10)
    st.pyplot(fig2, use_container_width=True)

    st.subheader("📋 Fundamental Metrics")
    fund_table = pd.DataFrame.from_dict(fundamentals, orient="index", columns=["Value"])
    st.table(fund_table.round(2))

    # -------------------- MODEL EVALUATION --------------------
    st.markdown("---")
    st.subheader("📊 CNN-LSTM Model Evaluation")
    y_pred_labels = np.argmax(preds[:len(y_test)], axis=1)
    metrics_df, cm = get_metrics(y_test, y_pred_labels)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig_cm, ax_cm = plt.subplots(figsize=(6,4))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=ax_cm, cbar=False)
    ax_cm.set_xlabel("Predicted", fontsize=10)
    ax_cm.set_ylabel("Actual", fontsize=10)
    ax_cm.tick_params(axis='both', labelsize=10)
    st.pyplot(fig_cm, use_container_width=True)

    st.subheader("Classification Metrics (Precision, Recall, F1)")
    st.dataframe(metrics_df.iloc[:3][["precision","recall","f1-score"]].round(2))

    fig_metrics, ax_metrics = plt.subplots(figsize=(6,4))
    metrics_df.iloc[:3][["precision","recall","f1-score"]].plot(kind='bar', ax=ax_metrics)
    ax_metrics.set_title("Precision / Recall / F1 per Class", fontsize=12)
    ax_metrics.set_xticklabels(["Bearish","Neutral","Bullish"], rotation=0, fontsize=10)
    ax_metrics.legend(fontsize=10)
    st.pyplot(fig_metrics, use_container_width=True)
