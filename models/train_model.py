import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from models.cnn_lstm_model import build_cnn_lstm
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# Sequence Builder (CNN-LSTM compatible)
# --------------------------------------------------
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# --------------------------------------------------
# Target Creation
# --------------------------------------------------
def create_investment_target(df, horizon=5):
    """
    0 = Avoid
    1 = Hold
    2 = Invest
    """

    # ✅ FORCE 1-D ARRAY (THIS FIXES YOUR ERROR)
    close = df["Close"].to_numpy().ravel()

    future_return = (np.roll(close, -horizon) / close) - 1

    volatility = (
        pd.Series(close)
        .pct_change()
        .rolling(horizon)
        .std()
        .to_numpy()
    )

    target = np.zeros(len(close), dtype=int)

    bullish_mask = (
        (future_return > 0.02) &
        (volatility < np.nanmedian(volatility))
    )

    neutral_mask = (
        (future_return >= -0.01) &
        (future_return <= 0.015)
    )

    target[bullish_mask] = 2
    target[neutral_mask] = 1

    return pd.Series(target, index=df.index)



# --------------------------------------------------
# MAIN TRAIN FUNCTION
# --------------------------------------------------
def train_model(df, time_steps=10):

    df = df.copy()

    # --------- TARGET ---------
    df["Target"] = create_investment_target(df)

    # --------- FEATURES ---------
    feature_cols = [
        "Close", "MA20", "MA50", "RSI", "Volatility",
        "NIFTY_Return", "NIFTY_Volatility"
    ]

    df_model = df[feature_cols + ["Target"]].dropna()

    # 🔐 SAFETY CHECK
    if len(df_model) < time_steps + 10:
        raise ValueError(
            f"Not enough data after preprocessing. "
            f"Need at least {time_steps + 10}, got {len(df_model)}"
        )

    X = df_model[feature_cols].values
    y = df_model["Target"].values

    # --------- SCALE ---------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------- SEQUENCES ---------
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

    # --------- TRAIN / TEST SPLIT ---------
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
    )

    # --------- MODEL ---------
    model = build_cnn_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
    
    # ---------- CLASS WEIGHTS ----------
    classes = np.unique(y_seq)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_seq
    )

    class_weight_dict = dict(zip(classes, class_weights))


    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1,
        class_weight=class_weight_dict,   # 🔥 IMPORTANT
        verbose=1
    )


    return model, X_test, y_test

