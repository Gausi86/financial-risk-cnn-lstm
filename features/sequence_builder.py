import numpy as np

def create_sequences(X, y, time_steps=10):
    """
    Converts tabular data into sequences for CNN-LSTM
    X: numpy array (samples, features)
    y: numpy array (labels)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])   # ✅ FIXED
    return np.array(Xs), np.array(ys)

