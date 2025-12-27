from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        LSTM(64),
        Dropout(0.3),
        Dense(3, activation="softmax")   # 🔥 3 classes
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

