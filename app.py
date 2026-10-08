import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Bitcoin Forecast Dashboard",
    page_icon="₿",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def get_data():

    try:
        df = yf.download(
            "BTC-USD",
            period="max",
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        required_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()

        df = df[required_cols]

        return df.dropna()

    except Exception as e:
        st.error(f"Error mengambil data: {e}")
        return pd.DataFrame()

# ==================================================
# PREPARE DATA
# ==================================================
def prepare_data(series, lookback=60):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        series.values.reshape(-1, 1)
    )

    X = []
    y = []

    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])

    return (
        np.array(X),
        np.array(y),
        scaler
    )

# ==================================================
# MODEL
# ==================================================
def create_model(input_shape):

    model = Sequential()

    model.add(
        LSTM(
            100,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        LSTM(100)
    )

    model.add(
        Dropout(0.3)
    )

    model.add(
        Dense(
            50,
            activation="relu"
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0005
        ),
        loss="mse"
    )

    return model

# ==================================================
# FUTURE PREDICTION
# ==================================================
def predict_future(
    model,
    last_data,
    days=90
):

    predictions = []

    current = last_data.copy()

    for _ in range(days):

        pred = model.predict(
            current.reshape(
                1,
                len(current),
                1
            ),
            verbose=0
        )[0][0]

        predictions.append(pred)

        current = np.append(
            current[1:],
            pred
        )

    return np.array(predictions)

# ==================================================
# TITLE
# ==================================================
st.title("₿ Bitcoin Forecast Dashboard")
st.markdown("### LSTM Deep Learning Prediction")

# ==================================================
# GET DATA
# ==================================================
df = get_data()

if df.empty:
    st.error(
        "Data Bitcoin gagal diambil dari Yahoo Finance."
    )
    st.stop()

if len(df) < 100:
    st.error(
        "Data tidak cukup untuk training model."
    )
    st.stop()

# ==================================================
# METRICS
# ==================================================
col1, col2, col3 = st.columns(3)

last_price = float(
    df["Close"].iloc[-1]
)

daily_change = (
    (
        df["Close"].iloc[-1]
        - df["Close"].iloc[-2]
    )
    /
    df["Close"].iloc[-2]
) * 100

market_volume = float(
    df["Volume"].iloc[-1]
)

with col1:
    st.metric(
        "BTC Price",
        f"${last_price:,.2f}"
    )

with col2:
    st.metric(
        "Daily Change",
        f"{daily_change:.2f}%"
    )

with col3:
    st.metric(
        "Volume",
        f"{market_volume:,.0f}"
    )

# ==================================================
# PRICE CHART
# ==================================================
st.subheader("Bitcoin Candlestick Chart")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="BTC"
    )
)

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==================================================
# TRAIN MODEL
# ==================================================
if st.button("🚀 Train Model & Predict"):

    with st.spinner("Training model..."):

        lookback = 60

        X, y, scaler = prepare_data(
            df["Close"],
            lookback
        )

        split = int(
            len(X) * 0.85
        )

        X_train = X[:split].reshape(
            -1,
            lookback,
            1
        )

        X_test = X[split:].reshape(
            -1,
            lookback,
            1
        )

        y_train = y[:split]
        y_test = y[split:]

        model = create_model(
            (
                lookback,
                1
            )
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=30,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )

        pred = model.predict(
            X_test,
            verbose=0
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        st.success(
            f"Training selesai | RMSE = {rmse:.5f}"
        )

        # ==========================================
        # ACTUAL VS PREDICTION
        # ==========================================
        actual = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        )

        predicted = scaler.inverse_transform(
            pred
        )

        compare_df = pd.DataFrame(
            {
                "Actual": actual.flatten(),
                "Prediction": predicted.flatten()
            }
        )

        st.subheader(
            "Actual vs Prediction"
        )

        st.line_chart(
            compare_df
        )

        # ==========================================
        # FUTURE FORECAST
        # ==========================================
        last_60 = scaler.transform(
            df["Close"]
            .tail(lookback)
            .values
            .reshape(-1, 1)
        ).flatten()

        future_scaled = predict_future(
            model,
            last_60,
            90
        )

        future_prices = scaler.inverse_transform(
            future_scaled.reshape(-1, 1)
        ).flatten()

        future_dates = pd.date_range(
            start=pd.Timestamp.today(),
            periods=90,
            freq="D"
        )

        future_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Predicted Price": future_prices
            }
        )

        st.subheader(
            "90 Days Forecast"
        )

        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=future_df["Date"],
                y=future_df["Predicted Price"],
                mode="lines",
                name="Forecast"
            )
        )

        forecast_fig.update_layout(
            height=500
        )

        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )

        st.dataframe(
            future_df,
            use_container_width=True
        )

        csv = future_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Forecast CSV",
            data=csv,
            file_name="btc_forecast.csv",
            mime="text/csv"
        )

# ==================================================
# DEBUG INFO
# ==================================================
with st.expander("Debug Info"):

    st.write("Rows:", len(df))
    st.write("Columns:", list(df.columns))
    st.dataframe(df.tail())