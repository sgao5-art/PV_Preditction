import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="PV Forecasting Demo", layout="wide")

st.title("PV Forecasting Demo")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


def create_sequences(values, window_size):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    return np.array(X), np.array(y)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "date" not in df.columns or "power_KW" not in df.columns:
        st.error("The CSV file must contain columns: date and power_KW")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
    df = df.dropna(subset=["date", "power_KW"])
    df = df.sort_values("date").reset_index(drop=True)

    st.subheader("Original PV Power Curve")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df["date"], df["power_KW"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Original PV Power Curve")
    plt.xticks(rotation=30)
    st.pyplot(fig1)

    st.subheader("Forecasting")
    window_size = st.slider("Window Size", min_value=3, max_value=48, value=24)
    run_forecast = st.button("Run Forecast")

    if run_forecast:
        values = df["power_KW"].values
        X, y = create_sequences(values, window_size)

        if len(X) == 0:
            st.error("Not enough data for forecasting.")
            st.stop()

        split_index = int(len(X) * 0.8)

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_dates = df["date"].iloc[window_size + split_index:].reset_index(drop=True)

        result_df = pd.DataFrame({
            "date": test_dates,
            "Actual": y_test,
            "Predicted": y_pred
        })

        st.subheader("Actual vs Predicted")
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(result_df["date"], result_df["Actual"], label="Actual")
        ax2.plot(result_df["date"], result_df["Predicted"], label="Predicted")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Power (kW)")
        ax2.set_title("Actual vs Predicted")
        ax2.legend()
        plt.xticks(rotation=30)
        st.pyplot(fig2)
