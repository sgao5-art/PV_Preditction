import streamlit as st
import pandas as pd
import numpy as np

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
    original_chart = df.set_index("date")[["power_KW"]]
    st.line_chart(original_chart)

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

        X_train_aug = np.column_stack([X_train, np.ones(len(X_train))])
        coef, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

        X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])
        y_pred = X_test_aug @ coef

        test_dates = df["date"].iloc[window_size + split_index:].reset_index(drop=True)

        result_df = pd.DataFrame({
            "date": test_dates,
            "Actual": y_test,
            "Predicted": y_pred
        }).set_index("date")

        st.subheader("Actual vs Predicted")
        st.line_chart(result_df)
