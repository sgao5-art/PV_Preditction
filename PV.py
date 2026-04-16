import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PV Forecasting App", layout="wide")

st.title("Photovoltaic Power Forecasting App")

uploaded_file = st.file_uploader("Upload one CSV file containing 2017–2019 data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "date" not in df.columns or "power_KW" not in df.columns:
        st.error("CSV must contain 'date' and 'power_KW'")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
    df = df.dropna(subset=["date", "power_KW"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day"] = df["date"].dt.date
    daily_df = df.groupby("day", as_index=False)["power_KW"].sum()
    daily_df["day"] = pd.to_datetime(daily_df["day"])
    daily_df["year"] = daily_df["day"].dt.year

    st.header("1. View Historical Generation Data")

    selected_day = st.date_input(
        "Select any historical day from 2017–2019",
        value=daily_df["day"].min().date(),
        min_value=daily_df["day"].min().date(),
        max_value=daily_df["day"].max().date()
    )

    selected_day_ts = pd.to_datetime(selected_day)
    selected_row = daily_df[daily_df["day"] == selected_day_ts]

    if not selected_row.empty:
        historical_value = float(selected_row["power_KW"].iloc[0])
        st.write(f"Historical PV generation on {selected_day}: {historical_value:.2f} kW")
    else:
        st.write(f"No historical data found for {selected_day}.")

    st.header("2. Historical Data Visualization")

    year_options = ["All"] + sorted(daily_df["year"].unique().tolist())
    selected_year = st.selectbox("Select year for visualization", year_options)

    if selected_year == "All":
        plot_df = daily_df.copy()
    else:
        plot_df = daily_df[daily_df["year"] == selected_year].copy()

    st.line_chart(plot_df.set_index("day")["power_KW"])

    st.header("3. Predict Future Generation")

    future_date = st.date_input("Select any future date to predict")

    if st.button("Predict"):
        daily_df["day_of_year"] = daily_df["day"].dt.dayofyear
        daily_df["month"] = daily_df["day"].dt.month

        X = daily_df[["day_of_year", "month"]].values
        y = daily_df["power_KW"].values

        X_aug = np.column_stack([X, np.ones(len(X))])
        coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        future_date_ts = pd.to_datetime(future_date)
        future_features = np.array([[future_date_ts.dayofyear, future_date_ts.month]])
        future_aug = np.column_stack([future_features, [1]])

        prediction = future_aug @ coef
        predicted_value = max(float(prediction[0]), 0.0)

        st.write(f"Predicted PV generation on {future_date}: {predicted_value:.2f} kW")
else:
    st.info("Please upload one CSV file containing all 2017–2019 data.")
