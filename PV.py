import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PV Forecasting App", layout="wide")

st.title("Photovoltaic Power Forecasting App")

uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    df_list = []

    for file in uploaded_files:
        temp_df = pd.read_csv(file)

        if "date" not in temp_df.columns or "power_KW" not in temp_df.columns:
            st.error(f"File {file.name} must contain 'date' and 'power_KW'")
            st.stop()

        temp_df = temp_df[["date", "power_KW"]].copy()
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
    df = df.dropna(subset=["date", "power_KW"])
    df = df.sort_values("date").reset_index(drop=True)

    df["day"] = df["date"].dt.date
    daily_df = df.groupby("day")["power_KW"].sum().reset_index()
    daily_df["day"] = pd.to_datetime(daily_df["day"])
    daily_df["year"] = daily_df["day"].dt.year

    st.header("1. Historical Data Visualization")

    available_years = sorted(daily_df["year"].dropna().unique().tolist())
    year_options = ["All"] + available_years

    year_option = st.selectbox("Select Year", options=year_options)

    if year_option == "All":
        plot_df = daily_df.copy()
    else:
        plot_df = daily_df[daily_df["year"] == year_option].copy()

    st.line_chart(plot_df.set_index("day")["power_KW"])

    st.header("2. Predict Future Date")

    input_date = st.date_input("Select a future date")

    if st.button("Predict"):
        daily_df["day_of_year"] = daily_df["day"].dt.dayofyear
        daily_df["month"] = daily_df["day"].dt.month

        X = daily_df[["day_of_year", "month"]].values
        y = daily_df["power_KW"].values

        X_aug = np.column_stack([X, np.ones(len(X))])
        coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        input_date = pd.to_datetime(input_date)
        input_features = np.array([[input_date.dayofyear, input_date.month]])
        input_aug = np.column_stack([input_features, [1]])

        prediction = input_aug @ coef
        predicted_value = max(float(prediction[0]), 0.0)

        st.header("3. Prediction Result")
        st.write(f"Predicted PV generation on {input_date.date()}: {predicted_value:.2f} kW")
