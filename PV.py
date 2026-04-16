import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PV Forecasting App", layout="wide")

st.title("Photovoltaic Power Forecasting App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 检查列
    if "date" not in df.columns or "power_KW" not in df.columns:
        st.error("CSV must contain 'date' and 'power_KW'")
        st.stop()

    # 数据处理
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
    df = df.dropna(subset=["date", "power_KW"])
    df = df.sort_values("date")

    # 按天汇总（关键）
    df["day"] = df["date"].dt.date
    daily_df = df.groupby("day")["power_KW"].sum().reset_index()
    daily_df["day"] = pd.to_datetime(daily_df["day"])

    # 提取年份
    daily_df["year"] = daily_df["day"].dt.year

    # =========================
    # 1. 年份选择 + 可视化
    # =========================
    st.header("1. Historical Data Visualization")

    year_option = st.selectbox(
        "Select Year",
        options=["All", 2017, 2018, 2019]
    )

    if year_option == "All":
        plot_df = daily_df
    else:
        plot_df = daily_df[daily_df["year"] == year_option]

    st.line_chart(plot_df.set_index("day")["power_KW"])

    # =========================
    # 2. 预测未来某一天
    # =========================
    st.header("2. Predict Future Date")

    input_date = st.date_input("Select a future date")

    if st.button("Predict"):
        # 特征：用简单时间特征
        daily_df["day_of_year"] = daily_df["day"].dt.dayofyear
        daily_df["month"] = daily_df["day"].dt.month

        X = daily_df[["day_of_year", "month"]].values
        y = daily_df["power_KW"].values

        # 简单线性回归（numpy实现）
        X_aug = np.column_stack([X, np.ones(len(X))])
        coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

        # 预测输入日期
        input_date = pd.to_datetime(input_date)
        input_features = np.array([[input_date.dayofyear, input_date.month]])
        input_aug = np.column_stack([input_features, [1]])

        prediction = input_aug @ coef

        st.subheader("Prediction Result")
        st.write(
            f"Predicted PV generation on {input_date.date()} : {prediction[0]:.2f} kW"
        )

        st.write(
            "This prediction is based on historical seasonal patterns learned from 2017–2019 data."
        )

else:
    st.info("Please upload a CSV file.")
