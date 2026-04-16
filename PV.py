import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="PV Forecasting App", layout="wide")

st.title("Photovoltaic Power Forecasting App")

# 读取三个文件
FILES = ["2017.csv", "2018.csv", "2019.csv"]

df_list = []

for file in FILES:
    try:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    except Exception as e:
        st.error(f"Failed to load {file}: {e}")
        st.stop()

df = pd.concat(df_list, ignore_index=True)

# 检查列
if "date" not in df.columns or "power_KW" not in df.columns:
    st.error("All CSV files must contain 'date' and 'power_KW'")
    st.stop()

# 数据清洗
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
df = df.dropna(subset=["date", "power_KW"])
df = df.sort_values("date").reset_index(drop=True)

# 按天汇总
df["day"] = df["date"].dt.date
daily_df = df.groupby("day", as_index=False)["power_KW"].sum()
daily_df["day"] = pd.to_datetime(daily_df["day"])
daily_df["year"] = daily_df["day"].dt.year

# =========================
# 1. 查看历史某一天
# =========================
st.header("1. View Historical Generation Data")

selected_day = st.date_input(
    "Select any historical day from 2017–2019",
    value=daily_df["day"].min().date(),
    min_value=daily_df["day"].min().date(),
    max_value=daily_df["day"].max().date()
)

selected_row = daily_df[daily_df["day"] == pd.to_datetime(selected_day)]

if not selected_row.empty:
    value = float(selected_row["power_KW"].iloc[0])
    st.write(f"Historical PV generation on {selected_day}: {value:.2f} kW")
else:
    st.write("No data found for selected day")

# =========================
# 2. 可视化
# =========================
st.header("2. Historical Data Visualization")

year_options = ["All"] + sorted(daily_df["year"].unique().tolist())
selected_year = st.selectbox("Select year", year_options)

if selected_year == "All":
    plot_df = daily_df
else:
    plot_df = daily_df[daily_df["year"] == selected_year]

st.line_chart(plot_df.set_index("day")["power_KW"])

# =========================
# 3. 预测
# =========================
st.header("3. Predict Future Generation")

future_date = st.date_input("Select a future date")

if st.button("Predict"):
    daily_df["day_of_year"] = daily_df["day"].dt.dayofyear
    daily_df["month"] = daily_df["day"].dt.month

    X = daily_df[["day_of_year", "month"]].values
    y = daily_df["power_KW"].values

    X_aug = np.column_stack([X, np.ones(len(X))])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)

    future = pd.to_datetime(future_date)
    features = np.array([[future.dayofyear, future.month]])
    features_aug = np.column_stack([features, [1]])

    prediction = features_aug @ coef
    prediction = max(float(prediction[0]), 0.0)

    st.write(f"Predicted PV generation on {future_date}: {prediction:.2f} kW")
