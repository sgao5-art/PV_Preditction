import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PV Forecasting App",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 自定义样式
# ---------------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 10px;
    }
    .sub-text {
        font-size: 16px;
        color: #555555;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 15px;
        border-radius: 12px;
        background-color: #f4f8fb;
        border: 1px solid #d9e6f2;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# 读取本地三年数据
# ---------------------------
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent

    files = [
        base_dir / "2017_pv_raw.csv",
        base_dir / "2018_pv_raw.csv",
        base_dir / "2019_pv_raw.csv"
    ]

    for f in files:
        if not f.exists():
            st.error(f"File not found: {f}")
            st.stop()

    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    if "date" not in df.columns or "power_KW" not in df.columns:
        st.error("Each CSV must contain 'date' and 'power_KW' columns.")
        st.stop()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
    df = df.dropna(subset=["date", "power_KW"])
    df = df.sort_values("date").reset_index(drop=True)

    # 按天汇总
    df["day"] = df["date"].dt.date
    daily_df = df.groupby("day", as_index=False)["power_KW"].sum()
    daily_df["day"] = pd.to_datetime(daily_df["day"])
    daily_df["year"] = daily_df["day"].dt.year
    daily_df["month"] = daily_df["day"].dt.month
    daily_df["day_of_year"] = daily_df["day"].dt.dayofyear
    daily_df["month_day"] = daily_df["day"].dt.strftime("%m-%d")

    return daily_df

daily_df = load_data()

# ---------------------------
# 简单预测模型
# ---------------------------
@st.cache_data
def train_model(data):
    X = data[["day_of_year", "month"]].values
    y = data["power_KW"].values
    X_aug = np.column_stack([X, np.ones(len(X))])
    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    return coef

coef = train_model(daily_df)

def predict_power(target_date, coef):
    target_ts = pd.to_datetime(target_date)
    features = np.array([[target_ts.dayofyear, target_ts.month]])
    features_aug = np.column_stack([features, [1]])
    prediction = features_aug @ coef
    return max(float(prediction[0]), 0.0)

# ---------------------------
# 页面标题
# ---------------------------
st.markdown('<div class="main-title">☀️ Photovoltaic Power Forecasting App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Explore historical PV generation, predict future power output, and compare forecasts with historical patterns.</div>',
    unsafe_allow_html=True
)

# ---------------------------
# 左侧功能栏
# ---------------------------
st.sidebar.title("Control Panel")
st.sidebar.write("Use the options below to interact with the app.")

st.sidebar.header("1. Historical Data")
selected_day = st.sidebar.date_input(
    "Select any historical day",
    value=daily_df["day"].min().date(),
    min_value=daily_df["day"].min().date(),
    max_value=daily_df["day"].max().date()
)

st.sidebar.header("2. Visualization")
year_options = ["All"] + sorted(daily_df["year"].unique().tolist())
selected_year = st.sidebar.selectbox("Select year", year_options)

st.sidebar.header("3. Future Prediction")
future_date = st.sidebar.date_input("Select a future date to predict")
predict_button = st.sidebar.button("Predict Future Generation")

# ---------------------------
# 历史结果 + 预测结果
# ---------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Historical Generation Result")
    selected_day_ts = pd.to_datetime(selected_day)
    selected_row = daily_df[daily_df["day"] == selected_day_ts]

    if not selected_row.empty:
        historical_value = float(selected_row["power_KW"].iloc[0])
        st.markdown(
            f"""
            <div class="result-box">
                <h4>Selected Date: {selected_day}</h4>
                <p><b>Historical PV Generation:</b> {historical_value:.2f} kW</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"No historical data found for {selected_day}.")

predicted_value = None
future_date_ts = pd.to_datetime(future_date)

with col2:
    st.subheader("Prediction Result")
    if predict_button:
        predicted_value = predict_power(future_date_ts, coef)
        st.markdown(
            f"""
            <div class="result-box">
                <h4>Future Date: {future_date}</h4>
                <p><b>Predicted PV Generation:</b> {predicted_value:.2f} kW</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Choose a future date in the sidebar and click the prediction button.")

# ---------------------------
# 历史年份趋势图
# ---------------------------
st.subheader("Historical Data Visualization")

if selected_year == "All":
    plot_df = daily_df.copy()
else:
    plot_df = daily_df[daily_df["year"] == selected_year].copy()

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(plot_df["day"], plot_df["power_KW"], linewidth=1.5)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power (kW)")
ax1.set_title(f"Historical PV Generation - {selected_year}")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ---------------------------
# 升级功能 1：同月同日历史对比
# ---------------------------
st.subheader("Prediction vs Historical Same-Day Comparison")

if predict_button:
    target_month_day = future_date_ts.strftime("%m-%d")
    same_day_df = daily_df[daily_df["month_day"] == target_month_day].copy()

    comparison_rows = []
    for _, row in same_day_df.iterrows():
        comparison_rows.append({
            "Label": row["day"].strftime("%Y-%m-%d"),
            "Power_KW": row["power_KW"],
            "Type": "Historical"
        })

    comparison_rows.append({
        "Label": future_date_ts.strftime("%Y-%m-%d"),
        "Power_KW": predicted_value,
        "Type": "Predicted"
    })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(comparison_df, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(comparison_df["Label"], comparison_df["Power_KW"], marker="o")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Predicted Value vs Historical Same-Day Values")
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
else:
    st.info("Click the prediction button to compare the predicted value with historical same-day values.")

# ---------------------------
# 升级功能 2：历史同期前后7天趋势对比
# ---------------------------
st.subheader("Historical 7-Day Window Comparison")

if predict_button:
    compare_years = sorted(daily_df["year"].unique().tolist())
    future_month = future_date_ts.month
    future_day = future_date_ts.day

    trend_data = []

    for year in compare_years:
        try:
            center_date = pd.Timestamp(year=year, month=future_month, day=future_day)
        except ValueError:
            continue

        start_date = center_date - pd.Timedelta(days=7)
        end_date = center_date + pd.Timedelta(days=7)

        temp = daily_df[(daily_df["day"] >= start_date) & (daily_df["day"] <= end_date)].copy()
        temp["offset"] = (temp["day"] - center_date).dt.days
        temp["Series"] = str(year)
        trend_data.append(temp[["offset", "power_KW", "Series"]])

    if trend_data:
        trend_df = pd.concat(trend_data, ignore_index=True)

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        for series_name in trend_df["Series"].unique():
            temp = trend_df[trend_df["Series"] == series_name]
            ax3.plot(temp["offset"], temp["power_KW"], marker="o", label=series_name)

        ax3.axvline(x=0, linestyle="--")
        ax3.scatter(0, predicted_value, s=100, label="Predicted Future Date")

        ax3.set_xlabel("Days Relative to Target Date")
        ax3.set_ylabel("Power (kW)")
        ax3.set_title("Predicted Value vs Historical ±7-Day Patterns")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3)

        st.caption("Offset 0 represents the same month/day in each historical year.")
    else:
        st.warning("No valid historical 7-day comparison window could be generated.")
else:
    st.info("Click the prediction button to see the ±7-day historical comparison.")

# ---------------------------
# 数据表
# ---------------------------
with st.expander("Show Processed Daily Data"):
    st.dataframe(daily_df, use_container_width=True)
