import streamlit as st
import pandas as pd
import numpy as np

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
# 读取本地数据
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/daily_generation.csv")
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
    df = df.dropna(subset=["day", "power_KW"])
    df = df.sort_values("day").reset_index(drop=True)
    df["year"] = df["day"].dt.year
    return df

daily_df = load_data()

# ---------------------------
# 标题区域
# ---------------------------
st.markdown('<div class="main-title">☀️ Photovoltaic Power Forecasting App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Explore historical generation, visualize yearly trends, and predict future output.</div>',
    unsafe_allow_html=True
)

# ---------------------------
# 左侧功能栏
# ---------------------------
st.sidebar.title("Control Panel")
st.sidebar.write("Use the options below to interact with the app.")

# 左侧历史日期选择
st.sidebar.header("1. Historical Data")
selected_day = st.sidebar.date_input(
    "Select any historical day",
    value=daily_df["day"].min().date(),
    min_value=daily_df["day"].min().date(),
    max_value=daily_df["day"].max().date()
)

# 左侧年份选择
st.sidebar.header("2. Visualization")
year_options = ["All"] + sorted(daily_df["year"].unique().tolist())
selected_year = st.sidebar.selectbox("Select year", year_options)

# 左侧未来预测日期
st.sidebar.header("3. Future Prediction")
future_date = st.sidebar.date_input("Select a future date to predict")
predict_button = st.sidebar.button("Predict Future Generation")

# ---------------------------
# 右侧显示区域
# ---------------------------
col1, col2 = st.columns([1, 1])

# 历史数据显示
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

# 预测结果显示
with col2:
    st.subheader("Prediction Result")

    if predict_button:
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

# 图表显示
st.subheader("Historical Data Visualization")

if selected_year == "All":
    plot_df = daily_df.copy()
else:
    plot_df = daily_df[daily_df["year"] == selected_year].copy()

st.line_chart(plot_df.set_index("day")["power_KW"])

# 显示处理后的数据
with st.expander("Show Processed Daily Data"):
    st.dataframe(plot_df, use_container_width=True)
