Python 3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> import streamlit as st
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... 
... from sklearn.linear_model import LinearRegression
... 
... st.set_page_config(page_title="PV Power Forecasting Demo", layout="wide")
... 
... st.title("Photovoltaic Power Forecasting Demo")
... st.write(
...     "This demo uploads PV generation data, visualizes the original power curve, "
...     "performs a simple forecast, and compares actual values with predicted values."
... )
... 
... st.header("1. Upload Data")
... st.write("Upload a CSV file containing two columns: `date` and `power_KW`.")
... 
... uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
... 
... 
... def create_sequences(values, window_size):
...     X = []
...     y = []
...     for i in range(window_size, len(values)):
...         X.append(values[i - window_size:i])
...         y.append(values[i])
...     return np.array(X), np.array(y)
... 
... 
... if uploaded_file is not None:
...     try:
...         df = pd.read_csv(uploaded_file)
... 
...         # Check required columns
...         if "date" not in df.columns or "power_KW" not in df.columns:
...             st.error("The CSV file must contain columns named 'date' and 'power_KW'.")
...             st.stop()

        # Convert and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Keep only required columns
        df = df[["date", "power_KW"]].copy()

        # Make sure power is numeric
        df["power_KW"] = pd.to_numeric(df["power_KW"], errors="coerce")
        df = df.dropna()

        if len(df) < 50:
            st.error("The dataset is too small for forecasting. Please upload a larger dataset.")
            st.stop()

        st.success("File uploaded successfully.")

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.header("2. Original PV Power Curve")
        st.write("This plot shows the historical PV power generation trend over time.")

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(df["date"], df["power_KW"])
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title("Original PV Power Curve")
        plt.xticks(rotation=30)
        st.pyplot(fig1)

        st.header("3. Forecasting")
        st.write(
            "This section uses historical PV generation data to train a simple forecasting model."
        )

        window_size = st.slider("Select historical window size", 3, 48, 24)

        run_forecast = st.button("Run Forecast")

        if run_forecast:
            values = df["power_KW"].values
            X, y = create_sequences(values, window_size)

            if len(X) == 0:
                st.error("Not enough data for the selected window size.")
                st.stop()

            # 80% train, 20% test
            split_index = int(len(X) * 0.8)

            X_train = X[:split_index]
            X_test = X[split_index:]
            y_train = y[:split_index]
            y_test = y[split_index:]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Match dates for test data
            test_dates = df["date"].iloc[window_size + split_index:].reset_index(drop=True)

            result_df = pd.DataFrame({
                "date": test_dates,
                "Actual": y_test,
                "Predicted": y_pred
            })

            st.header("4. Actual vs Predicted")
            st.write("This plot compares actual PV power values with predicted results.")

            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(result_df["date"], result_df["Actual"], label="Actual")
            ax2.plot(result_df["date"], result_df["Predicted"], label="Predicted")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Power (kW)")
            ax2.set_title("Actual vs Predicted PV Power")
            ax2.legend()
            plt.xticks(rotation=30)
            st.pyplot(fig2)

            st.subheader("Prediction Preview")
            st.dataframe(result_df.head(20))

            st.subheader("Brief Explanation")
            st.write(
                "The model uses previous PV power values to predict future output. "
                "The predicted curve generally follows the overall trend of the actual data, "
                "although some differences may appear during periods of rapid fluctuation."
            )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
