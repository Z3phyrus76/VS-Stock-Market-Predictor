import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 30-Day Stock Price Forecast")

# User input for ticker symbol
ticker = st.text_input("Enter a stock ticker symbol (e.g., AAPL, MSFT):", value="AAPL").upper()

# Date setup
start_date = "2023-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

# Fetch data
if ticker:
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        if stock.empty:
            st.error("❌ No data found. Please check the ticker symbol.")
        else:
            stock.reset_index(inplace=True)
            stock['Days'] = (stock['Date'] - stock['Date'].min()).dt.days
            X = stock[['Days']]
            y = stock[['Close']]

            # Train linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict next 30 days
            last_day = stock['Days'].max()
            future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
            future_dates = [stock['Date'].max() + timedelta(days=i) for i in range(1, 31)]
            future_y = model.predict(future_days)

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(stock['Date'], stock['Close'], label="Actual Closing Price")
            ax.plot(future_dates, future_y, label="Predicted Price", linestyle="dashed")
            ax.set_title(f"{ticker} Closing Price ({start_date} to {end_date}) + 30-Day Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Display tables
            st.subheader("📅 Last 5 Days of Actual Data")
            st.dataframe(stock[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].tail())

            st.subheader("🔮 30-Day Forecast")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close': future_y.flatten()
            })
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"⚠️ Error fetching or processing data: {e}")
