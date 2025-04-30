import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("Stock Market Predictor")

ticker = st.text_input("Enter a stock ticker symbol (e.g. AAPL, MSFT)", "AAPL").upper()

if st.button("Predict"):
    try:
        start_date = "2023-01-01"
        end_date = datetime.today().strftime('%Y-%m-%d')
        stock = yf.download(ticker, start=start_date, end=end_date)

        if stock.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            stock = stock.reset_index()
            stock['Days'] = (stock['Date'] - stock['Date'].min()).dt.days
            X = stock[['Days']]
            y = stock[['Close']]

            model = LinearRegression()
            model.fit(X, y)

            # Predict future
            last_day = stock['Days'].max()
            future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
            future_dates = [stock['Date'].max() + timedelta(days=i) for i in range(1, 31)]
            future_y = model.predict(future_days)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(stock['Date'], y, label="Actual Price")
            ax.plot(future_dates, future_y, label="Predicted Price", linestyle='dashed')
            ax.set_title(f"{ticker} Closing Price Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Show tables
            st.subheader("Last 5 Days of Data")
            st.dataframe(stock[['Date', 'Close']].tail())

            st.subheader("30-Day Forecast")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close': future_y.flatten()
            })
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"Error: {e}")
