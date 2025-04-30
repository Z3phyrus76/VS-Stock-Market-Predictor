import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# User input for ticker symbol
ticker = input("Enter a valid stock ticker (e.g., AAPL, MSFT): ").upper()

# Today's date and start date
today = datetime.now()
start_date = "2023-01-01"
end_date = today.strftime("%Y-%m-%d")

# Downloads the stock data
try:
    stock = yf.download(ticker, start=start_date, end=end_date)
    if stock.empty:
        raise ValueError("No data found. Check the ticker or date range.")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Prepare's the data
stock = stock.reset_index()
stock['Days'] = (stock['Date'] - stock['Date'].min()).dt.days
X = stock[['Days']]
y = stock[['Close']]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict's the next 30 days
last_day = stock['Days'].max()
future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
future_dates = [stock['Date'].max() + timedelta(days=i) for i in range(1, 31)]
future_y = model.predict(future_days)

# Plot
plt.plot(stock['Date'], y, label="Actual Closing Price")
plt.plot(future_dates, future_y, label="Predicted Price", linestyle="dashed")
plt.title(f"{ticker} Closing Price ({start_date} to {end_date}) + 30 Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Display the list
print("\nLast 5 Days of Data:")
print(stock[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].tail())

print("\n30-Day Forecast:")
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_y.flatten()})
print(forecast_df)
