
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

#2. DATA COLLECTION

TICKER = "^OEX"                          # S&P 100 index
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

print(f"Fetching data for {TICKER} from {START_DATE} to {END_DATE}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

if data.empty:
    print("ERROR: Could not fetch data. Check the ticker symbol and internet connection.")
else:
    print("Head of the raw data:")
    print(data.head())

    # Save raw data to Excel
    data.to_excel("sp100 data raw.xlsx")
    print("Saved raw data to sp100 data raw.xlsx")


# 3. DATA CLEANING & FEATURE ENGINEERING


# Daily Return: how much price changed relative to opening price
data["Daily Return"] = (data["Close"] - data["Open"]) / data["Open"]

# Moving Averages
data["50 MA"] = data["Close"].rolling(window=50).mean()    # Short-term trend
data["200 MA"] = data["Close"].rolling(window=200).mean()  # Long-term trend

# 20-day Annualised Volatility
# Standard deviation of daily returns over last 20 days, scaled to 1 year (252 trading days)
data["20D Volatility"] = data["Daily Return"].rolling(window=20).std() * np.sqrt(252)

# Drop rows with NaN values (from rolling calculations)
data.dropna(inplace=True)

# Target variable: next day's closing price
data["Target Close"] = data["Close"].shift(-1)
data.dropna(inplace=True)

print("\nHead of processed data:")
print(data.tail())


# 4. DATA VISUALISATION


# A. Closing Price and Moving Averages
plt.style.use("ggplot")

plt.figure(figsize=(12, 6))
plt.plot(data["Close"],  label="S&P 100 Close Price",    color="blue",  linewidth=1)
plt.plot(data["50 MA"],  label="50 Day Moving Average",  color="red",   linewidth=1)
plt.plot(data["200 MA"], label="200 Day Moving Average", color="green", linewidth=1)
plt.title("S&P 100 Index: Price and Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.savefig("sp100 trend analysis.png")
plt.show()

# B. 20-Day Annualised Volatility
plt.figure(figsize=(12, 4))
plt.plot(data["20D Volatility"], label="20 Day Annualized Volatility", color="orange")
plt.title("S&P 100 20 Day Annualized Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility (%)")
plt.legend()
plt.savefig("sp100 volatility analysis.png")
plt.show()


# 5. MODEL BUILDING


# Features used for prediction
features = ["Close", "Volume", "50 MA", "20D Volatility"]
X = data[features]
y = data["Target Close"]

# Split data — shuffle=False is critical for time-series to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
predictions = model.predict(X_test)

# 6. MODEL EVALUATION


rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2   = r2_score(y_test, predictions)

print("\nModel Evaluation:")
print(f"RMSE:     {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 7. FINAL OUTCOMES


# A. Save results to CSV
X_test["Predicted Close"] = predictions
X_test["Actual Close"]    = y_test

final_results = X_test[["Actual Close", "Predicted Close", "Volume", "50 MA", "20D Volatility"]]
final_results.to_csv("sp100 final results.csv", index=True)
print("\nFinal results (Actual vs. Predicted) saved to sp100 final results.csv")

# B. Final Plot: Actual vs Predicted
plt.figure(figsize=(12, 8))
plt.plot(final_results["Actual Close"],    label="Actual Closing Price",    color="blue")
plt.plot(final_results["Predicted Close"], label="Predicted Closing Price", color="red", linestyle="--")
plt.title("S&P 100 Price: Actual vs. Predicted (Test Period)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()

# Annotate with metrics
plt.text(
    x=final_results.index[int(len(final_results) * 0.05)],
    y=max(final_results["Actual Close"]),
    s=f"RMSE = {rmse:.2f}\nR² = {r2:.1f}",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.8, pad=5)
)

plt.savefig("sp100 prediction vs actual with metrics.png")
plt.show()

print("\nDone! All files saved.")
