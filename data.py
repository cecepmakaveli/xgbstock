# generate_synthetic_data.py
import pandas as pd
import numpy as np

np.random.seed(42)

def generate_data(n=200):
    base_date = pd.date_range(start="2022-01-01", periods=n, freq="D")
    open_prices = np.random.uniform(100, 200, size=n)
    high_prices = open_prices + np.random.uniform(0, 10, size=n)
    low_prices = open_prices - np.random.uniform(0, 10, size=n)
    close_prices = np.random.uniform(low_prices, high_prices)
    volume = np.random.randint(100000, 1000000, size=n)

    df = pd.DataFrame({
        "Date": base_date.strftime('%Y-%m-%d'),
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume
    })

    df.to_csv("synthetic_stock_data.csv", index=False)
    print("Synthetic data saved to 'synthetic_stock_data.csv'")

generate_data()
