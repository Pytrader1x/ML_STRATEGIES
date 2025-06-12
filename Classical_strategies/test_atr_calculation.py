"""
Test ATR calculation to find infinity issue
"""

import pandas as pd
import numpy as np

# Reproduce the data generation from simple_optimization_test.py
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', periods=100, freq='1H')

# Generate more realistic trending/ranging periods
regime_length = len(dates) // 4

# Create 4 different market regimes
trend_up = np.cumsum(np.random.normal(0.0002, 0.001, regime_length))
range_period = np.cumsum(np.random.normal(0, 0.002, regime_length))  
trend_down = np.cumsum(np.random.normal(-0.0002, 0.001, regime_length))
volatile_period = np.cumsum(np.random.normal(0, 0.003, len(dates) - 3*regime_length))

returns = np.concatenate([trend_up, range_period, trend_down, volatile_period])
prices = np.cumprod(1 + returns) * 0.75

df = pd.DataFrame({
    'Open': prices + np.random.normal(0, 0.0001, len(prices)),
    'High': prices + abs(np.random.normal(0, 0.0002, len(prices))),
    'Low': prices - abs(np.random.normal(0, 0.0002, len(prices))),
    'Close': prices
}, index=dates)

print("Basic data created:")
print(f"Close price range: {df['Close'].min():.6f} to {df['Close'].max():.6f}")

# Fix OHLC consistency
for i in range(len(df)):
    df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i][['Open', 'High', 'Low', 'Close']])
    df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i][['Open', 'High', 'Low', 'Close']])

# ATR calculation
df['TR'] = np.maximum(
    df['High'] - df['Low'],
    np.maximum(
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    )
)

print(f"TR range: {df['TR'].min():.10f} to {df['TR'].max():.10f}")
print(f"TR NaN count: {df['TR'].isna().sum()}")

df['ATR'] = df['TR'].rolling(14).mean()
print(f"ATR range: {df['ATR'].min():.10f} to {df['ATR'].max():.10f}")
print(f"ATR NaN count: {df['ATR'].isna().sum()}")

# This is the problematic line
df['IC_ATR_Normalized'] = np.clip((df['ATR'] / df['Close'] * 10000), 10, 100)

print(f"ATR/Close calculation: {(df['ATR'] / df['Close']).describe()}")
print(f"ATR/Close * 10000: {(df['ATR'] / df['Close'] * 10000).describe()}")
print(f"IC_ATR_Normalized range: {df['IC_ATR_Normalized'].min():.6f} to {df['IC_ATR_Normalized'].max():.6f}")
print(f"IC_ATR_Normalized infinity count: {np.isinf(df['IC_ATR_Normalized']).sum()}")
print(f"IC_ATR_Normalized NaN count: {df['IC_ATR_Normalized'].isna().sum()}")