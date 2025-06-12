"""
Minimal test to isolate infinity error
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig

# Create minimal config
config = OptimizedStrategyConfig(
    relaxed_mode=False,
    exit_on_signal_flip=True,
    signal_flip_min_profit_pips=5.0,
    verbose=True  # Enable verbose logging
)

# Create minimal test data
np.random.seed(42)
dates = pd.date_range(start='2010-01-01', periods=100, freq='1H')
prices = np.linspace(0.75, 0.76, 100)

df = pd.DataFrame({
    'Open': prices,
    'High': prices + 0.0001,
    'Low': prices - 0.0001,
    'Close': prices
}, index=dates)

# Add minimal indicators
df['NTI_Direction'] = [1 if i % 10 < 5 else -1 for i in range(100)]
df['MB_Bias'] = [1 if i % 8 < 4 else -1 for i in range(100)]
df['IC_Regime'] = [1 if i % 6 < 3 else 2 for i in range(100)]
df['IC_ATR_Normalized'] = 25.0  # Fixed ATR
df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend'})

print("Test data created")
print(f"Data shape: {df.shape}")
print(f"First few rows:")
print(df.head())

try:
    strategy = OptimizedProdStrategy(config)
    print("Strategy created successfully")
    
    result = strategy.run_backtest(df)
    print("Backtest completed successfully")
    print(f"Results: {result}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print("Full traceback:")
    traceback.print_exc()