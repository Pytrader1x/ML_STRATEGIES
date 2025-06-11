#!/usr/bin/env python3
"""Debug FX strategy execution"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Classical_strategies.strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC

# Load AUDUSD data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'data', 'AUDUSD_MASTER_15M.csv')
df = pd.read_csv(data_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.set_index('DateTime')

# Filter to 2024 data only for faster testing
df = df[df.index.year == 2024]
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Add indicators
print("\nAdding indicators...")
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df)
df = TIC.add_intelligent_chop(df)

# Check indicators
print("\nIndicator columns:", [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])

# Initialize strategy
config = OptimizedStrategyConfig()
config.initial_capital = 10000
config.risk_per_trade = 0.001  # 0.1% risk
config.verbose = True  # Enable verbose logging

strategy = OptimizedProdStrategy(config)

# Run backtest
print("\nRunning backtest...")
results = strategy.run_backtest(df)

print(f"\nResults type: {type(results)}")
print(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")

if isinstance(results, dict):
    print(f"\nMetrics: {results}")
    
    # Check trades
    if hasattr(strategy, 'trades'):
        print(f"\nNumber of trades: {len(strategy.trades)}")
        if strategy.trades:
            print(f"First trade: {strategy.trades[0]}")
            print(f"Total PnL from trades: {sum(t.pnl for t in strategy.trades)}")
else:
    print(f"\nDirect metrics result: {results}")