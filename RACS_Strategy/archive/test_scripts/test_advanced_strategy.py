"""
Quick test of advanced momentum strategy
"""

from advanced_momentum_strategy import AdvancedMomentumStrategy
import pandas as pd
import numpy as np

# Load data
print("Loading data...")
data = pd.read_csv('../data/AUDUSD_MASTER_15M.csv', parse_dates=['DateTime'], index_col='DateTime')

# Use small sample for testing
data = data[-5000:]  # Last 5000 bars only
print(f"Testing on {len(data)} bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Run strategy with default parameters
print("\nRunning strategy...")
strategy = AdvancedMomentumStrategy(
    data,
    sl_atr_multiplier=2.0,
    tp_atr_multiplier=3.0,
    trailing_sl_atr=1.5
)

# Run backtest
df = strategy.run_backtest()

# Calculate metrics
metrics = strategy.calculate_metrics(df)

print("\nResults:")
print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
print(f"Total Returns: {metrics['returns']:.1f}%")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Max Drawdown: {metrics['max_dd']:.1f}%")
print(f"Total Trades: {metrics['trades']}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")

if metrics['exit_analysis']:
    print("\nExit Reasons:")
    for reason, count in metrics['exit_analysis'].items():
        print(f"  {reason}: {count}")

# Save a sample of trades
if strategy.trades:
    trade_df = pd.DataFrame(strategy.trades[:10])  # First 10 trades
    print("\nSample trades:")
    print(trade_df[['entry_date', 'position', 'entry_price', 'stop_loss', 'take_profit']].to_string())