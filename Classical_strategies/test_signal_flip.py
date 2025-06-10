import pandas as pd
import numpy as np
from Strategy_1 import Strategy_1
from technical_indicators_custom import TIC

# Load data
df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Use a fixed sample for comparison
sample_size = 2000
np.random.seed(42)
random_start = np.random.randint(0, len(df) - sample_size)
df_test = df.iloc[random_start:random_start + sample_size].copy()

print(f"Testing on sample from {df_test.index[0]} to {df_test.index[-1]}")

# Calculate indicators
print("Calculating indicators...")
df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
df_test = TIC.add_intelligent_chop(df_test)

# Test WITH signal flip exits
print("\n" + "="*60)
print("TEST 1: WITH Signal Flip Exits (exit_on_signal_flip=True)")
print("="*60)

strategy_with_flip = Strategy_1(
    initial_capital=100000,
    risk_per_trade=0.02,
    tp_atr_multipliers=(0.8, 1.5, 2.5),
    sl_atr_multiplier=2.0,
    trailing_atr_multiplier=1.2,
    tsl_activation_pips=15,
    tsl_min_profit_pips=5,
    exit_on_signal_flip=True,  # ENABLED
    max_tp_percent=0.01,
    min_lot_size=1000000,
    pip_value_per_million=100,
    verbose=False
)

results_with_flip = strategy_with_flip.run_backtest(df_test)

print(f"Total Trades: {results_with_flip['total_trades']}")
print(f"Win Rate: {results_with_flip['win_rate']:.2f}%")
print(f"Total P&L: ${results_with_flip['total_pnl']:.2f}")
print(f"Sharpe Ratio: {results_with_flip['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_with_flip['max_drawdown']:.2f}%")
print("\nExit Reasons:")
for reason, count in results_with_flip['exit_reasons'].items():
    print(f"  {reason}: {count}")

# Test WITHOUT signal flip exits
print("\n" + "="*60)
print("TEST 2: WITHOUT Signal Flip Exits (exit_on_signal_flip=False)")
print("="*60)

strategy_no_flip = Strategy_1(
    initial_capital=100000,
    risk_per_trade=0.02,
    tp_atr_multipliers=(0.8, 1.5, 2.5),
    sl_atr_multiplier=2.0,
    trailing_atr_multiplier=1.2,
    tsl_activation_pips=15,
    tsl_min_profit_pips=5,
    exit_on_signal_flip=False,  # DISABLED
    max_tp_percent=0.01,
    min_lot_size=1000000,
    pip_value_per_million=100,
    verbose=False
)

results_no_flip = strategy_no_flip.run_backtest(df_test)

print(f"Total Trades: {results_no_flip['total_trades']}")
print(f"Win Rate: {results_no_flip['win_rate']:.2f}%")
print(f"Total P&L: ${results_no_flip['total_pnl']:.2f}")
print(f"Sharpe Ratio: {results_no_flip['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_no_flip['max_drawdown']:.2f}%")
print("\nExit Reasons:")
for reason, count in results_no_flip['exit_reasons'].items():
    print(f"  {reason}: {count}")

# Comparison
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"Trades Difference: {results_no_flip['total_trades'] - results_with_flip['total_trades']}")
print(f"Win Rate Change: {results_no_flip['win_rate'] - results_with_flip['win_rate']:.2f}%")
print(f"P&L Difference: ${results_no_flip['total_pnl'] - results_with_flip['total_pnl']:.2f}")
print(f"Sharpe Difference: {results_no_flip['sharpe_ratio'] - results_with_flip['sharpe_ratio']:.2f}")

# Calculate how many trades would have been exited by signal flip
signal_flip_exits = results_with_flip['exit_reasons'].get('signal_flip', 0)
print(f"\nSignal flip exits prevented: {signal_flip_exits}")
print(f"Percentage of trades affected: {(signal_flip_exits / results_with_flip['total_trades'] * 100):.1f}%")