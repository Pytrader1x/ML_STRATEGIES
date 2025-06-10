import pandas as pd
import numpy as np
from Strategy_1 import Strategy_1
from technical_indicators_custom import TIC

# Load data
df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Use a fixed sample for comparison
sample_size = 3000
np.random.seed(42)
random_start = np.random.randint(0, len(df) - sample_size)
df_test = df.iloc[random_start:random_start + sample_size].copy()

print(f"Testing on sample from {df_test.index[0]} to {df_test.index[-1]}")

# Calculate indicators
print("Calculating indicators...")
df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
df_test = TIC.add_intelligent_chop(df_test)

# Test NORMAL mode (strict confluence)
print("\n" + "="*60)
print("TEST 1: NORMAL MODE (Strict 3-indicator confluence)")
print("="*60)

strategy_normal = Strategy_1(
    initial_capital=100000,
    risk_per_trade=0.02,
    tp_atr_multipliers=(0.8, 1.5, 2.5),
    sl_atr_multiplier=2.0,
    trailing_atr_multiplier=1.2,
    tsl_activation_pips=15,
    tsl_min_profit_pips=5,
    exit_on_signal_flip=True,
    relaxed_mode=False,  # NORMAL MODE
    verbose=False
)

results_normal = strategy_normal.run_backtest(df_test)

print(f"Total Trades: {results_normal['total_trades']}")
print(f"Win Rate: {results_normal['win_rate']:.2f}%")
print(f"Total P&L: ${results_normal['total_pnl']:.2f}")
print(f"Sharpe Ratio: {results_normal['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_normal['max_drawdown']:.2f}%")
print(f"Average Win: ${results_normal['avg_win']:.2f}")
print(f"Average Loss: ${results_normal['avg_loss']:.2f}")

# Test RELAXED mode (NeuroTrend only)
print("\n" + "="*60)
print("TEST 2: RELAXED MODE (NeuroTrend only with tighter risk)")
print("="*60)

strategy_relaxed = Strategy_1(
    initial_capital=100000,
    risk_per_trade=0.02,
    tp_atr_multipliers=(0.8, 1.5, 2.5),
    sl_atr_multiplier=2.0,
    trailing_atr_multiplier=1.2,
    tsl_activation_pips=15,
    tsl_min_profit_pips=5,
    exit_on_signal_flip=True,
    relaxed_mode=True,  # RELAXED MODE
    relaxed_tp_multiplier=0.5,  # 50% of normal TP distances
    relaxed_position_multiplier=0.5,  # 50% of normal position size
    relaxed_tsl_activation_pips=8,  # Faster TSL activation
    verbose=True
)

results_relaxed = strategy_relaxed.run_backtest(df_test)

print(f"\nTotal Trades: {results_relaxed['total_trades']}")
print(f"Win Rate: {results_relaxed['win_rate']:.2f}%")
print(f"Total P&L: ${results_relaxed['total_pnl']:.2f}")
print(f"Sharpe Ratio: {results_relaxed['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results_relaxed['max_drawdown']:.2f}%")
print(f"Average Win: ${results_relaxed['avg_win']:.2f}")
print(f"Average Loss: ${results_relaxed['avg_loss']:.2f}")

# Count how many trades were relaxed
relaxed_trades = sum(1 for trade in results_relaxed['trades'] if trade.is_relaxed)
normal_trades = results_relaxed['total_trades'] - relaxed_trades

print(f"\nTrade Breakdown:")
print(f"  Normal trades (3-indicator confluence): {normal_trades}")
print(f"  Relaxed trades (NeuroTrend only): {relaxed_trades}")
print(f"  Percentage of relaxed trades: {(relaxed_trades / results_relaxed['total_trades'] * 100):.1f}%")

# Comparison
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"Total Trades: {results_normal['total_trades']} → {results_relaxed['total_trades']} ({results_relaxed['total_trades'] - results_normal['total_trades']:+d})")
print(f"Win Rate: {results_normal['win_rate']:.2f}% → {results_relaxed['win_rate']:.2f}% ({results_relaxed['win_rate'] - results_normal['win_rate']:+.2f}%)")
print(f"Total P&L: ${results_normal['total_pnl']:.2f} → ${results_relaxed['total_pnl']:.2f} ({results_relaxed['total_pnl'] - results_normal['total_pnl']:+.2f})")
print(f"Sharpe Ratio: {results_normal['sharpe_ratio']:.2f} → {results_relaxed['sharpe_ratio']:.2f} ({results_relaxed['sharpe_ratio'] - results_normal['sharpe_ratio']:+.2f})")
print(f"Max Drawdown: {results_normal['max_drawdown']:.2f}% → {results_relaxed['max_drawdown']:.2f}% ({results_relaxed['max_drawdown'] - results_normal['max_drawdown']:+.2f}%)")

# Analyze relaxed trade performance
if relaxed_trades > 0:
    relaxed_only = [t for t in results_relaxed['trades'] if t.is_relaxed]
    relaxed_wins = sum(1 for t in relaxed_only if t.pnl > 0)
    relaxed_pnl = sum(t.pnl for t in relaxed_only)
    relaxed_win_rate = (relaxed_wins / len(relaxed_only)) * 100 if relaxed_only else 0
    
    print(f"\nRelaxed Trades Only Performance:")
    print(f"  Win Rate: {relaxed_win_rate:.2f}%")
    print(f"  Total P&L: ${relaxed_pnl:.2f}")
    print(f"  Average P&L per trade: ${relaxed_pnl / len(relaxed_only):.2f}")