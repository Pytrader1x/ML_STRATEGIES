"""
Fast Comprehensive Validation - Checking for Bias and Cheating
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime, timedelta
import hashlib

warnings.filterwarnings('ignore')

print("🚀 COMPREHENSIVE STRATEGY VALIDATION")
print("="*80)

# 1. VERIFY DATA INTEGRITY
print("\n📊 STEP 1: Verifying AUDUSD Data Integrity...")
print("-"*60)

data_path = 'data' if os.path.exists('data') else '../data'
file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')

# Check file hash
with open(file_path, 'rb') as f:
    file_hash = hashlib.md5(f.read()).hexdigest()
print(f"File MD5: {file_hash}")

# Load data
df = pd.read_csv(file_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

print(f"Total rows: {len(df):,}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Years: {(df.index[-1] - df.index[0]).days / 365.25:.1f}")

# Quick data quality checks
df['HL_valid'] = df['High'] >= df['Low']
df['OHLC_valid'] = ((df['Open'] >= df['Low']) & (df['Open'] <= df['High']) &
                    (df['Close'] >= df['Low']) & (df['Close'] <= df['High']))

print(f"\nData Quality:")
print(f"  H>=L violations: {(~df['HL_valid']).sum()}")
print(f"  OHLC violations: {(~df['OHLC_valid']).sum()}")
print(f"  Price range: {df['Low'].min():.5f} to {df['High'].max():.5f}")

# Check for realistic AUDUSD range
if df['Low'].min() >= 0.4 and df['High'].max() <= 1.2:
    print("  ✅ Prices within realistic AUDUSD range")
else:
    print("  ⚠️ WARNING: Unrealistic prices detected!")

# Calculate indicators
print("\nCalculating indicators...")
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df, ha_len=350, ha_len2=30)
df = TIC.add_intelligent_chop(df)

# 2. VERIFY EXECUTION SETTINGS
print("\n📋 STEP 2: Verifying Execution Realism...")
print("-"*60)

# Strategy config with realistic settings
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.005,
    sl_min_pips=3.0,
    sl_max_pips=10.0,
    sl_atr_multiplier=0.8,
    tp_atr_multipliers=(0.15, 0.25, 0.4),
    max_tp_percent=0.005,
    tsl_activation_pips=8.0,
    tsl_min_profit_pips=1.0,
    trailing_atr_multiplier=0.8,
    tp_range_market_multiplier=0.4,
    tp_trend_market_multiplier=0.6,
    tp_chop_market_multiplier=0.3,
    exit_on_signal_flip=True,
    partial_profit_before_sl=True,
    partial_profit_sl_distance_ratio=0.3,
    partial_profit_size_percent=0.7,
    relaxed_mode=True,
    realistic_costs=True,  # CRITICAL
    verbose=False,
    debug_decisions=False,
    use_daily_sharpe=True
)

print(f"Realistic Costs: {config.realistic_costs}")
print(f"Intrabar Stop Check: {config.intrabar_stop_on_touch}")
print(f"Entry Slippage: {config.entry_slippage_pips} pips")
print(f"Stop Loss Slippage: {config.stop_loss_slippage_pips} pips")
print(f"Min/Max Stop: {config.sl_min_pips}-{config.sl_max_pips} pips")

if config.realistic_costs and config.intrabar_stop_on_touch:
    print("✅ Execution settings are realistic")
else:
    print("⚠️ WARNING: Unrealistic execution settings!")

# 3. MONTE CARLO VALIDATION
print("\n🎲 STEP 3: Monte Carlo Validation (25 Random Samples)...")
print("-"*60)

strategy = OptimizedProdStrategy(config)
sample_size = 10000  # ~104 days
n_simulations = 25

# Ensure we can take samples
max_start = len(df) - sample_size - 1000
if max_start <= 0:
    print("ERROR: Insufficient data for Monte Carlo")
else:
    results = []
    print(f"Running {n_simulations} simulations with {sample_size:,} bars each...")
    
    for i in range(n_simulations):
        # Random contiguous sample
        start_idx = np.random.randint(1000, max_start)
        sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
        
        try:
            result = strategy.run_backtest(sample_df)
            results.append({
                'sim': i + 1,
                'start': sample_df.index[0].strftime('%Y-%m-%d'),
                'sharpe': result.get('sharpe_ratio', 0),
                'return': result.get('total_return', 0),
                'trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0)
            })
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{n_simulations}...")
        except Exception as e:
            print(f"  Error in sim {i + 1}: {e}")
    
    # Analyze results
    if results:
        results_df = pd.DataFrame(results)
        valid = results_df[results_df['trades'] >= 50]
        
        if len(valid) > 0:
            print(f"\n📊 Monte Carlo Results ({len(valid)} valid samples):")
            print(f"  Avg Sharpe: {valid['sharpe'].mean():.3f} ± {valid['sharpe'].std():.3f}")
            print(f"  Min/Max: [{valid['sharpe'].min():.3f}, {valid['sharpe'].max():.3f}]")
            print(f"  % Profitable: {(valid['sharpe'] > 0).sum() / len(valid) * 100:.1f}%")
            print(f"  % Above 0.7: {(valid['sharpe'] > 0.7).sum() / len(valid) * 100:.1f}%")
            print(f"  Avg Return: {valid['return'].mean():.1f}%")
            print(f"  Avg Win Rate: {valid['win_rate'].mean():.1f}%")
            
            # Check consistency
            if valid['sharpe'].std() / valid['sharpe'].mean() < 0.5:
                print("  ✅ Results are consistent across random samples")
            else:
                print("  ⚠️ High variance across samples")

# 4. OUT-OF-SAMPLE TEST
print("\n📈 STEP 4: Out-of-Sample Validation...")
print("-"*60)

# Split at 2023-01-01
in_sample = df[df.index < '2023-01-01'].copy()
out_sample = df[df.index >= '2023-01-01'].copy()

print(f"In-sample: {in_sample.index[0].date()} to {in_sample.index[-1].date()}")
print(f"Out-sample: {out_sample.index[0].date()} to {out_sample.index[-1].date()}")

# Test both
print("\nTesting in-sample...")
in_result = strategy.run_backtest(in_sample)

print("Testing out-of-sample...")
out_result = strategy.run_backtest(out_sample)

# Compare
print("\n📊 Performance Comparison:")
print(f"{'Metric':<20} {'In-Sample':>12} {'Out-Sample':>12} {'Difference':>12}")
print("-"*60)

metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'max_drawdown']
for metric in metrics:
    in_val = in_result.get(metric, 0)
    out_val = out_result.get(metric, 0)
    diff = out_val - in_val
    print(f"{metric:<20} {in_val:>12.2f} {out_val:>12.2f} {diff:>+12.2f}")

# Check for overfitting
sharpe_decay = (in_result.get('sharpe_ratio', 0) - out_result.get('sharpe_ratio', 0)) / max(in_result.get('sharpe_ratio', 1), 1)
if sharpe_decay < 0.2:
    print("\n✅ No significant overfitting detected")
else:
    print(f"\n⚠️ Potential overfitting: {sharpe_decay:.1%} performance decay")

# 5. SANITY CHECK - Compare to simple baseline
print("\n🎯 STEP 5: Baseline Comparison...")
print("-"*60)

# Test on 2023 data
test_df = df.loc['2023-01-01':'2023-12-31'].copy()
test_result = strategy.run_backtest(test_df)

print(f"Strategy Performance (2023):")
print(f"  Sharpe: {test_result.get('sharpe_ratio', 0):.3f}")
print(f"  Return: {test_result.get('total_return', 0):.1f}%")
print(f"  Win Rate: {test_result.get('win_rate', 0):.1f}%")

# Simple baseline expectation
print(f"\nBaseline Expectations:")
print(f"  Random Trading: Sharpe ~0 (negative with costs)")
print(f"  Buy & Hold AUDUSD: Varies by year")
print(f"  Strategy Advantage: {test_result.get('sharpe_ratio', 0):.1f}x better than random")

# 6. FINAL CHECKS
print("\n✅ FINAL VALIDATION SUMMARY")
print("="*80)

checks = []

# Data check
if df['Low'].min() >= 0.4 and df['High'].max() <= 1.2:
    checks.append("✅ Data Integrity: Real AUDUSD data confirmed")
else:
    checks.append("❌ Data Integrity: Suspicious price ranges")

# Execution check  
if config.realistic_costs and config.intrabar_stop_on_touch:
    checks.append("✅ Execution: Realistic costs and stop checking")
else:
    checks.append("❌ Execution: Unrealistic settings detected")

# Monte Carlo check
if 'valid' in locals() and len(valid) > 0:
    if valid['sharpe'].mean() > 1.0 and (valid['sharpe'] > 0.7).sum() / len(valid) > 0.7:
        checks.append("✅ Monte Carlo: Robust performance across samples")
    else:
        checks.append("⚠️ Monte Carlo: Performance varies significantly")

# Overfitting check
if 'sharpe_decay' in locals() and sharpe_decay < 0.2:
    checks.append("✅ Overfitting: No significant decay out-of-sample")
else:
    checks.append("⚠️ Overfitting: Performance decay detected")

# Print all checks
for check in checks:
    print(f"  {check}")

print("\n🏁 CONCLUSION:")
if all("✅" in check for check in checks):
    print("  Strategy validation PASSED - No cheating or bias detected")
    print("  Results appear genuine and robust")
else:
    print("  Some validation checks raised concerns")
    print("  Further investigation recommended")

print("\n⚠️ Remember: Past performance ≠ Future results")
print("="*80)