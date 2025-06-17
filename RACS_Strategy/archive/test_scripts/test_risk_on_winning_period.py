"""
Test risk management on the period where original strategy achieved Sharpe 1.286
"""

import pandas as pd
from advanced_momentum_strategy import AdvancedMomentumStrategy
import json
from datetime import datetime

# Load the original winning period data
print("Loading data from winning period...")
data = pd.read_csv('../data/AUDUSD_MASTER_15M.csv', parse_dates=['DateTime'], index_col='DateTime')

# Use the 50k period that achieved good results
data = data[-50000:]
print(f"Testing on {len(data):,} bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Test a few risk management configurations
configs = [
    {'sl': 3.0, 'tp': 4.5, 'trail': 2.5, 'name': 'Conservative'},
    {'sl': 2.5, 'tp': 5.0, 'trail': 2.0, 'name': 'Balanced'},
    {'sl': 2.0, 'tp': 6.0, 'trail': 1.5, 'name': 'Aggressive'},
    {'sl': 3.5, 'tp': 7.0, 'trail': 3.0, 'name': 'Wide Stops'},
    {'sl': 4.0, 'tp': 8.0, 'trail': 3.5, 'name': 'Very Wide'},
]

best_result = None
best_sharpe = -999

print("\nTesting risk configurations...")
print("-" * 80)

for config in configs:
    print(f"\n{config['name']} Configuration:")
    print(f"  SL: {config['sl']}x ATR, TP: {config['tp']}x ATR, Trail: {config['trail']}x ATR")
    
    try:
        strategy = AdvancedMomentumStrategy(
            data.copy(),
            sl_atr_multiplier=config['sl'],
            tp_atr_multiplier=config['tp'],
            trailing_sl_atr=config['trail'],
            risk_per_trade=0.02
        )
        
        df = strategy.run_backtest()
        metrics = strategy.calculate_metrics(df)
        
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Returns: {metrics['returns']:.1f}%")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Max DD: {metrics['max_dd']:.1f}%")
        print(f"  Trades: {metrics['trades']}")
        
        if metrics['exit_analysis']:
            print("  Exits:", end="")
            for reason, count in sorted(metrics['exit_analysis'].items()):
                print(f" {reason}={count}", end="")
            print()
        
        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_result = {
                'config': config,
                'metrics': metrics
            }
            
    except Exception as e:
        print(f"  Error: {str(e)}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

original_sharpe = 1.286
print(f"Original Strategy (no risk mgmt): Sharpe = {original_sharpe:.3f}")

if best_result:
    config = best_result['config']
    metrics = best_result['metrics']
    
    print(f"\nBest Risk Config: {config['name']}")
    print(f"  Parameters: SL={config['sl']}, TP={config['tp']}, Trail={config['trail']}")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Returns: {metrics['returns']:.1f}%")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    
    if metrics['sharpe'] > original_sharpe:
        improvement = (metrics['sharpe'] / original_sharpe - 1) * 100
        print(f"\n✅ IMPROVED by {improvement:.1f}%!")
    else:
        gap = original_sharpe - metrics['sharpe']
        print(f"\n❌ Still {gap:.3f} below original")

# Now test with NO risk management (just momentum exits)
print("\n" + "-"*80)
print("Testing pure momentum exits (like original)...")
print("-"*80)

# Very wide stops that won't be hit
strategy = AdvancedMomentumStrategy(
    data.copy(),
    sl_atr_multiplier=10.0,  # Very wide - basically never hit
    tp_atr_multiplier=20.0,  # Very wide - basically never hit
    trailing_sl_atr=15.0,    # Very wide - basically never hit
    risk_per_trade=1.0       # Use full position like original
)

df = strategy.run_backtest()
metrics = strategy.calculate_metrics(df)

print(f"Sharpe: {metrics['sharpe']:.3f}")
print(f"Returns: {metrics['returns']:.1f}%")
print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Max DD: {metrics['max_dd']:.1f}%")
print(f"Trades: {metrics['trades']}")

if metrics['exit_analysis']:
    print("\nExit Types:")
    for reason, count in sorted(metrics['exit_analysis'].items()):
        pct = (count / metrics['trades'] * 100) if metrics['trades'] > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")