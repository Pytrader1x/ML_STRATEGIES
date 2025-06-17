"""
Test risk management on periods where original strategy performed well
"""

import pandas as pd
from advanced_momentum_strategy import AdvancedMomentumStrategy
from ultimate_optimizer import AdvancedBacktest

# Load data
print("Loading data...")
data = pd.read_csv('../data/AUDUSD_MASTER_15M.csv', parse_dates=['DateTime'], index_col='DateTime')

# Test on the period where we got Sharpe 1.93 (Recent 50k bars from backtest_results_summary.csv)
test_data = data[-50000:]
print(f"Testing on recent 50k bars where original had Sharpe 1.93")
print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")

# First, verify original strategy performance
print("\n1. Testing ORIGINAL strategy (no risk management):")
original = AdvancedBacktest(test_data)
original_result = original.strategy_momentum(lookback=40, entry_z=1.5, exit_z=0.5)
print(f"   Sharpe: {original_result['sharpe']:.3f}")
print(f"   Returns: {original_result['returns']:.1f}%")
print(f"   Win Rate: {original_result['win_rate']:.1f}%")
print(f"   Max DD: {original_result['max_dd']:.1f}%")

# Test advanced strategy with different risk parameters
print("\n2. Testing ADVANCED strategy with various risk parameters:")

risk_configs = [
    # More conservative parameters
    {'sl': 4.0, 'tp': 6.0, 'trail': 3.0},  # Wider stops
    {'sl': 5.0, 'tp': 8.0, 'trail': 4.0},  # Very wide stops
    {'sl': 3.0, 'tp': 9.0, 'trail': 2.5},  # High R:R ratio
    # Disable certain features
    {'sl': 10.0, 'tp': 15.0, 'trail': 8.0},  # Almost no stops
]

for i, config in enumerate(risk_configs):
    print(f"\n   Config {i+1}: SL={config['sl']}, TP={config['tp']}, Trail={config['trail']}")
    
    strategy = AdvancedMomentumStrategy(
        test_data.copy(),
        sl_atr_multiplier=config['sl'],
        tp_atr_multiplier=config['tp'],
        trailing_sl_atr=config['trail'],
        risk_per_trade=0.02
    )
    
    df = strategy.run_backtest()
    metrics = strategy.calculate_metrics(df)
    
    print(f"   Sharpe: {metrics['sharpe']:.3f}")
    print(f"   Returns: {metrics['returns']:.1f}%")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Trades: {metrics['trades']}")
    
    if metrics['exit_analysis']:
        print("   Exits:", end=" ")
        for reason, count in metrics['exit_analysis'].items():
            print(f"{reason}={count}", end=" ")
        print()
    
    # Check if we beat original
    if metrics['sharpe'] > original_result['sharpe']:
        print(f"   *** BEATS ORIGINAL! Improvement: +{metrics['sharpe'] - original_result['sharpe']:.3f} ***")

# Test with minimal risk management (essentially original + position sizing)
print("\n3. Testing MINIMAL risk management (very wide stops):")
minimal_strategy = AdvancedMomentumStrategy(
    test_data.copy(),
    sl_atr_multiplier=20.0,  # Essentially no stop loss
    tp_atr_multiplier=30.0,  # Essentially no take profit
    trailing_sl_atr=15.0,    # Very wide trailing stop
    risk_per_trade=0.02
)

df = minimal_strategy.run_backtest()
metrics = minimal_strategy.calculate_metrics(df)

print(f"   Sharpe: {metrics['sharpe']:.3f}")
print(f"   Returns: {metrics['returns']:.1f}%")
print(f"   Win Rate: {metrics['win_rate']:.1f}%")
print(f"   Trades: {metrics['trades']}")

if metrics['exit_analysis']:
    print("   Exits:", end=" ")
    for reason, count in metrics['exit_analysis'].items():
        print(f"{reason}={count}", end=" ")
    print()

print("\n" + "="*60)
print("CONCLUSION:")
if metrics['sharpe'] > original_result['sharpe']:
    print("Risk management CAN improve the strategy with right parameters!")
else:
    print("Original strategy without stops performs better on this data.")
    print("The momentum strategy's statistical exits are sufficient.")
print("="*60)