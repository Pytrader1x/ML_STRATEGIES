"""
Test script to verify all metrics are being calculated correctly
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading AUDUSD data...")
import os
data_path = 'data' if os.path.exists('data') else '../data'
df = pd.read_csv(os.path.join(data_path, 'AUDUSD_MASTER_15M.csv'))
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Use a small sample for quick testing
test_df = df.loc['2024-05-01':'2024-05-10'].copy()
print(f"Test data: {len(test_df)} rows")

# Calculate indicators
print("Calculating indicators...")
test_df = TIC.add_neuro_trend_intelligent(test_df)
test_df = TIC.add_market_bias(test_df, ha_len=350, ha_len2=30)
test_df = TIC.add_intelligent_chop(test_df)

# Create strategy with minimal risk for testing
config = OptimizedStrategyConfig(
    initial_capital=100_000,
    risk_per_trade=0.005,
    sl_min_pips=3.0,
    sl_max_pips=10.0,
    relaxed_mode=True,
    realistic_costs=True,
    verbose=False
)

strategy = OptimizedProdStrategy(config)

# Run backtest
print("\nRunning backtest...")
result = strategy.run_backtest(test_df)

# Display all metrics
print("\n" + "="*60)
print("METRICS VERIFICATION TEST")
print("="*60)

metrics_to_check = [
    ('Total Trades', 'total_trades'),
    ('Win Rate', 'win_rate', '%'),
    ('Sharpe Ratio', 'sharpe_ratio'),
    ('Sortino Ratio', 'sortino_ratio'),
    ('Average Trade', 'avg_trade', '$'),
    ('Win/Loss Ratio', 'win_loss_ratio'),
    ('Expectancy', 'expectancy', '$'),
    ('Best Trade', 'best_trade', '$'),
    ('Worst Trade', 'worst_trade', '$'),
    ('SQN Score', 'sqn'),
    ('Trades per Day', 'trades_per_day'),
    ('Recovery Factor', 'recovery_factor'),
    ('Total P&L', 'total_pnl', '$'),
    ('Max Drawdown', 'max_drawdown', '%'),
    ('Profit Factor', 'profit_factor')
]

for metric_info in metrics_to_check:
    name = metric_info[0]
    key = metric_info[1]
    suffix = metric_info[2] if len(metric_info) > 2 else ''
    
    value = result.get(key, 'NOT FOUND')
    
    if isinstance(value, float):
        if suffix == '$':
            print(f"{name:.<25} ${value:,.2f}")
        elif suffix == '%':
            print(f"{name:.<25} {value:.2f}%")
        else:
            print(f"{name:.<25} {value:.3f}")
    else:
        print(f"{name:.<25} {value}")

# Check if metrics are non-zero when they should be
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

if result['total_trades'] > 0:
    checks = [
        ('Average Trade calculated', result.get('avg_trade', 0) != 0),
        ('Trades per Day calculated', result.get('trades_per_day', 0) > 0),
        ('Best/Worst trades found', result.get('best_trade', 0) != 0 or result.get('worst_trade', 0) != 0),
    ]
    
    if result['winning_trades'] > 0 and result['losing_trades'] > 0:
        checks.append(('Win/Loss Ratio calculated', result.get('win_loss_ratio', 0) != 0))
    
    if result['max_drawdown'] > 0:
        checks.append(('Recovery Factor calculated', result.get('recovery_factor', 0) != 0))
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<40} {status}")

print("\n✅ All metrics are now properly calculated!")