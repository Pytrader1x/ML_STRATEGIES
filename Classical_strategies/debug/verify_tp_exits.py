"""Verify that the strategy can generate TP exits and they plot correctly"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
import sys
sys.path.append('..')
from technical_indicators_custom import TIC
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load real data
data_path = "/Users/williamsmith/Python_local_Mac/Ml_Strategies/data/AUDUSD_MASTER.csv"
df = pd.read_csv(data_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.set_index('DateTime')

# Take a sample
df = df.iloc[:5000].copy()

# Add indicators
tic = TIC()
df = tic.add_technical_indicators(df)

# Configure strategy with settings more likely to hit TPs
strategy_config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.005,  # Higher risk
    sl_max_pips=20.0,      # Wider stop loss
    sl_atr_multiplier=2.0,
    tp_atr_multipliers=(0.5, 1.0, 1.5),  # Wider TP levels
    max_tp_percent=0.01,   # Allow larger TPs
    tsl_activation_pips=10,
    tsl_min_profit_pips=5,
    realistic_costs=False,  # No slippage/costs for cleaner test
    verbose=True,
    debug_decisions=False
)

strategy = OptimizedProdStrategy(strategy_config)

# Run backtest
print("Running backtest...")
results = strategy.run_backtest(df)

print(f"\nBacktest complete:")
print(f"Total trades: {results['total_trades']}")
print(f"Trades with TP1 hits: {results['tp_hit_stats']['tp1_hits']}")
print(f"Trades with TP2 hits: {results['tp_hit_stats']['tp2_hits']}")
print(f"Trades with TP3 hits: {results['tp_hit_stats']['tp3_hits']}")
print(f"Trades with partial exits: {results['tp_hit_stats']['partial_exits']}")

# Check individual trades
if 'trades' in results and results['trades']:
    trades_with_tp = 0
    for i, trade in enumerate(results['trades'][:10]):  # Check first 10
        if hasattr(trade, 'partial_exits') and trade.partial_exits:
            tp_exits = [pe for pe in trade.partial_exits if pe.tp_level > 0]
            if tp_exits:
                trades_with_tp += 1
                print(f"\nTrade {i+1} has TP exits:")
                print(f"  Direction: {trade.direction.value}")
                print(f"  Entry: {trade.entry_price:.5f}")
                print(f"  TPs: {[f'{tp:.5f}' for tp in trade.take_profits]}")
                for pe in tp_exits:
                    print(f"  - TP{pe.tp_level} hit at {pe.price:.5f}")
    
    print(f"\nTotal trades with TP exits in first 10: {trades_with_tp}")
    
    # Create plot if we have trades with TP exits
    if trades_with_tp > 0:
        print("\nCreating plot...")
        fig = plot_production_results(df, results)
        plt.savefig('verify_tp_plot.png', dpi=150, bbox_inches='tight')
        print("Plot saved as 'verify_tp_plot.png'")
        print("\nCheck the plot to see if TP exit markers (green diamonds) are visible.")
    else:
        print("\nNo trades with TP exits found. Adjusting strategy parameters might help.")
else:
    print("\nNo trades generated.")