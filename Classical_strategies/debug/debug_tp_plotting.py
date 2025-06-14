"""Debug script to check why TP exit markers are not being plotted"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, Trade, PartialExit, TradeDirection, ExitReason
from strategy_code.Prod_plotting import ProductionPlotter, PlotConfig
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Let's first check a recent trades detail file to see if TP exits are recorded
trades_file = "/Users/williamsmith/Python_local_Mac/Ml_Strategies/Classical_strategies/results/AUDUSD_config_1_ultra-tight_risk_management_trades_detail_20250614_151841.csv"
trades_df = pd.read_csv(trades_file)

print("=== Checking trades with TP hits ===")
# Filter trades that have TP hits
tp_trades = trades_df[trades_df['tp_hits'] > 0]
print(f"Total trades with TP hits: {len(tp_trades)} out of {len(trades_df)}")

if len(tp_trades) > 0:
    print(f"\nFirst few trades with TP hits:")
    print(tp_trades[['trade_id', 'direction', 'tp_hits', 'exit_reason', 'partial_exit_1_type', 
                    'partial_exit_2_type', 'partial_exit_3_type']].head(10))
    
    # Count TP exit types
    tp1_count = (trades_df['partial_exit_1_type'] == 'TP0').sum()
    tp2_count = (trades_df['partial_exit_2_type'] == 'TP0').sum()
    tp3_count = (trades_df['partial_exit_3_type'] == 'TP0').sum()
    
    print(f"\nTP exit counts from CSV:")
    print(f"TP1 exits: {tp1_count}")
    print(f"TP2 exits: {tp2_count}")
    print(f"TP3 exits: {tp3_count}")

# Now let's create a mock trade with partial exits to test plotting
print("\n=== Creating test trade with TP exits ===")

# Create test timestamps
base_time = pd.Timestamp('2024-01-01 10:00:00')

# Create a test trade with TP exits
test_trade = Trade(
    entry_time=base_time,
    entry_price=0.7500,
    direction=TradeDirection.SHORT,
    position_size=1_000_000,
    stop_loss=0.7510,
    take_profits=[0.7490, 0.7480, 0.7470],
    exit_time=base_time + timedelta(hours=4),
    exit_price=0.7485,
    exit_reason=ExitReason.TRAILING_STOP
)

# Add partial exits
test_trade.partial_exits = [
    PartialExit(
        time=base_time + timedelta(hours=1),
        price=0.7490,
        size=500_000,
        tp_level=1,
        pnl=500.0
    ),
    PartialExit(
        time=base_time + timedelta(hours=2),
        price=0.7480,
        size=165_000,
        tp_level=2,
        pnl=330.0
    )
]

print(f"Test trade created:")
print(f"  Direction: {test_trade.direction.value}")
print(f"  Entry: {test_trade.entry_price}")
print(f"  TPs: {test_trade.take_profits}")
print(f"  Partial exits: {len(test_trade.partial_exits)}")
for pe in test_trade.partial_exits:
    print(f"    - TP{pe.tp_level} at {pe.price}, size: {pe.size/1e6:.2f}M")

# Now test how the plotting code would handle this
print("\n=== Testing plotting code handling ===")

# Convert trade to dict as plotting does
trade_dict = {
    'entry_time': test_trade.entry_time,
    'exit_time': test_trade.exit_time,
    'entry_price': test_trade.entry_price,
    'exit_price': test_trade.exit_price,
    'direction': test_trade.direction.value if isinstance(test_trade.direction, TradeDirection) else test_trade.direction,
    'exit_reason': test_trade.exit_reason.value if isinstance(test_trade.exit_reason, ExitReason) else test_trade.exit_reason,
    'take_profits': test_trade.take_profits,
    'stop_loss': test_trade.stop_loss,
    'partial_exits': test_trade.partial_exits
}

# Check partial exits
partial_exits = trade_dict.get('partial_exits', [])
print(f"\nPartial exits from dict: {len(partial_exits)}")
print(f"Type of partial_exits: {type(partial_exits)}")
if partial_exits:
    print(f"Type of first partial exit: {type(partial_exits[0])}")
    print(f"First partial exit has tp_level attr: {hasattr(partial_exits[0], 'tp_level')}")

# Test the filtering logic from _plot_partial_exits
tp_exits = [p for p in partial_exits 
           if (hasattr(p, 'tp_level') and p.tp_level > 0) or 
              (isinstance(p, dict) and p.get('tp_level', 0) > 0)]

print(f"\nTP exits after filtering: {len(tp_exits)}")
for i, pe in enumerate(tp_exits):
    if hasattr(pe, 'tp_level'):
        print(f"  TP{pe.tp_level}: time={pe.time}, price={pe.price}")
    else:
        print(f"  TP{pe.get('tp_level', '?')}: time={pe.get('time', '?')}, price={pe.get('price', '?')}")

# Check if the issue might be with the trade data structure
print("\n=== Checking actual strategy output ===")
print("The issue might be that trades returned by the strategy don't have PartialExit objects")
print("but instead have the data flattened into the CSV columns.")
print("\nThe plotting code expects trade.partial_exits to be a list of PartialExit objects,")
print("but the CSV export shows partial exits are stored as separate columns.")
print("\nThis suggests the plotting might be using a different data source than expected.")

# Configure strategy
strategy_config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    sl_atr_multiplier=1.0,
    tp_atr_multipliers=(0.2, 0.3, 0.5),
    max_tp_percent=0.003,
    tsl_activation_pips=3,
    tsl_min_profit_pips=1,
    tsl_initial_buffer_multiplier=1.0,
    trailing_atr_multiplier=0.8,
    realistic_costs=True,
    verbose=False,
    debug_decisions=False
)

strategy = OptimizedProdStrategy(strategy_config)

# Run backtest on a small sample
sample_df = df.iloc[:1000].copy()
results = strategy.run_backtest(sample_df)

# Check if we have trades with TP exits
if 'trades' in results and results['trades']:
    print(f"Total trades: {len(results['trades'])}")
    
    # Check for trades with partial exits
    trades_with_partial_exits = 0
    trades_with_tp_exits = 0
    
    for i, trade in enumerate(results['trades']):
        if hasattr(trade, 'partial_exits') and trade.partial_exits:
            trades_with_partial_exits += 1
            
            # Check if any partial exit is a TP exit
            tp_exits = [pe for pe in trade.partial_exits if pe.tp_level > 0]
            if tp_exits:
                trades_with_tp_exits += 1
                print(f"\nTrade {i+1} has {len(tp_exits)} TP exits:")
                for pe in tp_exits:
                    print(f"  - TP{pe.tp_level} at {pe.time}, price: {pe.price:.5f}, size: {pe.size/1e6:.2f}M, P&L: ${pe.pnl:.2f}")
    
    print(f"\nTrades with partial exits: {trades_with_partial_exits}")
    print(f"Trades with TP exits: {trades_with_tp_exits}")
    
    # Now let's debug the plotting
    if trades_with_tp_exits > 0:
        print("\nDebugging plotting...")
        
        # Create plotter
        plot_config = PlotConfig()
        plotter = ProductionPlotter(plot_config)
        
        # Check how _plot_partial_exits handles the data
        # We'll manually test with the first trade that has TP exits
        for trade in results['trades']:
            if hasattr(trade, 'partial_exits') and trade.partial_exits:
                tp_exits = [pe for pe in trade.partial_exits if pe.tp_level > 0]
                if tp_exits:
                    print(f"\nTesting plotting for trade:")
                    print(f"  Entry: {trade.entry_time} @ {trade.entry_price:.5f}")
                    print(f"  Direction: {trade.direction.value}")
                    print(f"  Partial exits: {len(trade.partial_exits)}")
                    
                    # Convert trade to dict as plotting does
                    trade_dict = {
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'direction': trade.direction.value,
                        'exit_reason': trade.exit_reason.value if trade.exit_reason else None,
                        'take_profits': trade.take_profits,
                        'stop_loss': trade.stop_loss,
                        'partial_exits': trade.partial_exits
                    }
                    
                    # Check how partial exits are accessed in plotting
                    partial_exits = trade_dict.get('partial_exits', [])
                    print(f"\n  Partial exits from dict: {len(partial_exits)}")
                    
                    # Check TP exits filtering
                    tp_exits_filtered = [p for p in partial_exits 
                                       if (hasattr(p, 'tp_level') and p.tp_level > 0)]
                    print(f"  TP exits after filtering: {len(tp_exits_filtered)}")
                    
                    if tp_exits_filtered:
                        for pe in tp_exits_filtered[:3]:  # Show first 3
                            print(f"    - TP{pe.tp_level}: hasattr check = {hasattr(pe, 'tp_level')}, value = {pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'}")
                    
                    break
        
        # Create actual plot
        fig = plotter.plot_production_results(sample_df, results)
        plt.savefig('debug_tp_plot.png', dpi=150, bbox_inches='tight')
        print("\nDebug plot saved as 'debug_tp_plot.png'")
else:
    print("No trades found in results")