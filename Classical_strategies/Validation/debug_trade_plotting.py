"""
Debug script to test trade plotting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_code.Prod_strategy import OptimizedStrategyConfig
from real_time_strategy_simulator import RealTimeStrategySimulator
import pandas as pd

# Create simple test
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    debug_decisions=False,
    verbose=False
)

simulator = RealTimeStrategySimulator(config)

# Run very small simulation
print("Running small simulation...")
results = simulator.run_real_time_simulation(
    currency_pair='AUDUSD',
    rows_to_simulate=200,
    verbose=False
)

print(f"\nSimulation Results:")
print(f"Total trades: {results['trade_statistics']['total_trades']}")
print(f"Trades in results: {len(results['detailed_data']['trades'])}")

# Check first trade structure
if results['detailed_data']['trades']:
    first_trade = results['detailed_data']['trades'][0]
    print(f"\nFirst trade type: {type(first_trade)}")
    print(f"Has entry_time: {hasattr(first_trade, 'entry_time')}")
    
    if hasattr(first_trade, 'entry_time'):
        print(f"Entry: {first_trade.entry_time} @ {first_trade.entry_price}")
        print(f"Exit: {first_trade.exit_time} @ {first_trade.exit_price}")
        print(f"Direction: {first_trade.direction}")
        print(f"Exit reason: {first_trade.exit_reason}")
        print(f"Stop loss: {first_trade.stop_loss}")
        print(f"Take profits: {first_trade.take_profits}")
        print(f"P&L: ${first_trade.pnl:.2f}")
        
        # Check if timestamps are in the dataframe
        from real_time_data_generator import RealTimeDataGenerator
        generator = RealTimeDataGenerator('AUDUSD')
        data_path = generator._find_data_path()
        df = pd.read_csv(data_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # Find if entry time exists in data
        entry_time_str = str(first_trade.entry_time)
        matching_times = df[df['DateTime'].astype(str) == entry_time_str]
        print(f"\nEntry time found in data: {len(matching_times) > 0}")
        if len(matching_times) > 0:
            print(f"Index in data: {matching_times.index[0]}")
else:
    print("\nNo trades found!")

print("\nDebug complete.")