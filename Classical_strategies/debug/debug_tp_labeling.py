"""
Debug script to check why partial exits are labeled as TP0 instead of TP1/TP2/TP3
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Take a small sample for debugging
df = df.iloc[-20000:]  # Last 20k rows

# Calculate indicators
print("Calculating indicators...")
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df)
df = TIC.add_intelligent_chop(df)

# Create strategy with debug mode enabled
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    debug_decisions=True,  # Enable debug output
    verbose=True
)

strategy = OptimizedProdStrategy(config)

# Run backtest
print("\n" + "="*80)
print("Running backtest with debug mode...")
print("="*80 + "\n")

results = strategy.run_backtest(df)

# Analyze partial exits
print("\n" + "="*80)
print("ANALYZING PARTIAL EXITS")
print("="*80)

if 'trades' in results and results['trades']:
    total_trades = len(results['trades'])
    trades_with_partials = 0
    tp0_count = 0
    tp1_count = 0
    tp2_count = 0
    tp3_count = 0
    
    for i, trade in enumerate(results['trades'][:10]):  # Check first 10 trades
        if trade.partial_exits:
            trades_with_partials += 1
            print(f"\nTrade {i+1}:")
            print(f"  Direction: {trade.direction.value}")
            print(f"  Entry: {trade.entry_price:.5f}")
            print(f"  Take Profits: {[f'{tp:.5f}' for tp in trade.take_profits]}")
            print(f"  TP Hits: {trade.tp_hits}")
            print(f"  Exit Reason: {trade.exit_reason.value if trade.exit_reason else 'None'}")
            print(f"  Number of Partial Exits: {len(trade.partial_exits)}")
            
            for j, pe in enumerate(trade.partial_exits):
                print(f"\n  Partial Exit {j+1}:")
                print(f"    Time: {pe.time}")
                print(f"    Price: {pe.price:.5f}")
                print(f"    Size: {pe.size/1e6:.2f}M")
                print(f"    TP Level: {pe.tp_level}")  # This should be 1, 2, or 3
                print(f"    P&L: ${pe.pnl:,.2f}")
                
                # Check if it matches any TP level
                if hasattr(pe, 'tp_level'):
                    if pe.tp_level == 0:
                        tp0_count += 1
                        print(f"    ⚠️  WARNING: TP Level is 0!")
                    elif pe.tp_level == 1:
                        tp1_count += 1
                    elif pe.tp_level == 2:
                        tp2_count += 1
                    elif pe.tp_level == 3:
                        tp3_count += 1
                
                # Verify the price matches expected TP
                for tp_idx, tp_price in enumerate(trade.take_profits):
                    if abs(pe.price - tp_price) < 0.00001:  # Within rounding error
                        print(f"    ✓ Matches TP{tp_idx+1} at {tp_price:.5f}")
                        if pe.tp_level != tp_idx + 1:
                            print(f"    ❌ ERROR: tp_level={pe.tp_level} but should be {tp_idx+1}")
    
    print(f"\n" + "="*80)
    print("SUMMARY:")
    print(f"Total trades: {total_trades}")
    print(f"Trades with partial exits: {trades_with_partials}")
    print(f"TP0 labels: {tp0_count} (should be 0)")
    print(f"TP1 labels: {tp1_count}")
    print(f"TP2 labels: {tp2_count}")
    print(f"TP3 labels: {tp3_count}")
    
    # Check the trade export logic
    print(f"\n" + "="*80)
    print("TESTING EXPORT LOGIC")
    print("="*80)
    
    # Test the export logic on first trade with partial exits
    for trade in results['trades']:
        if trade.partial_exits:
            print(f"\nTesting export for trade:")
            print(f"  Direction: {trade.direction.value}")
            
            for j, pe in enumerate(trade.partial_exits[:3], 1):
                print(f"\n  Partial Exit {j}:")
                print(f"    Has exit_type attr: {hasattr(pe, 'exit_type')}")
                print(f"    Has tp_level attr: {hasattr(pe, 'tp_level')}")
                if hasattr(pe, 'tp_level'):
                    print(f"    tp_level value: {pe.tp_level}")
                    print(f"    Expected label: TP{pe.tp_level}")
                
                # Test the exact logic from run_strategy_oop.py
                pe_type = pe.exit_type if hasattr(pe, 'exit_type') else f'TP{pe.tp_level}' if hasattr(pe, 'tp_level') else 'PARTIAL'
                print(f"    Export label: {pe_type}")
                
                # Diagnose why it might be TP0
                if hasattr(pe, 'tp_level') and pe.tp_level == 0:
                    print(f"    ❌ ERROR: tp_level is 0, which creates 'TP0' label")
            break

else:
    print("No trades found!")

print("\nDebug script completed.")