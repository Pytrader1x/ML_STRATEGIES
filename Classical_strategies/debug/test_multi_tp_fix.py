"""Test script to verify multiple TP exits in same candle work correctly"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection

# Create test configuration
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    sl_max_pips=10.0,
    tp_atr_multipliers=(0.5, 1.0, 1.5),  # Close TPs for testing
    max_tp_percent=0.005,
    tsl_activation_pips=15,
    tsl_min_profit_pips=5,
    realistic_costs=False,  # Disable for cleaner testing
    verbose=False,
    debug_decisions=True
)

# Create test data that will trigger multiple TPs in one candle
dates = pd.date_range(start='2024-01-01 00:00', periods=20, freq='15min')

# Base price
base_price = 0.7500

# Create data
data = {
    'Open': [base_price] * 20,
    'High': [base_price + 0.0001] * 20,
    'Low': [base_price - 0.0001] * 20,
    'Close': [base_price] * 20,
    'NTI_Direction': [0] * 20,
    'MB_Bias': [0] * 20,
    'IC_Regime': [0] * 20,
    'IC_RegimeName': ['quiet_range'] * 20,
    'IC_ATR_Normalized': [0.0010] * 20
}

# Set up entry signal at index 5
data['NTI_Direction'][5] = 1
data['MB_Bias'][5] = 1
data['IC_Regime'][5] = 1
data['IC_RegimeName'][5] = 'strong_trend'

# Create a big candle at index 10 that sweeps through all TPs
# For a LONG trade entered at 0.7500:
# TP1 = 0.7500 + 0.5 * 0.0010 = 0.7505
# TP2 = 0.7500 + 1.0 * 0.0010 = 0.7510  
# TP3 = 0.7500 + 1.5 * 0.0010 = 0.7515

data['High'][10] = 0.7520  # This should hit all three TPs
data['Close'][10] = 0.7518  # Close near the high

df = pd.DataFrame(data, index=dates)

print("=== TEST: Multiple TP Exits in Same Candle ===")
print(f"Test setup: Entry at {base_price:.4f}")
print(f"Expected TPs: TP1=0.7505, TP2=0.7510, TP3=0.7515")
print(f"Candle 10 High: {data['High'][10]:.4f} (should hit all TPs)")
print("=" * 50)

# Run strategy
strategy = OptimizedProdStrategy(config)
results = strategy.run_backtest(df)

# Check results
print("\n=== RESULTS ===")
print(f"Total trades: {len(results['trades'])}")

if len(results['trades']) > 0:
    trade = results['trades'][0]
    print(f"\nTrade details:")
    print(f"  Entry: {trade.entry_time} @ {trade.entry_price:.5f}")
    print(f"  Direction: {trade.direction.value}")
    print(f"  Position size: {trade.position_size/1e6:.2f}M")
    print(f"  Take profits: {[f'{tp:.5f}' for tp in trade.take_profits]}")
    print(f"  TP hits: {trade.tp_hits}")
    print(f"  Exit count: {trade.exit_count}")
    print(f"  Partial exits: {len(trade.partial_exits)}")
    
    if trade.partial_exits:
        print("\nPartial exits:")
        for i, pe in enumerate(trade.partial_exits):
            print(f"  {i+1}. TP{pe.tp_level} @ {pe.price:.5f}, size: {pe.size/1e6:.2f}M, P&L: ${pe.pnl:.2f}")
    
    print(f"\nFinal exit:")
    print(f"  Time: {trade.exit_time}")
    print(f"  Price: {trade.exit_price:.5f}")
    print(f"  Reason: {trade.exit_reason.value if trade.exit_reason else 'None'}")
    print(f"  Total P&L: ${trade.pnl:.2f}")
    
    # Verify all TPs were hit
    expected_tp_hits = 3
    if trade.tp_hits == expected_tp_hits:
        print(f"\n✅ SUCCESS: All {expected_tp_hits} TPs were hit in the same candle!")
    else:
        print(f"\n❌ ISSUE: Only {trade.tp_hits} TPs were hit, expected {expected_tp_hits}")
else:
    print("❌ No trades executed")

# Test edge case: Price touches TP1 and TP2 but not TP3
print("\n\n=== TEST 2: Partial TP Sweep ===")

# Reset data
data2 = data.copy()
data2['High'][10] = 0.7512  # Should hit TP1 and TP2 but not TP3
data2['Close'][10] = 0.7511

df2 = pd.DataFrame(data2, index=dates)

strategy2 = OptimizedProdStrategy(config)
results2 = strategy2.run_backtest(df2)

if len(results2['trades']) > 0:
    trade2 = results2['trades'][0]
    print(f"TP hits with high={data2['High'][10]:.4f}: {trade2.tp_hits}")
    print(f"Exit count: {trade2.exit_count}")
    print(f"Remaining size: {trade2.remaining_size/1e6:.2f}M")
    print(f"Exit reason: {trade2.exit_reason.value if trade2.exit_reason else 'None'}")
    
    # Check partial exits
    if trade2.partial_exits:
        print(f"Partial exits: {len(trade2.partial_exits)}")
        for pe in trade2.partial_exits:
            print(f"  - TP{pe.tp_level} @ {pe.price:.5f}")
    
    # The strategy should hit TP1 and TP2, but also trigger partial profit or TP1 pullback
    if trade2.tp_hits == 2:
        print("✅ SUCCESS: Correctly hit TP1 and TP2")
    else:
        print(f"❌ ISSUE: Expected 2 TP hits, got {trade2.tp_hits}")