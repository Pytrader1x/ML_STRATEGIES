"""
Analyze why trades are exiting on TSL instead of hitting full TP levels
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Read the most recent trade detail file
csv_file = 'results/AUDUSD_config_1_ultra-tight_risk_management_trades_detail_20250614_164203.csv'
print(f"Loading trades from: {csv_file}")

df = pd.read_csv(csv_file)

print(f"\nTotal trades: {len(df)}")
print(f"\n" + "="*80)
print("EXIT REASON BREAKDOWN:")
print("="*80)

# Count exit reasons
exit_counts = df['exit_reason'].value_counts()
for reason, count in exit_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{reason:20} {count:4d} ({percentage:5.1f}%)")

# Analyze TP distances
print(f"\n" + "="*80)
print("TAKE PROFIT ANALYSIS:")
print("="*80)

# Calculate average TP distances in pips
tp1_distances = []
tp2_distances = []
tp3_distances = []

for _, trade in df.iterrows():
    entry = trade['entry_price']
    direction = trade['direction']
    
    if pd.notna(trade['tp1_price']):
        if direction == 'long':
            tp1_dist = (trade['tp1_price'] - entry) * 10000
        else:
            tp1_dist = (entry - trade['tp1_price']) * 10000
        tp1_distances.append(tp1_dist)
    
    if pd.notna(trade['tp2_price']):
        if direction == 'long':
            tp2_dist = (trade['tp2_price'] - entry) * 10000
        else:
            tp2_dist = (entry - trade['tp2_price']) * 10000
        tp2_distances.append(tp2_dist)
    
    if pd.notna(trade['tp3_price']):
        if direction == 'long':
            tp3_dist = (trade['tp3_price'] - entry) * 10000
        else:
            tp3_dist = (entry - trade['tp3_price']) * 10000
        tp3_distances.append(tp3_dist)

print(f"Average TP1 distance: {np.mean(tp1_distances):.1f} pips")
print(f"Average TP2 distance: {np.mean(tp2_distances):.1f} pips")
print(f"Average TP3 distance: {np.mean(tp3_distances):.1f} pips")

# Analyze trades with partial exits
print(f"\n" + "="*80)
print("PARTIAL EXIT ANALYSIS:")
print("="*80)

partial_exit_counts = defaultdict(int)
trades_with_partials = 0

for _, trade in df.iterrows():
    has_partial = False
    
    # Check all three possible partial exits
    for i in range(1, 4):
        pe_type = trade.get(f'partial_exit_{i}_type')
        if pd.notna(pe_type):
            has_partial = True
            partial_exit_counts[pe_type] += 1
    
    if has_partial:
        trades_with_partials += 1

print(f"Trades with partial exits: {trades_with_partials} ({trades_with_partials/len(df)*100:.1f}%)")
print(f"\nPartial exit type breakdown:")
for pe_type, count in sorted(partial_exit_counts.items()):
    print(f"  {pe_type:10} {count:4d}")

# Analyze trades that hit trailing stop
print(f"\n" + "="*80)
print("TRAILING STOP ANALYSIS:")
print("="*80)

tsl_trades = df[df['exit_reason'] == 'trailing_stop']
print(f"Total TSL exits: {len(tsl_trades)}")

# Check how many had partial exits before TSL
tsl_with_partials = 0
tsl_pips = []

for _, trade in tsl_trades.iterrows():
    # Check for partial exits
    has_partial = False
    for i in range(1, 4):
        if pd.notna(trade.get(f'partial_exit_{i}_type')):
            has_partial = True
            break
    
    if has_partial:
        tsl_with_partials += 1
    
    # Calculate exit pips
    tsl_pips.append(trade['final_exit_pips'])

print(f"TSL exits with partial exits: {tsl_with_partials} ({tsl_with_partials/len(tsl_trades)*100:.1f}%)")
print(f"Average TSL exit pips: {np.mean(tsl_pips):.1f}")
print(f"Median TSL exit pips: {np.median(tsl_pips):.1f}")

# Show some profitable TSL examples
print(f"\nProfitable TSL exits (top 10):")
profitable_tsl = tsl_trades[tsl_trades['final_pnl'] > 0].nlargest(10, 'final_pnl')
for _, trade in profitable_tsl.iterrows():
    print(f"  P&L: ${trade['final_pnl']:8,.2f} | Pips: {trade['final_exit_pips']:6.1f} | Duration: {trade['trade_duration_hours']:5.1f}h")

# Analyze why TPs might not be hit
print(f"\n" + "="*80)
print("TP HIT ANALYSIS:")
print("="*80)

tp_hit_counts = df['tp_hits'].value_counts().sort_index()
for hits, count in tp_hit_counts.items():
    percentage = (count / len(df)) * 100
    print(f"TP hits = {hits}: {count:4d} trades ({percentage:5.1f}%)")

# Configuration insights
print(f"\n" + "="*80)
print("CONFIGURATION INSIGHTS:")
print("="*80)
print("Config 1: Ultra-Tight Risk Management")
print("- TP multipliers: (0.2, 0.3, 0.5) * ATR")
print("- TSL activation: 3 pips")
print("- TSL min profit: 1 pip")
print("- This means TSL activates very quickly and locks in small profits")
print("- The tight TP levels (especially TP1 at 0.2*ATR) should be hit often")
print("- But TSL might be taking profits before reaching full TP levels")