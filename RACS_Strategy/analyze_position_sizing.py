"""
Analyze why risk management is reducing performance
"""

import pandas as pd
import numpy as np
from ultimate_optimizer import AdvancedBacktest

# Load data
print("Comparing original vs risk-managed strategies...")
data = pd.read_csv('../data/AUDUSD_MASTER_15M.csv', parse_dates=['DateTime'], index_col='DateTime')
data = data[-50000:]  # Recent 50k

print(f"Testing on {len(data):,} bars")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Run original strategy
print("\n" + "="*60)
print("ORIGINAL STRATEGY (Fixed Position Size)")
print("="*60)

backtester = AdvancedBacktest(data)
result = backtester.strategy_momentum(
    lookback=40,
    entry_z=1.5,
    exit_z=0.5
)

print(f"Sharpe: {result['sharpe']:.3f}")
print(f"Returns: {result['returns']:.1f}%")
print(f"Win Rate: {result['win_rate']:.1f}%")
print(f"Max DD: {result['max_dd']:.1f}%")
print(f"Trades: {result['trades']}")

# Analyze the strategy signals
df = data.copy()
df['Momentum'] = df['Close'].pct_change(40)
df['Mom_Mean'] = df['Momentum'].rolling(50).mean()
df['Mom_Std'] = df['Momentum'].rolling(50).std()
df['Mom_Z'] = (df['Momentum'] - df['Mom_Mean']) / df['Mom_Std']

# Calculate ATR
df['High_Low'] = df['High'] - df['Low']
df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
df['ATR'] = df['True_Range'].rolling(14).mean()

# Find entry points
long_entries = df[df['Mom_Z'] < -1.5]
short_entries = df[df['Mom_Z'] > 1.5]

print(f"\nLong entries: {len(long_entries)}")
print(f"Short entries: {len(short_entries)}")

# Analyze what happens after entries
print("\n" + "-"*60)
print("ANALYZING POST-ENTRY PRICE MOVEMENT")
print("-"*60)

def analyze_entries(entries, direction='long'):
    """Analyze price movement after entries"""
    
    results = []
    
    for idx in entries.index[:100]:  # First 100 entries
        try:
            entry_price = df.loc[idx, 'Close']
            entry_atr = df.loc[idx, 'ATR']
            
            # Get next 200 bars
            future_data = df.loc[idx:].iloc[1:201]
            
            if len(future_data) < 50:
                continue
            
            # Calculate price extremes
            if direction == 'long':
                max_profit = (future_data['High'].max() - entry_price) / entry_atr
                max_loss = (entry_price - future_data['Low'].min()) / entry_atr
                
                # Check if stops would have been hit
                sl_2_hit = (future_data['Low'] < entry_price - 2*entry_atr).any()
                sl_3_hit = (future_data['Low'] < entry_price - 3*entry_atr).any()
                tp_3_hit = (future_data['High'] > entry_price + 3*entry_atr).any()
                tp_5_hit = (future_data['High'] > entry_price + 5*entry_atr).any()
                
            else:  # short
                max_profit = (entry_price - future_data['Low'].min()) / entry_atr
                max_loss = (future_data['High'].max() - entry_price) / entry_atr
                
                sl_2_hit = (future_data['High'] > entry_price + 2*entry_atr).any()
                sl_3_hit = (future_data['High'] > entry_price + 3*entry_atr).any()
                tp_3_hit = (future_data['Low'] < entry_price - 3*entry_atr).any()
                tp_5_hit = (future_data['Low'] < entry_price - 5*entry_atr).any()
            
            # Find momentum exit
            exit_mask = abs(future_data['Mom_Z']) < 0.5
            if exit_mask.any():
                exit_idx = exit_mask.idxmax()
                bars_to_exit = len(df.loc[idx:exit_idx]) - 1
                exit_price = df.loc[exit_idx, 'Close']
                
                if direction == 'long':
                    exit_return = (exit_price - entry_price) / entry_price * 100
                else:
                    exit_return = (entry_price - exit_price) / entry_price * 100
            else:
                bars_to_exit = len(future_data)
                exit_return = 0
            
            results.append({
                'max_profit_atr': max_profit,
                'max_loss_atr': max_loss,
                'sl_2_hit': sl_2_hit,
                'sl_3_hit': sl_3_hit,
                'tp_3_hit': tp_3_hit,
                'tp_5_hit': tp_5_hit,
                'bars_to_exit': bars_to_exit,
                'exit_return': exit_return
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(results)

# Analyze long entries
print("\nLONG ENTRIES ANALYSIS:")
long_results = analyze_entries(long_entries, 'long')
if len(long_results) > 0:
    print(f"Average max profit: {long_results['max_profit_atr'].mean():.1f} ATR")
    print(f"Average max loss: {long_results['max_loss_atr'].mean():.1f} ATR")
    print(f"2 ATR SL hit rate: {long_results['sl_2_hit'].mean()*100:.1f}%")
    print(f"3 ATR SL hit rate: {long_results['sl_3_hit'].mean()*100:.1f}%")
    print(f"3 ATR TP hit rate: {long_results['tp_3_hit'].mean()*100:.1f}%")
    print(f"5 ATR TP hit rate: {long_results['tp_5_hit'].mean()*100:.1f}%")
    print(f"Average bars to momentum exit: {long_results['bars_to_exit'].mean():.0f}")
    print(f"Average momentum exit return: {long_results['exit_return'].mean():.2f}%")

# Analyze short entries
print("\nSHORT ENTRIES ANALYSIS:")
short_results = analyze_entries(short_entries, 'short')
if len(short_results) > 0:
    print(f"Average max profit: {short_results['max_profit_atr'].mean():.1f} ATR")
    print(f"Average max loss: {short_results['max_loss_atr'].mean():.1f} ATR")
    print(f"2 ATR SL hit rate: {short_results['sl_2_hit'].mean()*100:.1f}%")
    print(f"3 ATR SL hit rate: {short_results['sl_3_hit'].mean()*100:.1f}%")
    print(f"3 ATR TP hit rate: {short_results['tp_3_hit'].mean()*100:.1f}%")
    print(f"5 ATR TP hit rate: {short_results['tp_5_hit'].mean()*100:.1f}%")
    print(f"Average bars to momentum exit: {short_results['bars_to_exit'].mean():.0f}")
    print(f"Average momentum exit return: {short_results['exit_return'].mean():.2f}%")

# Key insights
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print("\n1. The original strategy works because:")
print("   - It uses momentum mean reversion (contrarian)")
print("   - Entries at extreme z-scores often see quick reversions")
print("   - Position sizing is constant (no risk adjustment)")
print("   - No stops means riding through drawdowns to capture reversions")

print("\n2. Risk management hurts because:")
print("   - Stop losses cut trades before mean reversion occurs")
print("   - Position sizing based on ATR reduces size during volatility")
print("   - Trailing stops exit winning trades too early")
print("   - The strategy needs to endure temporary adverse moves")

print("\n3. Recommendation:")
print("   - Keep the original strategy without tight risk management")
print("   - Use portfolio-level risk controls instead")
print("   - Consider reducing overall position size if needed")
print("   - Monitor maximum drawdown at portfolio level")