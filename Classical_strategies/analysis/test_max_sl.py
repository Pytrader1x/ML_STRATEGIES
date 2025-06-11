"""
Test maximum stop loss of 45 pips
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import create_optimized_strategy
from technical_indicators_custom import TIC

def test_max_sl():
    """Test that stop loss is capped at 45 pips"""
    
    print("Testing Maximum Stop Loss (45 pips)")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use sample
    df_test = df.tail(5000).copy()
    
    # Calculate indicators
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    # Run strategy
    strategy = create_optimized_strategy(
        initial_capital=100_000,
        sl_max_pips=45.0,
        verbose=False
    )
    
    results = strategy.run_backtest(df_test)
    
    # Analyze stop losses
    print(f"\nTotal trades: {len(results['trades'])}")
    
    # Check all trades for SL distance
    max_sl_pips = 0
    sl_distances = []
    
    for trade in results['trades']:
        if trade.direction.value == 'long':
            sl_pips = (trade.entry_price - trade.stop_loss) * 10000
        else:
            sl_pips = (trade.stop_loss - trade.entry_price) * 10000
        
        sl_distances.append(sl_pips)
        max_sl_pips = max(max_sl_pips, sl_pips)
    
    avg_sl_pips = np.mean(sl_distances)
    
    print(f"\nStop Loss Analysis:")
    print(f"  Maximum SL distance: {max_sl_pips:.1f} pips")
    print(f"  Average SL distance: {avg_sl_pips:.1f} pips")
    print(f"  Min SL distance: {min(sl_distances):.1f} pips")
    
    # Check if max SL is respected
    if max_sl_pips <= 45.1:  # Allow small floating point difference
        print(f"\n✅ SUCCESS: All stop losses are within 45 pip maximum!")
    else:
        print(f"\n⚠️  WARNING: Some stop losses exceed 45 pips!")
        
    # Show distribution
    print(f"\nSL Distribution:")
    bins = [0, 15, 25, 35, 45, 100]
    hist, _ = np.histogram(sl_distances, bins=bins)
    
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(sl_distances) * 100
        print(f"  {bins[i]}-{bins[i+1]} pips: {count} trades ({pct:.1f}%)")
    
    # Check stop loss exits
    sl_exits = [t for t in results['trades'] 
                if t.exit_reason and t.exit_reason.value == 'stop_loss']
    
    if sl_exits:
        sl_losses = [t.pnl for t in sl_exits]
        avg_sl_loss = np.mean(sl_losses)
        print(f"\nStop Loss Exits:")
        print(f"  Total SL exits: {len(sl_exits)}")
        print(f"  Average loss per SL: ${avg_sl_loss:.2f}")

if __name__ == "__main__":
    test_max_sl()