"""
Debug the specific 2025-04-02 03:30 trade to check TP2 exit
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("DEBUG: 2025-04-02 03:30 TRADE - TP2 ANALYSIS")
    print("="*80)
    
    # Load data
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    # Take last 5000 rows (same as single run script)
    df = df_full.iloc[-5000:].copy()
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy (no debug to reduce output)
    strategy_config = OptimizedStrategyConfig(
        initial_capital=1_000_000, risk_per_trade=0.002, sl_max_pips=10.0,
        sl_atr_multiplier=1.0, tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003, tsl_activation_pips=15, tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0, trailing_atr_multiplier=1.2,
        tp_range_market_multiplier=0.5, tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3, sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False, partial_profit_before_sl=False,
        debug_decisions=False, use_daily_sharpe=True
    )
    
    strategy = OptimizedProdStrategy(strategy_config)
    results = strategy.run_backtest(df)
    
    # Find the trade closest to 2025-04-02 03:30
    target = pd.Timestamp('2025-04-02 03:30:00')
    closest_trade = None
    min_diff = float('inf')
    
    for trade in results['trades']:
        diff = abs((trade.entry_time - target).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_trade = trade
    
    if closest_trade:
        print(f"Found closest trade: Entry at {closest_trade.entry_time}")
        print(f"Final exit: {closest_trade.exit_time} via {closest_trade.exit_reason}")
        print(f"TP Hits: {closest_trade.tp_hits}")
        print(f"Total P&L: ${closest_trade.pnl:.2f}")
        
        print(f"\nTRADE LEVELS:")
        print(f"Entry: {closest_trade.entry_price:.5f}")
        print(f"Stop Loss: {closest_trade.stop_loss:.5f}")
        for i, tp in enumerate(closest_trade.take_profits, 1):
            print(f"TP{i}: {tp:.5f}")
        
        print(f"\nPARTIAL EXITS ANALYSIS:")
        print(f"Number of partial exits: {len(closest_trade.partial_exits)}")
        
        if len(closest_trade.partial_exits) == 0:
            print("❌ NO PARTIAL EXITS FOUND!")
            print("This trade went directly to final exit without hitting any TP levels.")
        else:
            # Analyze each partial exit
            tp_levels_hit = set()
            for i, pe in enumerate(closest_trade.partial_exits):
                tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
                tp_levels_hit.add(tp_level)
                
                print(f"\nPartial Exit {i+1}:")
                print(f"  Time: {pe.time}")
                print(f"  Price: {pe.price:.5f}")
                print(f"  TP Level: {tp_level}")
                print(f"  Size: {pe.size/1e6:.2f}M")
                print(f"  P&L: ${pe.pnl:.2f}")
                
                # Calculate pips
                if closest_trade.direction.value == 'short':
                    pips = (closest_trade.entry_price - pe.price) / 0.0001
                else:
                    pips = (pe.price - closest_trade.entry_price) / 0.0001
                print(f"  Pips: {pips:.1f}")
                
                # Check if this matches final exit time
                matches_final = pe.time == closest_trade.exit_time
                print(f"  Matches final exit time: {'YES' if matches_final else 'NO'}")
            
            print(f"\nTP LEVELS SUMMARY:")
            for tp_num in [1, 2, 3]:
                hit = tp_num in tp_levels_hit
                print(f"  TP{tp_num}: {'✅ HIT' if hit else '❌ NOT HIT'}")
            
            # Check what should show on chart
            print(f"\nCHART PLOTTING ANALYSIS:")
            final_exit_time = closest_trade.exit_time
            
            partial_markers_count = 0
            final_marker_count = 0
            
            for pe in closest_trade.partial_exits:
                tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 0
                
                # Simulate the plotting logic
                will_skip = final_exit_time and pe.time == final_exit_time
                
                if will_skip:
                    final_marker_count += 1
                    print(f"  TP{tp_level} at {pe.time}: Will show as FINAL EXIT marker")
                else:
                    partial_markers_count += 1
                    print(f"  TP{tp_level} at {pe.time}: Will show as PARTIAL EXIT marker")
            
            print(f"\nEXPECTED CHART MARKERS:")
            print(f"  Partial exit markers: {partial_markers_count}")
            print(f"  Final exit marker: 1 (TP{closest_trade.exit_reason.value.split('_')[-1] if 'take_profit' in str(closest_trade.exit_reason) else 'unknown'})")
            
            # Specific TP2 analysis
            tp2_exits = [pe for pe in closest_trade.partial_exits 
                        if hasattr(pe, 'tp_level') and pe.tp_level == 2]
            
            print(f"\nTP2 SPECIFIC ANALYSIS:")
            if len(tp2_exits) == 0:
                print("  ❌ NO TP2 EXITS FOUND!")
                print("  Possible reasons:")
                print("    1. Price never reached TP2 level")
                print("    2. Trade was closed before TP2 could be hit")
                print("    3. Bug in TP exit logic")
            else:
                for tp2 in tp2_exits:
                    print(f"  ✅ TP2 EXIT FOUND:")
                    print(f"    Time: {tp2.time}")
                    print(f"    Price: {tp2.price:.5f}")
                    print(f"    Expected TP2 price: {closest_trade.take_profits[1]:.5f}")
                    print(f"    Price match: {'YES' if abs(tp2.price - closest_trade.take_profits[1]) < 0.00001 else 'NO'}")
                    
                    # Check if this will be skipped in plotting
                    will_skip = tp2.time == final_exit_time
                    print(f"    Will be skipped in plotting: {'YES' if will_skip else 'NO'}")
                    if will_skip:
                        print(f"    Reason: Same time as final exit ({final_exit_time})")
        
        # Price movement analysis to see if TP2 was actually touched
        print(f"\nPRICE MOVEMENT VERIFICATION:")
        entry_idx = None
        exit_idx = None
        
        for i, timestamp in enumerate(df.index):
            if timestamp == closest_trade.entry_time:
                entry_idx = i
            if timestamp == closest_trade.exit_time:
                exit_idx = i
        
        if entry_idx is not None and exit_idx is not None:
            trade_data = df.iloc[entry_idx:exit_idx+1]
            direction = closest_trade.direction.value
            
            print(f"Trade duration: {len(trade_data)} bars")
            print(f"High during trade: {trade_data['High'].max():.5f}")
            print(f"Low during trade: {trade_data['Low'].min():.5f}")
            
            for i, tp_price in enumerate(closest_trade.take_profits, 1):
                if direction == 'short':
                    # For shorts, TP is below entry, check if Low touched TP
                    tp_touched = trade_data['Low'].min() <= tp_price
                else:
                    # For longs, TP is above entry, check if High touched TP
                    tp_touched = trade_data['High'].max() >= tp_price
                
                print(f"TP{i} ({tp_price:.5f}) touched by price: {'✅ YES' if tp_touched else '❌ NO'}")
    else:
        print("No suitable trade found")

if __name__ == "__main__":
    main()