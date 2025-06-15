"""
Find the trade that has TP1 and TP3 exits visible in the chart
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("FINDING TRADE WITH TP1 AND TP3 MARKERS FROM CHART")
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
    
    print(f"Total trades: {len(results['trades'])}")
    
    # Find trades with multiple TP hits, focusing on April period
    april_trades_with_tps = []
    
    for i, trade in enumerate(results['trades']):
        # Focus on April trades
        if trade.entry_time.month == 4 and trade.entry_time.year == 2025:
            if len(trade.partial_exits) >= 2:  # At least 2 partial exits
                april_trades_with_tps.append((i, trade))
    
    print(f"\nApril trades with multiple TP exits: {len(april_trades_with_tps)}")
    
    # Show details of each candidate trade
    for trade_idx, trade in april_trades_with_tps:
        print(f"\n" + "="*60)
        print(f"TRADE #{trade_idx + 1}: {trade.entry_time}")
        print(f"Exit: {trade.exit_time} via {trade.exit_reason}")
        print(f"Direction: {trade.direction}")
        print(f"TP Hits: {trade.tp_hits}")
        print(f"Total P&L: ${trade.pnl:.2f}")
        
        print(f"\nPartial Exits ({len(trade.partial_exits)}):")
        tp_levels_found = []
        for i, pe in enumerate(trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
            tp_levels_found.append(tp_level)
            
            # Check if this matches final exit time
            matches_final = pe.time == trade.exit_time
            marker_type = "FINAL" if matches_final else "PARTIAL"
            
            print(f"  {i+1}. TP{tp_level} at {pe.time} - {pe.size/1e6:.2f}M - ${pe.pnl:.2f} ({marker_type} marker)")
        
        print(f"TP Levels Hit: {sorted(set(tp_levels_found))}")
        
        # Check specifically for the pattern we see: TP1 + TP3 visible, TP2 missing
        has_tp1 = 1 in tp_levels_found
        has_tp2 = 2 in tp_levels_found  
        has_tp3 = 3 in tp_levels_found
        
        print(f"Pattern Analysis:")
        print(f"  Has TP1: {'✅' if has_tp1 else '❌'}")
        print(f"  Has TP2: {'✅' if has_tp2 else '❌'}")
        print(f"  Has TP3: {'✅' if has_tp3 else '❌'}")
        
        if has_tp1 and has_tp3 and not has_tp2:
            print(f"  🎯 POTENTIAL MATCH: This could be the trade in your chart!")
            print(f"  Problem: TP2 is missing - investigating why...")
            
            # Deep dive into this trade
            print(f"\n  DETAILED ANALYSIS:")
            print(f"  Entry: {trade.entry_price:.5f}")
            print(f"  TP Levels Set:")
            for j, tp in enumerate(trade.take_profits, 1):
                print(f"    TP{j}: {tp:.5f}")
            
            # Check price movement to see if TP2 was touched
            entry_idx = None
            exit_idx = None
            
            for k, timestamp in enumerate(df.index):
                if timestamp == trade.entry_time:
                    entry_idx = k
                if timestamp == trade.exit_time:
                    exit_idx = k
            
            if entry_idx is not None and exit_idx is not None:
                trade_data = df.iloc[entry_idx:exit_idx+1]
                direction = trade.direction.value
                
                print(f"  Price Movement Check:")
                tp2_price = trade.take_profits[1]
                
                if direction == 'short':
                    tp2_touched = trade_data['Low'].min() <= tp2_price
                else:
                    tp2_touched = trade_data['High'].max() >= tp2_price
                
                print(f"    TP2 ({tp2_price:.5f}) touched by price: {'✅' if tp2_touched else '❌'}")
                
                if tp2_touched and not has_tp2:
                    print(f"    🚨 BUG DETECTED: Price touched TP2 but no TP2 exit recorded!")
        
        print(f"\n  Chart Visibility Simulation:")
        final_exit_time = trade.exit_time
        visible_partials = []
        final_marker = None
        
        for pe in trade.partial_exits:
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 0
            will_skip = pe.time == final_exit_time
            
            if will_skip:
                final_marker = f"TP{tp_level}"
            else:
                visible_partials.append(f"TP{tp_level}")
        
        print(f"    Visible partial markers: {visible_partials}")
        print(f"    Final exit marker: {final_marker}")

if __name__ == "__main__":
    main()