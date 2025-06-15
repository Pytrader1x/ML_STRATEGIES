"""
Debug the March 30 21:15 trade TP3 display issue
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("DEBUG: March 30 21:15 Trade - TP3 Display Issue")
    print("="*80)
    
    # Load data
    data_path = '../data'
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    df_full = pd.read_csv(file_path)
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    
    # Take last 5000 rows
    df = df_full.iloc[-5000:].copy()
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy
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
    
    # Find the March 30 21:15 trade
    target = pd.Timestamp('2025-03-30 21:15:00')
    found_trade = None
    
    for trade in results['trades']:
        if trade.entry_time == target:
            found_trade = trade
            break
    
    if found_trade:
        print(f"\nFound trade: Entry at {found_trade.entry_time}")
        print(f"Direction: {found_trade.direction.value}")
        print(f"Position Size: {found_trade.position_size/1e6:.2f}M")
        print(f"Entry Price: {found_trade.entry_price:.5f}")
        
        print(f"\nTP Levels:")
        for i, tp in enumerate(found_trade.take_profits, 1):
            print(f"  TP{i}: {tp:.5f}")
        
        print(f"\n{'='*60}")
        print("PARTIAL EXITS ANALYSIS:")
        print(f"{'='*60}")
        
        total_exit_size = 0
        cumulative_pnl = 0
        
        for i, pe in enumerate(found_trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
            
            # Calculate pips
            if found_trade.direction.value == 'short':
                pips = (found_trade.entry_price - pe.price) / 0.0001
            else:
                pips = (pe.price - found_trade.entry_price) / 0.0001
            
            total_exit_size += pe.size
            cumulative_pnl += pe.pnl
            
            print(f"\nExit {i+1} - {'TP' + str(tp_level) if tp_level != 'N/A' else 'Final'}:")
            print(f"  Time: {pe.time}")
            print(f"  Price: {pe.price:.5f}")
            print(f"  Size: {pe.size/1e6:.2f}M")
            print(f"  Pips: {pips:.1f}")
            print(f"  P&L for this exit: ${pe.pnl:.2f}")
            print(f"  Cumulative P&L: ${cumulative_pnl:.2f}")
        
        print(f"\n{'='*60}")
        print("WHAT TP3 MARKER SHOULD SHOW:")
        print(f"{'='*60}")
        
        # Find TP3 exit
        tp3_exit = None
        for pe in found_trade.partial_exits:
            if hasattr(pe, 'tp_level') and pe.tp_level == 3:
                tp3_exit = pe
                break
        
        if tp3_exit:
            if found_trade.direction.value == 'short':
                tp3_pips = (found_trade.entry_price - tp3_exit.price) / 0.0001
            else:
                tp3_pips = (tp3_exit.price - found_trade.entry_price) / 0.0001
            
            print(f"\nTP3 Exit Details:")
            print(f"  Position closed: {tp3_exit.size/1e6:.2f}M (not 0M!)")
            print(f"  Pips: {tp3_pips:.1f}")
            print(f"  P&L for TP3 exit: ${tp3_exit.pnl:.2f}")
            print(f"  Total trade P&L: ${found_trade.pnl:.2f}")
            
            print(f"\nTP3 Marker Should Display:")
            print(f"  Line 1: TP3|+{tp3_pips:.1f}p|${tp3_exit.pnl:.0f}|{tp3_exit.size/1e6:.1f}M")
            print(f"  Line 2: Total: ${found_trade.pnl:.0f}")
        
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"{'='*60}")
        print(f"Total Position: {found_trade.position_size/1e6:.2f}M")
        print(f"Total Exited: {total_exit_size/1e6:.2f}M")
        print(f"Match: {'✅' if abs(found_trade.position_size - total_exit_size) < 1 else '❌'}")
        
        print(f"\nP&L Breakdown:")
        for i, pe in enumerate(found_trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'Final'
            print(f"  {'TP' + str(tp_level) if tp_level != 'Final' else tp_level}: ${pe.pnl:.2f}")
        print(f"  Total: ${found_trade.pnl:.2f}")
        
    else:
        print("Trade not found!")

if __name__ == "__main__":
    main()