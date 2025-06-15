"""
Debug the 2025-04-03 17:15 trade P&L calculations
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("DEBUG: 2025-04-03 17:15 TRADE - P&L CALCULATION ANALYSIS")
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
    
    # Find the trade around 2025-04-03 17:15
    target = pd.Timestamp('2025-04-03 17:15:00')
    closest_trade = None
    min_diff = float('inf')
    
    for trade in results['trades']:
        diff = abs((trade.entry_time - target).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_trade = trade
    
    if closest_trade:
        print(f"\nFound trade: Entry at {closest_trade.entry_time}")
        print(f"Direction: {closest_trade.direction.value}")
        print(f"Initial Position: {closest_trade.position_size/1e6:.2f}M units")
        print(f"Entry Price: {closest_trade.entry_price:.5f}")
        
        print(f"\nTRADE LEVELS:")
        print(f"Stop Loss: {closest_trade.stop_loss:.5f}")
        for i, tp in enumerate(closest_trade.take_profits, 1):
            print(f"TP{i}: {tp:.5f}")
        
        print(f"\n{'='*60}")
        print("PARTIAL EXITS BREAKDOWN:")
        print(f"{'='*60}")
        
        total_exit_size = 0
        total_pnl = 0
        tp_pnls = {}
        
        for i, pe in enumerate(closest_trade.partial_exits):
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
            
            print(f"\nPartial Exit {i+1} (TP{tp_level}):")
            print(f"  Time: {pe.time}")
            print(f"  Exit Price: {pe.price:.5f}")
            print(f"  Size: {pe.size/1e6:.2f}M units")
            
            # Manual P&L calculation
            if closest_trade.direction.value == 'short':
                price_change = closest_trade.entry_price - pe.price
            else:
                price_change = pe.price - closest_trade.entry_price
            
            pips = price_change / 0.0001
            manual_pnl = (pe.size / 1e6) * 100 * pips  # $100/pip per million
            
            print(f"  Price Change: {price_change:.5f}")
            print(f"  Pips: {pips:.1f}")
            print(f"  Reported P&L: ${pe.pnl:.2f}")
            print(f"  Manual Calc P&L: ${manual_pnl:.2f}")
            print(f"  Difference: ${abs(pe.pnl - manual_pnl):.2f}")
            
            total_exit_size += pe.size
            total_pnl += pe.pnl
            
            if tp_level != 'N/A':
                if tp_level not in tp_pnls:
                    tp_pnls[tp_level] = []
                tp_pnls[tp_level].append(pe.pnl)
        
        print(f"\n{'='*60}")
        print("FINAL EXIT:")
        print(f"{'='*60}")
        print(f"Final Exit Time: {closest_trade.exit_time}")
        print(f"Final Exit Price: {closest_trade.exit_price:.5f}")
        print(f"Exit Reason: {closest_trade.exit_reason}")
        
        # Check if final exit was also a TP exit
        final_tp_num = None
        if 'take_profit' in str(closest_trade.exit_reason):
            tp_str = str(closest_trade.exit_reason.value)
            if 'take_profit_1' in tp_str:
                final_tp_num = 1
            elif 'take_profit_2' in tp_str:
                final_tp_num = 2
            elif 'take_profit_3' in tp_str:
                final_tp_num = 3
        
        if final_tp_num:
            print(f"Final exit was TP{final_tp_num}")
        
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"{'='*60}")
        print(f"Initial Position: {closest_trade.position_size/1e6:.2f}M units")
        print(f"Total Exited: {total_exit_size/1e6:.2f}M units")
        print(f"Position Match: {'✅' if abs(closest_trade.position_size - total_exit_size) < 1 else '❌'}")
        
        print(f"\nP&L from Partial Exits: ${total_pnl:.2f}")
        print(f"Trade Total P&L: ${closest_trade.pnl:.2f}")
        print(f"Difference: ${abs(closest_trade.pnl - total_pnl):.2f}")
        
        # TP-level P&L breakdown
        print(f"\nP&L by TP Level:")
        for tp_level in sorted(tp_pnls.keys()):
            level_total = sum(tp_pnls[tp_level])
            print(f"  TP{tp_level}: ${level_total:.2f} ({len(tp_pnls[tp_level])} exits)")
        
        # Check for calculation issues
        print(f"\n{'='*60}")
        print("POTENTIAL ISSUES:")
        print(f"{'='*60}")
        
        if abs(closest_trade.pnl - total_pnl) > 0.01:
            print("❌ Total P&L doesn't match sum of partial exits!")
            print("   This could indicate:")
            print("   1. Missing partial exit records")
            print("   2. P&L calculation error in strategy")
            print("   3. Final exit P&L not included in partial exits")
        else:
            print("✅ Total P&L matches sum of partial exits")
        
        # Verify expected TP sizes
        expected_tp1_size = closest_trade.position_size * 0.3333
        expected_tp2_size = (closest_trade.position_size - expected_tp1_size) * 0.5
        expected_tp3_size = closest_trade.position_size - expected_tp1_size - expected_tp2_size
        
        print(f"\nExpected TP Sizes:")
        print(f"  TP1: {expected_tp1_size/1e6:.2f}M (33.33% of initial)")
        print(f"  TP2: {expected_tp2_size/1e6:.2f}M (50% of remaining)")
        print(f"  TP3: {expected_tp3_size/1e6:.2f}M (100% of remaining)")
        
        # Check actual TP sizes
        actual_tp_sizes = {}
        for pe in closest_trade.partial_exits:
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else None
            if tp_level and tp_level > 0:
                if tp_level not in actual_tp_sizes:
                    actual_tp_sizes[tp_level] = 0
                actual_tp_sizes[tp_level] += pe.size
        
        print(f"\nActual TP Sizes:")
        for tp_level in sorted(actual_tp_sizes.keys()):
            print(f"  TP{tp_level}: {actual_tp_sizes[tp_level]/1e6:.2f}M")
            
        # Detailed P&L verification
        print(f"\n{'='*60}")
        print("DETAILED P&L VERIFICATION:")
        print(f"{'='*60}")
        
        # Recalculate total P&L manually
        manual_total_pnl = 0
        
        for i, pe in enumerate(closest_trade.partial_exits):
            if closest_trade.direction.value == 'short':
                price_change = closest_trade.entry_price - pe.price
            else:
                price_change = pe.price - closest_trade.entry_price
            
            pips = price_change / 0.0001
            pnl = (pe.size / 1e6) * 100 * pips
            manual_total_pnl += pnl
            
            tp_level = pe.tp_level if hasattr(pe, 'tp_level') else 'N/A'
            print(f"Exit {i+1} (TP{tp_level}): {pe.size/1e6:.2f}M @ {pips:.1f} pips = ${pnl:.2f}")
        
        print(f"\nManual Total P&L: ${manual_total_pnl:.2f}")
        print(f"Reported Total P&L: ${closest_trade.pnl:.2f}")
        print(f"Match: {'✅' if abs(manual_total_pnl - closest_trade.pnl) < 0.01 else '❌'}")
            
    else:
        print("No trade found near the specified time")

if __name__ == "__main__":
    main()