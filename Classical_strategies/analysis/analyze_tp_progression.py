#!/usr/bin/env python3
"""
Analyze the true TP progression - how many trades hit TP1, TP2, TP3
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import os

def analyze_tp_progression():
    """Analyze how TPs are actually hit in sequence"""
    
    # Load data
    possible_paths = ['data', '../data']
    data_path = None
    for path in possible_paths:
        file_path = os.path.join(path, 'AUDUSD_MASTER_15M.csv')
        if os.path.exists(file_path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("Cannot find AUDUSD data")
    
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    
    print("Loading AUDUSD data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Filter to Feb-March 2025
    df = df[(df.index >= '2025-02-01') & (df.index <= '2025-03-31')]
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total data points: {len(df):,}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk Management", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            realistic_costs=True
        )),
        ("Config 2: Scalping Strategy", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            realistic_costs=True
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n{'='*80}")
        print(f"{config_name}")
        print(f"{'='*80}")
        
        # Run strategy
        strategy = OptimizedProdStrategy(config)
        results = strategy.run_backtest(df)
        
        # Analyze TP progression
        tp_analysis = {
            'never_hit_tp': 0,
            'hit_tp1_only': 0,
            'hit_tp2_but_not_tp3': 0,
            'hit_all_tps': 0,
            'tp1_final_exit': 0,
            'tp2_final_exit': 0,
            'tp3_final_exit': 0
        }
        
        # Categorize each trade
        for trade in results['trades']:
            # Check partial exits to understand TP progression
            tp_levels_hit = set()
            for pe in trade.partial_exits:
                if hasattr(pe, 'tp_level') and pe.tp_level > 0:
                    tp_levels_hit.add(pe.tp_level)
            
            # Count based on highest TP hit
            max_tp = max(tp_levels_hit) if tp_levels_hit else 0
            
            if max_tp == 0:
                tp_analysis['never_hit_tp'] += 1
            elif max_tp == 1:
                tp_analysis['hit_tp1_only'] += 1
            elif max_tp == 2:
                tp_analysis['hit_tp2_but_not_tp3'] += 1
            elif max_tp >= 3:
                tp_analysis['hit_all_tps'] += 1
            
            # Check if final exit was at a TP
            if trade.exit_reason and 'take_profit' in trade.exit_reason.value:
                tp_num = int(trade.exit_reason.value.split('_')[-1])
                if tp_num == 1:
                    tp_analysis['tp1_final_exit'] += 1
                elif tp_num == 2:
                    tp_analysis['tp2_final_exit'] += 1
                elif tp_num == 3:
                    tp_analysis['tp3_final_exit'] += 1
        
        # Print analysis
        total_trades = len(results['trades'])
        print(f"\nTotal Trades: {total_trades}")
        
        print("\n📊 TP PROGRESSION ANALYSIS:")
        print(f"Never hit any TP:        {tp_analysis['never_hit_tp']:>4} ({tp_analysis['never_hit_tp']/total_trades*100:>5.1f}%)")
        print(f"Hit TP1 only:           {tp_analysis['hit_tp1_only']:>4} ({tp_analysis['hit_tp1_only']/total_trades*100:>5.1f}%)")
        print(f"Hit TP1+TP2 (not TP3):  {tp_analysis['hit_tp2_but_not_tp3']:>4} ({tp_analysis['hit_tp2_but_not_tp3']/total_trades*100:>5.1f}%)")
        print(f"Hit all 3 TPs:          {tp_analysis['hit_all_tps']:>4} ({tp_analysis['hit_all_tps']/total_trades*100:>5.1f}%)")
        
        print("\n🎯 FINAL EXIT AT TP:")
        print(f"Final exit at TP1:      {tp_analysis['tp1_final_exit']:>4} ({tp_analysis['tp1_final_exit']/total_trades*100:>5.1f}%)")
        print(f"Final exit at TP2:      {tp_analysis['tp2_final_exit']:>4} ({tp_analysis['tp2_final_exit']/total_trades*100:>5.1f}%)")
        print(f"Final exit at TP3:      {tp_analysis['tp3_final_exit']:>4} ({tp_analysis['tp3_final_exit']/total_trades*100:>5.1f}%)")
        
        # Check the confusing pattern
        print("\n🔍 INVESTIGATING THE ISSUE:")
        confusing_trades = []
        for trade in results['trades']:
            if trade.exit_reason and trade.exit_reason.value == 'take_profit_1':
                confusing_trades.append({
                    'tp_hits': trade.tp_hits,
                    'partial_exits': len(trade.partial_exits),
                    'position_size': trade.position_size,
                    'remaining_size': trade.remaining_size,
                    'pnl': trade.pnl
                })
        
        print(f"\nTrades with exit_reason='take_profit_1': {len(confusing_trades)}")
        if confusing_trades:
            print("Sample of these trades:")
            for i, t in enumerate(confusing_trades[:5]):
                print(f"  Trade {i+1}: tp_hits={t['tp_hits']}, partial_exits={t['partial_exits']}, "
                      f"position={t['position_size']/1e6:.0f}M, remaining={t['remaining_size']/1e6:.1f}M")
        
        print("\n💡 EXPLANATION:")
        print("The issue is that 'exit_reason=take_profit_1' with 'tp_hits=0' means:")
        print("1. The trade hit TP1 and exited COMPLETELY (no partials)")
        print("2. This happens when the remaining position is small")
        print("3. The tp_hits counter only increments for PARTIAL exits")
        print("4. When TP causes FULL exit, tp_hits stays at 0")

if __name__ == "__main__":
    analyze_tp_progression()