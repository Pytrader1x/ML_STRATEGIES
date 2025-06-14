#!/usr/bin/env python3
"""
Analyze true exit patterns - distinguish between pure SL, mixed TP+SL, and pure TP exits
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import os
import warnings
import time

warnings.filterwarnings('ignore')

def load_and_prepare_data(currency_pair, start_date, end_date):
    """Load and prepare data for a specific currency pair and date range"""
    
    possible_paths = ['data', '../data']
    data_path = None
    for path in possible_paths:
        file_path = os.path.join(path, f'{currency_pair}_MASTER_15M.csv')
        if os.path.exists(file_path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Cannot find data for {currency_pair}")
    
    file_path = os.path.join(data_path, f'{currency_pair}_MASTER_15M.csv')
    
    print(f"Loading {currency_pair} data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total data points: {len(df):,}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    return df

def analyze_true_exit_patterns(trades):
    """Analyze the true exit patterns of trades"""
    
    exit_patterns = {
        'pure_sl': 0,           # 100% exit at stop loss
        'pure_tp': 0,           # All TPs hit (TP3 reached)
        'partial_tp_then_sl': 0, # Some TP hits, then SL
        'partial_tp_then_exit': 0, # Some TP hits, then other exit
        'partial_exit_then_sl': 0,  # Partial profit exit, then SL
        'other': 0              # Other patterns
    }
    
    detailed_patterns = []
    
    for i, trade in enumerate(trades):
        pattern_info = {
            'trade_num': i + 1,
            'direction': trade.direction.value,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'tp_hits': trade.tp_hits,
            'partial_exits': len(trade.partial_exits),
            'exit_reason': trade.exit_reason.value if trade.exit_reason else 'unknown',
            'pnl': trade.pnl,
            'pattern': ''
        }
        
        # Analyze the pattern
        if trade.tp_hits == 0 and len(trade.partial_exits) == 0:
            # No TPs or partials - pure stop loss
            if trade.exit_reason and 'stop_loss' in trade.exit_reason.value:
                exit_patterns['pure_sl'] += 1
                pattern_info['pattern'] = 'Pure Stop Loss'
            else:
                exit_patterns['other'] += 1
                pattern_info['pattern'] = f'Pure {trade.exit_reason.value}'
                
        elif trade.tp_hits >= 3:
            # All TPs hit - pure TP exit
            exit_patterns['pure_tp'] += 1
            pattern_info['pattern'] = 'Pure Take Profit (TP3)'
            
        elif trade.tp_hits > 0:
            # Some TPs hit
            if trade.exit_reason and 'stop_loss' in trade.exit_reason.value:
                exit_patterns['partial_tp_then_sl'] += 1
                pattern_info['pattern'] = f'TP{trade.tp_hits} then Stop Loss'
            else:
                exit_patterns['partial_tp_then_exit'] += 1
                pattern_info['pattern'] = f'TP{trade.tp_hits} then {trade.exit_reason.value}'
                
        elif len(trade.partial_exits) > 0:
            # Partial exits (not TP) occurred
            if trade.exit_reason and 'stop_loss' in trade.exit_reason.value:
                exit_patterns['partial_exit_then_sl'] += 1
                pattern_info['pattern'] = 'Partial Exit then Stop Loss'
            else:
                exit_patterns['other'] += 1
                pattern_info['pattern'] = f'Partial Exit then {trade.exit_reason.value}'
        else:
            exit_patterns['other'] += 1
            pattern_info['pattern'] = 'Other'
            
        detailed_patterns.append(pattern_info)
    
    return exit_patterns, detailed_patterns

def print_exit_pattern_analysis(exit_patterns, total_trades, config_name):
    """Print detailed exit pattern analysis"""
    
    print(f"\n{'='*80}")
    print(f"{config_name} - TRUE EXIT PATTERN ANALYSIS")
    print(f"{'='*80}")
    
    print("\n━━━ Exit Pattern Breakdown ━━━")
    print(f"Total Trades: {total_trades}")
    print("")
    
    # Pure exits
    pure_sl_pct = (exit_patterns['pure_sl'] / total_trades * 100) if total_trades > 0 else 0
    pure_tp_pct = (exit_patterns['pure_tp'] / total_trades * 100) if total_trades > 0 else 0
    
    print(f"PURE EXITS:")
    print(f"  Pure Stop Loss (100% at SL):         {exit_patterns['pure_sl']:>4} ({pure_sl_pct:>5.1f}%)")
    print(f"  Pure Take Profit (reached TP3):      {exit_patterns['pure_tp']:>4} ({pure_tp_pct:>5.1f}%)")
    
    # Mixed exits
    mixed_tp_sl_pct = (exit_patterns['partial_tp_then_sl'] / total_trades * 100) if total_trades > 0 else 0
    mixed_partial_sl_pct = (exit_patterns['partial_exit_then_sl'] / total_trades * 100) if total_trades > 0 else 0
    partial_tp_exit_pct = (exit_patterns['partial_tp_then_exit'] / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\nMIXED EXITS:")
    print(f"  Partial TP then Stop Loss:            {exit_patterns['partial_tp_then_sl']:>4} ({mixed_tp_sl_pct:>5.1f}%)")
    print(f"  Partial Exit then Stop Loss:          {exit_patterns['partial_exit_then_sl']:>4} ({mixed_partial_sl_pct:>5.1f}%)")
    print(f"  Partial TP then Other Exit:           {exit_patterns['partial_tp_then_exit']:>4} ({partial_tp_exit_pct:>5.1f}%)")
    
    # Other
    other_pct = (exit_patterns['other'] / total_trades * 100) if total_trades > 0 else 0
    print(f"\nOTHER:")
    print(f"  Other patterns:                       {exit_patterns['other']:>4} ({other_pct:>5.1f}%)")
    
    # Summary stats
    total_sl_involved = exit_patterns['pure_sl'] + exit_patterns['partial_tp_then_sl'] + exit_patterns['partial_exit_then_sl']
    total_sl_pct = (total_sl_involved / total_trades * 100) if total_trades > 0 else 0
    
    total_profitable_exits = exit_patterns['pure_tp'] + exit_patterns['partial_tp_then_sl'] + exit_patterns['partial_tp_then_exit']
    profitable_exit_pct = (total_profitable_exits / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n━━━ Summary Statistics ━━━")
    print(f"Trades involving Stop Loss:             {total_sl_involved:>4} ({total_sl_pct:>5.1f}%)")
    print(f"Trades with at least one TP hit:        {total_profitable_exits:>4} ({profitable_exit_pct:>5.1f}%)")

def main():
    """Run true exit pattern analysis"""
    
    print("="*80)
    print("TRUE EXIT PATTERN ANALYSIS")
    print("Distinguishing between pure SL, mixed TP+SL, and pure TP exits")
    print("="*80)
    
    # Load data
    currency = 'AUDUSD'
    start_date = '2025-02-01'
    end_date = '2025-03-31'
    
    df = load_and_prepare_data(currency, start_date, end_date)
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk Management", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            realistic_costs=True,
            use_daily_sharpe=True
        )),
        ("Config 2: Scalping Strategy", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            realistic_costs=True,
            use_daily_sharpe=True
        ))
    ]
    
    for config_name, config in configs:
        # Create and run strategy
        strategy = OptimizedProdStrategy(config)
        print(f"\nRunning {config_name}...")
        results = strategy.run_backtest(df)
        
        # Analyze exit patterns
        if 'trades' in results and results['trades']:
            exit_patterns, detailed_patterns = analyze_true_exit_patterns(results['trades'])
            print_exit_pattern_analysis(exit_patterns, results['total_trades'], config_name)
            
            # Show sample trades with mixed exits
            print(f"\n━━━ Sample Mixed Exit Trades ━━━")
            mixed_trades = [p for p in detailed_patterns if 'then Stop Loss' in p['pattern']][:5]
            
            for trade in mixed_trades:
                print(f"\nTrade #{trade['trade_num']}:")
                print(f"  Pattern: {trade['pattern']}")
                print(f"  Direction: {trade['direction'].upper()} @ {trade['entry_price']:.5f}")
                print(f"  TP Hits: {trade['tp_hits']}, Partial Exits: {trade['partial_exits']}")
                print(f"  Final Exit: {trade['exit_reason']} @ {trade['exit_price']:.5f}")
                print(f"  P&L: ${trade['pnl']:,.2f}")

if __name__ == "__main__":
    main()