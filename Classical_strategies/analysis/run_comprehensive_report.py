#!/usr/bin/env python3
"""
Comprehensive strategy report with all exit metrics
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

def print_comprehensive_report(results, config_name):
    """Print comprehensive performance report with all metrics"""
    
    print(f"\n{'='*80}")
    print(f"{config_name}")
    print(f"{'='*80}")
    
    # Basic Performance Metrics
    print("\n━━━ Performance Metrics ━━━")
    print(f"  Total Trades:         {results['total_trades']}")
    print(f"  Win Rate:             {results['win_rate']:.1f}%")
    print(f"  Sharpe Ratio:         {results['sharpe_ratio']:.3f}")
    print(f"  Total Return:         {results['total_return']:.1f}%")
    print(f"  Total P&L:            ${results['total_pnl']:,.0f}")
    print(f"  Max Drawdown:         {results['max_drawdown']:.1f}%")
    print(f"  Profit Factor:        {results['profit_factor']:.2f}")
    print(f"  Avg Win:              ${results['avg_win']:,.0f}")
    print(f"  Avg Loss:             ${results['avg_loss']:,.0f}")
    
    # Exit Pattern Analysis
    if 'exit_pattern_stats' in results:
        patterns = results['exit_pattern_stats']
        total_trades = results['total_trades']
        
        print("\n━━━ Exit Pattern Analysis ━━━")
        print(f"  Pure Stop Loss:       {patterns['pure_sl']:>4} ({patterns['pure_sl']/total_trades*100:>5.1f}%)")
        print(f"  Partial → Stop Loss:  {patterns['partial_then_sl']:>4} ({patterns['partial_then_sl']/total_trades*100:>5.1f}%)")
        print(f"  Pure Take Profit:     {patterns['pure_tp']:>4} ({patterns['pure_tp']/total_trades*100:>5.1f}%)")
        print(f"  TP → Other Exit:      {patterns['tp_then_other']:>4} ({patterns['tp_then_other']/total_trades*100:>5.1f}%)")
        print(f"  Other Exits:          {patterns['other']:>4} ({patterns['other']/total_trades*100:>5.1f}%)")
    
    # Stop Loss Outcome Analysis
    if 'sl_outcome_stats' in results:
        sl_stats = results['sl_outcome_stats']
        sl_total = sl_stats['sl_total']
        
        print("\n━━━ Stop Loss Outcome Analysis ━━━")
        print(f"  Total SL Exits:       {sl_total:>4} ({sl_total/total_trades*100:>5.1f}%)")
        
        if sl_total > 0:
            print(f"\n  Of trades hitting Stop Loss:")
            print(f"    → Resulted in LOSS:     {sl_stats['sl_loss']:>4} ({sl_stats['sl_loss']/sl_total*100:>5.1f}%)")
            print(f"    → BREAKEVEN (±$50):     {sl_stats['sl_breakeven']:>4} ({sl_stats['sl_breakeven']/sl_total*100:>5.1f}%)")
            print(f"    → Resulted in PROFIT:   {sl_stats['sl_profit']:>4} ({sl_stats['sl_profit']/sl_total*100:>5.1f}%)")
    
    # Take Profit Analysis
    if 'tp_hit_stats' in results:
        tp_stats = results['tp_hit_stats']
        
        print("\n━━━ Take Profit & Partial Exit Analysis ━━━")
        print(f"  Trades with TP1 hit:  {tp_stats['tp1_hits']:>4} ({tp_stats['tp1_hits']/total_trades*100:>5.1f}%)")
        print(f"  Trades with TP2 hit:  {tp_stats['tp2_hits']:>4} ({tp_stats['tp2_hits']/total_trades*100:>5.1f}%)")
        print(f"  Trades with TP3 hit:  {tp_stats['tp3_hits']:>4} ({tp_stats['tp3_hits']/total_trades*100:>5.1f}%)")
        print(f"  Trades with Partial:  {tp_stats['partial_exits']:>4} ({tp_stats['partial_exits']/total_trades*100:>5.1f}%)")
    
    # Key Insights
    print("\n━━━ Key Insights ━━━")
    
    # Calculate true loss rate
    if 'sl_outcome_stats' in results and 'exit_pattern_stats' in results:
        actual_losses = sl_stats['sl_loss']
        print(f"  Trades ending in actual loss:     {actual_losses:>4} ({actual_losses/total_trades*100:>5.1f}%)")
        
        profitable_sl = sl_stats['sl_profit']
        print(f"  SL exits that were profitable:    {profitable_sl:>4} ({profitable_sl/total_trades*100:>5.1f}%)")
        
        # Mixed outcome trades
        mixed_outcomes = patterns['partial_then_sl'] + patterns['tp_then_other']
        print(f"  Mixed exit patterns:              {mixed_outcomes:>4} ({mixed_outcomes/total_trades*100:>5.1f}%)")

def main():
    """Run comprehensive report with all metrics"""
    
    print("="*80)
    print("COMPREHENSIVE STRATEGY PERFORMANCE REPORT")
    print("Including Stop Loss Outcome Analysis")
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
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            realistic_costs=True,
            use_daily_sharpe=True
        )),
        ("Config 2: Scalping Strategy", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            sl_atr_multiplier=0.5,
            tp_atr_multipliers=(0.1, 0.2, 0.3),
            realistic_costs=True,
            use_daily_sharpe=True
        ))
    ]
    
    all_results = []
    
    for config_name, config in configs:
        # Create and run strategy
        strategy = OptimizedProdStrategy(config)
        print(f"\nRunning {config_name}...")
        results = strategy.run_backtest(df)
        
        # Print comprehensive report
        print_comprehensive_report(results, config_name)
        
        # Store results
        all_results.append((config_name, results))
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Config 1':>20} {'Config 2':>20}")
    print("-" * 70)
    
    for metric in ['total_trades', 'win_rate', 'sharpe_ratio', 'total_return', 'max_drawdown']:
        c1_val = all_results[0][1][metric]
        c2_val = all_results[1][1][metric]
        
        if metric in ['win_rate', 'total_return', 'max_drawdown']:
            print(f"{metric.replace('_', ' ').title():<30} {c1_val:>19.1f}% {c2_val:>19.1f}%")
        elif metric == 'sharpe_ratio':
            print(f"{metric.replace('_', ' ').title():<30} {c1_val:>20.3f} {c2_val:>20.3f}")
        else:
            print(f"{metric.replace('_', ' ').title():<30} {c1_val:>20.0f} {c2_val:>20.0f}")
    
    # Exit pattern comparison
    if 'sl_outcome_stats' in all_results[0][1]:
        print("\nStop Loss Outcomes:")
        c1_sl = all_results[0][1]['sl_outcome_stats']
        c2_sl = all_results[1][1]['sl_outcome_stats']
        
        print(f"{'SL with Loss':<30} {c1_sl['sl_loss']:>20} {c2_sl['sl_loss']:>20}")
        print(f"{'SL with Profit':<30} {c1_sl['sl_profit']:>20} {c2_sl['sl_profit']:>20}")
    
    print("\n" + "="*80)
    print("Report complete. Exit statistics are now included in all performance metrics.")

if __name__ == "__main__":
    main()