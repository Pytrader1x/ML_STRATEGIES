#!/usr/bin/env python3
"""
Run backtest with detailed exit statistics for Feb-March 2025
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
    
    # Try multiple paths
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
    
    # Filter date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total data points: {len(df):,}")
    
    # Calculate indicators
    print("Calculating indicators...")
    
    # Neuro Trend Intelligent
    print("  Calculating Neuro Trend Intelligent...")
    start_time = time.time()
    df = TIC.add_neuro_trend_intelligent(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed in {elapsed_time:.1f}s")
    
    # Market Bias
    print("  Calculating Market Bias...")
    start_time = time.time()
    df = TIC.add_market_bias(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed in {elapsed_time:.1f}s")
    
    # Intelligent Chop
    print("  Calculating Intelligent Chop...")
    start_time = time.time()
    df = TIC.add_intelligent_chop(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed in {elapsed_time:.1f}s")
    
    return df

def print_exit_statistics(results):
    """Print detailed exit statistics"""
    
    print("\n━━━ Exit Statistics ━━━")
    
    # Exit reasons
    if 'exit_reasons' in results:
        print("\nExit Reasons:")
        total_exits = sum(results['exit_reasons'].values())
        for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_exits * 100) if total_exits > 0 else 0
            print(f"  {reason:.<30} {count:>4} ({percentage:>5.1f}%)")
    
    # TP hit statistics
    if 'tp_hit_stats' in results:
        tp_stats = results['tp_hit_stats']
        total_trades = results['total_trades']
        
        print("\nTake Profit Statistics:")
        if total_trades > 0:
            print(f"  TP1 Hits: {tp_stats['tp1_hits']:>4} ({tp_stats['tp1_hits']/total_trades*100:>5.1f}%)")
            print(f"  TP2 Hits: {tp_stats['tp2_hits']:>4} ({tp_stats['tp2_hits']/total_trades*100:>5.1f}%)")
            print(f"  TP3 Hits: {tp_stats['tp3_hits']:>4} ({tp_stats['tp3_hits']/total_trades*100:>5.1f}%)")
            print(f"  Trades with Partial Exits: {tp_stats['partial_exits']:>4} ({tp_stats['partial_exits']/total_trades*100:>5.1f}%)")

def verify_position_sizes(trade_log_df):
    """Verify that no trade exits more than it entered"""
    
    print("\n━━━ Position Size Verification ━━━")
    
    errors = []
    for trade_id in trade_log_df['trade_id'].unique():
        trade_data = trade_log_df[trade_log_df['trade_id'] == trade_id]
        
        # Get entry size
        entry = trade_data[trade_data['action'] == 'ENTRY']
        if len(entry) == 0:
            continue
            
        entry_size = entry.iloc[0]['size']
        
        # Sum all exit sizes
        exits = trade_data[trade_data['action'].str.contains('EXIT')]
        total_exit_size = exits['size'].sum()
        
        # Check if exit size exceeds entry size
        if total_exit_size > entry_size * 1.01:  # Allow 1% tolerance for rounding
            errors.append({
                'trade_id': trade_id,
                'entry_size': entry_size,
                'total_exit_size': total_exit_size,
                'excess': total_exit_size - entry_size
            })
    
    if errors:
        print(f"⚠️  Found {len(errors)} trades with exit size errors:")
        for error in errors[:5]:  # Show first 5
            print(f"    Trade {error['trade_id']}: Entry {error['entry_size']/1e6:.1f}M, "
                  f"Exit {error['total_exit_size']/1e6:.1f}M, "
                  f"Excess {error['excess']/1e6:.3f}M")
    else:
        print("✅ All trades have valid exit sizes (no over-exiting detected)")

def main():
    """Run backtest with detailed exit statistics"""
    
    print("="*80)
    print("Running Backtest with Exit Statistics")
    print("Period: Feb-March 2025")
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
            max_tp_percent=0.003,
            realistic_costs=True,
            verbose=False,
            debug_decisions=False,
            use_daily_sharpe=True
        )),
        ("Config 2: Scalping Strategy", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            sl_atr_multiplier=0.5,
            tp_atr_multipliers=(0.1, 0.2, 0.3),
            max_tp_percent=0.002,
            realistic_costs=True,
            verbose=False,
            debug_decisions=False,
            use_daily_sharpe=True
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n{'='*80}")
        print(f"Testing {config_name}")
        print(f"{'='*80}")
        
        # Create strategy
        strategy = OptimizedProdStrategy(config)
        strategy.enable_trade_logging = True
        
        # Run backtest
        print("\nRunning backtest...")
        results = strategy.run_backtest(df)
        
        # Print standard results
        print("\n━━━ Performance Metrics ━━━")
        print(f"  Total Trades:     {results['total_trades']}")
        print(f"  Win Rate:         {results['win_rate']:.1f}%")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
        print(f"  Total Return:     {results['total_return']:.1f}%")
        print(f"  Total P&L:        ${results['total_pnl']:,.0f}")
        print(f"  Max Drawdown:     {results['max_drawdown']:.1f}%")
        print(f"  Profit Factor:    {results['profit_factor']:.2f}")
        
        # Print exit statistics
        print_exit_statistics(results)
        
        # Verify position sizes if trade log exists
        if 'trade_log' in results and results['trade_log']:
            trade_log_df = pd.DataFrame(results['trade_log'])
            verify_position_sizes(trade_log_df)
            
            # Save trade log
            filename = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_verified_trade_log.csv'
            os.makedirs('results', exist_ok=True)
            trade_log_df.to_csv(filename, index=False)
            print(f"\nTrade log saved to {filename}")
    
    print("\n" + "="*80)
    print("Backtest complete!")

if __name__ == "__main__":
    main()