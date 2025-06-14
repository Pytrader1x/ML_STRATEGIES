#!/usr/bin/env python3
"""
Run backtest with detailed trade logging for Feb-March 2025
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

def save_trade_log(trade_log, filename):
    """Save detailed trade log to CSV"""
    if not trade_log:
        print("No trades to log")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trade_log)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Reorder columns for better readability
    column_order = [
        'timestamp', 'action', 'direction', 'entry_price', 'current_price',
        'size', 'remaining_size', 'reason', 'sl_level', 'tp1_level', 'tp2_level', 'tp3_level',
        'pips', 'pnl', 'cumulative_pnl', 'confidence', 'is_relaxed',
        'nti_direction', 'mb_bias', 'ic_regime', 'ic_regime_name', 'atr', 'trade_id'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Trade log saved to {filename}")
    
    # Print summary
    print("\nTrade Log Summary:")
    print(f"Total actions logged: {len(df)}")
    print(f"Entries: {len(df[df['action'] == 'ENTRY'])}")
    print(f"Exits: {len(df[df['action'].str.contains('EXIT')])}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

def main():
    """Run backtest with detailed logging"""
    
    print("="*80)
    print("Running Backtest with Detailed Trade Logging")
    print("Period: Feb-March 2025")
    print("="*80)
    
    # Load data
    currency = 'AUDUSD'
    start_date = '2025-02-01'
    end_date = '2025-03-31'
    
    df = load_and_prepare_data(currency, start_date, end_date)
    
    # Create strategy config (Config 1: Ultra-Tight Risk Management)
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
        sl_max_pips=10.0,
        sl_atr_multiplier=1.0,
        tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003,
        realistic_costs=True,  # Enable realistic slippage
        verbose=False,
        debug_decisions=False,
        use_daily_sharpe=True
    )
    
    # Create strategy
    strategy = OptimizedProdStrategy(config)
    strategy.enable_trade_logging = True
    
    # Run backtest
    print("\nRunning backtest...")
    results = strategy.run_backtest(df)
    
    # Print results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Total Return: {results['total_return']:.1f}%")
    print(f"Total P&L: ${results['total_pnl']:,.0f}")
    print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    
    # Save trade log
    if 'trade_log' in results and results['trade_log']:
        filename = f'results/{currency}_detailed_trade_log_feb_mar_2025.csv'
        os.makedirs('results', exist_ok=True)
        save_trade_log(results['trade_log'], filename)
    else:
        print("\nNo trade log found in results")
    
    print("\nBacktest complete!")

if __name__ == "__main__":
    main()