#!/usr/bin/env python3
"""
Main entry point for ADX Trend Strategy.

This script provides a command-line interface to run backtests,
optimizations, and generate reports for the ADX trend-following strategy.
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import run_backtest, optimize_strategy
from utils import download_data, generate_performance_report
from config import get_strategy_config, validate_config


def download_historical_data(config):
    """Download historical data for specified symbols."""
    print("Downloading historical data...")
    
    data_config = config['data']
    os.makedirs(data_config['save_path'], exist_ok=True)
    
    for symbol in data_config['symbols']:
        print(f"Downloading {symbol}...")
        try:
            data = download_data(
                symbol=symbol,
                start_date='2020-01-01',
                end_date='2024-01-01',
                interval=data_config['interval']
            )
            
            filename = f"{data_config['save_path']}{symbol}_{data_config['interval']}.csv"
            data.to_csv(filename)
            print(f"  Saved to {filename}")
        except Exception as e:
            print(f"  Error downloading {symbol}: {e}")


def run_single_backtest(config, data_path=None, plot=True):
    """Run a single backtest with current configuration."""
    print("\n=== RUNNING BACKTEST ===")
    
    strategy_config = config['strategy']
    backtest_config = config['backtest']
    
    # Run backtest
    results = run_backtest(
        data_path=data_path,
        start_date=backtest_config['start_date'],
        end_date=backtest_config['end_date'],
        initial_cash=backtest_config['initial_cash'],
        commission=backtest_config['commission'],
        plot=plot,
        **strategy_config
    )
    
    # Load trade history if available
    trade_history_path = 'ADX_Strategy/trade_history.csv'
    if os.path.exists(trade_history_path):
        trades_df = pd.read_csv(trade_history_path)
        
        # Generate performance report
        if config['report']['generate_html']:
            generate_performance_report(results, trades_df)
    
    return results


def run_optimization(config, data_path=None):
    """Run parameter optimization."""
    print("\n=== RUNNING OPTIMIZATION ===")
    
    backtest_config = config['backtest']
    
    optimization_results = optimize_strategy(
        data_path=data_path,
        start_date=backtest_config['start_date'],
        end_date=backtest_config['end_date'],
        initial_cash=backtest_config['initial_cash'],
        commission=backtest_config['commission']
    )
    
    return optimization_results


def run_walk_forward_analysis(config, data_path=None, window_size=365, step_size=90):
    """Run walk-forward analysis."""
    print("\n=== RUNNING WALK-FORWARD ANALYSIS ===")
    
    # Load data to get date range
    if data_path and os.path.exists(data_path):
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Download data if not provided
        symbol = config['backtest']['symbol']
        data = download_data(symbol, '2020-01-01', '2024-01-01', '1h')
    
    results = []
    
    # Implement walk-forward logic here
    # This is a placeholder for the actual implementation
    print("Walk-forward analysis not yet fully implemented")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='ADX Trend Strategy - Backtest and Optimization Tool'
    )
    
    parser.add_argument(
        'mode',
        choices=['backtest', 'optimize', 'download', 'walkforward'],
        help='Mode to run the strategy in'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV file with OHLCV data'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Symbol to trade (default: SPY)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--cash',
        type=float,
        help='Initial cash amount'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_strategy_config()
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.start:
        config['backtest']['start_date'] = args.start
    if args.end:
        config['backtest']['end_date'] = args.end
    if args.cash:
        config['backtest']['initial_cash'] = args.cash
    if args.symbol:
        config['backtest']['symbol'] = args.symbol
    
    # Execute based on mode
    if args.mode == 'download':
        download_historical_data(config)
        
    elif args.mode == 'backtest':
        results = run_single_backtest(
            config,
            data_path=args.data,
            plot=not args.no_plot
        )
        
        # Check performance thresholds
        perf_thresholds = config['performance']
        
        print("\n=== PERFORMANCE CHECK ===")
        if results['sharpe_ratio'] is None:
            print("⚠️  Sharpe ratio: N/A (no trades executed)")
        elif results['sharpe_ratio'] < perf_thresholds['min_sharpe_ratio']:
            print(f"⚠️  Sharpe ratio ({results['sharpe_ratio']:.2f}) below threshold ({perf_thresholds['min_sharpe_ratio']})")
        else:
            print(f"✅ Sharpe ratio: {results['sharpe_ratio']:.2f}")
            
        if abs(results['max_drawdown']) > perf_thresholds['max_drawdown']:
            print(f"⚠️  Max drawdown ({abs(results['max_drawdown']):.2f}%) exceeds threshold ({perf_thresholds['max_drawdown']}%)")
        else:
            print(f"✅ Max drawdown: {abs(results['max_drawdown']):.2f}%")
            
    elif args.mode == 'optimize':
        optimization_results = run_optimization(config, data_path=args.data)
        
        # Display best parameters
        if not optimization_results.empty:
            best_params = optimization_results.iloc[0]
            print("\n=== BEST PARAMETERS ===")
            print(f"ADX Threshold: {best_params['adx_threshold']}")
            print(f"Williams Period: {best_params['williams_period']}")
            print(f"SMA Period: {best_params['sma_period']}")
            print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.3f}")
            print(f"Total Return: {best_params['total_return'] * 100:.2f}%")
            
    elif args.mode == 'walkforward':
        results = run_walk_forward_analysis(config, data_path=args.data)
    
    print("\nDone!")


if __name__ == '__main__':
    main()