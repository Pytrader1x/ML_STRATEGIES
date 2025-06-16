#!/usr/bin/env python3
"""
Resample 15M data to 1H bars for better performance with the ADX strategy.
"""

import pandas as pd
import os
import sys

def resample_ohlc_data(input_file, output_file, source_timeframe='15min', target_timeframe='1h'):
    """
    Resample OHLC data from one timeframe to another.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to output CSV file
    source_timeframe : str
        Source timeframe (e.g., '15min', '5min')
    target_timeframe : str
        Target timeframe (e.g., '1h', '4h', '1d')
    """
    print(f"Reading data from {input_file}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Convert DateTime column to datetime type and set as index
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Original data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Define aggregation rules for OHLC data
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum' if 'Volume' in df.columns else None
    }
    
    # Remove None values
    agg_rules = {k: v for k, v in agg_rules.items() if v is not None and k in df.columns}
    
    print(f"\nResampling from {source_timeframe} to {target_timeframe}...")
    
    # Resample the data
    resampled_df = df.resample(target_timeframe).agg(agg_rules)
    
    # Remove any rows with NaN values
    resampled_df = resampled_df.dropna()
    
    print(f"Resampled data shape: {resampled_df.shape}")
    
    # Reset index to have DateTime as a column
    resampled_df.reset_index(inplace=True)
    
    # Save to CSV
    resampled_df.to_csv(output_file, index=False)
    print(f"\nResampled data saved to {output_file}")
    
    # Display first few rows
    print("\nFirst 5 rows of resampled data:")
    print(resampled_df.head())
    
    return resampled_df


def main():
    # Define file paths
    input_file = '../data/AUDUSD_MASTER_15M.csv'
    output_file = '../data/AUDUSD_MASTER_1H.csv'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    # Resample the data
    resampled_data = resample_ohlc_data(
        input_file=input_file,
        output_file=output_file,
        source_timeframe='15min',
        target_timeframe='1h'
    )
    
    print("\n" + "="*50)
    print("Resampling complete!")
    print("="*50)
    
    # Now run backtest on the resampled data
    print("\nRunning backtest on 1H resampled data...")
    
    from backtest import run_backtest
    from config import STRATEGY_PARAMS
    
    # Run backtest with 1H data
    # Turn off verbose logging for cleaner output
    params = STRATEGY_PARAMS.copy()
    params['printlog'] = False
    
    results = run_backtest(
        data_path=output_file,
        start_date='2015-01-01',
        end_date='2023-12-31',
        initial_cash=10000,
        commission=0.0002,  # Lower commission for forex
        plot=False,
        **params
    )
    
    if results:
        print("\n=== BACKTEST RESULTS (1H Data) ===")
        print(f"Final Portfolio Value: ${results['final_value']:.2f}")
        print(f"Total Return: {results['total_return'] * 100:.2f}%")
        if results['sharpe_ratio']:
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")


if __name__ == '__main__':
    main()