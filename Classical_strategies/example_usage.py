"""
Example Usage of Production Strategy
Demonstrates how to use the clean production trading strategy
"""

import pandas as pd
import numpy as np
from Prod_strategy import create_strategy
from Prod_plotting import plot_production_results
from technical_indicators_custom import TIC

def main():
    """Run a complete example of the production strategy"""
    
    print("Production Strategy Example")
    print("=" * 50)
    
    # Load data
    print("Loading AUDUSD 15M data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent data sample
    sample_size = 1500
    df_sample = df.tail(sample_size).copy()
    
    print(f"Using {len(df_sample)} bars from {df_sample.index[0]} to {df_sample.index[-1]}")
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    df_sample = TIC.add_neuro_trend_intelligent(df_sample, base_fast=10, base_slow=50, confirm_bars=3)
    df_sample = TIC.add_market_bias(df_sample, ha_len=350, ha_len2=30)
    df_sample = TIC.add_intelligent_chop(df_sample)
    
    # Create and run strategy
    print("\nRunning production strategy...")
    strategy = create_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        exit_on_signal_flip=True,
        intelligent_sizing=True,
        relaxed_mode=False,
        verbose=False
    )
    
    results = strategy.run_backtest(df_sample)
    
    # Display results
    print("\nStrategy Results:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  Total P&L: ${results['total_pnl']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
    
    # Create plot
    print("\nGenerating strategy chart...")
    plot_production_results(
        df=df_sample,
        results=results,
        title="Production Strategy - AUDUSD 15M",
        show_pnl=True,
        show_position_sizes=True,
        save_path="charts/production_example.png",
        show=True
    )
    
    print("\nExample completed! Chart saved to charts/production_example.png")

if __name__ == "__main__":
    main()