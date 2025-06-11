"""
Run a 2-year backtest on the production strategy
No plotting, just results display
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import create_strategy
from technical_indicators_custom import TIC
import time
from datetime import datetime, timedelta

def main():
    """Run a 2-year backtest without plotting"""
    
    print("Production Strategy - 2 Year Backtest")
    print("=" * 80)
    
    # Load data
    print("\nLoading AUDUSD 15M data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Get last 2 years of data
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=730)  # 2 years
    
    # Filter data
    df_2years = df[df.index >= start_date].copy()
    
    print(f"\nBacktest Period:")
    print(f"  Start: {df_2years.index[0]}")
    print(f"  End: {df_2years.index[-1]}")
    print(f"  Total Bars: {len(df_2years):,}")
    print(f"  Duration: {(df_2years.index[-1] - df_2years.index[0]).days} days")
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    start_time = time.time()
    df_2years = TIC.add_neuro_trend_intelligent(df_2years, base_fast=10, base_slow=50, confirm_bars=3)
    df_2years = TIC.add_market_bias(df_2years, ha_len=350, ha_len2=30)
    df_2years = TIC.add_intelligent_chop(df_2years)
    indicator_time = time.time() - start_time
    print(f"Indicators calculated in {indicator_time:.2f} seconds")
    
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
    
    start_time = time.time()
    results = strategy.run_backtest(df_2years)
    backtest_time = time.time() - start_time
    
    print(f"Backtest completed in {backtest_time:.2f} seconds")
    print(f"Processing speed: {len(df_2years)/backtest_time:.0f} bars/second")
    
    # Display comprehensive results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS - 2 YEAR PERIOD")
    print("=" * 80)
    
    print("\nPerformance Metrics:")
    print(f"  Total Trades:        {results['total_trades']}")
    print(f"  Winning Trades:      {results['winning_trades']}")
    print(f"  Losing Trades:       {results['losing_trades']}")
    print(f"  Win Rate:            {results['win_rate']:.2f}%")
    print(f"  Total P&L:           ${results['total_pnl']:,.2f}")
    print(f"  Total Return:        {results['total_return']:.2f}%")
    print(f"  Average Win:         ${results['avg_win']:,.2f}")
    print(f"  Average Loss:        ${results['avg_loss']:,.2f}")
    print(f"  Profit Factor:       {results['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:        {results['max_drawdown']:.2f}%")
    print(f"  Final Capital:       ${results['final_capital']:,.2f}")
    
    # Exit reasons breakdown
    print("\nExit Reasons Breakdown:")
    for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['total_trades']) * 100
        print(f"  {reason:20} {count:4} ({percentage:5.1f}%)")
    
    # Monthly breakdown
    print("\nMonthly Performance:")
    monthly_pnl = {}
    for trade in results['trades']:
        if trade.exit_time:
            month_key = trade.exit_time.strftime('%Y-%m')
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = {'pnl': 0, 'trades': 0}
            monthly_pnl[month_key]['pnl'] += trade.pnl
            monthly_pnl[month_key]['trades'] += 1
    
    print(f"  {'Month':<10} {'Trades':<8} {'P&L':<12} {'Avg/Trade':<10}")
    print("  " + "-" * 40)
    total_positive_months = 0
    for month, data in sorted(monthly_pnl.items()):
        avg_trade = data['pnl'] / data['trades']
        if data['pnl'] > 0:
            total_positive_months += 1
        print(f"  {month:<10} {data['trades']:<8} ${data['pnl']:<11,.0f} ${avg_trade:<9,.0f}")
    
    print(f"\n  Positive Months: {total_positive_months}/{len(monthly_pnl)} ({total_positive_months/len(monthly_pnl)*100:.1f}%)")
    
    # Risk metrics
    print("\nRisk Analysis:")
    print(f"  Average Trade Duration: {np.mean([(t.exit_time - t.entry_time).total_seconds()/3600 for t in results['trades'] if t.exit_time]):.1f} hours")
    print(f"  Risk/Reward Ratio: {abs(results['avg_loss'])/results['avg_win']:.2f}" if results['avg_win'] > 0 else "  Risk/Reward Ratio: N/A")
    print(f"  Win Rate Required for Breakeven: {100/(1 + results['avg_win']/abs(results['avg_loss'])):.1f}%" if results['avg_loss'] < 0 else "  Win Rate Required for Breakeven: N/A")
    
    # Position size analysis
    print("\nPosition Size Analysis:")
    position_sizes = {}
    for trade in results['trades']:
        size_m = trade.position_size / 1_000_000
        if size_m not in position_sizes:
            position_sizes[size_m] = 0
        position_sizes[size_m] += 1
    
    for size, count in sorted(position_sizes.items()):
        percentage = (count / results['total_trades']) * 100
        print(f"  {size:.0f}M positions: {count} ({percentage:.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Over the 2-year period, the strategy generated a {results['total_return']:.2f}% return")
    print(f"with a Sharpe ratio of {results['sharpe_ratio']:.2f} and maximum drawdown of {results['max_drawdown']:.2f}%.")
    print(f"The strategy maintained a {results['win_rate']:.1f}% win rate across {results['total_trades']} trades.")
    
    # Annual metrics
    days_in_period = (df_2years.index[-1] - df_2years.index[0]).days
    annual_return = results['total_return'] * (365 / days_in_period)
    annual_sharpe = results['sharpe_ratio'] * np.sqrt(252 / (days_in_period / 365))
    
    print(f"\nAnnualized Metrics:")
    print(f"  Annual Return: {annual_return:.2f}%")
    print(f"  Annual Sharpe: {annual_sharpe:.2f}")
    print(f"  Trades per Month: {results['total_trades'] / (days_in_period / 30.44):.1f}")
    
    print("\n2-year backtest completed successfully!")

if __name__ == "__main__":
    main()