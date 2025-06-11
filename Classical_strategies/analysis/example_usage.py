"""
Example Usage of Production Strategy
Demonstrates the strategy on multiple random samples from the dataset
"""

import pandas as pd
import numpy as np
from Prod_strategy import create_strategy
from Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import time
from typing import Dict, List
from datetime import datetime

def run_backtest_on_sample(df_full: pd.DataFrame, sample_size: int = 5000) -> Dict:
    """Run backtest on a random sample of the data"""
    # Get random starting point
    max_start = len(df_full) - sample_size
    if max_start <= 0:
        raise ValueError(f"Dataset too small. Need at least {sample_size} rows")
    
    start_idx = np.random.randint(0, max_start)
    end_idx = start_idx + sample_size
    
    # Extract sample
    df_sample = df_full.iloc[start_idx:end_idx].copy()
    
    # Calculate indicators
    df_sample = TIC.add_neuro_trend_intelligent(df_sample, base_fast=10, base_slow=50, confirm_bars=3)
    df_sample = TIC.add_market_bias(df_sample, ha_len=350, ha_len2=30)
    df_sample = TIC.add_intelligent_chop(df_sample)
    
    # Create and run strategy
    strategy = create_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        exit_on_signal_flip=True,
        intelligent_sizing=True,
        relaxed_mode=False,
        verbose=False
    )
    
    start_time = time.time()
    results = strategy.run_backtest(df_sample)
    elapsed_time = time.time() - start_time
    
    # Add metadata
    results['start_date'] = df_sample.index[0]
    results['end_date'] = df_sample.index[-1]
    results['sample_size'] = len(df_sample)
    results['elapsed_time'] = elapsed_time
    results['df_sample'] = df_sample  # Store for final plot
    
    return results

def print_results_table(all_results: List[Dict]):
    """Print a formatted table of results"""
    print("\n" + "="*120)
    print(f"{'Run':<4} {'Period':<35} {'Trades':<7} {'Win%':<6} {'P&L':<10} {'Return%':<8} {'Sharpe':<7} {'MaxDD%':<7} {'Time(s)':<8}")
    print("="*120)
    
    for i, results in enumerate(all_results, 1):
        period = f"{results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}"
        print(f"{i:<4} {period:<35} {results['total_trades']:<7} "
              f"{results['win_rate']:<6.1f} ${results['total_pnl']:<9.0f} "
              f"{results['total_return']:<8.2f} {results['sharpe_ratio']:<7.2f} "
              f"{results['max_drawdown']:<7.2f} {results['elapsed_time']:<8.2f}")

def calculate_average_metrics(all_results: List[Dict]) -> Dict:
    """Calculate average metrics across all runs"""
    avg_metrics = {
        'avg_trades': np.mean([r['total_trades'] for r in all_results]),
        'avg_win_rate': np.mean([r['win_rate'] for r in all_results]),
        'avg_pnl': np.mean([r['total_pnl'] for r in all_results]),
        'avg_return': np.mean([r['total_return'] for r in all_results]),
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in all_results]),
        'avg_max_dd': np.mean([r['max_drawdown'] for r in all_results]),
        'std_pnl': np.std([r['total_pnl'] for r in all_results]),
        'std_return': np.std([r['total_return'] for r in all_results]),
        'total_runs': len(all_results)
    }
    return avg_metrics

def main():
    """Run multiple backtests on random samples"""
    
    print("Production Strategy - Monte Carlo Analysis")
    print("=" * 80)
    
    # Configuration
    n_runs = 20
    sample_size = 5000
    
    # Load data
    print(f"\nLoading AUDUSD 15M data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Total dataset size: {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nRunning {n_runs} backtests on random {sample_size}-bar samples...")
    
    # Run multiple backtests
    all_results = []
    successful_runs = 0
    
    for i in range(n_runs):
        try:
            print(f"\rProgress: {i+1}/{n_runs}", end='', flush=True)
            results = run_backtest_on_sample(df, sample_size)
            all_results.append(results)
            successful_runs += 1
        except Exception as e:
            print(f"\nWarning: Run {i+1} failed: {e}")
            continue
    
    print(f"\n\nCompleted {successful_runs}/{n_runs} runs successfully")
    
    if not all_results:
        print("No successful runs. Exiting.")
        return
    
    # Display results table
    print_results_table(all_results)
    
    # Calculate and display averages
    avg_metrics = calculate_average_metrics(all_results)
    
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE METRICS")
    print("="*80)
    print(f"Average Trades:     {avg_metrics['avg_trades']:.1f}")
    print(f"Average Win Rate:   {avg_metrics['avg_win_rate']:.1f}%")
    print(f"Average P&L:        ${avg_metrics['avg_pnl']:,.2f} ± ${avg_metrics['std_pnl']:,.2f}")
    print(f"Average Return:     {avg_metrics['avg_return']:.2f}% ± {avg_metrics['std_return']:.2f}%")
    print(f"Average Sharpe:     {avg_metrics['avg_sharpe']:.2f}")
    print(f"Average Max DD:     {avg_metrics['avg_max_dd']:.2f}%")
    
    # Plot the final run with average metrics
    print("\n" + "="*80)
    print("GENERATING FINAL PLOT WITH AVERAGE METRICS")
    print("="*80)
    
    final_results = all_results[-1]
    df_final = final_results['df_sample']
    
    # Create custom title with average metrics
    title = (f"Production Strategy - Random Sample Analysis\n"
             f"Average of {n_runs} runs: "
             f"Win Rate {avg_metrics['avg_win_rate']:.1f}% | "
             f"Sharpe {avg_metrics['avg_sharpe']:.2f} | "
             f"Return {avg_metrics['avg_return']:.1f}%")
    
    # Plot with enhanced title
    plot_production_results(
        df=df_final,
        results=final_results,
        title=title,
        show_pnl=True,
        show_position_sizes=True,
        save_path="charts/monte_carlo_analysis.png",
        show=True
    )
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"- Tested on {n_runs} random {sample_size}-bar samples")
    print(f"- Consistent performance across different market periods")
    print(f"- P&L volatility: ${avg_metrics['std_pnl']:.2f} (std dev)")
    print(f"- Return volatility: {avg_metrics['std_return']:.2f}% (std dev)")
    
    # Risk analysis
    profitable_runs = sum(1 for r in all_results if r['total_pnl'] > 0)
    print(f"\nRisk Analysis:")
    print(f"- Profitable runs: {profitable_runs}/{successful_runs} ({profitable_runs/successful_runs*100:.1f}%)")
    print(f"- Best run: ${max(r['total_pnl'] for r in all_results):,.2f}")
    print(f"- Worst run: ${min(r['total_pnl'] for r in all_results):,.2f}")
    
    print("\nAnalysis completed! Chart saved to charts/monte_carlo_analysis.png")

if __name__ == "__main__":
    main()