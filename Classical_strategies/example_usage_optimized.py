"""
Example Usage of Optimized Production Strategy
Demonstrates improvements from exit analysis insights
Includes Monte Carlo analysis on random samples
"""

import pandas as pd
import numpy as np
from Prod_strategy import create_strategy
from Prod_strategy_optimized import create_optimized_strategy
from Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import time
from datetime import timedelta
from typing import Dict, List
import sys
import argparse

def run_comparison_backtest():
    """Run both original and optimized strategies for comparison"""
    
    print("Production Strategy Comparison - Original vs Optimized")
    print("=" * 80)
    
    # Load data
    print("\nLoading AUDUSD 15M data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use last 5 years for comparison (default)
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=5*365)
    df_test = df[df.index >= start_date].copy()
    
    print(f"Test Period: {df_test.index[0]} to {df_test.index[-1]}")
    print(f"Total Bars: {len(df_test):,}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    start_time = time.time()
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    
    # Add additional columns that optimized strategy needs
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    indicator_time = time.time() - start_time
    print(f"Indicators calculated in {indicator_time:.2f}s")
    
    # Run original strategy
    print("\n" + "=" * 60)
    print("RUNNING ORIGINAL STRATEGY")
    print("=" * 60)
    
    strategy_original = create_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        exit_on_signal_flip=True,
        intelligent_sizing=True,
        relaxed_mode=False,
        verbose=False
    )
    
    start_time = time.time()
    results_original = strategy_original.run_backtest(df_test)
    original_time = time.time() - start_time
    
    print(f"Execution time: {original_time:.2f}s")
    print_results_summary("Original Strategy", results_original)
    
    # Run optimized strategy
    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZED STRATEGY")
    print("=" * 60)
    
    strategy_optimized = create_optimized_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=2.0,
        signal_flip_partial_exit_percent=0.5,
        partial_profit_before_sl=True,
        sl_volatility_adjustment=True,
        intelligent_sizing=True,
        relaxed_mode=False,
        verbose=False
    )
    
    start_time = time.time()
    results_optimized = strategy_optimized.run_backtest(df_test)
    optimized_time = time.time() - start_time
    
    print(f"Execution time: {optimized_time:.2f}s")
    print_results_summary("Optimized Strategy", results_optimized)
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    
    print(f"\nPerformance Improvement:")
    pnl_improvement = results_optimized['total_pnl'] - results_original['total_pnl']
    pnl_improvement_pct = (pnl_improvement / abs(results_original['total_pnl'])) * 100 if results_original['total_pnl'] != 0 else 0
    
    print(f"  P&L Improvement:     ${pnl_improvement:,.2f} ({pnl_improvement_pct:+.1f}%)")
    print(f"  Return Improvement:  {results_optimized['total_return'] - results_original['total_return']:+.2f}%")
    print(f"  Win Rate Change:     {results_optimized['win_rate'] - results_original['win_rate']:+.2f}%")
    print(f"  Sharpe Improvement:  {results_optimized['sharpe_ratio'] - results_original['sharpe_ratio']:+.2f}")
    print(f"  Max DD Change:       {results_optimized['max_drawdown'] - results_original['max_drawdown']:+.2f}%")
    
    # Exit reason analysis
    print("\nExit Reason Comparison:")
    print(f"{'Exit Reason':<20} {'Original':<15} {'Optimized':<15} {'Change':<15}")
    print("-" * 65)
    
    all_reasons = set(list(results_original['exit_reasons'].keys()) + 
                     list(results_optimized['exit_reasons'].keys()))
    
    for reason in sorted(all_reasons):
        orig_count = results_original['exit_reasons'].get(reason, 0)
        opt_count = results_optimized['exit_reasons'].get(reason, 0)
        change = opt_count - orig_count
        print(f"{reason:<20} {orig_count:<15} {opt_count:<15} {change:+<15}")
    
    # Signal flip analysis
    print("\nSignal Flip Analysis:")
    orig_flips = results_original['exit_reasons'].get('signal_flip', 0)
    opt_flips = results_optimized['exit_reasons'].get('signal_flip', 0)
    
    # Calculate signal flip P&L
    orig_flip_pnl = sum(t.pnl for t in results_original['trades'] 
                       if t.exit_reason and t.exit_reason.value == 'signal_flip')
    opt_flip_pnl = sum(t.pnl for t in results_optimized['trades'] 
                      if t.exit_reason and t.exit_reason.value == 'signal_flip')
    
    print(f"  Original: {orig_flips} flips, P&L: ${orig_flip_pnl:,.2f}")
    print(f"  Optimized: {opt_flips} flips, P&L: ${opt_flip_pnl:,.2f}")
    print(f"  Improvement: ${opt_flip_pnl - orig_flip_pnl:,.2f}")
    
    print("\nComparison completed!")

def print_results_summary(name: str, results: dict):
    """Print a summary of backtest results"""
    print(f"\n{name} Results:")
    print(f"  Total Trades:    {results['total_trades']}")
    print(f"  Win Rate:        {results['win_rate']:.2f}%")
    print(f"  Total P&L:       ${results['total_pnl']:,.2f}")
    print(f"  Total Return:    {results['total_return']:.2f}%")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:    {results['max_drawdown']:.2f}%")
    print(f"  Profit Factor:   {results['profit_factor']:.2f}")

def run_optimization_test():
    """Test specific optimization features"""
    
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZATION FEATURES")
    print("=" * 80)
    
    # Load a smaller sample for detailed testing
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent 3 months for detailed test
    df_test = df.tail(8640).copy()  # ~3 months of 15M data
    
    # Calculate indicators
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    
    # Test different optimization settings
    configs = [
        ("Base (No Optimizations)", {
            "signal_flip_min_profit_pips": 0,
            "signal_flip_min_time_hours": 0,
            "partial_profit_before_sl": False,
            "sl_volatility_adjustment": False
        }),
        ("Signal Flip Filter Only", {
            "signal_flip_min_profit_pips": 5.0,
            "signal_flip_min_time_hours": 2.0,
            "partial_profit_before_sl": False,
            "sl_volatility_adjustment": False
        }),
        ("Partial Exit on Flip", {
            "signal_flip_min_profit_pips": 5.0,
            "signal_flip_min_time_hours": 2.0,
            "signal_flip_partial_exit_percent": 0.5,
            "partial_profit_before_sl": False,
            "sl_volatility_adjustment": False
        }),
        ("All Optimizations", {
            "signal_flip_min_profit_pips": 5.0,
            "signal_flip_min_time_hours": 2.0,
            "signal_flip_partial_exit_percent": 0.5,
            "partial_profit_before_sl": True,
            "sl_volatility_adjustment": True
        })
    ]
    
    print(f"\nTesting on {len(df_test)} bars ({(df_test.index[-1] - df_test.index[0]).days} days)")
    print(f"{'Configuration':<25} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Sharpe':<8}")
    print("-" * 65)
    
    for config_name, settings in configs:
        strategy = create_optimized_strategy(
            initial_capital=100_000,
            **settings
        )
        results = strategy.run_backtest(df_test)
        
        print(f"{config_name:<25} {results['total_trades']:<8} "
              f"{results['win_rate']:<8.1f} ${results['total_pnl']:<11,.0f} "
              f"{results['sharpe_ratio']:<8.2f}")
    
    print("\nOptimization test completed!")

def run_monte_carlo_analysis():
    """Run Monte Carlo analysis with random samples"""
    
    print("Optimized Strategy - Monte Carlo Analysis")
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
    
    # Store results
    original_results = []
    optimized_results = []
    final_df = None
    
    # Run multiple backtests
    for i in range(n_runs):
        print(f"\rProgress: {i+1}/{n_runs}", end='', flush=True)
        
        # Get random sample
        max_start = len(df) - sample_size
        if max_start <= 0:
            raise ValueError(f"Dataset too small. Need at least {sample_size} rows")
        
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + sample_size
        df_sample = df.iloc[start_idx:end_idx].copy()
        
        # Calculate indicators
        df_sample = TIC.add_neuro_trend_intelligent(df_sample, base_fast=10, base_slow=50, confirm_bars=3)
        df_sample = TIC.add_market_bias(df_sample, ha_len=350, ha_len2=30)
        df_sample = TIC.add_intelligent_chop(df_sample)
        df_sample['IC_ATR_MA'] = df_sample['IC_ATR_Normalized'].rolling(20).mean()
        df_sample['NTI_Strength'] = abs(df_sample['NTI_Direction'].rolling(5).mean())
        
        # Run original strategy
        strategy_orig = create_strategy(
            initial_capital=100_000,
            risk_per_trade=0.02,
            exit_on_signal_flip=True,
            intelligent_sizing=True,
            verbose=False
        )
        results_orig = strategy_orig.run_backtest(df_sample)
        results_orig['start_date'] = df_sample.index[0]
        results_orig['end_date'] = df_sample.index[-1]
        original_results.append(results_orig)
        
        # Run optimized strategy
        strategy_opt = create_optimized_strategy(
            initial_capital=100_000,
            risk_per_trade=0.02,
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=2.0,
            signal_flip_partial_exit_percent=0.5,
            partial_profit_before_sl=True,
            sl_volatility_adjustment=True,
            sl_max_pips=45.0,  # Maximum 45 pip stop loss
            intelligent_sizing=True,
            verbose=False
        )
        results_opt = strategy_opt.run_backtest(df_sample)
        results_opt['start_date'] = df_sample.index[0]
        results_opt['end_date'] = df_sample.index[-1]
        optimized_results.append(results_opt)
        
        # Store last sample for plotting
        if i == n_runs - 1:
            final_df = df_sample
            final_results = results_opt
    
    print(f"\n\nCompleted {n_runs} runs successfully")
    
    # Display comparison table
    print("\n" + "="*140)
    print("MONTE CARLO RESULTS COMPARISON")
    print("="*140)
    print(f"{'Run':<4} {'Period':<35} {'Orig P&L':<12} {'Opt P&L':<12} {'Improvement':<12} {'Win%':<6} {'Sharpe':<7} {'DD%':<7}")
    print("-"*140)
    
    for i in range(n_runs):
        orig = original_results[i]
        opt = optimized_results[i]
        period = f"{orig['start_date'].strftime('%Y-%m-%d')} to {orig['end_date'].strftime('%Y-%m-%d')}"
        improvement = opt['total_pnl'] - orig['total_pnl']
        
        print(f"{i+1:<4} {period:<35} ${orig['total_pnl']:<11,.0f} ${opt['total_pnl']:<11,.0f} "
              f"${improvement:<11,.0f} {opt['win_rate']:<5.1f} {opt['sharpe_ratio']:<6.2f} {opt['max_drawdown']:<6.1f}")
    
    # Calculate averages
    avg_orig_pnl = np.mean([r['total_pnl'] for r in original_results])
    avg_opt_pnl = np.mean([r['total_pnl'] for r in optimized_results])
    avg_orig_wr = np.mean([r['win_rate'] for r in original_results])
    avg_opt_wr = np.mean([r['win_rate'] for r in optimized_results])
    avg_orig_sharpe = np.mean([r['sharpe_ratio'] for r in original_results])
    avg_opt_sharpe = np.mean([r['sharpe_ratio'] for r in optimized_results])
    avg_orig_dd = np.mean([r['max_drawdown'] for r in original_results])
    avg_opt_dd = np.mean([r['max_drawdown'] for r in optimized_results])
    avg_improvement = avg_opt_pnl - avg_orig_pnl
    avg_improvement_pct = (avg_improvement / abs(avg_orig_pnl)) * 100 if avg_orig_pnl != 0 else 0
    
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE")
    print("="*80)
    print(f"Original Strategy:")
    print(f"  Average P&L: ${avg_orig_pnl:,.2f}")
    print(f"  Average Win Rate: {avg_orig_wr:.1f}%")
    print(f"  Average Sharpe: {avg_orig_sharpe:.2f}")
    print(f"  Average Max Drawdown: {avg_orig_dd:.2f}%")
    
    print(f"\nOptimized Strategy:")
    print(f"  Average P&L: ${avg_opt_pnl:,.2f}")
    print(f"  Average Win Rate: {avg_opt_wr:.1f}%")
    print(f"  Average Sharpe: {avg_opt_sharpe:.2f}")
    print(f"  Average Max Drawdown: {avg_opt_dd:.2f}%")
    
    print(f"\nImprovement:")
    print(f"  Average P&L Improvement: ${avg_improvement:,.2f} ({avg_improvement_pct:+.1f}%)")
    print(f"  Win Rate Improvement: {avg_opt_wr - avg_orig_wr:+.1f}%")
    print(f"  Sharpe Improvement: {avg_opt_sharpe - avg_orig_sharpe:+.2f}")
    print(f"  Drawdown Improvement: {avg_opt_dd - avg_orig_dd:+.2f}% (lower is better)")
    
    # Count profitable runs
    orig_profitable = sum(1 for r in original_results if r['total_pnl'] > 0)
    opt_profitable = sum(1 for r in optimized_results if r['total_pnl'] > 0)
    
    print(f"\nProfitable Runs:")
    print(f"  Original: {orig_profitable}/{n_runs} ({orig_profitable/n_runs*100:.1f}%)")
    print(f"  Optimized: {opt_profitable}/{n_runs} ({opt_profitable/n_runs*100:.1f}%)")
    
    # Plot the final run
    print("\n" + "="*80)
    print("PLOTTING FINAL RUN")
    print("="*80)
    
    title = (f"Optimized Strategy - Monte Carlo Analysis (Final Run)\n"
             f"Average of {n_runs} runs: "
             f"Win Rate {avg_opt_wr:.1f}% | "
             f"Sharpe {avg_opt_sharpe:.2f} | "
             f"Drawdown {avg_opt_dd:.1f}% | "
             f"P&L ${avg_opt_pnl:,.0f}")
    
    plot_production_results(
        df=final_df,
        results=final_results,
        title=title,
        show_pnl=True,
        show_position_sizes=True,
        save_path="charts/optimized_monte_carlo.png",
        show=True
    )
    
    print("\nMonte Carlo analysis completed! Chart saved to charts/optimized_monte_carlo.png")

def run_n_year_backtest(n_years=5):
    """Run optimized strategy on n years of data"""
    
    print(f"Optimized Strategy - {n_years} Year Backtest")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading AUDUSD 15M data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Get n years of data
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=n_years*365)
    df_test = df[df.index >= start_date].copy()
    
    print(f"Test Period: {df_test.index[0]} to {df_test.index[-1]}")
    print(f"Total Bars: {len(df_test):,}")
    print(f"Duration: {(df_test.index[-1] - df_test.index[0]).days} days")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    start_time = time.time()
    df_test = TIC.add_neuro_trend_intelligent(df_test, base_fast=10, base_slow=50, confirm_bars=3)
    df_test = TIC.add_market_bias(df_test, ha_len=350, ha_len2=30)
    df_test = TIC.add_intelligent_chop(df_test)
    df_test['IC_ATR_MA'] = df_test['IC_ATR_Normalized'].rolling(20).mean()
    df_test['NTI_Strength'] = abs(df_test['NTI_Direction'].rolling(5).mean())
    indicator_time = time.time() - start_time
    print(f"Indicators calculated in {indicator_time:.2f}s")
    
    # Run optimized strategy
    print("\nRunning optimized strategy...")
    strategy = create_optimized_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=2.0,
        signal_flip_partial_exit_percent=0.5,
        partial_profit_before_sl=True,
        sl_volatility_adjustment=True,
        sl_max_pips=45.0,  # Maximum 45 pip stop loss
        intelligent_sizing=True,
        verbose=False
    )
    
    start_time = time.time()
    results = strategy.run_backtest(df_test)
    backtest_time = time.time() - start_time
    
    print(f"Backtest completed in {backtest_time:.2f}s")
    print(f"Processing speed: {len(df_test)/backtest_time:.0f} bars/second")
    
    # Display results
    print("\n" + "="*80)
    print(f"OPTIMIZED STRATEGY RESULTS - {n_years} YEARS")
    print("="*80)
    
    print_results_summary("Optimized Strategy", results)
    
    # Exit reason breakdown
    print("\nExit Reason Breakdown:")
    for reason, count in sorted(results['exit_reasons'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['total_trades']) * 100
        print(f"  {reason:20} {count:4} ({percentage:5.1f}%)")
    
    # Annual metrics
    days_in_period = (df_test.index[-1] - df_test.index[0]).days
    annual_return = results['total_return'] * (365 / days_in_period)
    
    print(f"\nAnnualized Metrics:")
    print(f"  Annual Return: {annual_return:.2f}%")
    print(f"  Annual Sharpe: {results['sharpe_ratio'] * np.sqrt(252 / (days_in_period / 365)):.2f}")
    print(f"  Trades per Month: {results['total_trades'] / (days_in_period / 30.44):.1f}")
    
    print(f"\n{n_years}-year backtest completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized trading strategy analysis')
    parser.add_argument('--years', type=int, help='Run n-year backtest instead of Monte Carlo')
    args = parser.parse_args()
    
    if args.years:
        # Run n-year backtest
        run_n_year_backtest(args.years)
    else:
        # Default: Run Monte Carlo analysis
        run_monte_carlo_analysis()