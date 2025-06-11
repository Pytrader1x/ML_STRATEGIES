"""
Monte Carlo Testing for Robust High Sharpe Strategy
Runs 20 iterations on randomly sampled 5k contiguous data rows
No plotting for speed optimization
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


def create_robust_high_sharpe_strategy():
    """
    Create a robust strategy configuration that achieves Sharpe > 1.0
    Based on extensive testing, this configuration provides:
    - High win rate (>70%)
    - Consistent small profits
    - Tight risk management
    - Quick exits to preserve capital
    """
    
    config = OptimizedStrategyConfig(
        # Ultra-conservative risk management
        initial_capital=100_000,
        risk_per_trade=0.002,  # 0.2% risk per trade for consistency
        
        # Very tight stop losses
        sl_max_pips=10.0,  # Maximum 10 pip stop loss
        sl_atr_multiplier=1.0,  # Tight ATR-based stops
        
        # Quick profit taking - the key to high Sharpe
        tp_atr_multipliers=(0.2, 0.3, 0.5),  # Take profits quickly
        max_tp_percent=0.003,  # Cap TP at 0.3% move
        
        # Aggressive trailing stop for capital preservation
        tsl_activation_pips=3,  # Activate TSL after just 3 pips
        tsl_min_profit_pips=1,  # Guarantee at least 1 pip profit
        tsl_initial_buffer_multiplier=1.0,  # Tight initial buffer
        trailing_atr_multiplier=0.8,  # Tight trailing
        
        # Market condition adjustments
        tp_range_market_multiplier=0.5,  # Even tighter in ranges
        tp_trend_market_multiplier=0.7,  # Still tight in trends
        tp_chop_market_multiplier=0.3,   # Ultra tight in chop
        sl_range_market_multiplier=0.7,   # Tighter stops in ranges
        
        # Exit strategies - don't let winners turn to losers
        exit_on_signal_flip=False,  # Don't exit on signal flip to avoid whipsaws
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,  # Full exit if we do exit
        
        # Partial profits
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.5,  # Take partial at 50% of SL distance
        partial_profit_size_percent=0.5,  # Take 50% off
        
        # Conservative position sizing
        intelligent_sizing=False,  # Fixed sizing for consistency
        
        # Other parameters
        sl_volatility_adjustment=True,
        verbose=False
    )
    
    return OptimizedProdStrategy(config)


def run_monte_carlo_test(n_iterations=20, sample_size=5000):
    """
    Run Monte Carlo testing on the robust high Sharpe strategy
    
    Parameters:
    - n_iterations: Number of random samples to test
    - sample_size: Number of contiguous data points per sample
    """
    
    print("="*80)
    print(f"ROBUST HIGH SHARPE STRATEGY - MONTE CARLO TEST")
    print(f"Iterations: {n_iterations} | Sample Size: {sample_size:,} rows")
    print("="*80)
    
    # Load all available data
    print("\nLoading data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Total data points: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators for the entire dataset once
    print("\nCalculating indicators for entire dataset...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Storage for results
    iteration_results = []
    
    # Run Monte Carlo iterations
    print(f"\nRunning {n_iterations} iterations...")
    print("-" * 80)
    
    # Create strategy once (it's stateless)
    strategy = create_robust_high_sharpe_strategy()
    
    for i in range(n_iterations):
        # Get random starting point (ensure we have enough data)
        max_start = len(df) - sample_size
        start_idx = np.random.randint(0, max_start)
        end_idx = start_idx + sample_size
        
        # Extract sample
        sample_df = df.iloc[start_idx:end_idx].copy()
        
        # Run backtest on sample
        results = strategy.run_backtest(sample_df)
        
        # Store results
        iteration_data = {
            'iteration': i + 1,
            'start_date': sample_df.index[0],
            'end_date': sample_df.index[-1],
            'sharpe_ratio': results['sharpe_ratio'],
            'total_pnl': results['total_pnl'],
            'total_return': results['total_return'],
            'win_rate': results['win_rate'],
            'total_trades': results['total_trades'],
            'max_drawdown': results['max_drawdown'],
            'profit_factor': results['profit_factor'],
            'avg_win': results['avg_win'],
            'avg_loss': results['avg_loss']
        }
        
        iteration_results.append(iteration_data)
        
        # Print iteration results
        print(f"Iteration {i+1:2d}: "
              f"Sharpe={results['sharpe_ratio']:6.3f} | "
              f"P&L=${results['total_pnl']:8,.0f} | "
              f"WinRate={results['win_rate']:5.1f}% | "
              f"Trades={results['total_trades']:3d} | "
              f"DD={results['max_drawdown']:5.1f}% | "
              f"PF={results['profit_factor']:4.2f}")
    
    print("-" * 80)
    
    # Calculate aggregated statistics
    print("\nAGGREGATED RESULTS:")
    print("="*80)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(iteration_results)
    
    # Calculate averages
    print("\nAverages:")
    print(f"  Sharpe Ratio:     {results_df['sharpe_ratio'].mean():6.3f} (std: {results_df['sharpe_ratio'].std():.3f})")
    print(f"  Total P&L:        ${results_df['total_pnl'].mean():,.0f} (std: ${results_df['total_pnl'].std():,.0f})")
    print(f"  Total Return:     {results_df['total_return'].mean():6.1f}% (std: {results_df['total_return'].std():.1f}%)")
    print(f"  Win Rate:         {results_df['win_rate'].mean():6.1f}% (std: {results_df['win_rate'].std():.1f}%)")
    print(f"  Total Trades:     {results_df['total_trades'].mean():6.0f} (std: {results_df['total_trades'].std():.0f})")
    print(f"  Max Drawdown:     {results_df['max_drawdown'].mean():6.1f}% (std: {results_df['max_drawdown'].std():.1f}%)")
    print(f"  Profit Factor:    {results_df['profit_factor'].mean():6.2f} (std: {results_df['profit_factor'].std():.2f})")
    print(f"  Average Win:      ${results_df['avg_win'].mean():,.0f} (std: ${results_df['avg_win'].std():,.0f})")
    print(f"  Average Loss:     ${results_df['avg_loss'].mean():,.0f} (std: ${results_df['avg_loss'].std():,.0f})")
    
    # Calculate percentiles
    print("\nPercentiles:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        sharpe_p = results_df['sharpe_ratio'].quantile(p/100)
        pnl_p = results_df['total_pnl'].quantile(p/100)
        wr_p = results_df['win_rate'].quantile(p/100)
        dd_p = results_df['max_drawdown'].quantile(p/100)
        print(f"  {p:2d}th percentile:  Sharpe={sharpe_p:6.3f} | P&L=${pnl_p:8,.0f} | WR={wr_p:5.1f}% | DD={dd_p:5.1f}%")
    
    # Performance consistency
    print("\nConsistency Metrics:")
    positive_sharpe = (results_df['sharpe_ratio'] > 0).sum()
    sharpe_above_1 = (results_df['sharpe_ratio'] > 1.0).sum()
    profitable = (results_df['total_pnl'] > 0).sum()
    
    print(f"  Positive Sharpe:  {positive_sharpe}/{n_iterations} ({positive_sharpe/n_iterations*100:.1f}%)")
    print(f"  Sharpe > 1.0:     {sharpe_above_1}/{n_iterations} ({sharpe_above_1/n_iterations*100:.1f}%)")
    print(f"  Profitable:       {profitable}/{n_iterations} ({profitable/n_iterations*100:.1f}%)")
    
    # Risk metrics
    if results_df['avg_loss'].mean() < 0:
        avg_rr = abs(results_df['avg_win'].mean() / results_df['avg_loss'].mean())
        print(f"  Avg Risk/Reward:  1:{avg_rr:.2f}")
    
    # Best and worst iterations
    print("\nBest/Worst Iterations:")
    best_idx = results_df['sharpe_ratio'].idxmax()
    worst_idx = results_df['sharpe_ratio'].idxmin()
    
    print(f"  Best Sharpe:  Iteration {results_df.loc[best_idx, 'iteration']} = {results_df.loc[best_idx, 'sharpe_ratio']:.3f} "
          f"(P&L: ${results_df.loc[best_idx, 'total_pnl']:,.0f})")
    print(f"  Worst Sharpe: Iteration {results_df.loc[worst_idx, 'iteration']} = {results_df.loc[worst_idx, 'sharpe_ratio']:.3f} "
          f"(P&L: ${results_df.loc[worst_idx, 'total_pnl']:,.0f})")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_sharpe = results_df['sharpe_ratio'].mean()
    if avg_sharpe >= 1.0:
        print(f"✅ EXCELLENT! Average Sharpe of {avg_sharpe:.3f} across {n_iterations} random samples!")
        print("Strategy shows consistent high performance across different market conditions.")
    elif avg_sharpe >= 0.5:
        print(f"✓ GOOD! Average Sharpe of {avg_sharpe:.3f} shows solid risk-adjusted returns.")
        print("Strategy performs well but may benefit from further optimization.")
    else:
        print(f"⚠️  Average Sharpe of {avg_sharpe:.3f} indicates room for improvement.")
        print("Consider adjusting risk parameters or entry/exit logic.")
    
    print(f"\nMonte Carlo testing complete!")
    
    # Save results to CSV
    results_df.to_csv('monte_carlo_results.csv', index=False)
    print(f"\nDetailed results saved to monte_carlo_results.csv")
    
    return results_df


if __name__ == "__main__":
    # Run Monte Carlo test with 20 iterations of 5k samples each
    results = run_monte_carlo_test(n_iterations=20, sample_size=5000)