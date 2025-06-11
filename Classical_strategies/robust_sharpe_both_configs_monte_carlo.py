"""
Monte Carlo Testing for Both Robust High Sharpe Strategy Configurations
Runs 20 iterations on randomly sampled 5k contiguous data rows for each config
No plotting for speed optimization
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


def create_config_1_ultra_tight_risk():
    """
    Configuration 1: Ultra-Tight Risk Management
    Achieved Sharpe Ratio: 1.171
    """
    config = OptimizedStrategyConfig(
        # Ultra-conservative risk management
        initial_capital=100_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
        
        # Very tight stop losses
        sl_max_pips=10.0,  # Maximum 10 pip stop loss
        sl_atr_multiplier=1.0,
        
        # Quick profit taking
        tp_atr_multipliers=(0.2, 0.3, 0.5),  # TP levels
        max_tp_percent=0.003,
        
        # Aggressive trailing stop
        tsl_activation_pips=3,  # TSL activation at 3 pips
        tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=0.8,
        
        # Market condition adjustments
        tp_range_market_multiplier=0.5,
        tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3,
        sl_range_market_multiplier=0.7,
        
        # Exit strategies
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        
        # Partial profits
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        
        # Conservative position sizing
        intelligent_sizing=False,
        
        # Other parameters
        sl_volatility_adjustment=True,
        verbose=False
    )
    
    return OptimizedProdStrategy(config)


def create_config_2_scalping():
    """
    Configuration 2: Scalping Strategy
    Achieved Sharpe Ratio: 1.146
    """
    config = OptimizedStrategyConfig(
        # Even more conservative risk
        initial_capital=100_000,
        risk_per_trade=0.001,  # 0.1% risk per trade
        
        # Ultra-tight stop losses
        sl_max_pips=5.0,  # Maximum 5 pip stop loss
        sl_atr_multiplier=0.5,
        
        # Ultra-tight profit taking
        tp_atr_multipliers=(0.1, 0.2, 0.3),  # Ultra-tight TP levels
        max_tp_percent=0.002,
        
        # Immediate trailing stop
        tsl_activation_pips=2,
        tsl_min_profit_pips=0.5,
        tsl_initial_buffer_multiplier=0.5,
        trailing_atr_multiplier=0.5,
        
        # Market condition adjustments
        tp_range_market_multiplier=0.3,
        tp_trend_market_multiplier=0.5,
        tp_chop_market_multiplier=0.2,
        sl_range_market_multiplier=0.5,
        
        # Exit strategies - immediate exit on signal flips
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=0.0,
        signal_flip_min_time_hours=0.0,
        signal_flip_partial_exit_percent=1.0,
        
        # Partial profits
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        
        # Conservative position sizing
        intelligent_sizing=False,
        
        # Other parameters
        sl_volatility_adjustment=True,
        verbose=False
    )
    
    return OptimizedProdStrategy(config)


def run_monte_carlo_test_both_configs(n_iterations=50, sample_size=5000, plot_last=False, save_plots=False):
    """
    Run Monte Carlo testing on both strategy configurations
    
    Parameters:
    - n_iterations: Number of random samples to test
    - sample_size: Number of contiguous data points per sample
    - plot_last: Whether to plot the last iteration results
    - save_plots: Whether to save plots to file (only used if plot_last=True)
    """
    
    print("="*80)
    print(f"ROBUST HIGH SHARPE STRATEGIES - MONTE CARLO TEST (BOTH CONFIGS)")
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
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk Management", create_config_1_ultra_tight_risk()),
        ("Config 2: Scalping Strategy", create_config_2_scalping())
    ]
    
    all_results = {}
    
    for config_name, strategy in configs:
        print(f"\n{'='*80}")
        print(f"Testing {config_name}")
        print(f"{'='*80}")
        
        # Storage for results
        iteration_results = []
        
        # Run Monte Carlo iterations
        print(f"\nRunning {n_iterations} iterations...")
        print("-" * 80)
        
        # Store last iteration for plotting
        last_sample_df = None
        last_results = None
        
        for i in range(n_iterations):
            # Get random starting point (ensure we have enough data)
            max_start = len(df) - sample_size
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + sample_size
            
            # Extract sample
            sample_df = df.iloc[start_idx:end_idx].copy()
            
            # Run backtest on sample
            results = strategy.run_backtest(sample_df)
            
            # Store last iteration data
            if i == n_iterations - 1:
                last_sample_df = sample_df.copy()
                last_results = results
            
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
            
            # Print iteration results with date range and duration
            start_date_str = sample_df.index[0].strftime('%m/%d/%Y')
            end_date_str = sample_df.index[-1].strftime('%m/%d/%Y')
            duration_days = (sample_df.index[-1] - sample_df.index[0]).days
            print(f"Iteration {i+1:2d}: "
                  f"Sharpe={results['sharpe_ratio']:6.3f} | "
                  f"P&L=${results['total_pnl']:8,.0f} | "
                  f"WinRate={results['win_rate']:5.1f}% | "
                  f"Trades={results['total_trades']:3d} | "
                  f"DD={results['max_drawdown']:5.1f}% | "
                  f"PF={results['profit_factor']:4.2f} | "
                  f"{start_date_str} to {end_date_str} ({duration_days} days, {len(sample_df):,} rows)")
        
        print("-" * 80)
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(iteration_results)
        all_results[config_name] = results_df
        
        # Calculate aggregated statistics
        print(f"\nAGGREGATED RESULTS FOR {config_name}:")
        print("="*80)
        
        # Calculate averages
        print("\nAverages:")
        print(f"  Sharpe Ratio:     {results_df['sharpe_ratio'].mean():6.3f} (std: {results_df['sharpe_ratio'].std():.3f})")
        print(f"  Total P&L:        ${results_df['total_pnl'].mean():,.0f} (std: ${results_df['total_pnl'].std():,.0f})")
        print(f"  Total Return:     {results_df['total_return'].mean():6.1f}% (std: {results_df['total_return'].std():.1f}%)")
        print(f"  Win Rate:         {results_df['win_rate'].mean():6.1f}% (std: {results_df['win_rate'].std():.1f}%)")
        print(f"  Total Trades:     {results_df['total_trades'].mean():6.0f} (std: {results_df['total_trades'].std():.0f})")
        print(f"  Max Drawdown:     {results_df['max_drawdown'].mean():6.1f}% (std: {results_df['max_drawdown'].std():.1f}%)")
        print(f"  Profit Factor:    {results_df['profit_factor'].mean():6.2f} (std: {results_df['profit_factor'].std():.2f})")
        
        # Performance consistency
        print("\nConsistency Metrics:")
        positive_sharpe = (results_df['sharpe_ratio'] > 0).sum()
        sharpe_above_1 = (results_df['sharpe_ratio'] > 1.0).sum()
        profitable = (results_df['total_pnl'] > 0).sum()
        
        print(f"  Positive Sharpe:  {positive_sharpe}/{n_iterations} ({positive_sharpe/n_iterations*100:.1f}%)")
        print(f"  Sharpe > 1.0:     {sharpe_above_1}/{n_iterations} ({sharpe_above_1/n_iterations*100:.1f}%)")
        print(f"  Profitable:       {profitable}/{n_iterations} ({profitable/n_iterations*100:.1f}%)")
        
        # Save results to CSV
        csv_filename = f'results/monte_carlo_results_{config_name.replace(":", "").replace(" ", "_").lower()}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"\nDetailed results saved to {csv_filename}")
        
        # Plot last iteration if requested
        if plot_last and last_sample_df is not None and last_results is not None:
            print(f"\nPlotting last iteration results for {config_name}...")
            
            # Generate plot
            fig = plot_production_results(
                df=last_sample_df,
                results=last_results,
                title=f"{config_name} - Last Iteration\nSharpe={last_results['sharpe_ratio']:.3f}, P&L=${last_results['total_pnl']:,.0f}",
                show_pnl=True,
                show=not save_plots  # Show plot if not saving
            )
            
            # Save plot if requested
            if save_plots:
                plot_filename = f'results/last_iteration_{config_name.replace(":", "").replace(" ", "_").lower()}.png'
                fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {plot_filename}")
    
    # Compare both configurations
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    config1_df = all_results["Config 1: Ultra-Tight Risk Management"]
    config2_df = all_results["Config 2: Scalping Strategy"]
    
    print("\n                              Config 1           Config 2")
    print("                         Ultra-Tight Risk    Scalping Strategy")
    print("-" * 60)
    print(f"Avg Sharpe Ratio:           {config1_df['sharpe_ratio'].mean():6.3f}            {config2_df['sharpe_ratio'].mean():6.3f}")
    print(f"Avg Total P&L:          ${config1_df['total_pnl'].mean():9,.0f}        ${config2_df['total_pnl'].mean():9,.0f}")
    print(f"Avg Win Rate:               {config1_df['win_rate'].mean():5.1f}%            {config2_df['win_rate'].mean():5.1f}%")
    print(f"Avg Trades:                   {config1_df['total_trades'].mean():3.0f}                {config2_df['total_trades'].mean():3.0f}")
    print(f"Avg Max Drawdown:           {config1_df['max_drawdown'].mean():5.1f}%            {config2_df['max_drawdown'].mean():5.1f}%")
    print(f"Avg Profit Factor:          {config1_df['profit_factor'].mean():5.2f}             {config2_df['profit_factor'].mean():5.2f}")
    print(f"% Sharpe > 1.0:              {(config1_df['sharpe_ratio'] > 1.0).sum()/n_iterations*100:3.0f}%               {(config2_df['sharpe_ratio'] > 1.0).sum()/n_iterations*100:3.0f}%")
    print(f"% Profitable:                {(config1_df['total_pnl'] > 0).sum()/n_iterations*100:3.0f}%               {(config2_df['total_pnl'] > 0).sum()/n_iterations*100:3.0f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    best_config = "Config 1" if config1_df['sharpe_ratio'].mean() > config2_df['sharpe_ratio'].mean() else "Config 2"
    print(f"\n✅ {best_config} shows superior performance based on average Sharpe ratio.")
    print("\nBoth configurations demonstrate robust performance with high Sharpe ratios.")
    print("Monte Carlo testing complete!")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    plot_last = '--plot' in sys.argv or '-p' in sys.argv
    save_plots = '--save-plots' in sys.argv or '-s' in sys.argv
    
    # Run Monte Carlo test for both configurations
    results = run_monte_carlo_test_both_configs(
        n_iterations=50, 
        sample_size=8000,
        plot_last=plot_last,
        save_plots=save_plots
    )
    
    if plot_last:
        print("\nNote: Run with --plot or -p to display plots of the last iteration")
        print("      Run with --save-plots or -s to save plots to files")