"""
Unified Strategy Runner - Monte Carlo Testing Framework
Supports multiple modes: single currency, multi-currency, with various analysis options
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

warnings.filterwarnings('ignore')

__version__ = "2.0.0"


def create_config_1_ultra_tight_risk():
    """
    Configuration 1: Ultra-Tight Risk Management
    Achieved Sharpe Ratio: 1.171 on AUDUSD
    """
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
        sl_max_pips=10.0,
        sl_atr_multiplier=1.0,
        tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003,
        tsl_activation_pips=3,
        tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=0.8,
        tp_range_market_multiplier=0.5,
        tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3,
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )
    return OptimizedProdStrategy(config)


def create_config_2_scalping():
    """
    Configuration 2: Scalping Strategy
    Achieved Sharpe Ratio: 1.146 on AUDUSD
    """
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.001,  # 0.1% risk per trade
        sl_max_pips=5.0,
        sl_atr_multiplier=0.5,
        tp_atr_multipliers=(0.1, 0.2, 0.3),
        max_tp_percent=0.002,
        tsl_activation_pips=2,
        tsl_min_profit_pips=0.5,
        tsl_initial_buffer_multiplier=0.5,
        trailing_atr_multiplier=0.5,
        tp_range_market_multiplier=0.3,
        tp_trend_market_multiplier=0.5,
        tp_chop_market_multiplier=0.2,
        sl_range_market_multiplier=0.5,
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=0.0,
        signal_flip_min_time_hours=0.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.3,
        partial_profit_size_percent=0.7,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )
    return OptimizedProdStrategy(config)


def load_and_prepare_data(currency_pair, data_path='../data'):
    """Load and prepare data for a specific currency pair"""
    file_path = os.path.join(data_path, f'{currency_pair}_MASTER_15M.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading {currency_pair} data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Total data points: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    return df


def run_single_monte_carlo(df, strategy, n_iterations, sample_size):
    """Run Monte Carlo simulation for a single strategy configuration"""
    iteration_results = []
    
    for i in range(n_iterations):
        # Get random starting point
        max_start = len(df) - sample_size
        if max_start < 0:
            raise ValueError(f"Insufficient data: need {sample_size} rows, have {len(df)}")
        
        start_idx = np.random.randint(0, max_start)
        sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
        
        # Run backtest
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
        
        # Print progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations...")
    
    return pd.DataFrame(iteration_results)


def generate_calendar_year_analysis(results_df, config_name, currency=None, show_plots=False, save_plots=True):
    """Generate calendar year analysis and visualizations"""
    # Add year columns
    results_df['primary_year'] = results_df.apply(
        lambda row: row['start_date'].year if (row['end_date'] - row['start_date']).days <= 365 
        else row['start_date'].year if row['start_date'].month <= 6 
        else row['end_date'].year, axis=1
    )
    
    # Calculate year-by-year statistics
    yearly_stats = results_df.groupby('primary_year').agg({
        'sharpe_ratio': ['mean', 'std', 'count'],
        'total_return': ['mean', 'std'],
        'win_rate': ['mean', 'std'],
        'total_trades': ['mean', 'sum'],
        'max_drawdown': ['mean', 'std'],
        'profit_factor': ['mean', 'std'],
        'total_pnl': ['mean', 'sum']
    }).round(2)
    
    print(f"\nCalendar Year Analysis for {config_name}:")
    print("-" * 100)
    print(f"{'Year':<6} {'Count':<6} {'Avg Sharpe':<12} {'Avg Return%':<12} {'Avg WinRate%':<13} "
          f"{'Avg MaxDD%':<12} {'Avg PF':<8}")
    print("-" * 100)
    
    for year in sorted(yearly_stats.index):
        count = int(yearly_stats.loc[year, ('sharpe_ratio', 'count')])
        avg_sharpe = yearly_stats.loc[year, ('sharpe_ratio', 'mean')]
        avg_return = yearly_stats.loc[year, ('total_return', 'mean')]
        avg_winrate = yearly_stats.loc[year, ('win_rate', 'mean')]
        avg_dd = yearly_stats.loc[year, ('max_drawdown', 'mean')]
        avg_pf = yearly_stats.loc[year, ('profit_factor', 'mean')]
        
        print(f"{year:<6} {count:<6} {avg_sharpe:<12.3f} {avg_return:<12.1f} {avg_winrate:<13.1f} "
              f"{avg_dd:<12.1f} {avg_pf:<8.2f}")
    
    # Generate calendar year visualization if we have enough data
    if len(yearly_stats) > 2 and currency:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{currency} - {config_name} Calendar Year Analysis', fontsize=14)
        
        # Plot 1: Sharpe Ratio by Year
        years = sorted(yearly_stats.index)
        sharpe_means = [yearly_stats.loc[year, ('sharpe_ratio', 'mean')] for year in years]
        sharpe_stds = [yearly_stats.loc[year, ('sharpe_ratio', 'std')] for year in years]
        
        ax1.bar(years, sharpe_means, yerr=sharpe_stds, capsize=5, alpha=0.7)
        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Average Sharpe Ratio')
        ax1.set_title('Sharpe Ratio by Year (with std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win Rate and Return by Year
        ax2_twin = ax2.twinx()
        
        win_rates = [yearly_stats.loc[year, ('win_rate', 'mean')] for year in years]
        returns = [yearly_stats.loc[year, ('total_return', 'mean')] for year in years]
        
        line1 = ax2.plot(years, win_rates, 'b-o', label='Win Rate %', linewidth=2)
        line2 = ax2_twin.plot(years, returns, 'r-s', label='Avg Return %', linewidth=2)
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Win Rate (%)', color='b')
        ax2_twin.set_ylabel('Average Return (%)', color='r')
        ax2.set_title('Win Rate and Returns by Year')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='best')
        
        plt.tight_layout()
        
        # Save if requested
        if save_plots:
            plot_filename = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_calendar_year.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"Calendar year plot saved to {plot_filename}")
        
        # Show if requested
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return yearly_stats


def run_single_currency_mode(currency='AUDUSD', n_iterations=50, sample_size=8000, 
                           enable_plots=True, enable_calendar_analysis=True,
                           show_plots=False, save_plots=True):
    """Run Monte Carlo testing on a single currency pair with both configurations"""
    
    print("="*80)
    print(f"SINGLE CURRENCY MODE - {currency}")
    print(f"Iterations: {n_iterations} | Sample Size: {sample_size:,} rows")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(currency)
    
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
        
        # Run Monte Carlo
        results_df = run_single_monte_carlo(df, strategy, n_iterations, sample_size)
        all_results[config_name] = results_df
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Average Sharpe Ratio:  {results_df['sharpe_ratio'].mean():.3f} (std: {results_df['sharpe_ratio'].std():.3f})")
        print(f"  Average Total Return:  {results_df['total_return'].mean():.1f}% (std: {results_df['total_return'].std():.1f}%)")
        print(f"  Average Win Rate:      {results_df['win_rate'].mean():.1f}% (std: {results_df['win_rate'].std():.1f}%)")
        print(f"  Average Max Drawdown:  {results_df['max_drawdown'].mean():.1f}% (std: {results_df['max_drawdown'].std():.1f}%)")
        print(f"  % Sharpe > 1.0:        {(results_df['sharpe_ratio'] > 1.0).sum()/n_iterations*100:.1f}%")
        print(f"  % Profitable:          {(results_df['total_pnl'] > 0).sum()/n_iterations*100:.1f}%")
        
        # Calendar year analysis
        if enable_calendar_analysis and len(results_df) > 0:
            generate_calendar_year_analysis(results_df, config_name, currency, show_plots, save_plots)
        
        # Save results
        csv_filename = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_monte_carlo.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
    
    # Generate comparison visualizations if enabled
    if enable_plots:
        generate_comparison_plots(all_results, currency, show_plots, save_plots)
    
    return all_results


def run_multi_currency_mode(currencies=['GBPUSD', 'EURUSD', 'USDJPY', 'NZDUSD', 'USDCAD'],
                          n_iterations=30, sample_size=8000):
    """Run Monte Carlo testing on multiple currency pairs"""
    
    print("="*80)
    print("MULTI-CURRENCY MODE")
    print(f"Testing {len(currencies)} currency pairs")
    print(f"Iterations per pair: {n_iterations} | Sample Size: {sample_size:,} rows")
    print("="*80)
    
    all_results = {}
    
    for currency in currencies:
        try:
            # Load data
            df = load_and_prepare_data(currency)
            
            # Test both configurations
            configs = [
                ("Config 1: Ultra-Tight Risk", create_config_1_ultra_tight_risk()),
                ("Config 2: Scalping", create_config_2_scalping())
            ]
            
            currency_results = {}
            
            for config_name, strategy in configs:
                print(f"\n{currency} - {config_name}")
                
                # Run Monte Carlo
                results_df = run_single_monte_carlo(df, strategy, n_iterations, sample_size)
                
                # Store summary stats
                currency_results[config_name] = {
                    'avg_sharpe': results_df['sharpe_ratio'].mean(),
                    'avg_pnl': results_df['total_pnl'].mean(),
                    'avg_win_rate': results_df['win_rate'].mean(),
                    'avg_drawdown': results_df['max_drawdown'].mean(),
                    'avg_profit_factor': results_df['profit_factor'].mean(),
                    'sharpe_above_1_pct': (results_df['sharpe_ratio'] > 1.0).sum() / n_iterations * 100,
                    'full_results': results_df
                }
                
                print(f"  Average Sharpe: {currency_results[config_name]['avg_sharpe']:.3f}")
                print(f"  % Sharpe > 1.0: {currency_results[config_name]['sharpe_above_1_pct']:.1f}%")
            
            all_results[currency] = currency_results
            
        except Exception as e:
            print(f"Error processing {currency}: {str(e)}")
            continue
    
    # Generate summary report
    generate_multi_currency_summary(all_results, currencies)
    
    # Save consolidated results
    save_multi_currency_results(all_results)
    
    return all_results


def generate_comparison_plots(all_results, currency, show_plots=False, save_plots=True):
    """Generate comparison plots for single currency mode"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{currency} - Monte Carlo Analysis Comparison', fontsize=16)
    
    config1_df = all_results["Config 1: Ultra-Tight Risk Management"]
    config2_df = all_results["Config 2: Scalping Strategy"]
    
    # Plot 1: Sharpe Ratio Distribution
    ax1 = axes[0, 0]
    ax1.hist(config1_df['sharpe_ratio'], bins=20, alpha=0.5, label='Config 1', color='blue')
    ax1.hist(config2_df['sharpe_ratio'], bins=20, alpha=0.5, label='Config 2', color='red')
    ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Sharpe Ratio Distribution')
    ax1.legend()
    
    # Plot 2: Returns Distribution
    ax2 = axes[0, 1]
    ax2.hist(config1_df['total_return'], bins=20, alpha=0.5, label='Config 1', color='blue')
    ax2.hist(config2_df['total_return'], bins=20, alpha=0.5, label='Config 2', color='red')
    ax2.set_xlabel('Total Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Returns Distribution')
    ax2.legend()
    
    # Plot 3: Win Rate vs Sharpe Scatter
    ax3 = axes[1, 0]
    ax3.scatter(config1_df['win_rate'], config1_df['sharpe_ratio'], alpha=0.5, label='Config 1', color='blue')
    ax3.scatter(config2_df['win_rate'], config2_df['sharpe_ratio'], alpha=0.5, label='Config 2', color='red')
    ax3.set_xlabel('Win Rate (%)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Win Rate vs Sharpe Ratio')
    ax3.legend()
    
    # Plot 4: Performance Metrics Comparison
    ax4 = axes[1, 1]
    metrics = ['Sharpe', 'Win Rate', 'Profit Factor']
    config1_values = [
        config1_df['sharpe_ratio'].mean(),
        config1_df['win_rate'].mean() / 100,  # Scale to match other metrics
        config1_df['profit_factor'].mean() / 3  # Scale for visibility
    ]
    config2_values = [
        config2_df['sharpe_ratio'].mean(),
        config2_df['win_rate'].mean() / 100,
        config2_df['profit_factor'].mean() / 3
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, config1_values, width, label='Config 1', alpha=0.8, color='blue')
    ax4.bar(x + width/2, config2_values, width, label='Config 2', alpha=0.8, color='red')
    ax4.set_ylabel('Normalized Values')
    ax4.set_title('Average Performance Metrics')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots:
        plot_filename = f'results/{currency}_monte_carlo_comparison.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to {plot_filename}")
    
    # Show plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close()


def generate_multi_currency_summary(all_results, currencies):
    """Generate summary report for multi-currency mode"""
    print("\n" + "="*80)
    print("MULTI-CURRENCY SUMMARY REPORT")
    print("="*80)
    
    # Config 1 Summary
    print("\nConfig 1: Ultra-Tight Risk Management")
    print("-" * 60)
    print(f"{'Currency':<10} {'Avg Sharpe':>12} {'Avg P&L':>12} {'Win Rate':>10} {'Sharpe>1':>10}")
    print("-" * 60)
    
    for currency in currencies:
        if currency in all_results:
            config1 = all_results[currency].get('Config 1: Ultra-Tight Risk', {})
            print(f"{currency:<10} {config1.get('avg_sharpe', 0):>12.3f} "
                  f"${config1.get('avg_pnl', 0):>11,.0f} "
                  f"{config1.get('avg_win_rate', 0):>9.1f}% "
                  f"{config1.get('sharpe_above_1_pct', 0):>9.0f}%")
    
    # Config 2 Summary
    print("\n\nConfig 2: Scalping Strategy")
    print("-" * 60)
    print(f"{'Currency':<10} {'Avg Sharpe':>12} {'Avg P&L':>12} {'Win Rate':>10} {'Sharpe>1':>10}")
    print("-" * 60)
    
    for currency in currencies:
        if currency in all_results:
            config2 = all_results[currency].get('Config 2: Scalping', {})
            print(f"{currency:<10} {config2.get('avg_sharpe', 0):>12.3f} "
                  f"${config2.get('avg_pnl', 0):>11,.0f} "
                  f"{config2.get('avg_win_rate', 0):>9.1f}% "
                  f"{config2.get('sharpe_above_1_pct', 0):>9.0f}%")
    
    # Find best performing currencies
    best_c1_currency = max(all_results.keys(), 
                          key=lambda x: all_results[x].get('Config 1: Ultra-Tight Risk', {}).get('avg_sharpe', 0))
    best_c2_currency = max(all_results.keys(),
                          key=lambda x: all_results[x].get('Config 2: Scalping', {}).get('avg_sharpe', 0))
    
    print("\n" + "="*80)
    print("BEST PERFORMING CURRENCIES")
    print("="*80)
    print(f"Config 1 Best: {best_c1_currency} (Sharpe: {all_results[best_c1_currency]['Config 1: Ultra-Tight Risk']['avg_sharpe']:.3f})")
    print(f"Config 2 Best: {best_c2_currency} (Sharpe: {all_results[best_c2_currency]['Config 2: Scalping']['avg_sharpe']:.3f})")


def save_multi_currency_results(all_results):
    """Save multi-currency results to CSV"""
    rows = []
    
    for currency, currency_results in all_results.items():
        for config_name, config_data in currency_results.items():
            if 'full_results' in config_data:
                results_df = config_data['full_results']
                for idx, row in results_df.iterrows():
                    rows.append({
                        'currency': currency,
                        'config': config_name,
                        'iteration': row['iteration'],
                        'sharpe_ratio': row['sharpe_ratio'],
                        'total_pnl': row['total_pnl'],
                        'total_return': row['total_return'],
                        'win_rate': row['win_rate'],
                        'max_drawdown': row['max_drawdown'],
                        'profit_factor': row['profit_factor'],
                        'total_trades': row['total_trades']
                    })
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv('results/multi_currency_monte_carlo_results.csv', index=False)
    print(f"\nDetailed results saved to results/multi_currency_monte_carlo_results.csv")


def main():
    """Main function with command line interface"""
    epilog_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           USAGE EXAMPLES                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 BASIC USAGE:
  python run_Strategy.py
    → Runs default: AUDUSD, 50 iterations, 8,000 rows per test

📊 SINGLE CURRENCY TESTING:
  python run_Strategy.py --currency GBPUSD
    → Test British Pound with default settings
  
  python run_Strategy.py --iterations 100 --sample-size 10000
    → Run 100 tests with 10,000 rows each (more robust results)
  
  python run_Strategy.py --show-plots
    → Display interactive charts after analysis

🌍 MULTI-CURRENCY TESTING:
  python run_Strategy.py --mode multi
    → Test all major pairs: GBPUSD, EURUSD, USDJPY, NZDUSD, USDCAD
  
  python run_Strategy.py --mode multi --currencies EURUSD GBPUSD
    → Test only Euro and British Pound

📈 VISUALIZATION OPTIONS:
  python run_Strategy.py --show-plots --no-save-plots
    → View plots interactively without saving PNG files
  
  python run_Strategy.py --no-plots
    → Skip all visualizations (faster, analysis only)

⚡ PERFORMANCE OPTIONS:
  python run_Strategy.py --no-calendar
    → Skip calendar year analysis for faster results
  
  python run_Strategy.py --iterations 20 --sample-size 3000
    → Quick test with fewer iterations and smaller samples

═══════════════════════════════════════════════════════════════════════════════
                          WHAT THE FLAGS DO                                    
═══════════════════════════════════════════════════════════════════════════════

MODE SELECTION:
  --mode        Choose between 'single' (one currency) or 'multi' (many currencies)
                Default: single

DATA PARAMETERS:
  --currency    Which currency pair to test (e.g., AUDUSD, GBPUSD, EURUSD)
                Default: AUDUSD
  
  --currencies  List of currencies for multi-mode testing
                Default: GBPUSD EURUSD USDJPY NZDUSD USDCAD
  
  --iterations  How many random samples to test (more = more reliable)
                Default: 50
  
  --sample-size Number of data rows per test (more = longer time period)
                Default: 8000 (~3 months of 15-minute data)

VISUALIZATION:
  --show-plots      Open charts in a window for interactive viewing
  --no-save-plots   Don't save charts as PNG files
  --no-plots        Skip all chart generation (faster)

ANALYSIS:
  --no-calendar     Skip year-by-year performance breakdown

═══════════════════════════════════════════════════════════════════════════════

💡 TIPS:
  • Use more iterations (100+) for production decisions
  • Larger sample sizes test strategy robustness over longer periods
  • Multi-currency mode helps identify which pairs work best
  • Calendar analysis reveals performance in different market conditions

📧 Need help? Check the README.md for detailed documentation
"""
    
    parser = argparse.ArgumentParser(
        description='''
╔══════════════════════════════════════════════════════════════════════════════╗
║          MONTE CARLO STRATEGY TESTER - High Performance Trading             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Test two proven trading strategies across multiple currency pairs:          ║
║  • Config 1: Ultra-Tight Risk Management (0.2% risk, 10 pip stops)         ║
║  • Config 2: Scalping Strategy (0.1% risk, 5 pip stops)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
        ''',
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'multi', 'custom'],
                       help='Testing mode - "single": test one currency pair, "multi": test multiple pairs (default: single)')
    parser.add_argument('--currency', type=str, default='AUDUSD',
                       help='Currency pair to test in single mode, e.g. GBPUSD, EURUSD (default: AUDUSD)')
    parser.add_argument('--currencies', type=str, nargs='+',
                       default=['GBPUSD', 'EURUSD', 'USDJPY', 'NZDUSD', 'USDCAD'],
                       help='List of currency pairs for multi mode (default: GBPUSD EURUSD USDJPY NZDUSD USDCAD)')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of random samples to test - more iterations = more reliable results (default: 50)')
    parser.add_argument('--sample-size', type=int, default=8000,
                       help='Data rows per test (~3 months for 8000 rows of 15-min data) (default: 8000)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip all chart generation for faster processing')
    parser.add_argument('--no-calendar', action='store_true',
                       help='Skip year-by-year performance analysis')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display charts in GUI window for interactive viewing')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save charts as PNG files in results folder (default: enabled)')
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false',
                       help='Disable saving charts to PNG files')
    parser.add_argument('--version', '-v', action='version', 
                       version=f'Monte Carlo Strategy Tester v{__version__}')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print(f"Starting Strategy Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == 'single':
        run_single_currency_mode(
            currency=args.currency,
            n_iterations=args.iterations,
            sample_size=args.sample_size,
            enable_plots=not args.no_plots,
            enable_calendar_analysis=not args.no_calendar,
            show_plots=args.show_plots,
            save_plots=args.save_plots
        )
    
    elif args.mode == 'multi':
        run_multi_currency_mode(
            currencies=args.currencies,
            n_iterations=args.iterations,
            sample_size=args.sample_size
        )
    
    elif args.mode == 'custom':
        # Custom mode - can be extended for specific use cases
        print("Custom mode - implement your specific testing logic here")
        # Example: Test specific date ranges, custom configurations, etc.
    
    print(f"\nTesting completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()