"""
Run the Winning Momentum Strategy
Achieved Sharpe Ratio: 1.286

This file runs the exact strategy configuration that achieved our target.
"""

import pandas as pd
import numpy as np
import json
from ultimate_optimizer import AdvancedBacktest
import matplotlib.pyplot as plt
from datetime import datetime


# {
#   "success": true,
#   "best_sharpe": 1.286292010827033,
#   "best_strategy": "momentum",
#   "best_params": {
#     "lookback": 40,
#     "entry_z": 1.5
#   },
#   "timestamp": "2025-06-17T00:31:57.341574"
# }


def run_winning_strategy(data_path='../data/AUDUSD_MASTER_15M.csv', 
                        plot_results=True,
                        save_trades=True,
                        plot_period=None,
                        start_date=None,
                        end_date=None,
                        last_n_bars=None,
                        test_segments=True):
    """
    Run the momentum strategy that achieved Sharpe > 1
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    plot_results : bool
        Whether to plot the results
    save_trades : bool
        Whether to save trade log to CSV
    plot_period : str, optional
        Which period to plot: 'best', 'worst', 'both', or None for default behavior
    start_date : str, optional
        Start date for custom period (YYYY-MM-DD or YYYY)
    end_date : str, optional
        End date for custom period (YYYY-MM-DD or YYYY)
    last_n_bars : int, optional
        Test on last N bars of data
    test_segments : bool
        Whether to test on predefined segments (default: True)
    """
    
    print("="*60)
    print("Running Winning Momentum Strategy")
    print("="*60)
    
    
    # Hardcoded winning parameters from SUCCESS_SHARPE_ABOVE_1.json
    lookback = 40
    entry_z = 1.5
    exit_z = 0.5
    
    # Load the success parameters to display
    try:
        with open('SUCCESS_SHARPE_ABOVE_1.json', 'r') as f:
            success_data = json.load(f)
            print(f"\nLoaded successful parameters from SUCCESS_SHARPE_ABOVE_1.json:")
            print(f"Best Sharpe: {success_data['best_sharpe']:.3f}")
            print(f"Strategy: {success_data['best_strategy']}")
            print(f"Parameters: lookback={lookback}, entry_z={entry_z}")
    except:
        print(f"\nUsing hardcoded winning parameters:")
        print(f"Lookback: {lookback}, Entry Z: {entry_z}")
    
    # Load data
    print(f"\nLoading data from {data_path}")
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    print(f"Total data points: {len(data):,}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    results_summary = []
    
    # Handle last_n_bars if specified
    if last_n_bars:
        print(f"\n--- Testing on last {last_n_bars:,} bars ---")
        test_data = data[-last_n_bars:] if len(data) > last_n_bars else data
        print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")
        
        backtester = AdvancedBacktest(test_data)
        result = backtester.strategy_momentum(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z
        )
        
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"Total Returns: {result['returns']:.1f}%")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Max Drawdown: {result['max_dd']:.1f}%")
        print(f"Total Trades: {result['trades']}")
        
        results_summary.append({
            'segment': f'Last {last_n_bars:,} bars',
            'sharpe': result['sharpe'],
            'returns': result['returns'],
            'win_rate': result['win_rate'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'data': test_data,
            'backtester': backtester
        })
        
        if plot_results:
            plot_strategy_performance(test_data, backtester, 
                                    title_suffix=f"Last {last_n_bars:,} bars")
        
        # Don't run other segments if last_n_bars is specified
        test_segments = False
    
    # Handle custom date range
    elif start_date or end_date:
        # Parse dates - handle both YYYY and YYYY-MM-DD formats
        if start_date and len(start_date) == 4:  # Just year
            start_date = f"{start_date}-01-01"
        if end_date and len(end_date) == 4:  # Just year
            end_date = f"{end_date}-12-31"
        
        custom_data = data.copy()
        if start_date:
            custom_data = custom_data[custom_data.index >= start_date]
        if end_date:
            custom_data = custom_data[custom_data.index <= end_date]
        
        print(f"\n--- Testing custom period ---")
        print(f"Date range: {custom_data.index[0]} to {custom_data.index[-1]}")
        print(f"Total bars: {len(custom_data):,}")
        
        backtester = AdvancedBacktest(custom_data)
        result = backtester.strategy_momentum(lookback=lookback, entry_z=entry_z, exit_z=exit_z)
        
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"Total Returns: {result['returns']:.1f}%")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Max Drawdown: {result['max_dd']:.1f}%")
        print(f"Total Trades: {result['trades']}")
        
        date_label = f"{start_date or 'start'} to {end_date or 'end'}"
        results_summary.append({
            'segment': f'Custom period ({date_label})',
            'sharpe': result['sharpe'],
            'returns': result['returns'],
            'win_rate': result['win_rate'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'data': custom_data,
            'backtester': backtester
        })
        
        if plot_results:
            plot_strategy_performance(custom_data, backtester, 
                                    title_suffix=f"Custom Period ({date_label})")
        
        # Don't run other segments if custom date range is specified
        test_segments = False
    
    # Test on predefined segments if requested
    if test_segments:
        segments = {
            'Recent 50k bars': data[-50000:],
            'Previous 50k bars': data[-100000:-50000] if len(data) > 100000 else None,
            'Last 20k bars': data[-20000:],
            'Full available': data
        }
        
        for segment_name, segment_data in segments.items():
            if segment_data is None or len(segment_data) < 1000:
                continue
            
            print(f"\n--- Testing on {segment_name} ---")
            print(f"Date range: {segment_data.index[0]} to {segment_data.index[-1]}")
            
            # Run backtest
            backtester = AdvancedBacktest(segment_data)
            result = backtester.strategy_momentum(
                lookback=lookback,
                entry_z=entry_z,
                exit_z=exit_z
            )
            
            print(f"Sharpe Ratio: {result['sharpe']:.3f}")
            print(f"Total Returns: {result['returns']:.1f}%")
            print(f"Win Rate: {result['win_rate']:.1f}%")
            print(f"Max Drawdown: {result['max_dd']:.1f}%")
            print(f"Total Trades: {result['trades']}")
            
            results_summary.append({
                'segment': segment_name,
                'sharpe': result['sharpe'],
                'returns': result['returns'],
                'win_rate': result['win_rate'],
                'max_dd': result['max_dd'],
                'trades': result['trades']
            })
            
            # Store data and backtester for plotting
            results_summary[-1]['data'] = segment_data
            results_summary[-1]['backtester'] = backtester
    
    # Handle plotting based on user preferences
    if plot_results and test_segments and not (start_date or end_date or last_n_bars):
        if plot_period:
            # Sort results by sharpe ratio
            sorted_results = sorted([r for r in results_summary if 'data' in r], 
                                  key=lambda x: x['sharpe'])
            
            if plot_period == 'worst' or plot_period == 'both':
                worst = sorted_results[0]
                print(f"\n--- Plotting WORST performing period ---")
                print(f"Segment: {worst['segment']}")
                print(f"Sharpe: {worst['sharpe']:.3f}")
                plot_strategy_performance(worst['data'], worst['backtester'], 
                                        title_suffix=f"WORST Period - {worst['segment']} (Sharpe: {worst['sharpe']:.3f})")
            
            if plot_period == 'best' or plot_period == 'both':
                best = sorted_results[-1]
                print(f"\n--- Plotting BEST performing period ---")
                print(f"Segment: {best['segment']}")
                print(f"Sharpe: {best['sharpe']:.3f}")
                plot_strategy_performance(best['data'], best['backtester'], 
                                        title_suffix=f"BEST Period - {best['segment']} (Sharpe: {best['sharpe']:.3f})")
        
        else:
            # Default behavior - plot worst then best
            sorted_results = sorted([r for r in results_summary if 'data' in r], 
                                  key=lambda x: x['sharpe'])
            
            # Plot worst
            worst = sorted_results[0]
            print(f"\n--- Plotting WORST performing period (default) ---")
            print(f"Segment: {worst['segment']}")
            print(f"Sharpe: {worst['sharpe']:.3f}")
            plot_strategy_performance(worst['data'], worst['backtester'], 
                                    title_suffix=f"WORST Period - {worst['segment']} (Sharpe: {worst['sharpe']:.3f})")
            
            # Plot best
            best = sorted_results[-1]
            print(f"\n--- Plotting BEST performing period (default) ---")
            print(f"Segment: {best['segment']}")
            print(f"Sharpe: {best['sharpe']:.3f}")
            plot_strategy_performance(best['data'], best['backtester'], 
                                    title_suffix=f"BEST Period - {best['segment']} (Sharpe: {best['sharpe']:.3f})")
    
    # Save results summary (without data/backtester objects)
    summary_df = pd.DataFrame([{k: v for k, v in r.items() 
                               if k not in ['data', 'backtester']} 
                              for r in results_summary])
    summary_df.to_csv('backtest_results_summary.csv', index=False)
    print(f"\n\nResults saved to backtest_results_summary.csv")
    
    return results_summary


def plot_strategy_performance(data, backtester, title_suffix=""):
    """Create detailed performance plots"""
    
    # Generate signals for plotting
    df = data.copy()
    
    # Calculate momentum and z-score - using hardcoded parameters
    lookback = 40
    entry_z = 1.5
    exit_z = 0.5
    
    df['Momentum'] = df['Close'].pct_change(lookback)
    df['Mom_Mean'] = df['Momentum'].rolling(50).mean()
    df['Mom_Std'] = df['Momentum'].rolling(50).std()
    df['Z_Score'] = (df['Momentum'] - df['Mom_Mean']) / df['Mom_Std']
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['Z_Score'] < -entry_z, 'Signal'] = 1
    df.loc[df['Z_Score'] > entry_z, 'Signal'] = -1
    df.loc[abs(df['Z_Score']) < exit_z, 'Signal'] = 0
    
    # Position tracking
    df['Position'] = df['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Buy_Hold_Returns'] = (1 + df['Returns']).cumprod()
    
    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 1. Price and signals
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], 'b-', alpha=0.5, linewidth=0.5)
    
    # Mark entries
    long_entries = df[(df['Position'] == 1) & (df['Position'].shift(1) != 1)]
    short_entries = df[(df['Position'] == -1) & (df['Position'].shift(1) != -1)]
    exits = df[(df['Position'] == 0) & (df['Position'].shift(1) != 0)]
    
    ax1.scatter(long_entries.index, long_entries['Close'], 
                color='green', marker='^', s=50, alpha=0.7, label='Long Entry')
    ax1.scatter(short_entries.index, short_entries['Close'], 
                color='red', marker='v', s=50, alpha=0.7, label='Short Entry')
    ax1.scatter(exits.index, exits['Close'], 
                color='yellow', marker='o', s=30, alpha=0.5, label='Exit')
    
    title = 'AUDUSD Price and Trading Signals'
    if title_suffix:
        title += f' - {title_suffix}'
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Z-Score
    ax2 = axes[1]
    ax2.plot(df.index, df['Z_Score'], 'purple', alpha=0.7)
    ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Short Entry')
    ax2.axhline(y=-1.5, color='green', linestyle='--', alpha=0.5, label='Long Entry')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.3)
    ax2.fill_between(df.index, -0.5, 0.5, alpha=0.1, color='gray', label='Exit Zone')
    ax2.set_title('Momentum Z-Score', fontsize=14)
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df['Position'], 
                     where=(df['Position'] > 0), color='green', alpha=0.3, label='Long')
    ax3.fill_between(df.index, 0, df['Position'], 
                     where=(df['Position'] < 0), color='red', alpha=0.3, label='Short')
    ax3.set_title('Position Over Time', fontsize=14)
    ax3.set_ylabel('Position')
    ax3.set_ylim(-1.5, 1.5)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative returns
    ax4 = axes[3]
    ax4.plot(df.index, df['Cumulative_Returns'], 'green', linewidth=2, label='Strategy')
    ax4.plot(df.index, df['Buy_Hold_Returns'], 'blue', linewidth=1, alpha=0.5, label='Buy & Hold')
    ax4.set_title('Cumulative Returns', fontsize=14)
    ax4.set_ylabel('Cumulative Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save with unique filename based on title
    filename = 'winning_strategy_performance'
    if title_suffix:
        # Clean filename
        clean_suffix = title_suffix.replace(' ', '_').replace('-', '').replace('(', '').replace(')', '').replace(':', '')
        filename += f'_{clean_suffix}'
    filename += '.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {filename}")
    plt.show()
    
    # Print some statistics
    print("\n--- Trade Statistics ---")
    total_trades = (df['Position'].diff() != 0).sum() // 2
    long_trades = ((df['Position'] == 1) & (df['Position'].shift(1) != 1)).sum()
    short_trades = ((df['Position'] == -1) & (df['Position'].shift(1) != -1)).sum()
    
    print(f"Total trades: {total_trades}")
    print(f"Long trades: {long_trades}")
    print(f"Short trades: {short_trades}")
    
    # Calculate average holding period
    position_changes = df['Position'].diff() != 0
    position_blocks = position_changes.cumsum()
    holding_periods = df.groupby(position_blocks).size()
    avg_holding = holding_periods[df.groupby(position_blocks)['Position'].first() != 0].mean()
    print(f"Average holding period: {avg_holding:.1f} bars ({avg_holding * 15 / 60:.1f} hours)")
    
    # Save trade log
    trades_df = pd.DataFrame()
    entry_mask = (df['Position'] != 0) & (df['Position'].shift(1) == 0)
    exit_mask = (df['Position'] == 0) & (df['Position'].shift(1) != 0)
    
    if entry_mask.sum() > 0:
        trades_data = []
        entry_dates = df.index[entry_mask]
        exit_dates = df.index[exit_mask]
        
        for i in range(min(len(entry_dates), len(exit_dates))):
            entry_idx = df.index.get_loc(entry_dates[i])
            exit_idx = df.index.get_loc(exit_dates[i])
            
            trade_data = {
                'entry_date': entry_dates[i],
                'exit_date': exit_dates[i],
                'entry_price': df['Close'].iloc[entry_idx],
                'exit_price': df['Close'].iloc[exit_idx],
                'position': df['Position'].iloc[entry_idx],
                'entry_z_score': df['Z_Score'].iloc[entry_idx],
                'exit_z_score': df['Z_Score'].iloc[exit_idx],
                'return': df.iloc[entry_idx:exit_idx+1]['Strategy_Returns'].sum()
            }
            trades_data.append(trade_data)
        
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv('winning_strategy_trades.csv', index=False)
        print(f"\nTrade log saved to winning_strategy_trades.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the winning momentum strategy backtest')
    parser.add_argument('--data', default='../data/AUDUSD_MASTER_15M.csv', help='Path to data file')
    parser.add_argument('--plot', default='both', choices=['best', 'worst', 'both', 'none'], 
                       help='Which period to plot (default: both)')
    parser.add_argument('--start', help='Start date for custom period (YYYY-MM-DD or YYYY)')
    parser.add_argument('--end', help='End date for custom period (YYYY-MM-DD or YYYY)')
    parser.add_argument('--last', type=int, help='Test on last N bars')
    parser.add_argument('--no-save', action='store_true', help='Do not save trade log')
    
    args = parser.parse_args()
    
    # Run the backtest
    results = run_winning_strategy(
        data_path=args.data,
        plot_results=(args.plot != 'none'),
        save_trades=not args.no_save,
        plot_period=args.plot if not (args.start or args.end or args.last) else None,
        start_date=args.start,
        end_date=args.end,
        last_n_bars=args.last,
        test_segments=not (args.start or args.end or args.last)
    )
    
    print("\n" + "="*60)
    print("Backtest Complete!")
    print("="*60)
    
    # Print usage examples
    print("\nUsage examples:")
    print("  python run_winning_strategy.py                    # Default: plot worst and best periods")
    print("  python run_winning_strategy.py --plot best        # Plot only best period")
    print("  python run_winning_strategy.py --plot worst       # Plot only worst period")
    print("  python run_winning_strategy.py --last 10000       # Test on last 10,000 bars")
    print("  python run_winning_strategy.py --start 2024-01-01 # Test from 2024 to present")
    print("  python run_winning_strategy.py --start 2023-06-01 --end 2023-12-31  # Specific period")
    print("  python run_winning_strategy.py --start 2020 --end 2025  # Test 2020-2025 (year format)")
    print("  python run_winning_strategy.py --last 50000 --plot none  # Quick test without plotting")