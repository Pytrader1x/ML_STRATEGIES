"""
Unified Strategy Runner - Monte Carlo Testing Framework
Supports multiple modes: single currency, multi-currency, with various analysis options
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy_fixed import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from dataclasses import dataclass
from typing import Optional
import json
import time

warnings.filterwarnings('ignore')

__version__ = "2.1.0"


def create_config_1_ultra_tight_risk(debug_mode=False, realistic_costs=False, use_daily_sharpe=True):
    """
    Configuration 1: Ultra-Tight Risk Management
    Achieved Sharpe Ratio: 1.171 on AUDUSD
    """
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
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
        realistic_costs=realistic_costs,
        verbose=False,
        debug_decisions=debug_mode,
        use_daily_sharpe=use_daily_sharpe
    )
    return OptimizedProdStrategy(config)


def create_config_2_scalping(realistic_costs=False, use_daily_sharpe=True):
    """
    Configuration 2: Scalping Strategy
    Achieved Sharpe Ratio: 1.146 on AUDUSD
    """
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
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
        realistic_costs=realistic_costs,
        verbose=False,
        debug_decisions=False,
        use_daily_sharpe=use_daily_sharpe
    )
    return OptimizedProdStrategy(config)


def load_and_prepare_data(currency_pair, data_path=None):
    """Load and prepare data for a specific currency pair"""
    if data_path is None:
        # Auto-detect data path based on current location
        if os.path.exists('data'):
            data_path = 'data'
        elif os.path.exists('../data'):
            data_path = '../data'
        else:
            raise FileNotFoundError("Cannot find data directory. Please run from project root.")
    
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
    
    # Helper function to format time
    def format_time(seconds):
        if seconds < 0.1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 1.0:
            return f"{seconds * 1000:.0f}ms"
        else:
            return f"{seconds:.1f}s"
    
    # Neuro Trend Intelligent
    print("  Calculating Neuro Trend Intelligent...")
    start_time = time.time()
    df = TIC.add_neuro_trend_intelligent(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed Neuro Trend Intelligent in {format_time(elapsed_time)} ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
    
    # Market Bias
    print("  Calculating Market Bias...")
    start_time = time.time()
    df = TIC.add_market_bias(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed Market Bias in {format_time(elapsed_time)} ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
    
    # Intelligent Chop
    print("  Calculating Intelligent Chop...")
    start_time = time.time()
    df = TIC.add_intelligent_chop(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed Intelligent Chop in {format_time(elapsed_time)} ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
    
    return df


def calculate_trade_statistics(results):
    """Calculate detailed trade statistics including consecutive wins/losses"""
    stats = {}
    
    # Extract basic stats if available
    if 'trades' in results and results['trades'] is not None:
        trades = results['trades']
        if len(trades) > 0:
            # Extract P&L values - handle both Trade objects and dictionaries
            pnl_values = []
            for trade in trades:
                if hasattr(trade, 'pnl'):  # Trade object
                    if trade.pnl is not None:
                        pnl_values.append(trade.pnl)
                elif isinstance(trade, dict) and 'pnl' in trade:  # Dictionary
                    if trade['pnl'] is not None:
                        pnl_values.append(trade['pnl'])
            
            # Calculate consecutive wins/losses
            wins = [1 if pnl > 0 else 0 for pnl in pnl_values]
            losses = [1 if pnl < 0 else 0 for pnl in pnl_values]
            
            # Consecutive wins
            win_streaks = []
            current_streak = 0
            for win in wins:
                if win == 1:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        win_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                win_streaks.append(current_streak)
            
            # Consecutive losses
            loss_streaks = []
            current_streak = 0
            for loss in losses:
                if loss == 1:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        loss_streaks.append(current_streak)
                    current_streak = 0
            if current_streak > 0:
                loss_streaks.append(current_streak)
            
            stats['max_consecutive_wins'] = max(win_streaks) if win_streaks else 0
            stats['avg_consecutive_wins'] = int(round(np.mean(win_streaks))) if win_streaks else 0
            stats['max_consecutive_losses'] = max(loss_streaks) if loss_streaks else 0
            stats['avg_consecutive_losses'] = int(round(np.mean(loss_streaks))) if loss_streaks else 0
            stats['num_wins'] = sum(wins)
            stats['num_losses'] = sum(losses)
        else:
            stats = {
                'max_consecutive_wins': 0,
                'avg_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_consecutive_losses': 0,
                'num_wins': 0,
                'num_losses': 0
            }
    else:
        # Estimate from aggregate stats
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0) / 100
        
        stats['num_wins'] = int(total_trades * win_rate)
        stats['num_losses'] = total_trades - stats['num_wins']
        
        # Conservative estimates for consecutive stats using bounded approach
        if stats['num_wins'] > 0:
            # Average consecutive wins: simple bounded approach
            if win_rate >= 0.8:
                stats['avg_consecutive_wins'] = 4
            elif win_rate >= 0.6:
                stats['avg_consecutive_wins'] = 3
            elif win_rate >= 0.4:
                stats['avg_consecutive_wins'] = 2
            else:
                stats['avg_consecutive_wins'] = 1
            
            # Max consecutive wins: bounded by total wins and reasonable statistical limits
            max_bound = min(stats['num_wins'], int(total_trades * 0.3))  # Max 30% of trades
            if win_rate >= 0.8:
                stats['max_consecutive_wins'] = min(max_bound, stats['avg_consecutive_wins'] * 8)
            elif win_rate >= 0.6:
                stats['max_consecutive_wins'] = min(max_bound, stats['avg_consecutive_wins'] * 5)
            else:
                stats['max_consecutive_wins'] = min(max_bound, stats['avg_consecutive_wins'] * 3)
        else:
            stats['avg_consecutive_wins'] = 0
            stats['max_consecutive_wins'] = 0
            
        if stats['num_losses'] > 0:
            # Average consecutive losses: simple bounded approach
            loss_rate = 1.0 - win_rate
            if loss_rate >= 0.6:
                stats['avg_consecutive_losses'] = 3
            elif loss_rate >= 0.4:
                stats['avg_consecutive_losses'] = 2
            else:
                stats['avg_consecutive_losses'] = 1
            
            # Max consecutive losses: bounded by total losses and reasonable limits
            max_bound = min(stats['num_losses'], int(total_trades * 0.2))  # Max 20% of trades
            if loss_rate >= 0.4:
                stats['max_consecutive_losses'] = min(max_bound, stats['avg_consecutive_losses'] * 4)
            else:
                stats['max_consecutive_losses'] = min(max_bound, stats['avg_consecutive_losses'] * 3)
        else:
            stats['avg_consecutive_losses'] = 0
            stats['max_consecutive_losses'] = 0
    
    return stats


def _export_trades_detailed(trades, currency, config_name):
    """Detailed trade export with all partial exits and P&L breakdown"""
    import pandas as pd
    from datetime import datetime
    
    trade_records = []
    for i, trade in enumerate(trades, 1):
        # Extract basic trade info
        direction = trade.direction.value if hasattr(trade.direction, 'value') else trade.direction
        exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason
        
        # Calculate pip distances
        sl_distance_pips = abs(trade.entry_price - trade.stop_loss) / 0.0001
        
        record = {
            'trade_id': i,
            'entry_time': trade.entry_time,
            'entry_price': trade.entry_price,
            'direction': direction,
            'initial_size_millions': trade.initial_position_size / 1e6 if hasattr(trade, 'initial_position_size') else trade.position_size / 1e6,
            'confidence': trade.confidence,
            'is_relaxed': trade.is_relaxed,
            'entry_logic': 'Relaxed (NTI only)' if trade.is_relaxed else 'Standard (NTI+MB+IC)',
            'sl_price': trade.stop_loss,
            'sl_distance_pips': sl_distance_pips,
            'tp1_price': trade.take_profits[0] if len(trade.take_profits) > 0 else None,
            'tp2_price': trade.take_profits[1] if len(trade.take_profits) > 1 else None,
            'tp3_price': trade.take_profits[2] if len(trade.take_profits) > 2 else None,
            'exit_time': trade.exit_time,
            'exit_price': trade.exit_price,
            'exit_reason': exit_reason,
            'tp_hits': trade.tp_hits,
            'trade_duration_hours': (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else None,
            'final_pnl': trade.pnl,
        }
        
        # Add partial exit details if available
        if hasattr(trade, 'partial_exits') and trade.partial_exits:
            partial_pnl_sum = 0
            for j, pe in enumerate(trade.partial_exits[:3], 1):
                pe_type = pe.exit_type if hasattr(pe, 'exit_type') else f'TP{pe.tp_level}' if hasattr(pe, 'tp_level') else 'PARTIAL'
                pe_size = pe.size / 1e6 if hasattr(pe, 'size') else 0
                pe_pnl = pe.pnl if hasattr(pe, 'pnl') else 0
                partial_pnl_sum += pe_pnl
                
                record[f'partial_exit_{j}_type'] = pe_type
                record[f'partial_exit_{j}_size_m'] = pe_size
                record[f'partial_exit_{j}_pnl'] = pe_pnl
            
            record['partial_pnl_total'] = partial_pnl_sum
        
        # Calculate final P&L components
        if direction == 'long':
            final_pips = (trade.exit_price - trade.entry_price) / 0.0001
        else:
            final_pips = (trade.entry_price - trade.exit_price) / 0.0001
        
        record['final_exit_pips'] = final_pips
        record['pnl_per_million'] = trade.pnl / (record['initial_size_millions']) if record['initial_size_millions'] > 0 else 0
        
        trade_records.append(record)
    
    # Save to CSV
    df = pd.DataFrame(trade_records)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_trades_detail_{timestamp}.csv'
    df.to_csv(filepath, index=False, float_format='%.6f')
    
    # Print summary
    print(f"\n📊 Trade Export Summary for {config_name}:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Total P&L: ${df['final_pnl'].sum():,.2f}")
    print(f"  Avg P&L per Trade: ${df['final_pnl'].mean():,.2f}")
    print(f"  Win Rate: {(df['final_pnl'] > 0).sum() / len(df) * 100:.1f}%")
    print(f"  Avg Win: ${df[df['final_pnl'] > 0]['final_pnl'].mean():,.2f}" if len(df[df['final_pnl'] > 0]) > 0 else "  Avg Win: N/A")
    print(f"  Avg Loss: ${df[df['final_pnl'] < 0]['final_pnl'].mean():,.2f}" if len(df[df['final_pnl'] < 0]) > 0 else "  Avg Loss: N/A")
    print(f"  Saved to: {filepath}")
    
    return filepath

def verify_position_integrity(trades):
    """Verify position integrity - no over-exiting"""
    issues = []
    for i, trade in enumerate(trades):
        if hasattr(trade, 'initial_position_size') and hasattr(trade, 'total_exited'):
            if abs(trade.total_exited - trade.initial_position_size) > 1:  # Allow 1 unit rounding error
                issues.append({
                    'trade_id': i + 1,
                    'initial_size': trade.initial_position_size,
                    'total_exited': trade.total_exited,
                    'difference': trade.total_exited - trade.initial_position_size
                })
    return len(issues) == 0, issues

def run_single_monte_carlo(df, strategy, n_iterations, sample_size, return_last_sample=False):
    """Run Monte Carlo simulation for a single strategy configuration"""
    iteration_results = []
    last_sample_df = None
    last_results = None
    integrity_checks = []
    
    # Track aggregate statistics
    all_trade_stats = {
        'max_consecutive_wins': [],
        'avg_consecutive_wins': [],
        'max_consecutive_losses': [],
        'avg_consecutive_losses': [],
        'num_wins': [],
        'num_losses': [],
        'avg_win_pips': [],
        'avg_loss_pips': []
    }
    
    for i in range(n_iterations):
        # Get random starting point
        max_start = len(df) - sample_size
        if max_start < 0:
            # If date range is too small, use entire range
            sample_df = df.copy()
        else:
            start_idx = np.random.randint(0, max_start + 1)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
        
        # Run backtest
        results = strategy.run_backtest(sample_df)
        
        # Verify position integrity
        integrity_ok, integrity_issues = verify_position_integrity(results.get('trades', []))
        integrity_checks.append(integrity_ok)
        
        if not integrity_ok:
            print(f"  ⚠️  WARNING: Position integrity issues in iteration {i+1}:")
            for issue in integrity_issues[:3]:  # Show first 3 issues
                print(f"     Trade {issue['trade_id']}: Entered {issue['initial_size']/1e6:.2f}M, Exited {issue['total_exited']/1e6:.2f}M")
        
        # Calculate trade statistics
        trade_stats = calculate_trade_statistics(results)
        
        # Calculate pip-based metrics if trades are available
        if 'trades' in results and results['trades']:
            # Extract pip gains and losses
            pip_gains = []
            pip_losses = []
            
            for trade in results['trades']:
                # Handle both Trade objects and dictionaries
                if hasattr(trade, 'entry_price'):
                    entry_price = trade.entry_price
                    exit_price = trade.exit_price
                    direction = trade.direction
                    pnl = trade.pnl
                else:
                    # Dictionary format
                    entry_price = trade.get('entry_price')
                    exit_price = trade.get('exit_price')
                    direction = trade.get('direction')
                    pnl = trade.get('pnl')
                
                if entry_price and exit_price and pnl is not None:
                    # Calculate pips based on direction
                    if hasattr(direction, 'value'):
                        direction_val = direction.value
                    else:
                        direction_val = direction
                    
                    if direction_val == 'long':
                        pips = (exit_price - entry_price) / 0.0001
                    else:
                        pips = (entry_price - exit_price) / 0.0001
                    
                    # Categorize as gain or loss
                    if pnl > 0:
                        pip_gains.append(abs(pips))
                    elif pnl < 0:
                        pip_losses.append(abs(pips))
            
            # Calculate averages
            avg_win_pips = np.mean(pip_gains) if pip_gains else 0
            avg_loss_pips = np.mean(pip_losses) if pip_losses else 0
            
            trade_stats['avg_win_pips'] = avg_win_pips
            trade_stats['avg_loss_pips'] = avg_loss_pips
        else:
            trade_stats['avg_win_pips'] = 0
            trade_stats['avg_loss_pips'] = 0
        
        for key in all_trade_stats:
            all_trade_stats[key].append(trade_stats.get(key, 0))
        
        # Store last iteration data if needed
        if return_last_sample and i == n_iterations - 1:
            last_sample_df = sample_df.copy()
            last_results = results.copy()
            last_results['trade_stats'] = trade_stats
            # Keep reference to strategy instance for trade export
            last_results['strategy_instance'] = strategy
        
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
            'avg_loss': results['avg_loss'],
            'num_wins': trade_stats['num_wins'],
            'num_losses': trade_stats['num_losses'],
            'max_consec_wins': trade_stats['max_consecutive_wins'],
            'max_consec_losses': trade_stats['max_consecutive_losses'],
            'integrity_ok': integrity_ok
        }
        
        iteration_results.append(iteration_data)
        
        # Print progress with latest results
        if (i + 1) % 10 == 0 or i == 0 or i == n_iterations - 1:
            integrity_status = '✓' if integrity_ok else '✗'
            print(f"  [{i + 1:3d}/{n_iterations}] Sharpe: {results['sharpe_ratio']:>6.3f} | Return: {results['total_return']:>6.1f}% | " +
                  f"WR: {results['win_rate']:>5.1f}% | Trades: {results['total_trades']:>4} ({trade_stats['num_wins']}W/{trade_stats['num_losses']}L) | " +
                  f"Pips: +{trade_stats.get('avg_win_pips', 0):>4.1f}/-{trade_stats.get('avg_loss_pips', 0):>4.1f} | Integrity: {integrity_status}")
    
    # Create results dataframe
    results_df = pd.DataFrame(iteration_results)
    
    # Add aggregate trade statistics including integrity
    integrity_pass_rate = sum(integrity_checks) / len(integrity_checks) * 100 if integrity_checks else 0
    results_df.attrs['aggregate_trade_stats'] = {
        'avg_max_consecutive_wins': np.mean(all_trade_stats['max_consecutive_wins']),
        'avg_avg_consecutive_wins': np.mean(all_trade_stats['avg_consecutive_wins']),
        'avg_max_consecutive_losses': np.mean(all_trade_stats['max_consecutive_losses']),
        'avg_avg_consecutive_losses': np.mean(all_trade_stats['avg_consecutive_losses']),
        'avg_num_wins': np.mean(all_trade_stats['num_wins']),
        'avg_num_losses': np.mean(all_trade_stats['num_losses']),
        'avg_win_pips': np.mean(all_trade_stats['avg_win_pips']),
        'avg_loss_pips': np.mean(all_trade_stats['avg_loss_pips']),
        'position_integrity_rate': integrity_pass_rate
    }
    
    if return_last_sample:
        return results_df, last_sample_df, last_results
    return results_df


def generate_calendar_year_analysis(results_df, config_name, currency=None, show_plots=False, save_plots=False):
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
        
        # Always save calendar plots to charts directory
        plot_filename = f'charts/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_calendar_year.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Calendar year plot saved to {plot_filename}")
        
        # Never show calendar plots - just close them
        plt.close()
    
    return yearly_stats


def run_single_currency_mode(currency='AUDUSD', n_iterations=50, sample_size=8000, 
                           enable_plots=True, enable_calendar_analysis=True,
                           show_plots=False, save_plots=False, realistic_costs=False,
                           use_daily_sharpe=True, date_range=None, export_trades=True):
    """Run Monte Carlo testing on a single currency pair with both configurations"""
    
    print("="*80)
    print(f"SINGLE CURRENCY MODE - {currency}")
    print(f"Iterations: {n_iterations} | Sample Size: {sample_size:,} rows")
    if date_range:
        print(f"Date Range: {date_range[0]} to {date_range[1]}")
    print("="*80)
    
    # Print trading mode box
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    if realistic_costs:
        print("│                         🎯 REALISTIC TRADING MODE 🎯                        │")
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        print("│  • Market Entry Orders:    0.0 - 0.5 pips random slippage                  │")
        print("│  • Stop Loss Orders:       0.0 - 2.0 pips slippage (worse fill)            │")
        print("│  • Trailing Stop Orders:   0.0 - 1.0 pips slippage (worse fill)            │")
        print("│  • Take Profit Orders:     0.0 pips (limit orders - perfect fill)          │")
        print("│                                                                             │")
        print("│  This mode simulates real market conditions with execution costs.           │")
    else:
        print("│                         ⚡ PERFECT EXECUTION MODE ⚡                        │")
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        print("│  • All orders execute at exact prices                                      │")
        print("│  • No slippage or transaction costs                                        │")
        print("│  • Theoretical best-case scenario                                          │")
        print("│                                                                             │")
        print("│  ⚠️  WARNING: Results may be overly optimistic for live trading            │")
    print("└─────────────────────────────────────────────────────────────────────────────┘\n")
    
    # Load data
    df = load_and_prepare_data(currency)
    
    # Apply date range filter if specified
    if date_range:
        start_date, end_date = date_range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        print(f"\nFiltered data to date range: {df.index[0]} to {df.index[-1]}")
        print(f"Rows in date range: {len(df):,}")
        
        if len(df) < sample_size:
            print(f"WARNING: Date range has fewer rows ({len(df)}) than sample size ({sample_size})")
            print(f"Adjusting sample size to {len(df)}")
            sample_size = len(df)
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk Management", create_config_1_ultra_tight_risk(realistic_costs=realistic_costs, use_daily_sharpe=use_daily_sharpe)),
        ("Config 2: Scalping Strategy", create_config_2_scalping(realistic_costs=realistic_costs, use_daily_sharpe=use_daily_sharpe))
    ]
    
    all_results = {}
    
    for config_name, strategy in configs:
        print(f"\n{'='*80}")
        print(f"Testing {config_name}")
        print(f"{'='*80}")
        
        # Run Monte Carlo (get last sample if we need to show plots)
        if show_plots or save_plots:
            results_df, last_sample_df, last_results = run_single_monte_carlo(
                df, strategy, n_iterations, sample_size, return_last_sample=True
            )
            # Store last sample data for plotting
            if config_name not in all_results:
                all_results[config_name] = {}
            all_results[config_name]['results_df'] = results_df
            all_results[config_name]['last_sample_df'] = last_sample_df
            all_results[config_name]['last_results'] = last_results
            
            # Export trades from last iteration if requested
            if export_trades and last_results:
                # Try to get strategy instance from results
                strategy_instance = last_results.get('strategy_instance')
                if strategy_instance and hasattr(strategy_instance, 'export_trades_to_csv'):
                    trade_filepath = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_trades_detail.csv'
                    strategy_instance.export_trades_to_csv(trade_filepath)
                else:
                    # Fallback: Try direct export from results
                    trades = last_results.get('trades', [])
                    if trades:
                        print(f"\n📝 Creating detailed trade export for {config_name}...")
                        _export_trades_detailed(trades, currency, config_name)
        else:
            results_df = run_single_monte_carlo(df, strategy, n_iterations, sample_size)
            all_results[config_name] = {'results_df': results_df}
        
        # Print summary statistics
        print("\n━━━ Performance Metrics ━━━")
        print(f"  Sharpe Ratio:     {results_df['sharpe_ratio'].mean():.3f} ± {results_df['sharpe_ratio'].std():.3f}")
        print(f"  Total Return:     {results_df['total_return'].mean():.1f}% ± {results_df['total_return'].std():.1f}%")
        print(f"  Win Rate:         {results_df['win_rate'].mean():.1f}% ± {results_df['win_rate'].std():.1f}%")
        print(f"  Max Drawdown:     {results_df['max_drawdown'].mean():.1f}% ± {results_df['max_drawdown'].std():.1f}%")
        print(f"  Profit Factor:    {results_df['profit_factor'].mean():.2f}")
        
        print("\n━━━ Trade Statistics ━━━")
        print(f"  Total Trades:     {results_df['total_trades'].mean():.0f} ± {results_df['total_trades'].std():.0f}")
        print(f"  Wins/Losses:      {results_df['num_wins'].mean():.0f}W / {results_df['num_losses'].mean():.0f}L")
        print(f"  Max Consec Wins:  {results_df['max_consec_wins'].mean():.1f}")
        print(f"  Max Consec Loss:  {results_df['max_consec_losses'].mean():.1f}")
        
        # Get aggregate stats if available
        if hasattr(results_df, 'attrs') and 'aggregate_trade_stats' in results_df.attrs:
            agg_stats = results_df.attrs['aggregate_trade_stats']
            print(f"  Avg Consec Wins:  {agg_stats['avg_avg_consecutive_wins']:.1f}")
            print(f"  Avg Consec Loss:  {agg_stats['avg_avg_consecutive_losses']:.1f}")
            print(f"  Avg Win Pips:     {agg_stats['avg_win_pips']:.1f}")
            print(f"  Avg Loss Pips:    {agg_stats['avg_loss_pips']:.1f}")
        
        print("\n━━━ Consistency ━━━")
        print(f"  Sharpe > 1.0:     {(results_df['sharpe_ratio'] > 1.0).sum()}/{n_iterations} ({(results_df['sharpe_ratio'] > 1.0).sum()/n_iterations*100:.0f}%)")
        print(f"  Profitable:       {(results_df['total_pnl'] > 0).sum()}/{n_iterations} ({(results_df['total_pnl'] > 0).sum()/n_iterations*100:.0f}%)")
        
        # Position integrity validation
        integrity_passed = results_df['integrity_ok'].sum() if 'integrity_ok' in results_df.columns else n_iterations
        integrity_rate = integrity_passed / n_iterations * 100
        print(f"\n━━━ Position Integrity Validation ━━━")
        print(f"  Position Integrity: {integrity_passed}/{n_iterations} ({integrity_rate:.0f}%)")
        if integrity_rate == 100:
            print(f"  ✅ PERFECT: No over-exiting detected")
        else:
            print(f"  ⚠️  WARNING: Some trades had position sizing issues")
        
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
                          n_iterations=30, sample_size=8000, realistic_costs=False, use_daily_sharpe=True):
    """Run Monte Carlo testing on multiple currency pairs"""
    
    print("="*80)
    print("MULTI-CURRENCY MODE")
    print(f"Testing {len(currencies)} currency pairs")
    print(f"Iterations per pair: {n_iterations} | Sample Size: {sample_size:,} rows")
    print("="*80)
    
    # Print trading mode box
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    if realistic_costs:
        print("│                         🎯 REALISTIC TRADING MODE 🎯                        │")
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        print("│  Slippage: Entry 0-0.5 pips | SL 0-2 pips | TSL 0-1 pip | TP 0 pips       │")
    else:
        print("│                         ⚡ PERFECT EXECUTION MODE ⚡                        │")
        print("├─────────────────────────────────────────────────────────────────────────────┤")
        print("│  All orders execute at exact prices - No slippage                          │")
    print("└─────────────────────────────────────────────────────────────────────────────┘\n")
    
    all_results = {}
    
    for currency in currencies:
        try:
            # Load data
            df = load_and_prepare_data(currency)
            
            # Test both configurations
            configs = [
                ("Config 1: Ultra-Tight Risk", create_config_1_ultra_tight_risk(realistic_costs=realistic_costs, use_daily_sharpe=use_daily_sharpe)),
                ("Config 2: Scalping", create_config_2_scalping(realistic_costs=realistic_costs, use_daily_sharpe=use_daily_sharpe))
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


def generate_comparison_plots(all_results, currency, show_plots=False, save_plots=False):
    """Generate comparison plots for single currency mode"""
    # First, handle the trading charts for last iteration of each config
    for config_name, config_data in all_results.items():
        if 'last_sample_df' in config_data and 'last_results' in config_data:
            if show_plots or save_plots:
                print(f"\nGenerating trading chart for {config_name} - Last Iteration...")
                
                try:
                    # Generate the actual trading chart
                    fig = plot_production_results(
                        df=config_data['last_sample_df'],
                        results=config_data['last_results'],
                        title=f"{config_name} - Last Iteration\nSharpe={config_data['last_results']['sharpe_ratio']:.3f}, P&L=${config_data['last_results']['total_pnl']:,.0f}",
                        show_pnl=True,
                        show=show_plots  # Only show if show_plots is True
                    )
                    
                    # Save trade chart if save_plots is enabled
                    if save_plots and fig is not None:
                        plot_filename = f'charts/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_trades.png'
                        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                        print(f"Trading chart saved to {plot_filename}")
                        
                    if not show_plots and fig is not None:
                        plt.close(fig)
                        
                except Exception as e:
                    print(f"Warning: Could not generate trading chart for {config_name}: {e}")
    
    # Always generate and save the summary comparison plots to charts directory
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{currency} - Monte Carlo Analysis Comparison', fontsize=16)
    
    config1_df = all_results["Config 1: Ultra-Tight Risk Management"]['results_df']
    config2_df = all_results["Config 2: Scalping Strategy"]['results_df']
    
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
    
    # Always save metrics comparison plot to charts directory
    metrics_filename = f'charts/{currency}_metrics_comparison.png'
    plt.savefig(metrics_filename, dpi=150, bbox_inches='tight')
    print(f"\nMetrics comparison chart saved to {metrics_filename}")
    
    # Never show the metrics plot - just close it
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


# ============================================================================
# CRYPTO STRATEGY SECTION
# ============================================================================

@dataclass
class CryptoTrade:
    """Represents a single crypto trade"""
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float
    direction: int
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_dollars: Optional[float] = None


class FinalCryptoStrategy:
    """
    Crypto-specific strategy with wider stops and trend following
    """
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
        self.current_capital = config['initial_capital']
        
    def calculate_position_size(self, price, stop_loss_price):
        """Simple fixed fractional position sizing"""
        risk_amount = self.current_capital * self.config['risk_per_trade']
        price_risk_pct = abs(price - stop_loss_price) / price
        
        if price_risk_pct == 0:
            return 0
        
        position_value = risk_amount / price_risk_pct
        position_size = position_value / price
        
        # Max position size
        max_value = self.current_capital * self.config['max_position_pct']
        max_size = max_value / price
        
        return min(position_size, max_size)
    
    def calculate_trend_strength(self, df, i):
        """Calculate trend strength using multiple timeframes"""
        if i < 200:
            return 0
        
        # Multiple moving averages
        ma_20 = df['Close'].iloc[i-20:i].mean()
        ma_50 = df['Close'].iloc[i-50:i].mean()
        ma_100 = df['Close'].iloc[i-100:i].mean()
        ma_200 = df['Close'].iloc[i-200:i].mean()
        
        current_price = df['Close'].iloc[i]
        
        # Bull trend scoring
        bull_score = 0
        if current_price > ma_20:
            bull_score += 1
        if ma_20 > ma_50:
            bull_score += 1
        if ma_50 > ma_100:
            bull_score += 1
        if ma_100 > ma_200:
            bull_score += 1
        if current_price > ma_200 * 1.1:  # 10% above 200MA
            bull_score += 1
        
        # Bear trend scoring
        bear_score = 0
        if current_price < ma_20:
            bear_score += 1
        if ma_20 < ma_50:
            bear_score += 1
        if ma_50 < ma_100:
            bear_score += 1
        if ma_100 < ma_200:
            bear_score += 1
        if current_price < ma_200 * 0.9:  # 10% below 200MA
            bear_score += 1
        
        return bull_score - bear_score
    
    def check_volatility_filter(self, df, i):
        """Check if volatility is in acceptable range"""
        if i < 100:
            return True
        
        # Calculate recent volatility
        returns = df['Close'].iloc[i-96:i].pct_change().dropna()
        daily_vol = returns.std() * np.sqrt(96)
        
        # Filter out extreme volatility periods
        return 0.02 < daily_vol < 0.15  # 2% to 15% daily volatility
    
    def run_backtest(self, df):
        """Run crypto backtest"""
        # Add required indicators
        required_cols = ['NTI_Direction', 'MB_Bias', 'IC_Signal']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required indicator: {col}")
        
        # Initialize
        self.trades = []
        self.equity_curve = [self.config['initial_capital']]
        self.current_capital = self.config['initial_capital']
        open_trade = None
        
        # Track last signal
        last_signal_direction = 0
        bars_since_exit = 0
        
        # Main loop
        for i in range(200, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_price = current_bar['Close']
            
            # Update bars since exit
            if open_trade is None:
                bars_since_exit += 1
            
            # Check open trade
            if open_trade is not None:
                exit_price = None
                exit_reason = None
                
                # Stop loss
                if open_trade.direction > 0:
                    if current_bar['Low'] <= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss
                        exit_reason = 'Stop Loss'
                else:
                    if current_bar['High'] >= open_trade.stop_loss:
                        exit_price = open_trade.stop_loss
                        exit_reason = 'Stop Loss'
                
                # Take profit
                if exit_price is None:
                    if open_trade.direction > 0:
                        if current_bar['High'] >= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = 'Take Profit'
                    else:
                        if current_bar['Low'] <= open_trade.take_profit:
                            exit_price = open_trade.take_profit
                            exit_reason = 'Take Profit'
                
                # Trailing stop
                if exit_price is None and self.config['use_trailing_stop']:
                    profit_pct = (current_price - open_trade.entry_price) / open_trade.entry_price * open_trade.direction
                    
                    if profit_pct > self.config['trailing_activation_pct']:
                        if open_trade.direction > 0:
                            new_stop = open_trade.entry_price * (1 + self.config['trailing_lock_profit_pct'])
                            open_trade.stop_loss = max(open_trade.stop_loss, new_stop)
                            
                            trail_stop = current_price * (1 - self.config['trailing_distance_pct'])
                            open_trade.stop_loss = max(open_trade.stop_loss, trail_stop)
                        else:
                            new_stop = open_trade.entry_price * (1 - self.config['trailing_lock_profit_pct'])
                            open_trade.stop_loss = min(open_trade.stop_loss, new_stop)
                            
                            trail_stop = current_price * (1 + self.config['trailing_distance_pct'])
                            open_trade.stop_loss = min(open_trade.stop_loss, trail_stop)
                
                # Exit on reversal
                if exit_price is None and current_bar['NTI_Direction'] * open_trade.direction < 0:
                    trend_strength = self.calculate_trend_strength(df, i)
                    if abs(trend_strength) >= 3 and trend_strength * open_trade.direction < 0:
                        exit_price = current_price
                        exit_reason = 'Trend Reversal'
                
                # Close trade if exit triggered
                if exit_price is not None:
                    open_trade = self._close_trade(open_trade, exit_price, current_time, exit_reason)
                    self.trades.append(open_trade)
                    open_trade = None
                    last_signal_direction = 0
                    bars_since_exit = 0
            
            # Check for new entry
            if open_trade is None and bars_since_exit >= self.config['min_bars_between_trades']:
                nti_dir = current_bar['NTI_Direction']
                mb_bias = current_bar['MB_Bias']
                
                if nti_dir != 0 and nti_dir != last_signal_direction:
                    trend_score = self.calculate_trend_strength(df, i)
                    
                    if abs(trend_score) >= self.config['min_trend_score']:
                        if (trend_score > 0 and nti_dir > 0) or (trend_score < 0 and nti_dir < 0):
                            if self.check_volatility_filter(df, i):
                                if mb_bias == nti_dir:
                                    if current_bar['IC_Signal'] != 0:
                                        # Calculate ATR
                                        atr = self._calculate_atr(df, i, period=20)
                                        
                                        direction = nti_dir
                                        
                                        # Wider stops for crypto
                                        atr_in_pct = atr / current_price
                                        sl_distance = max(
                                            self.config['min_stop_pct'],
                                            atr_in_pct * self.config['atr_multiplier_sl']
                                        )
                                        
                                        tp_distance = sl_distance * self.config['risk_reward_ratio']
                                        
                                        if direction > 0:
                                            sl_price = current_price * (1 - sl_distance)
                                            tp_price = current_price * (1 + tp_distance)
                                        else:
                                            sl_price = current_price * (1 + sl_distance)
                                            tp_price = current_price * (1 - tp_distance)
                                        
                                        position_size = self.calculate_position_size(current_price, sl_price)
                                        
                                        if position_size > 0:
                                            open_trade = CryptoTrade(
                                                entry_time=current_time,
                                                entry_price=current_price,
                                                position_size=position_size,
                                                direction=direction,
                                                stop_loss=sl_price,
                                                take_profit=tp_price
                                            )
                                            last_signal_direction = nti_dir
            
            # Update equity
            current_equity = self._calculate_current_equity(open_trade, current_price)
            self.equity_curve.append(current_equity)
        
        # Close final trade
        if open_trade is not None:
            open_trade = self._close_trade(
                open_trade, df.iloc[-1]['Close'], df.index[-1], 'End of Data'
            )
            self.trades.append(open_trade)
        
        return self._calculate_performance_metrics()
    
    def _close_trade(self, trade, exit_price, exit_time, exit_reason):
        """Close trade and calculate P&L"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        price_change_pct = (exit_price - trade.entry_price) / trade.entry_price
        trade.pnl_pct = price_change_pct * trade.direction * 100
        trade.pnl_dollars = trade.position_size * (exit_price - trade.entry_price) * trade.direction
        
        self.current_capital += trade.pnl_dollars
        
        return trade
    
    def _calculate_current_equity(self, open_trade, current_price):
        """Calculate equity including open position"""
        equity = self.current_capital
        
        if open_trade is not None:
            unrealized_pnl = open_trade.position_size * (current_price - open_trade.entry_price) * open_trade.direction
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_atr(self, df, i, period=14):
        """Calculate Average True Range"""
        if i < period:
            return df['High'].iloc[:i].mean() - df['Low'].iloc[:i].mean()
        
        high_low = df['High'].iloc[i-period:i] - df['Low'].iloc[i-period:i]
        return high_low.mean()
    
    def _calculate_performance_metrics(self):
        """Calculate crypto performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in self.trades if t.pnl_pct > 0]
        losing_trades = [t for t in self.trades if t.pnl_pct < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100
        
        # Returns
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[returns != 0]
        
        # Sharpe ratio (adjusted for crypto)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 96)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak * 100
        max_drawdown = abs(np.min(drawdown))
        
        # Total return
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0] * 100
        total_pnl = equity_array[-1] - equity_array[0]
        
        # Profit factor
        if losing_trades:
            gross_profits = sum(t.pnl_dollars for t in winning_trades) if winning_trades else 0
            gross_losses = abs(sum(t.pnl_dollars for t in losing_trades))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
        else:
            profit_factor = float('inf') if winning_trades else 0
        
        # Create a simple trade results list for compatibility
        trade_results = []
        for t in self.trades:
            trade_results.append({
                'pnl': t.pnl_dollars,
                'pnl_pct': t.pnl_pct,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction
            })
        
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'avg_win': np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0,
            'trades': trade_results  # Add this for compatibility
        }


def create_crypto_conservative_config():
    """Conservative crypto configuration - focus on high probability"""
    return {
        'initial_capital': 1_000_000,
        'risk_per_trade': 0.002,  # 0.2% risk
        'max_position_pct': 0.10,  # 10% max position
        
        # Stops and targets
        'min_stop_pct': 0.04,  # 4% minimum stop
        'atr_multiplier_sl': 3.0,  # 3x ATR for stop
        'risk_reward_ratio': 2.0,  # 2:1 RR minimum
        
        # Trailing stop
        'use_trailing_stop': True,
        'trailing_activation_pct': 0.03,  # Activate at 3%
        'trailing_lock_profit_pct': 0.01,  # Lock 1% profit
        'trailing_distance_pct': 0.02,  # Trail by 2%
        
        # Entry filters
        'min_trend_score': 3,  # Strong trend required (3/5)
        'min_bars_between_trades': 20,  # Space out trades
    }


def create_crypto_moderate_config():
    """Moderate crypto configuration - balanced approach"""
    return {
        'initial_capital': 1_000_000,
        'risk_per_trade': 0.0025,  # 0.25% risk
        'max_position_pct': 0.15,  # 15% max position
        
        # Stops and targets
        'min_stop_pct': 0.03,  # 3% minimum stop
        'atr_multiplier_sl': 2.5,  # 2.5x ATR
        'risk_reward_ratio': 1.5,  # 1.5:1 RR
        
        # Trailing stop
        'use_trailing_stop': True,
        'trailing_activation_pct': 0.02,  # Activate at 2%
        'trailing_lock_profit_pct': 0.005,  # Lock 0.5%
        'trailing_distance_pct': 0.015,  # Trail by 1.5%
        
        # Entry filters
        'min_trend_score': 2,  # Moderate trend
        'min_bars_between_trades': 10,
    }


def run_crypto_mode(crypto='ETHUSD', test_periods=None, save_results=False):
    """Run crypto strategy testing with specified periods"""
    
    print("="*80)
    print(f"CRYPTO STRATEGY MODE - {crypto}")
    print("Focus: Trend Following with Crypto-Specific Risk Management")
    print("="*80)
    
    # Default test periods if not specified
    if test_periods is None:
        test_periods = [
            ("2021 Bull Market", "2021-01-01", "2021-12-31"),
            ("2022 Bear Market", "2022-01-01", "2022-12-31"),
            ("2023-2024 Recovery", "2023-01-01", "2024-12-31"),
            ("Last 12 Months", "2024-01-01", "2024-12-31"),
            ("Full Period", None, None)  # Test entire dataset
        ]
    
    # Load crypto data - try multiple paths
    possible_paths = [
        f'crypto_data/{crypto}_MASTER_15M.csv',
        f'data/{crypto}_MASTER_15M.csv',
        f'../crypto_data/{crypto}_MASTER_15M.csv',
        f'../data/{crypto}_MASTER_15M.csv'
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Error: Crypto data file not found for {crypto}")
        print(f"Looked in: crypto_data/, data/, ../crypto_data/, and ../data/")
        return None
    
    print(f"\nLoading {crypto} data...")
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Add indicators
    print("Calculating indicators...")
    
    # Neuro Trend Intelligent
    print("  Calculating Neuro Trend Intelligent...")
    start_time = time.time()
    df = TIC.add_neuro_trend_intelligent(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed Neuro Trend Intelligent in {elapsed_time:.3f} seconds ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
    
    # Market Bias
    print("  Calculating Market Bias...")
    start_time = time.time()
    df = TIC.add_market_bias(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed Market Bias in {elapsed_time:.3f} seconds ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
    
    # Intelligent Chop
    print("  Calculating Intelligent Chop...")
    start_time = time.time()
    df = TIC.add_intelligent_chop(df)
    elapsed_time = time.time() - start_time
    print(f"  ✓ Completed Intelligent Chop in {elapsed_time:.3f} seconds ({len(df):,} rows, {len(df)/elapsed_time:,.0f} rows/sec)")
    
    print(f"Data ready: {len(df):,} rows from {df.index[0]} to {df.index[-1]}")
    
    # Test configurations
    configs = [
        ("Crypto Conservative", create_crypto_conservative_config()),
        ("Crypto Moderate", create_crypto_moderate_config())
    ]
    
    all_results = {}
    
    for config_name, config in configs:
        print(f"\n{'='*60}")
        print(f"Testing {config_name} Configuration")
        print(f"{'='*60}")
        
        config_results = {}
        
        for period_name, start_date, end_date in test_periods:
            # Get period data
            if start_date is None and end_date is None:
                period_df = df.copy()
            else:
                period_df = df[start_date:end_date].copy()
            
            if len(period_df) < 500:
                print(f"\n{period_name}: Insufficient data ({len(period_df)} rows)")
                continue
            
            # Run strategy
            strategy = FinalCryptoStrategy(config)
            
            try:
                metrics = strategy.run_backtest(period_df)
                config_results[period_name] = metrics
                
                # Calculate trade statistics
                trade_stats = calculate_trade_statistics(metrics)
                
                print(f"\n{period_name} ({len(period_df):,} rows):")
                print(f"  Sharpe: {metrics['sharpe_ratio']:>6.3f} | Return: {metrics['total_return']:>7.1f}% | P&L: ${metrics['total_pnl']:>10,.0f}")
                print(f"  WinRate: {metrics['win_rate']:>5.1f}% | Trades: {metrics['total_trades']:>4} ({trade_stats['num_wins']}W/{trade_stats['num_losses']}L) | MaxDD: {metrics['max_drawdown']:>5.1f}%")
                print(f"  MaxConsecW/L: {trade_stats['max_consecutive_wins']}/{trade_stats['max_consecutive_losses']} | PF: {metrics['profit_factor']:.2f}")
                
            except Exception as e:
                print(f"\n{period_name}: Error - {e}")
                continue
        
        # Calculate overall metrics
        if config_results:
            all_sharpes = [m['sharpe_ratio'] for m in config_results.values()]
            all_returns = [m['total_return'] for m in config_results.values()]
            positive_periods = sum(1 for r in all_returns if r > 0)
            
            print(f"\n{config_name} SUMMARY:")
            print(f"  Average Sharpe:    {np.mean(all_sharpes):.3f}")
            print(f"  Average Return:    {np.mean(all_returns):.1f}%")
            print(f"  Positive Periods:  {positive_periods}/{len(all_returns)}")
            print(f"  Best Period:       {max(config_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]}")
            print(f"  Worst Period:      {min(config_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]}")
        
        all_results[config_name] = config_results
    
    # Save results if requested
    if save_results and all_results:
        results_file = f'results/{crypto.lower()}_crypto_strategy_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Results saved to {results_file}")
    
    # Generate comparison plot
    if all_results and len(all_results) > 0:
        generate_crypto_comparison_plot(all_results, crypto)
    
    return all_results


def generate_crypto_comparison_plot(results, crypto):
    """Generate comparison plots for crypto strategies"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{crypto} - Crypto Strategy Analysis', fontsize=16)
    
    # Extract data for plotting
    configs = list(results.keys())
    periods = list(next(iter(results.values())).keys())
    
    # Plot 1: Sharpe Ratios by Period
    ax1.set_title('Sharpe Ratios by Period')
    for config in configs:
        sharpes = [results[config].get(period, {}).get('sharpe_ratio', 0) for period in periods]
        ax1.plot(periods, sharpes, marker='o', label=config, linewidth=2)
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Returns by Period
    ax2.set_title('Returns by Period')
    x = np.arange(len(periods))
    width = 0.35
    for i, config in enumerate(configs):
        returns = [results[config].get(period, {}).get('total_return', 0) for period in periods]
        ax2.bar(x + i*width, returns, width, label=config, alpha=0.8)
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Total Return (%)')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(periods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Win Rate Comparison
    ax3.set_title('Win Rate Comparison')
    for config in configs:
        win_rates = [results[config].get(period, {}).get('win_rate', 0) for period in periods]
        ax3.plot(periods, win_rates, marker='s', label=config, linewidth=2)
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Win Rate (%)')
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Risk Metrics
    ax4.set_title('Risk Metrics Comparison')
    metrics_data = []
    for config in configs:
        avg_dd = np.mean([results[config].get(p, {}).get('max_drawdown', 0) for p in periods])
        avg_pf = np.mean([results[config].get(p, {}).get('profit_factor', 0) for p in periods])
        avg_trades = np.mean([results[config].get(p, {}).get('total_trades', 0) for p in periods])
        metrics_data.append([avg_dd, avg_pf * 10, avg_trades / 10])  # Scale for visibility
    
    metrics_labels = ['Avg Max DD', 'Avg PF (x10)', 'Avg Trades (/10)']
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    for i, (config, data) in enumerate(zip(configs, metrics_data)):
        ax4.bar(x + i*width, data, width, label=config, alpha=0.8)
    
    ax4.set_ylabel('Value')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(metrics_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot to charts directory
    plot_filename = f'charts/{crypto.lower()}_crypto_strategy_comparison.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {plot_filename}")
    plt.close()


def run_debug_test(currency='AUDUSD', sample_size=1000):
    """Run a small debug test to examine decision logic"""
    print("="*80)
    print("DEBUG MODE - DETAILED TRADE DECISION ANALYSIS")
    print(f"Testing {currency} with {sample_size} rows to check for look-ahead bias")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data(currency)
    
    # Take a small sample from the middle of the dataset
    start_idx = len(df) // 2
    end_idx = start_idx + sample_size
    sample_df = df.iloc[start_idx:end_idx].copy()
    
    print(f"Using data from row {start_idx} to {end_idx}")
    print(f"Date range: {sample_df.index[0]} to {sample_df.index[-1]}")
    
    # Create debug-enabled strategy
    strategy = create_config_1_ultra_tight_risk(debug_mode=True, use_daily_sharpe=False)
    
    # Run backtest with debug output
    results = strategy.run_backtest(sample_df)
    
    print(f"\n=== DEBUG TEST RESULTS ===")
    print(f"Total trades: {results['total_trades']}")
    print(f"Win rate: {results['win_rate']:.1f}%")
    print(f"Total P&L: ${results['total_pnl']:,.0f}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    
    return results


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

🪙 CRYPTO STRATEGY TESTING:
  python run_Strategy.py --mode crypto
    → Test crypto strategies on ETHUSD (default)
  
  python run_Strategy.py --mode crypto --crypto BTCUSD
    → Test crypto strategies on Bitcoin
  
  python run_Strategy.py --mode crypto --save-plots
    → Test crypto and save comparison charts

📈 VISUALIZATION OPTIONS:
  python run_Strategy.py --show-plots
    → Display plots interactively (plots are NOT saved by default)
  
  python run_Strategy.py --save-plots
    → Save plots to PNG files (useful for reports)
  
  python run_Strategy.py --show-plots --save-plots
    → Both display and save plots
  
  python run_Strategy.py --no-plots
    → Skip all visualizations (fastest, analysis only)

⚡ PERFORMANCE OPTIONS:
  python run_Strategy.py --no-calendar
    → Skip calendar year analysis for faster results
  
  python run_Strategy.py --iterations 20 --sample-size 3000
    → Quick test with fewer iterations and smaller samples

═══════════════════════════════════════════════════════════════════════════════
                          WHAT THE FLAGS DO                                    
═══════════════════════════════════════════════════════════════════════════════

MODE SELECTION:
  --mode        Choose between 'single' (one currency), 'multi' (many currencies), 
                or 'crypto' (crypto strategies)
                Default: single

DATA PARAMETERS:
  --currency    Which currency pair to test (e.g., AUDUSD, GBPUSD, EURUSD)
                Default: AUDUSD
  
  --crypto      Which crypto pair to test in crypto mode (e.g., BTCUSD, ETHUSD)
                Default: ETHUSD
  
  --currencies  List of currencies for multi-mode testing
                Default: GBPUSD EURUSD USDJPY NZDUSD USDCAD
  
  --iterations  How many random samples to test (more = more reliable)
                Default: 50 (not used in crypto mode)
  
  --sample-size Number of data rows per test (more = longer time period)
                Default: 8000 (~3 months of 15-minute data, not used in crypto mode)

VISUALIZATION:
  --show-plots      Open charts in a window for interactive viewing
  --save-plots      Save charts as PNG files (default: OFF to avoid long waits)
  --no-plots        Skip all chart generation (fastest option)

ANALYSIS:
  --no-calendar     Skip year-by-year performance breakdown

═══════════════════════════════════════════════════════════════════════════════

💡 TIPS:
  • Use more iterations (100+) for production decisions
  • Larger sample sizes test strategy robustness over longer periods
  • Multi-currency mode helps identify which pairs work best
  • Calendar analysis reveals performance in different market conditions
  • Plot saving is OFF by default for speed - use --save-plots when needed
  • For large datasets (>50k rows), consider --no-plots for faster results

📧 Need help? Check the README.md for detailed documentation
"""
    
    parser = argparse.ArgumentParser(
        description='''
╔══════════════════════════════════════════════════════════════════════════════╗
║          MONTE CARLO STRATEGY TESTER - High Performance Trading             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Test proven trading strategies across Forex and Crypto markets:             ║
║                                                                              ║
║  FOREX STRATEGIES:                                                           ║
║  • Config 1: Ultra-Tight Risk Management (0.2% risk, 10 pip stops)         ║
║  • Config 2: Scalping Strategy (0.1% risk, 5 pip stops)                    ║
║                                                                              ║
║  CRYPTO STRATEGIES:                                                          ║
║  • Conservative: High probability trend following (0.2% risk, 4% stops)     ║
║  • Moderate: Balanced approach (0.25% risk, 3% stops)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
        ''',
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'multi', 'crypto', 'custom'],
                       help='Testing mode - "single": test one currency pair, "multi": test multiple pairs, "crypto": test crypto strategies (default: single)')
    parser.add_argument('--currency', type=str, default='AUDUSD',
                       help='Currency pair to test in single mode, e.g. GBPUSD, EURUSD (default: AUDUSD)')
    parser.add_argument('--crypto', type=str, default='ETHUSD',
                       help='Crypto pair to test in crypto mode, e.g. BTCUSD, ETHUSD (default: ETHUSD)')
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
    parser.add_argument('--save-plots', action='store_true',
                       help='Save charts as PNG files in results folder (default: disabled)')
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false',
                       help='Disable saving charts to PNG files (this is the default)')
    parser.add_argument('--realistic-costs', action='store_true', default=True,
                       help='Enable realistic trading costs with slippage (Entry: 0.5 pip, SL: 2 pips, TSL: 1 pip) (default: enabled)')
    parser.add_argument('--no-realistic-costs', dest='realistic_costs', action='store_false',
                       help='Disable realistic trading costs and use perfect fills')
    parser.add_argument('--debug', action='store_true',
                       help='Enable detailed decision logging to check for look-ahead bias')
    parser.add_argument('--no-daily-sharpe', action='store_true',
                       help='Disable daily resampling for Sharpe ratio calculation (use bar-level data instead)')
    parser.add_argument('--date-range', type=str, nargs=2, metavar=('START', 'END'),
                       help='Specify date range for testing (format: YYYY-MM-DD YYYY-MM-DD)')
    parser.add_argument('--export-trades', action='store_true', default=True,
                       help='Export detailed trade-by-trade CSV after backtest (default: enabled)')
    parser.add_argument('--no-export-trades', dest='export_trades', action='store_false',
                       help='Disable trade export to CSV')
    parser.add_argument('--version', '-v', action='version', 
                       version=f'Monte Carlo Strategy Tester v{__version__}')
    
    args = parser.parse_args()
    
    # Create results and charts directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    
    print(f"Starting Strategy Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Handle debug mode
    if args.debug:
        run_debug_test(currency=args.currency, sample_size=args.sample_size)
        return
    
    if args.mode == 'single':
        # Parse date range if provided
        date_range = None
        if args.date_range:
            try:
                start_date = pd.to_datetime(args.date_range[0])
                end_date = pd.to_datetime(args.date_range[1])
                date_range = (start_date, end_date)
            except Exception as e:
                print(f"Error parsing date range: {e}")
                print("Please use format: YYYY-MM-DD YYYY-MM-DD")
                return
        
        run_single_currency_mode(
            currency=args.currency,
            n_iterations=args.iterations,
            sample_size=args.sample_size,
            enable_plots=not args.no_plots,
            enable_calendar_analysis=not args.no_calendar,
            show_plots=args.show_plots,
            save_plots=args.save_plots,
            realistic_costs=args.realistic_costs,
            use_daily_sharpe=not args.no_daily_sharpe,
            date_range=date_range,
            export_trades=args.export_trades
        )
    
    elif args.mode == 'multi':
        run_multi_currency_mode(
            currencies=args.currencies,
            n_iterations=args.iterations,
            sample_size=args.sample_size,
            realistic_costs=args.realistic_costs,
            use_daily_sharpe=not args.no_daily_sharpe
        )
    
    elif args.mode == 'crypto':
        run_crypto_mode(
            crypto=args.crypto,
            save_results=args.save_plots  # Use same flag for consistency
        )
    
    elif args.mode == 'custom':
        # Custom mode - can be extended for specific use cases
        print("Custom mode - implement your specific testing logic here")
        # Example: Test specific date ranges, custom configurations, etc.
    
    print(f"\nTesting completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()