"""
Comprehensive multi-currency test of the winning momentum strategy
Tests on both recent data and full historical data
"""

import pandas as pd
import numpy as np
import json
from ultimate_optimizer import AdvancedBacktest
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class EnhancedBacktest(AdvancedBacktest):
    """Enhanced backtesting with detailed trade tracking"""
    
    def __init__(self, data: pd.DataFrame):
        super().__init__(data)
        self.trades_detail = []
        
    def strategy_momentum_enhanced(self, lookback: int = 20, 
                                 entry_z: float = 2.0,
                                 exit_z: float = 0.5) -> dict:
        """Enhanced momentum strategy with trade tracking"""
        df = self.data.copy()
        
        # Calculate momentum
        df['Momentum'] = df['Close'].pct_change(lookback)
        df['Mom_Mean'] = df['Momentum'].rolling(50).mean()
        df['Mom_Std'] = df['Momentum'].rolling(50).std()
        df['Mom_Z'] = (df['Momentum'] - df['Mom_Mean']) / df['Mom_Std']
        
        # Signals (mean reversion)
        df['Signal'] = 0
        df.loc[df['Mom_Z'] < -entry_z, 'Signal'] = 1  # Buy on extreme negative
        df.loc[df['Mom_Z'] > entry_z, 'Signal'] = -1  # Sell on extreme positive
        
        # Exit when momentum normalizes
        df.loc[abs(df['Mom_Z']) < exit_z, 'Signal'] = 0
        
        # Track trades
        self.trades_detail = []
        position = 0
        entry_idx = None
        entry_price = None
        
        for idx in range(len(df)):
            current_signal = df.iloc[idx]['Signal']
            
            # Entry
            if position == 0 and current_signal != 0:
                position = current_signal
                entry_idx = idx
                entry_price = df.iloc[idx]['Close']
                entry_time = df.index[idx]
                
            # Exit
            elif position != 0 and (current_signal == 0 or current_signal == -position):
                exit_idx = idx
                exit_price = df.iloc[idx]['Close']
                exit_time = df.index[idx]
                
                # Calculate trade details
                pnl_pct = (exit_price - entry_price) / entry_price * position * 100
                holding_bars = exit_idx - entry_idx
                
                # Determine entry timing within bar
                entry_hour = entry_time.hour
                entry_minute = entry_time.minute
                entry_timing = f"{entry_hour:02d}:{entry_minute:02d}"
                
                self.trades_detail.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': 'Long' if position > 0 else 'Short',
                    'pnl_pct': pnl_pct,
                    'holding_bars': holding_bars,
                    'holding_hours': holding_bars * 0.25,  # 15-minute bars
                    'entry_hour': entry_hour,
                    'entry_minute': entry_minute,
                    'entry_timing': entry_timing
                })
                
                # Reset position
                position = current_signal if current_signal != 0 else 0
                if position != 0:
                    entry_idx = idx
                    entry_price = df.iloc[idx]['Close']
                    entry_time = df.index[idx]
        
        # Calculate standard metrics
        metrics = self._calculate_metrics(df)
        
        # Add enhanced metrics
        if self.trades_detail:
            trades_df = pd.DataFrame(self.trades_detail)
            metrics['avg_holding_bars'] = trades_df['holding_bars'].mean()
            metrics['avg_holding_hours'] = trades_df['holding_hours'].mean()
            metrics['median_holding_hours'] = trades_df['holding_hours'].median()
            
            # Entry timing analysis
            entry_hours = trades_df['entry_hour'].value_counts()
            metrics['entry_hour_distribution'] = entry_hours.to_dict()
            
            # Most common entry times
            top_hours = entry_hours.nlargest(3)
            metrics['top_entry_hours'] = top_hours.to_dict()
            
            # Long vs Short holding periods
            long_trades = trades_df[trades_df['position'] == 'Long']
            short_trades = trades_df[trades_df['position'] == 'Short']
            
            if len(long_trades) > 0:
                metrics['avg_long_holding_hours'] = long_trades['holding_hours'].mean()
            else:
                metrics['avg_long_holding_hours'] = 0
                
            if len(short_trades) > 0:
                metrics['avg_short_holding_hours'] = short_trades['holding_hours'].mean()
            else:
                metrics['avg_short_holding_hours'] = 0
        else:
            metrics['avg_holding_bars'] = 0
            metrics['avg_holding_hours'] = 0
            metrics['median_holding_hours'] = 0
            metrics['entry_hour_distribution'] = {}
            metrics['top_entry_hours'] = {}
            metrics['avg_long_holding_hours'] = 0
            metrics['avg_short_holding_hours'] = 0
        
        return metrics


def test_currency_pair(currency_pair, data_path, lookback=40, entry_z=1.5, exit_z=0.5, 
                      test_period='last_50000', save_details=False):
    """Test the winning strategy on a specific currency pair"""
    
    print(f"\n{'='*60}")
    print(f"Testing {currency_pair} - {test_period}")
    print(f"{'='*60}")
    
    try:
        # Load data
        data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
        print(f"Total data points: {len(data):,}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        # Select test period
        if test_period == 'last_50000':
            test_data = data[-50000:] if len(data) > 50000 else data
        elif test_period == 'last_20000':
            test_data = data[-20000:] if len(data) > 20000 else data
        elif test_period == 'full':
            test_data = data
        else:
            # Assume it's a tuple with start and end dates
            if isinstance(test_period, tuple):
                start_date, end_date = test_period
                test_data = data[start_date:end_date]
            else:
                test_data = data
        
        print(f"Testing on {len(test_data):,} bars")
        print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Run enhanced backtest
        backtester = EnhancedBacktest(test_data)
        result = backtester.strategy_momentum_enhanced(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z
        )
        
        # Print results
        print(f"\nResults:")
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"Total Returns: {result['returns']:.1f}%")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Max Drawdown: {result['max_dd']:.1f}%")
        print(f"Total Trades: {result['trades']}")
        
        # Print enhanced metrics
        print(f"\nTrade Timing Metrics:")
        print(f"Average Holding Period: {result['avg_holding_hours']:.1f} hours ({result['avg_holding_bars']:.1f} bars)")
        print(f"Median Holding Period: {result['median_holding_hours']:.1f} hours")
        if result['avg_long_holding_hours'] > 0:
            print(f"Average Long Position Holding: {result['avg_long_holding_hours']:.1f} hours")
        if result['avg_short_holding_hours'] > 0:
            print(f"Average Short Position Holding: {result['avg_short_holding_hours']:.1f} hours")
        
        # Entry timing analysis
        if result['top_entry_hours']:
            print(f"\nMost Common Entry Hours (UTC):")
            for hour, count in sorted(result['top_entry_hours'].items()):
                print(f"  Hour {hour:02d}:00 - {count} trades")
        
        return {
            'currency_pair': currency_pair,
            'test_period': test_period,
            'sharpe': result['sharpe'],
            'returns': result['returns'],
            'win_rate': result['win_rate'],
            'max_dd': result['max_dd'],
            'trades': result['trades'],
            'bars_tested': len(test_data),
            'test_range': f"{test_data.index[0]} to {test_data.index[-1]}",
            'years_tested': len(test_data) / (252 * 96),  # Approximate years (96 bars per day)
            'avg_holding_hours': result['avg_holding_hours'],
            'median_holding_hours': result['median_holding_hours'],
            'avg_long_holding_hours': result['avg_long_holding_hours'],
            'avg_short_holding_hours': result['avg_short_holding_hours'],
            'entry_hour_distribution': result['entry_hour_distribution'],
            'top_entry_hours': result['top_entry_hours']
        }
        
    except Exception as e:
        print(f"Error testing {currency_pair}: {str(e)}")
        return {
            'currency_pair': currency_pair,
            'test_period': test_period,
            'sharpe': None,
            'returns': None,
            'win_rate': None,
            'max_dd': None,
            'trades': None,
            'bars_tested': 0,
            'test_range': 'Error',
            'years_tested': 0,
            'avg_holding_hours': 0,
            'median_holding_hours': 0,
            'avg_long_holding_hours': 0,
            'avg_short_holding_hours': 0,
            'entry_hour_distribution': {},
            'top_entry_hours': {},
            'error': str(e)
        }


def run_comprehensive_test():
    """Run comprehensive multi-currency test on both recent and full data"""
    
    print("="*80)
    print("COMPREHENSIVE MULTI-CURRENCY BACKTEST - WINNING MOMENTUM STRATEGY")
    print("="*80)
    print(f"\nStrategy Parameters (from AUDUSD optimization):")
    print(f"Lookback: 40")
    print(f"Entry Z-Score: 1.5")
    print(f"Exit Z-Score: 0.5")
    print(f"Original AUDUSD Sharpe (last 50k): 1.286")
    
    # Define all currency pairs to test
    currency_pairs = {
        'AUDUSD': '../data/AUDUSD_MASTER_15M.csv',
        'AUDJPY': '../data/AUDJPY_MASTER_15M.csv',
        'AUDNZD': '../data/AUDNZD_MASTER_15M.csv',
        'GBPUSD': '../data/GBPUSD_MASTER_15M.csv',
        'USDCAD': '../data/USDCAD_MASTER_15M.csv',
        'NZDUSD': '../data/NZDUSD_MASTER_15M.csv',
        'EURUSD': '../data/EURUSD_MASTER_15M.csv',
        'EURGBP': '../data/EURGBP_MASTER_15M.csv'
    }
    
    # Test periods - only test on full data as requested
    test_periods = ['full']
    
    # Run tests
    all_results = []
    
    for period in test_periods:
        print(f"\n{'='*80}")
        print(f"TESTING PERIOD: {period.upper()}")
        print(f"{'='*80}")
        
        for pair, data_path in currency_pairs.items():
            result = test_currency_pair(
                currency_pair=pair,
                data_path=data_path,
                test_period=period,
                save_details=True
            )
            all_results.append(result)
    
    # Create comprehensive DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    results_df.to_csv('comprehensive_currency_test_results.csv', index=False)
    
    # Create summary reports for each test period
    for period in test_periods:
        period_results = results_df[results_df['test_period'] == period].copy()
        period_results = period_results.sort_values('sharpe', ascending=False, na_position='last')
        
        print(f"\n{'='*80}")
        print(f"SUMMARY - {period.upper()}")
        print(f"{'='*80}")
        print(f"\n{'Currency':<10} {'Sharpe':>8} {'Returns':>10} {'Win Rate':>10} {'Max DD':>10} {'Trades':>8} {'Avg Hold (hrs)':>15}")
        print("-"*95)
        
        for _, row in period_results.iterrows():
            if row['sharpe'] is not None:
                print(f"{row['currency_pair']:<10} {row['sharpe']:>8.3f} {row['returns']:>10.1f}% "
                      f"{row['win_rate']:>10.1f}% {row['max_dd']:>10.1f}% {row['trades']:>8} "
                      f"{row['avg_holding_hours']:>15.1f}")
            else:
                print(f"{row['currency_pair']:<10} {'Error':>8}")
        
        # Calculate statistics
        valid_results = period_results.dropna(subset=['sharpe'])
        if len(valid_results) > 0:
            print(f"\nStatistics for {period}:")
            print(f"Average Sharpe: {valid_results['sharpe'].mean():.3f}")
            print(f"Median Sharpe: {valid_results['sharpe'].median():.3f}")
            print(f"Std Dev Sharpe: {valid_results['sharpe'].std():.3f}")
            print(f"Min Sharpe: {valid_results['sharpe'].min():.3f}")
            print(f"Max Sharpe: {valid_results['sharpe'].max():.3f}")
            
            positive_sharpe = (valid_results['sharpe'] > 0).sum()
            above_one = (valid_results['sharpe'] > 1.0).sum()
            print(f"Positive Sharpe: {positive_sharpe}/{len(valid_results)} ({positive_sharpe/len(valid_results)*100:.1f}%)")
            print(f"Sharpe > 1.0: {above_one}/{len(valid_results)} ({above_one/len(valid_results)*100:.1f}%)")
            
            # Holding period statistics
            print(f"\nHolding Period Statistics:")
            print(f"Average Holding Period: {valid_results['avg_holding_hours'].mean():.1f} hours")
            print(f"Median Holding Period: {valid_results['median_holding_hours'].median():.1f} hours")
            print(f"Min Holding Period: {valid_results['avg_holding_hours'].min():.1f} hours")
            print(f"Max Holding Period: {valid_results['avg_holding_hours'].max():.1f} hours")
            
            # Analyze entry timing patterns
            print(f"\nEntry Timing Analysis:")
            all_entries = defaultdict(int)
            for _, row in valid_results.iterrows():
                if 'entry_hour_distribution' in row and row['entry_hour_distribution']:
                    for hour, count in row['entry_hour_distribution'].items():
                        all_entries[hour] += count
            
            if all_entries:
                sorted_hours = sorted(all_entries.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"Most Active Entry Hours (UTC):")
                for hour, count in sorted_hours:
                    print(f"  Hour {hour:02d}:00 - {count} total entries")
                
                # Check for bias patterns
                morning_entries = sum(count for hour, count in all_entries.items() if 6 <= hour < 12)
                afternoon_entries = sum(count for hour, count in all_entries.items() if 12 <= hour < 18)
                evening_entries = sum(count for hour, count in all_entries.items() if 18 <= hour < 24)
                night_entries = sum(count for hour, count in all_entries.items() if 0 <= hour < 6)
                
                total_entries = sum(all_entries.values())
                print(f"\nEntry Distribution by Time of Day:")
                print(f"  Morning (06:00-12:00): {morning_entries} ({morning_entries/total_entries*100:.1f}%)")
                print(f"  Afternoon (12:00-18:00): {afternoon_entries} ({afternoon_entries/total_entries*100:.1f}%)")
                print(f"  Evening (18:00-00:00): {evening_entries} ({evening_entries/total_entries*100:.1f}%)")
                print(f"  Night (00:00-06:00): {night_entries} ({night_entries/total_entries*100:.1f}%)")
    
    # Create comparison visualization
    create_comparison_plots(results_df)
    
    # Create holding period visualization
    create_holding_period_analysis(results_df)
    
    # Generate comprehensive report
    generate_comprehensive_report(results_df)
    
    return results_df


def create_comparison_plots(results_df):
    """Create comparison plots for different test periods"""
    
    # Separate results by period
    recent_results = results_df[results_df['test_period'] == 'last_50000'].dropna(subset=['sharpe'])
    full_results = results_df[results_df['test_period'] == 'full'].dropna(subset=['sharpe'])
    
    if len(recent_results) == 0 or len(full_results) == 0:
        print("Insufficient data for comparison plots")
        return
    
    # Ensure same currency order for comparison
    currencies = recent_results['currency_pair'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Sharpe Ratio comparison
    ax1 = axes[0, 0]
    x = np.arange(len(currencies))
    width = 0.35
    
    recent_sharpes = recent_results.set_index('currency_pair').loc[currencies]['sharpe'].values
    full_sharpes = full_results.set_index('currency_pair').loc[currencies]['sharpe'].values
    
    ax1.bar(x - width/2, recent_sharpes, width, label='Last 50k bars', alpha=0.8)
    ax1.bar(x + width/2, full_sharpes, width, label='Full history', alpha=0.8)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Currency Pair')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(currencies, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns comparison
    ax2 = axes[0, 1]
    recent_returns = recent_results.set_index('currency_pair').loc[currencies]['returns'].values
    full_returns = full_results.set_index('currency_pair').loc[currencies]['returns'].values
    
    ax2.bar(x - width/2, recent_returns, width, label='Last 50k bars', alpha=0.8)
    ax2.bar(x + width/2, full_returns, width, label='Full history', alpha=0.8)
    ax2.set_xlabel('Currency Pair')
    ax2.set_ylabel('Total Returns (%)')
    ax2.set_title('Returns Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(currencies, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Consistency plot (Sharpe recent vs full)
    ax3 = axes[1, 0]
    ax3.scatter(full_sharpes, recent_sharpes, s=100, alpha=0.6)
    for i, txt in enumerate(currencies):
        ax3.annotate(txt, (full_sharpes[i], recent_sharpes[i]), fontsize=8)
    
    # Add diagonal line
    min_val = min(min(full_sharpes), min(recent_sharpes))
    max_val = max(max(full_sharpes), max(recent_sharpes))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    ax3.set_xlabel('Sharpe Ratio (Full History)')
    ax3.set_ylabel('Sharpe Ratio (Last 50k bars)')
    ax3.set_title('Strategy Consistency', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance degradation
    ax4 = axes[1, 1]
    degradation = ((recent_sharpes - full_sharpes) / full_sharpes * 100)
    colors = ['green' if x >= 0 else 'red' for x in degradation]
    ax4.bar(currencies, degradation, color=colors, alpha=0.7)
    ax4.set_ylabel('Performance Change (%)')
    ax4.set_title('Recent vs Full History Performance', fontsize=14)
    ax4.set_xticklabels(currencies, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_currency_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plots saved to comprehensive_currency_comparison.png")
    plt.show()


def create_holding_period_analysis(results_df):
    """Create visualizations for holding period analysis"""
    
    # Only use full period data
    full_results = results_df[results_df['test_period'] == 'full'].dropna(subset=['sharpe'])
    
    if len(full_results) == 0:
        print("Insufficient data for holding period analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Holding periods by currency
    ax1 = axes[0, 0]
    currencies = full_results['currency_pair'].values
    holding_periods = full_results['avg_holding_hours'].values
    
    bars = ax1.bar(currencies, holding_periods, alpha=0.7)
    ax1.set_xlabel('Currency Pair')
    ax1.set_ylabel('Average Holding Period (hours)')
    ax1.set_title('Average Holding Period by Currency', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Color bars based on performance
    for i, (bar, sharpe) in enumerate(zip(bars, full_results['sharpe'].values)):
        if sharpe > 1.0:
            bar.set_color('green')
        elif sharpe > 0:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    # 2. Holding period vs Sharpe scatter
    ax2 = axes[0, 1]
    ax2.scatter(holding_periods, full_results['sharpe'].values, s=100, alpha=0.6)
    for i, txt in enumerate(currencies):
        ax2.annotate(txt, (holding_periods[i], full_results['sharpe'].values[i]), fontsize=8)
    
    ax2.set_xlabel('Average Holding Period (hours)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Holding Period vs Performance', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Entry hour distribution (aggregate)
    ax3 = axes[1, 0]
    all_entries = defaultdict(int)
    for _, row in full_results.iterrows():
        if 'entry_hour_distribution' in row and row['entry_hour_distribution']:
            for hour, count in row['entry_hour_distribution'].items():
                all_entries[hour] += count
    
    if all_entries:
        hours = list(range(24))
        counts = [all_entries.get(h, 0) for h in hours]
        
        ax3.bar(hours, counts, alpha=0.7, color='blue')
        ax3.set_xlabel('Hour of Day (UTC)')
        ax3.set_ylabel('Number of Entries')
        ax3.set_title('Entry Hour Distribution (All Currencies)', fontsize=14)
        ax3.set_xticks(range(0, 24, 3))
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Long vs Short holding comparison
    ax4 = axes[1, 1]
    long_holdings = []
    short_holdings = []
    labels = []
    
    for _, row in full_results.iterrows():
        if row['avg_long_holding_hours'] > 0 or row['avg_short_holding_hours'] > 0:
            labels.append(row['currency_pair'])
            long_holdings.append(row['avg_long_holding_hours'])
            short_holdings.append(row['avg_short_holding_hours'])
    
    if labels:
        x = np.arange(len(labels))
        width = 0.35
        
        ax4.bar(x - width/2, long_holdings, width, label='Long Positions', alpha=0.8, color='green')
        ax4.bar(x + width/2, short_holdings, width, label='Short Positions', alpha=0.8, color='red')
        
        ax4.set_xlabel('Currency Pair')
        ax4.set_ylabel('Average Holding Period (hours)')
        ax4.set_title('Long vs Short Position Holding Periods', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('holding_period_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nHolding period analysis saved to holding_period_analysis.png")
    plt.show()


def generate_comprehensive_report(results_df):
    """Generate a comprehensive report of all tests"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("COMPREHENSIVE MULTI-CURRENCY STRATEGY TEST REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\nStrategy: Momentum (Z-Score Mean Reversion)")
    report_lines.append("Parameters:")
    report_lines.append("  - Lookback: 40 bars")
    report_lines.append("  - Entry Z-Score: 1.5")
    report_lines.append("  - Exit Z-Score: 0.5")
    
    # Analyze each test period
    for period in ['last_50000', 'full']:
        period_results = results_df[results_df['test_period'] == period].dropna(subset=['sharpe'])
        
        if len(period_results) == 0:
            continue
            
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"ANALYSIS: {period.upper()}")
        report_lines.append(f"{'='*80}")
        
        # Best and worst
        best = period_results.nlargest(1, 'sharpe').iloc[0]
        worst = period_results.nsmallest(1, 'sharpe').iloc[0]
        
        report_lines.append(f"\nBest Performer: {best['currency_pair']}")
        report_lines.append(f"  - Sharpe: {best['sharpe']:.3f}")
        report_lines.append(f"  - Returns: {best['returns']:.1f}%")
        report_lines.append(f"  - Avg Holding Period: {best['avg_holding_hours']:.1f} hours")
        report_lines.append(f"  - Years tested: {best['years_tested']:.1f}")
        
        report_lines.append(f"\nWorst Performer: {worst['currency_pair']}")
        report_lines.append(f"  - Sharpe: {worst['sharpe']:.3f}")
        report_lines.append(f"  - Returns: {worst['returns']:.1f}%")
        report_lines.append(f"  - Avg Holding Period: {worst['avg_holding_hours']:.1f} hours")
        report_lines.append(f"  - Years tested: {worst['years_tested']:.1f}")
        
        # Statistics
        report_lines.append(f"\nStatistics:")
        report_lines.append(f"  - Average Sharpe: {period_results['sharpe'].mean():.3f}")
        report_lines.append(f"  - Sharpe Std Dev: {period_results['sharpe'].std():.3f}")
        report_lines.append(f"  - Positive Sharpe: {(period_results['sharpe'] > 0).sum()}/{len(period_results)}")
        report_lines.append(f"  - Sharpe > 1.0: {(period_results['sharpe'] > 1.0).sum()}/{len(period_results)}")
        
        # Holding period statistics
        report_lines.append(f"\nHolding Period Analysis:")
        report_lines.append(f"  - Average Holding Period: {period_results['avg_holding_hours'].mean():.1f} hours")
        report_lines.append(f"  - Range: {period_results['avg_holding_hours'].min():.1f} - {period_results['avg_holding_hours'].max():.1f} hours")
        
        # Entry timing summary
        all_entries = defaultdict(int)
        for _, row in period_results.iterrows():
            if 'entry_hour_distribution' in row and row['entry_hour_distribution']:
                for hour, count in row['entry_hour_distribution'].items():
                    all_entries[hour] += count
        
        if all_entries:
            sorted_hours = sorted(all_entries.items(), key=lambda x: x[1], reverse=True)[:3]
            report_lines.append(f"\nTop Entry Hours (UTC):")
            for hour, count in sorted_hours:
                report_lines.append(f"  - Hour {hour:02d}:00 - {count} entries")
    
    # Consistency analysis
    report_lines.append(f"\n{'='*80}")
    report_lines.append("CONSISTENCY ANALYSIS")
    report_lines.append(f"{'='*80}")
    
    # Compare recent vs full for each currency
    currencies = results_df['currency_pair'].unique()
    consistency_data = []
    
    for currency in currencies:
        recent = results_df[(results_df['currency_pair'] == currency) & 
                           (results_df['test_period'] == 'last_50000')]['sharpe'].values
        full = results_df[(results_df['currency_pair'] == currency) & 
                         (results_df['test_period'] == 'full')]['sharpe'].values
        
        if len(recent) > 0 and len(full) > 0 and not pd.isna(recent[0]) and not pd.isna(full[0]):
            consistency_data.append({
                'currency': currency,
                'recent_sharpe': recent[0],
                'full_sharpe': full[0],
                'difference': recent[0] - full[0],
                'pct_change': (recent[0] - full[0]) / full[0] * 100 if full[0] != 0 else 0
            })
    
    consistency_df = pd.DataFrame(consistency_data)
    
    if len(consistency_df) > 0:
        report_lines.append("\nPerformance consistency (Recent vs Full History):")
        for _, row in consistency_df.iterrows():
            report_lines.append(f"  {row['currency']}: Recent={row['recent_sharpe']:.3f}, "
                              f"Full={row['full_sharpe']:.3f}, Change={row['pct_change']:.1f}%")
        
        # Overall assessment
        avg_recent = consistency_df['recent_sharpe'].mean()
        avg_full = consistency_df['full_sharpe'].mean()
        
        report_lines.append(f"\nAverage Sharpe (Recent): {avg_recent:.3f}")
        report_lines.append(f"Average Sharpe (Full): {avg_full:.3f}")
        
        if avg_recent > avg_full * 0.8:
            report_lines.append("\nConsistency Assessment: GOOD")
            report_lines.append("  - Recent performance is consistent with historical performance")
        else:
            report_lines.append("\nConsistency Assessment: CONCERNING")
            report_lines.append("  - Significant performance degradation in recent period")
    
    # Calculate overall metrics first
    all_valid = results_df.dropna(subset=['sharpe'])
    
    # Entry timing bias analysis
    report_lines.append(f"\n{'='*80}")
    report_lines.append("ENTRY TIMING BIAS ANALYSIS")
    report_lines.append(f"{'='*80}")
    
    # Aggregate all entry hours across all currencies
    total_entries = defaultdict(int)
    for _, row in all_valid.iterrows():
        if 'entry_hour_distribution' in row and row['entry_hour_distribution']:
            for hour, count in row['entry_hour_distribution'].items():
                total_entries[hour] += count
    
    if total_entries:
        report_lines.append("\nGlobal Entry Hour Distribution (UTC):")
        sorted_all_hours = sorted(total_entries.items())
        for hour, count in sorted_all_hours:
            report_lines.append(f"  Hour {hour:02d}:00 - {count} entries")
        
        # Time of day analysis
        morning = sum(count for hour, count in total_entries.items() if 6 <= hour < 12)
        afternoon = sum(count for hour, count in total_entries.items() if 12 <= hour < 18)
        evening = sum(count for hour, count in total_entries.items() if 18 <= hour < 24)
        night = sum(count for hour, count in total_entries.items() if 0 <= hour < 6)
        
        total = sum(total_entries.values())
        report_lines.append(f"\nTime of Day Distribution:")
        report_lines.append(f"  Morning (06:00-12:00 UTC): {morning} entries ({morning/total*100:.1f}%)")
        report_lines.append(f"  Afternoon (12:00-18:00 UTC): {afternoon} entries ({afternoon/total*100:.1f}%)")
        report_lines.append(f"  Evening (18:00-00:00 UTC): {evening} entries ({evening/total*100:.1f}%)")
        report_lines.append(f"  Night (00:00-06:00 UTC): {night} entries ({night/total*100:.1f}%)")
        
        # Bias assessment
        expected_pct = 25.0  # Expected percentage for each period if uniform
        max_deviation = max(abs(morning/total*100 - expected_pct),
                           abs(afternoon/total*100 - expected_pct),
                           abs(evening/total*100 - expected_pct),
                           abs(night/total*100 - expected_pct))
        
        if max_deviation < 5:
            report_lines.append("\nEntry Timing Bias Assessment: LOW")
            report_lines.append("  - Entries are relatively evenly distributed throughout the day")
        elif max_deviation < 10:
            report_lines.append("\nEntry Timing Bias Assessment: MODERATE")
            report_lines.append("  - Some preference for certain times of day")
        else:
            report_lines.append("\nEntry Timing Bias Assessment: HIGH")
            report_lines.append("  - Strong bias towards specific trading hours")
    
    # Final recommendations
    report_lines.append(f"\n{'='*80}")
    report_lines.append("FINAL ASSESSMENT & RECOMMENDATIONS")
    report_lines.append(f"{'='*80}")
    
    if len(all_valid) > 0:
        overall_positive = (all_valid['sharpe'] > 0).sum() / len(all_valid)
        overall_above_one = (all_valid['sharpe'] > 1.0).sum() / len(all_valid)
        
        if overall_positive > 0.8 and overall_above_one > 0.3:
            report_lines.append("\nStrategy Robustness: EXCELLENT")
            report_lines.append("  - Consistently profitable across multiple currencies and time periods")
            report_lines.append("  - NOT overfitted to AUDUSD")
            report_lines.append("\nRecommendations:")
            report_lines.append("  1. Deploy strategy across multiple currency pairs")
            report_lines.append("  2. Use dynamic position sizing based on individual pair performance")
            report_lines.append("  3. Focus allocation on top performers")
            report_lines.append(f"  4. Average holding period is {all_valid['avg_holding_hours'].mean():.1f} hours - ensure sufficient capital for overnight positions")
            report_lines.append("  5. Monitor entry timing patterns for potential market microstructure effects")
        elif overall_positive > 0.6:
            report_lines.append("\nStrategy Robustness: MODERATE")
            report_lines.append("  - Mixed performance across currencies")
            report_lines.append("  - Shows some generalization beyond AUDUSD")
            report_lines.append("\nRecommendations:")
            report_lines.append("  1. Use strategy selectively on best-performing pairs")
            report_lines.append("  2. Consider pair-specific parameter optimization")
            report_lines.append("  3. Implement additional filters for weak performers")
        else:
            report_lines.append("\nStrategy Robustness: POOR")
            report_lines.append("  - Inconsistent performance")
            report_lines.append("  - May be overfitted to specific market conditions")
    
    # Save report
    report_content = '\n'.join(report_lines)
    with open('comprehensive_currency_test_report.txt', 'w') as f:
        f.write(report_content)
    
    print(f"\n\nComprehensive report saved to comprehensive_currency_test_report.txt")


if __name__ == "__main__":
    # Run comprehensive test
    results = run_comprehensive_test()
    
    print("\n" + "="*80)
    print("Comprehensive Multi-Currency Test Complete!")
    print("="*80)
    print("\nFiles created:")
    print("  - comprehensive_currency_test_results.csv")
    print("  - comprehensive_currency_test_report.txt")
    print("  - comprehensive_currency_comparison.png")
    print("  - holding_period_analysis.png")