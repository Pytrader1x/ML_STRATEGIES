"""
Object-oriented multi-currency strategy analyzer using the backtesting framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import warnings
import argparse
import os
from typing import Dict, List, Optional

from backtesting import Backtester, MomentumStrategy

warnings.filterwarnings('ignore')


class MultiCurrencyAnalyzer:
    """Comprehensive multi-currency strategy analyzer"""
    
    def __init__(self, strategy_params: Optional[Dict] = None):
        self.strategy_params = strategy_params or {
            'lookback': 40,
            'entry_z': 1.5,
            'exit_z': 0.5
        }
        self.results = []
        self.currency_pairs = {
            'AUDUSD': '../data/AUDUSD_MASTER_15M.csv',
            'AUDJPY': '../data/AUDJPY_MASTER_15M.csv',
            'AUDNZD': '../data/AUDNZD_MASTER_15M.csv',
            'GBPUSD': '../data/GBPUSD_MASTER_15M.csv',
            'USDCAD': '../data/USDCAD_MASTER_15M.csv',
            'NZDUSD': '../data/NZDUSD_MASTER_15M.csv',
            'EURUSD': '../data/EURUSD_MASTER_15M.csv',
            'EURGBP': '../data/EURGBP_MASTER_15M.csv'
        }
        
        # Ensure output directories exist
        os.makedirs('charts', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def test_currency_pair(self, currency_pair: str, data_path: str, 
                          test_period: str = 'full') -> Dict:
        """Test the strategy on a specific currency pair"""
        
        print(f"\n{'='*60}")
        print(f"Testing {currency_pair} - {test_period}")
        print(f"{'='*60}")
        
        try:
            # Load data
            data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
            print(f"Total data points: {len(data):,}")
            print(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            # Select test period
            test_data = self._select_test_period(data, test_period)
            print(f"Testing on {len(test_data):,} bars")
            print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Create strategy and backtester
            strategy = MomentumStrategy(**self.strategy_params)
            backtester = Backtester(test_data)
            
            # Run backtest with trade tracking
            result = backtester.run_backtest(strategy, track_trades=True)
            
            # Print results
            self._print_results(result)
            
            # Store results with additional metadata
            result_data = {
                'currency_pair': currency_pair,
                'test_period': test_period,
                'bars_tested': len(test_data),
                'test_range': f"{test_data.index[0]} to {test_data.index[-1]}",
                'years_tested': len(test_data) / (252 * 96),
                'backtester': backtester,
                **result  # Include all metrics from backtest
            }
            
            return result_data
            
        except Exception as e:
            print(f"Error testing {currency_pair}: {str(e)}")
            return self._create_error_result(currency_pair, test_period, str(e))
    
    def _select_test_period(self, data: pd.DataFrame, test_period: str) -> pd.DataFrame:
        """Select appropriate test period from data"""
        if test_period == 'last_50000':
            return data[-50000:] if len(data) > 50000 else data
        elif test_period == 'last_20000':
            return data[-20000:] if len(data) > 20000 else data
        elif test_period == 'full':
            return data
        else:
            return data
    
    def _print_results(self, result: Dict):
        """Print backtest results"""
        print(f"\nResults:")
        print(f"Sharpe Ratio: {result['sharpe']:.3f}")
        print(f"Total Returns: {result['returns']:.1f}%")
        print(f"Win Rate: {result['win_rate']:.1f}%")
        print(f"Max Drawdown: {result['max_dd']:.1f}%")
        print(f"Total Trades: {result['trades']}")
        
        # Print enhanced metrics if available
        if 'avg_holding_hours' in result:
            print(f"\nTrade Timing Metrics:")
            print(f"Average Holding Period: {result['avg_holding_hours']:.1f} hours")
            print(f"Median Holding Period: {result['median_holding_hours']:.1f} hours")
            
            if result.get('avg_long_holding_hours', 0) > 0:
                print(f"Average Long Position Holding: {result['avg_long_holding_hours']:.1f} hours")
            if result.get('avg_short_holding_hours', 0) > 0:
                print(f"Average Short Position Holding: {result['avg_short_holding_hours']:.1f} hours")
            
            # Entry timing analysis
            if result.get('top_entry_hours'):
                print(f"\nMost Common Entry Hours (UTC):")
                for hour, count in sorted(result['top_entry_hours'].items()):
                    print(f"  Hour {hour:02d}:00 - {count} trades")
    
    def _create_error_result(self, currency_pair: str, test_period: str, 
                           error_msg: str) -> Dict:
        """Create result dict for error cases"""
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
            'error': error_msg,
            'backtester': None
        }
    
    def run_comprehensive_test(self, test_periods: Optional[List[str]] = None) -> pd.DataFrame:
        """Run comprehensive multi-currency test"""
        
        test_periods = test_periods or ['full']
        
        self._print_header()
        
        # Run tests
        self.results = []
        
        for period in test_periods:
            print(f"\n{'='*80}")
            print(f"TESTING PERIOD: {period.upper()}")
            print(f"{'='*80}")
            
            for pair, data_path in self.currency_pairs.items():
                result = self.test_currency_pair(pair, data_path, period)
                self.results.append(result)
        
        # Create DataFrame (exclude backtester objects)
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'backtester'} 
            for r in self.results
        ])
        
        # Save and summarize results
        results_df.to_csv('results/comprehensive_currency_test_results.csv', index=False)
        self._generate_summary(results_df)
        
        return results_df
    
    def _print_header(self):
        """Print test header"""
        print("="*80)
        print("COMPREHENSIVE MULTI-CURRENCY BACKTEST - MOMENTUM STRATEGY")
        print("="*80)
        print(f"\nStrategy Parameters:")
        print(f"Lookback: {self.strategy_params['lookback']}")
        print(f"Entry Z-Score: {self.strategy_params['entry_z']}")
        print(f"Exit Z-Score: {self.strategy_params['exit_z']}")
    
    def _generate_summary(self, results_df: pd.DataFrame):
        """Generate summary statistics"""
        
        test_periods = results_df['test_period'].unique()
        
        for period in test_periods:
            period_results = results_df[results_df['test_period'] == period].copy()
            period_results = period_results.sort_values('sharpe', ascending=False, na_position='last')
            
            print(f"\n{'='*80}")
            print(f"SUMMARY - {period.upper()}")
            print(f"{'='*80}")
            
            # Print table header
            print(f"\n{'Currency':<10} {'Sharpe':>8} {'Returns':>10} {'Win Rate':>10} " +
                  f"{'Max DD':>10} {'Trades':>8} {'Avg Hold (hrs)':>15}")
            print("-"*95)
            
            # Print results
            for _, row in period_results.iterrows():
                if row['sharpe'] is not None:
                    holding_hrs = row.get('avg_holding_hours', 0)
                    print(f"{row['currency_pair']:<10} {row['sharpe']:>8.3f} " +
                          f"{row['returns']:>10.1f}% {row['win_rate']:>10.1f}% " +
                          f"{row['max_dd']:>10.1f}% {row['trades']:>8} " +
                          f"{holding_hrs:>15.1f}")
                else:
                    print(f"{row['currency_pair']:<10} {'Error':>8}")
            
            # Calculate statistics
            self._print_statistics(period_results)
    
    def _print_statistics(self, period_results: pd.DataFrame):
        """Print period statistics"""
        valid_results = period_results.dropna(subset=['sharpe'])
        
        if len(valid_results) == 0:
            return
            
        print(f"\nStatistics:")
        print(f"Average Sharpe: {valid_results['sharpe'].mean():.3f}")
        print(f"Median Sharpe: {valid_results['sharpe'].median():.3f}")
        print(f"Positive Sharpe: {(valid_results['sharpe'] > 0).sum()}/{len(valid_results)}")
        print(f"Sharpe > 1.0: {(valid_results['sharpe'] > 1.0).sum()}/{len(valid_results)}")
        
        # Holding period statistics if available
        if 'avg_holding_hours' in valid_results.columns:
            print(f"\nHolding Period Statistics:")
            print(f"Average: {valid_results['avg_holding_hours'].mean():.1f} hours")
            print(f"Range: {valid_results['avg_holding_hours'].min():.1f} - " +
                  f"{valid_results['avg_holding_hours'].max():.1f} hours")
    
    def show_plots(self, currency: str = 'AUDUSD'):
        """Show detailed plots for a specific currency"""
        
        # Find the result for the specified currency
        currency_result = None
        for result in self.results:
            if result['currency_pair'] == currency and result['backtester'] is not None:
                currency_result = result
                break
        
        if not currency_result:
            print(f"No results found for {currency}")
            return
        
        backtester = currency_result['backtester']
        if not backtester.trades:
            print(f"No trades found for {currency}")
            return
        
        # Get trades DataFrame
        trades_df = backtester.get_trades_df()
        
        # Create visualization with price data
        self._create_analysis_plots(trades_df, currency, backtester)
    
    def _create_analysis_plots(self, trades_df: pd.DataFrame, currency: str, backtester):
        """Create focused trading analysis with price chart and P&L metrics"""
        
        # Create 3-panel figure (3x1) - vertical layout
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
        fig.suptitle(f'{currency} Trading Analysis', fontsize=16)
        
        # 1. Price chart with trades (top panel)
        ax1 = axes[0]
        self._plot_price_with_trades_enhanced(ax1, trades_df, backtester, currency)
        
        # 2. Cumulative P&L (middle panel)
        ax2 = axes[1]
        trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
        ax2.plot(trades_df.index, trades_df['cumulative_pnl'], linewidth=2, color='blue')
        ax2.fill_between(trades_df.index, 0, trades_df['cumulative_pnl'], 
                        where=(trades_df['cumulative_pnl'] > 0), alpha=0.3, color='green')
        ax2.fill_between(trades_df.index, 0, trades_df['cumulative_pnl'], 
                        where=(trades_df['cumulative_pnl'] < 0), alpha=0.3, color='red')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L (%)')
        ax2.set_title('Cumulative Performance')
        ax2.grid(True, alpha=0.3)
        
        # Add final P&L text
        final_pnl = trades_df['cumulative_pnl'].iloc[-1]
        ax2.text(0.98, 0.95, f'Final P&L: {final_pnl:.1f}%', 
                transform=ax2.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Per-trade P&L (bottom panel)
        ax3 = axes[2]
        colors = ['green' if x > 0 else 'red' for x in trades_df['pnl_pct']]
        bars = ax3.bar(trades_df.index, trades_df['pnl_pct'], color=colors, alpha=0.7, width=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('P&L (%)')
        ax3.set_title('Individual Trade Performance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add win rate text
        win_rate = (trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (trades_df['pnl_pct'] < 0).any() else 0
        
        stats_text = f'Win Rate: {win_rate:.1f}% | Avg Win: {avg_win:.2f}% | Avg Loss: {avg_loss:.2f}%'
        ax3.text(0.5, 0.95, stats_text, transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        chart_path = f'charts/{currency}_detailed_analysis.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"\nDetailed analysis saved to {chart_path}")
        plt.show()
    
    def _plot_price_with_trades_enhanced(self, ax, trades_df: pd.DataFrame, backtester, currency: str):
        """Plot price chart with enhanced trade visualization and color-coded exits"""
        
        # Get signals DataFrame with price data
        df = backtester.signals_df
        
        # Limit to last 2000 bars for clarity
        plot_df = df.iloc[-2000:] if len(df) > 2000 else df
        
        # Plot price
        ax.plot(plot_df.index, plot_df['Close'], 'k-', linewidth=1, alpha=0.8, label='Price')
        
        # Get trades within the plot period
        plot_start = plot_df.index[0]
        plot_end = plot_df.index[-1]
        
        # Filter trades to plot period
        plot_trades = trades_df[
            (trades_df['entry_time'] >= plot_start) & 
            (trades_df['entry_time'] <= plot_end)
        ]
        
        # Plot trade entries and exits with enhanced visualization
        for _, trade in plot_trades.iterrows():
            try:
                entry_price = trade['entry_price']
                
                # Exit point
                if trade['exit_time'] <= plot_end:
                    exit_price = trade['exit_price']
                    
                    # Determine if it's a long or short trade based on position
                    if trade['position'] == 'Long':
                        marker_entry = '^'  # Up triangle for long entry
                        entry_color = 'darkgreen'
                    else:  # Short position
                        marker_entry = 'v'  # Down triangle for short entry
                        entry_color = 'darkred'
                    
                    # Color exit based on profit/loss
                    if trade['pnl_pct'] > 0:
                        exit_color = 'green'
                        edge_color = 'green'
                    else:
                        exit_color = 'red'
                        edge_color = 'red'
                    
                    # Draw entry marker (triangle)
                    ax.scatter(trade['entry_time'], entry_price, 
                              color=entry_color, marker=marker_entry, s=80, 
                              alpha=0.9, zorder=5, edgecolors='black', linewidth=1)
                    
                    # Draw exit marker (square, color-coded)
                    ax.scatter(trade['exit_time'], exit_price, 
                              color=exit_color, marker='s', s=80, 
                              alpha=0.9, zorder=5, edgecolors='black', linewidth=1)
                    
                    # Draw line connecting entry to exit
                    ax.plot([trade['entry_time'], trade['exit_time']], 
                           [entry_price, exit_price], 
                           color=edge_color, alpha=0.4, linewidth=1.5, linestyle='--')
            except:
                continue
        
        # Add position indicator (subtle colored background)
        if 'Position' in plot_df.columns:
            # Create position areas
            y_min = plot_df['Close'].min() * 0.995
            y_max = plot_df['Close'].max() * 1.005
            
            # Long positions
            long_mask = plot_df['Position'] > 0
            ax.fill_between(plot_df.index, y_min, y_max,
                           where=long_mask, alpha=0.05, color='green', label='Long Position')
            
            # Short positions
            short_mask = plot_df['Position'] < 0
            ax.fill_between(plot_df.index, y_min, y_max,
                           where=short_mask, alpha=0.05, color='red', label='Short Position')
        
        # Add legend with trade markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='k', lw=1, label='Price'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen', 
                   markersize=8, label='Buy Entry', markeredgecolor='black'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', 
                   markersize=8, label='Sell Entry', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=8, label='Profit Exit', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                   markersize=8, label='Loss Exit', markeredgecolor='black'),
        ]
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{currency} Price Chart with Trade Execution (Last 2000 bars)')
        ax.grid(True, alpha=0.3)
        ax.legend(handles=legend_elements, loc='best', fontsize=8)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add trade count text
        trade_count = len(plot_trades)
        winning_trades = (plot_trades['pnl_pct'] > 0).sum()
        ax.text(0.02, 0.98, f'Visible Trades: {trade_count} ({winning_trades} wins)', 
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def generate_report(self, results_df: pd.DataFrame):
        """Generate comprehensive report"""
        
        report_path = 'results/comprehensive_currency_test_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE MULTI-CURRENCY STRATEGY TEST REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nStrategy: Momentum (Z-Score Mean Reversion)\n")
            f.write("Parameters:\n")
            f.write(f"  - Lookback: {self.strategy_params['lookback']} bars\n")
            f.write(f"  - Entry Z-Score: {self.strategy_params['entry_z']}\n")
            f.write(f"  - Exit Z-Score: {self.strategy_params['exit_z']}\n")
            
            # Performance summary
            valid_results = results_df.dropna(subset=['sharpe'])
            if len(valid_results) > 0:
                f.write(f"\nOverall Performance:\n")
                f.write(f"  - Average Sharpe: {valid_results['sharpe'].mean():.3f}\n")
                f.write(f"  - Positive Sharpe: {(valid_results['sharpe'] > 0).sum()}/{len(valid_results)}\n")
                f.write(f"  - Sharpe > 1.0: {(valid_results['sharpe'] > 1.0).sum()}/{len(valid_results)}\n")
                
                if 'avg_holding_hours' in valid_results.columns:
                    f.write(f"  - Average Holding Period: {valid_results['avg_holding_hours'].mean():.1f} hours\n")
        
        print(f"\nReport saved to {report_path}")


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Multi-Currency Strategy Analyzer')
    parser.add_argument('--show-plots', type=str, default='AUDUSD',
                       help='Show detailed plots for specified currency (default: AUDUSD)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip showing plots')
    parser.add_argument('--test-period', type=str, default='full',
                       choices=['full', 'last_50000', 'last_20000'],
                       help='Test period to use (default: full)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MultiCurrencyAnalyzer()
    
    # Run comprehensive test
    results_df = analyzer.run_comprehensive_test([args.test_period])
    
    # Generate report
    analyzer.generate_report(results_df)
    
    # Show plots if requested
    if not args.no_plots:
        analyzer.show_plots(args.show_plots)
    
    print("\n" + "="*80)
    print("Multi-Currency Analysis Complete!")
    print("="*80)
    print("\nFiles created:")
    print("  - results/comprehensive_currency_test_results.csv")
    print("  - results/comprehensive_currency_test_report.txt")
    if not args.no_plots:
        print(f"  - charts/{args.show_plots}_detailed_analysis.png")


if __name__ == "__main__":
    main()