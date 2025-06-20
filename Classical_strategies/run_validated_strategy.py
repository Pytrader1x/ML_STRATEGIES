"""
Run the Validated Strategy - Production Ready Implementation
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import json
import argparse
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class ValidatedStrategyRunner:
    """Run the validated aggressive scalping strategy"""
    
    def __init__(self, currency_pair='AUDUSD', initial_capital=1_000_000, position_size_millions=1.0, show_plots=False, save_plots=False):
        self.currency_pair = currency_pair
        self.initial_capital = initial_capital
        self.position_size_millions = position_size_millions
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.df = None
        
    def load_data(self):
        """Load and prepare data"""
        data_path = 'data' if os.path.exists('data') else '../data'
        file_path = os.path.join(data_path, f'{self.currency_pair}_MASTER_15M.csv')
        
        print(f"Loading {self.currency_pair} data...")
        self.df = pd.read_csv(file_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        print(f"Data loaded: {len(self.df):,} rows from {self.df.index[0]} to {self.df.index[-1]}")
        
        # Calculate indicators
        print("Calculating indicators...")
        self.df = TIC.add_neuro_trend_intelligent(self.df)
        self.df = TIC.add_market_bias(self.df, ha_len=350, ha_len2=30)
        self.df = TIC.add_intelligent_chop(self.df)
        
        return self.df
    
    def create_strategy(self, verbose=None):
        """Create the validated strategy configuration"""
        config = OptimizedStrategyConfig(
            # Capital settings
            initial_capital=self.initial_capital,
            risk_per_trade=0.005,  # 0.5% risk per trade
            base_position_size_millions=self.position_size_millions,  # 1M or 2M AUD
            
            # Stop loss settings (tight scalping)
            sl_min_pips=3.0,
            sl_max_pips=10.0,
            sl_atr_multiplier=0.8,
            
            # Take profit settings (quick profits)
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            # Market regime adjustments
            tp_range_market_multiplier=0.4,
            tp_trend_market_multiplier=0.6,
            tp_chop_market_multiplier=0.3,
            
            # Trailing stop settings
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            trailing_atr_multiplier=0.8,
            
            # Exit logic (aggressive)
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=1.0,
            
            # Partial profit taking
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.3,
            partial_profit_size_percent=0.7,
            
            # Entry mode (relaxed for more trades)
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            # CRITICAL: Realistic execution
            realistic_costs=True,
            entry_slippage_pips=0.5,
            stop_loss_slippage_pips=2.0,
            trailing_stop_slippage_pips=1.0,
            take_profit_slippage_pips=0.0,
            
            # Other settings
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            verbose=verbose if verbose is not None else False,
            debug_decisions=False,
            use_daily_sharpe=True
        )
        
        return OptimizedProdStrategy(config)
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run backtest on specified period"""
        if self.df is None:
            self.load_data()
        
        # Filter data if dates specified
        test_df = self.df.copy()
        if start_date and end_date:
            test_df = test_df.loc[start_date:end_date]
            print(f"Testing period: {start_date} to {end_date}")
        elif start_date:
            test_df = test_df.loc[start_date:]
            print(f"Testing from: {start_date}")
        elif end_date:
            test_df = test_df.loc[:end_date]
            print(f"Testing until: {end_date}")
        else:
            print("Testing full dataset")
        
        print(f"Test data: {len(test_df):,} rows")
        
        # Create and run strategy
        strategy = self.create_strategy()
        result = strategy.run_backtest(test_df)
        
        return result, test_df
    
    def print_results(self, result, test_df=None, period_name=""):
        """Print formatted results and optionally show plots"""
        print("\n" + "="*80)
        print("🏆 STRATEGY PERFORMANCE RESULTS")
        if period_name:
            print(f"Period: {period_name}")
        print("="*80)
        
        # Key metrics
        print(f"\n📊 Key Performance Metrics:")
        print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio: {result.get('sortino_ratio', 0):.3f}")
        print(f"  Total Return: {result.get('total_return', 0):.2f}%")
        print(f"  Max Drawdown: {result.get('max_drawdown', 0):.2f}%")
        print(f"  Recovery Factor: {result.get('recovery_factor', 0):.2f}")
        
        print(f"\n💰 Trading Statistics:")
        print(f"  Position Size: {self.position_size_millions}M AUD")
        print(f"  Total Trades: {result.get('total_trades', 0):,}")
        print(f"  Win Rate: {result.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
        print(f"  Average Trade: ${result.get('avg_trade', 0):,.2f}")
        print(f"  Best Trade: ${result.get('best_trade', 0):,.2f}")
        print(f"  Worst Trade: ${result.get('worst_trade', 0):,.2f}")
        
        print(f"\n📈 Risk Metrics:")
        print(f"  Win/Loss Ratio: {result.get('win_loss_ratio', 0):.2f}")
        print(f"  Expectancy: ${result.get('expectancy', 0):.2f}")
        print(f"  SQN Score: {result.get('sqn', 0):.2f}")
        print(f"  Trades per Day: {result.get('trades_per_day', 0):.1f}")
        
        # Exit statistics
        if 'trades' in result and result['trades']:
            trades = result['trades']
            total_trades = len(trades)
            
            print(f"\n🎯 Exit Breakdown:")
            
            # Count exits by type
            exit_counts = {}
            for trade in trades:
                if hasattr(trade, 'exit_reason'):
                    exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else str(trade.exit_reason)
                    exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
            
            # Display exit breakdown
            for exit_type, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_trades * 100) if total_trades > 0 else 0
                print(f"  {exit_type}: {count} trades ({percentage:.1f}%)")
            
            # TP hit statistics
            tp1_hits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 1)
            tp2_hits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 2)
            tp3_hits = sum(1 for trade in trades if hasattr(trade, 'tp_hits') and trade.tp_hits >= 3)
            
            print(f"\n📍 Take Profit Statistics:")
            print(f"  TP1 Hit: {tp1_hits} trades ({tp1_hits/total_trades*100:.1f}%)")
            print(f"  TP2 Hit: {tp2_hits} trades ({tp2_hits/total_trades*100:.1f}%)")
            print(f"  TP3 Hit: {tp3_hits} trades ({tp3_hits/total_trades*100:.1f}%)")
        
        # Final P&L
        if 'total_pnl' in result:
            print(f"\n💵 Final Results:")
            print(f"  Starting Capital: ${self.initial_capital:,.0f}")
            print(f"  Ending Capital: ${self.initial_capital + result['total_pnl']:,.0f}")
            print(f"  Total P&L: ${result['total_pnl']:,.0f}")
        
        # Performance rating
        sharpe = result.get('sharpe_ratio', 0)
        if sharpe >= 2.0:
            rating = "🌟 EXCEPTIONAL"
        elif sharpe >= 1.5:
            rating = "⭐ EXCELLENT"
        elif sharpe >= 1.0:
            rating = "✅ VERY GOOD"
        elif sharpe >= 0.7:
            rating = "👍 GOOD"
        else:
            rating = "⚠️ NEEDS IMPROVEMENT"
        
        print(f"\n🎯 Performance Rating: {rating}")
        
        # Generate plot if requested and data is available
        if (self.show_plots or self.save_plots) and test_df is not None:
            self.generate_plot(test_df, result, period_name)
        
        print("="*80)
    
    def generate_plot(self, df, result, period_name=""):
        """Generate trading chart using the production plotting function"""
        print(f"\n📊 Generating trading chart...")
        
        try:
            # Create title with performance metrics
            title_lines = [
                f"Validated Aggressive Scalping Strategy - {self.currency_pair}",
                f"Period: {period_name}" if period_name else "",
                f"Sharpe: {result.get('sharpe_ratio', 0):.3f} | Return: {result.get('total_return', 0):.1f}% | P&L: ${result.get('total_pnl', 0):,.0f}"
            ]
            title = "\n".join([line for line in title_lines if line])
            
            # Generate the plot
            fig = plot_production_results(
                df=df,
                results=result,
                title=title,
                show_pnl=True,
                show=self.show_plots
            )
            
            # Save plot if requested
            if self.save_plots and fig is not None:
                # Ensure charts directory exists
                os.makedirs('charts', exist_ok=True)
                
                # Create filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                period_str = period_name.replace(" ", "_").replace(":", "").lower() if period_name else "test"
                filename = f'charts/validated_strategy_{self.currency_pair}_{period_str}_{timestamp}.png'
                
                # Save with high DPI
                fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
                print(f"  💾 Chart saved to: {filename}")
            
            # Close figure if not showing to free memory
            if not self.show_plots and fig is not None:
                plt.close(fig)
                
        except Exception as e:
            print(f"  ❌ Error generating plot: {str(e)}")
            print("  📊 Continuing without chart...")
    
    def run_monte_carlo(self, n_simulations=25, sample_size_days=90, validate=False):
        """Run Monte Carlo simulation with random contiguous samples
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            sample_size_days: Number of days in each sample
            validate: If True, use this as validation mode with stricter criteria
        """
        if self.df is None:
            self.load_data()
        
        print(f"\n🎲 MONTE CARLO {'VALIDATION' if validate else 'SIMULATION'}")
        print(f"="*80)
        print(f"Running {n_simulations} simulations with {sample_size_days}-day samples")
        print(f"Position Size: {self.position_size_millions}M AUD")
        if self.show_plots:
            print(f"📊 Will show plot for the last simulation")
        print(f"="*80)
        
        # Calculate sample size in bars (96 bars per day for 15-minute data)
        bars_per_day = 96
        sample_size = sample_size_days * bars_per_day
        
        # Ensure we have enough data
        min_required = sample_size + 1000  # Buffer for indicator calculation
        if len(self.df) < min_required:
            print(f"ERROR: Insufficient data. Need at least {min_required} bars, have {len(self.df)}")
            return None
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Storage for results
        all_results = []
        last_sample_df = None
        last_result = None
        
        # Run simulations
        for sim in range(n_simulations):
            # Select random starting point
            max_start_idx = len(self.df) - sample_size - 500  # Leave buffer
            start_idx = np.random.randint(500, max_start_idx)
            end_idx = start_idx + sample_size
            
            # Extract sample
            sample_df = self.df.iloc[start_idx:end_idx].copy()
            start_date = sample_df.index[0]
            end_date = sample_df.index[-1]
            
            # Print progress indicator
            if sim == 0:
                print(f"\nProgress: ", end='', flush=True)
            
            # Show progress dots
            if (sim + 1) % 5 == 0:
                print(f"{sim+1}", end='', flush=True)
                if (sim + 1) % 50 == 0:
                    print()  # New line every 50
                    if sim + 1 < n_simulations:
                        print("Progress: ", end='', flush=True)
                else:
                    print(".", end='', flush=True)
            
            # Print detailed info less frequently - every 25 simulations or for first/last
            if sim == 0 or sim == n_simulations - 1 or (sim + 1) % 25 == 0:
                if sim > 0:
                    print()  # New line before detailed output
                print(f"\nSimulation {sim+1}/{n_simulations}: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
            
            try:
                # Create and run strategy (suppress config printing)
                strategy = self.create_strategy(verbose=False)
                result = strategy.run_backtest(sample_df)
                
                # Store results with metadata
                result['sim_num'] = sim + 1
                result['start_date'] = start_date
                result['end_date'] = end_date
                result['days'] = (end_date - start_date).days
                all_results.append(result)
                
                # Store last sample for plotting
                if sim == n_simulations - 1:
                    last_sample_df = sample_df
                    last_result = result
                
                # Print detailed metrics only for milestone runs
                if sim == 0 or sim == n_simulations - 1 or (sim + 1) % 25 == 0:
                    print(f"  Sharpe: {result.get('sharpe_ratio', 0):>6.2f} | "
                          f"Return: {result.get('total_return', 0):>6.1f}% | "
                          f"Max DD: {result.get('max_drawdown', 0):>5.1f}% | "
                          f"Win Rate: {result.get('win_rate', 0):>5.1f}%")
                    print(f"  P&L: ${result.get('total_pnl', 0):>10,.0f} | "
                          f"Trades: {result.get('total_trades', 0):>4d} | "
                          f"PF: {result.get('profit_factor', 0):>4.2f} | "
                          f"Avg Trade: ${result.get('avg_trade', 0):>7,.0f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                continue
        
        # Ensure we end progress line
        if n_simulations > 0:
            print()  # Final newline after progress
        
        # Analyze results
        if all_results:
            self._analyze_monte_carlo_results(all_results, sample_size_days, validate=validate)
            
            # Show/save plot for last simulation if requested
            if (self.show_plots or self.save_plots) and last_sample_df is not None and last_result is not None:
                print(f"\n📊 {'Showing' if self.show_plots else 'Generating'} plot for last simulation (#{n_simulations})...")
                start_str = last_result['start_date'].strftime('%d %b %Y')
                end_str = last_result['end_date'].strftime('%d %b %Y')
                period_name = f"Monte Carlo Sample #{n_simulations}: {start_str} to {end_str}"
                self.generate_plot(last_sample_df, last_result, period_name)
        
        return all_results
    
    def _analyze_monte_carlo_results(self, results, sample_days, validate=False):
        """Analyze and display Monte Carlo results"""
        print("\n" + "="*80)
        print(f"📊 MONTE CARLO {'VALIDATION' if validate else 'ANALYSIS'}")
        print("="*80)
        
        # Convert to DataFrame for easy analysis
        df_results = pd.DataFrame([{
            'sim': r['sim_num'],
            'start_date': r['start_date'],
            'end_date': r['end_date'],
            'sharpe': r.get('sharpe_ratio', 0),
            'return': r.get('total_return', 0),
            'pnl': r.get('total_pnl', 0),
            'trades': r.get('total_trades', 0),
            'win_rate': r.get('win_rate', 0),
            'max_dd': r.get('max_drawdown', 0),
            'profit_factor': r.get('profit_factor', 0)
        } for r in results])
        
        # Calculate yearly metrics (scale from sample period to full year)
        scale_factor = 365 / sample_days
        df_results['annual_return'] = df_results['return'] * scale_factor
        df_results['annual_pnl'] = df_results['pnl'] * scale_factor
        
        # Summary statistics
        print(f"\n📈 PERFORMANCE STATISTICS ({len(df_results)} simulations):")
        print("-"*80)
        print(f"{'Metric':<20} {'Mean':>12} {'Std Dev':>12} {'Min':>12} {'Max':>12} {'Median':>12}")
        print("-"*80)
        
        metrics = [
            ('Sharpe Ratio', 'sharpe'),
            ('Return %', 'return'),
            ('Annual Return %', 'annual_return'),
            ('P&L ($)', 'pnl'),
            ('Annual P&L ($)', 'annual_pnl'),
            ('Win Rate %', 'win_rate'),
            ('Max Drawdown %', 'max_dd'),
            ('Profit Factor', 'profit_factor')
        ]
        
        for label, col in metrics:
            values = df_results[col]
            print(f"{label:<20} {values.mean():>12.2f} {values.std():>12.2f} "
                  f"{values.min():>12.2f} {values.max():>12.2f} {values.median():>12.2f}")
        
        # Distribution analysis
        print(f"\n📊 DISTRIBUTION ANALYSIS:")
        print("-"*80)
        sharpe_values = df_results['sharpe']
        print(f"Profitable Samples (Sharpe > 0): {(sharpe_values > 0).sum()} ({(sharpe_values > 0).sum()/len(sharpe_values)*100:.1f}%)")
        print(f"Good Performance (Sharpe > 0.7): {(sharpe_values > 0.7).sum()} ({(sharpe_values > 0.7).sum()/len(sharpe_values)*100:.1f}%)")
        print(f"Excellent (Sharpe > 1.0): {(sharpe_values > 1.0).sum()} ({(sharpe_values > 1.0).sum()/len(sharpe_values)*100:.1f}%)")
        print(f"Exceptional (Sharpe > 2.0): {(sharpe_values > 2.0).sum()} ({(sharpe_values > 2.0).sum()/len(sharpe_values)*100:.1f}%)")
        
        # Consistency metrics
        cv = sharpe_values.std() / sharpe_values.mean() if sharpe_values.mean() != 0 else float('inf')
        print(f"\n🎯 CONSISTENCY METRICS:")
        print("-"*80)
        print(f"Coefficient of Variation: {cv:.3f}")
        print(f"Success Rate (positive returns): {(df_results['return'] > 0).sum()/len(df_results)*100:.1f}%")
        print(f"Risk-Adjusted Annual Return: {sharpe_values.mean() * np.sqrt(252):.1f}%")
        
        # Time period coverage
        print(f"\n📅 TIME PERIOD COVERAGE:")
        print("-"*80)
        print(f"Earliest Sample: {df_results['start_date'].min().strftime('%Y-%m-%d')}")
        print(f"Latest Sample: {df_results['end_date'].max().strftime('%Y-%m-%d')}")
        print(f"Data Span: {(df_results['end_date'].max() - df_results['start_date'].min()).days} days")
        
        # Overall assessment
        avg_sharpe = sharpe_values.mean()
        avg_annual_return = df_results['annual_return'].mean()
        avg_annual_pnl = df_results['annual_pnl'].mean()
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print("="*80)
        print(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"Expected Annual Return: {avg_annual_return:.1f}%")
        print(f"Expected Annual P&L: ${avg_annual_pnl:,.0f}")
        print(f"Position Size: {self.position_size_millions}M AUD")
        
        if avg_sharpe >= 1.5:
            rating = "EXCELLENT - Strong edge across market conditions"
        elif avg_sharpe >= 1.0:
            rating = "VERY GOOD - Consistent profitable strategy"
        elif avg_sharpe >= 0.7:
            rating = "GOOD - Meets minimum requirements"
        elif avg_sharpe >= 0.5:
            rating = "ACCEPTABLE - But needs improvement"
        else:
            rating = "POOR - Strategy needs significant work"
        
        print(f"Strategy Rating: {rating}")
        
        # Validation criteria (if in validation mode)
        if validate:
            print(f"\n✅ VALIDATION CRITERIA:")
            print("="*80)
            
            # Define validation thresholds
            min_sharpe = 0.7
            min_success_rate = 60.0
            max_drawdown = 15.0
            min_profit_factor = 1.2
            
            # Check criteria
            passed_criteria = []
            failed_criteria = []
            
            # 1. Average Sharpe Ratio
            if avg_sharpe >= min_sharpe:
                passed_criteria.append(f"✓ Average Sharpe Ratio: {avg_sharpe:.3f} >= {min_sharpe}")
            else:
                failed_criteria.append(f"✗ Average Sharpe Ratio: {avg_sharpe:.3f} < {min_sharpe}")
            
            # 2. Success Rate
            success_rate = (df_results['return'] > 0).sum()/len(df_results)*100
            if success_rate >= min_success_rate:
                passed_criteria.append(f"✓ Success Rate: {success_rate:.1f}% >= {min_success_rate}%")
            else:
                failed_criteria.append(f"✗ Success Rate: {success_rate:.1f}% < {min_success_rate}%")
            
            # 3. Maximum Drawdown
            worst_dd = df_results['max_dd'].max()
            if worst_dd <= max_drawdown:
                passed_criteria.append(f"✓ Worst Drawdown: {worst_dd:.1f}% <= {max_drawdown}%")
            else:
                failed_criteria.append(f"✗ Worst Drawdown: {worst_dd:.1f}% > {max_drawdown}%")
            
            # 4. Average Profit Factor
            avg_pf = df_results['profit_factor'].mean()
            if avg_pf >= min_profit_factor:
                passed_criteria.append(f"✓ Average Profit Factor: {avg_pf:.2f} >= {min_profit_factor}")
            else:
                failed_criteria.append(f"✗ Average Profit Factor: {avg_pf:.2f} < {min_profit_factor}")
            
            # Print results
            for criterion in passed_criteria:
                print(criterion)
            for criterion in failed_criteria:
                print(criterion)
            
            # Overall validation result
            if len(failed_criteria) == 0:
                print(f"\n🎯 VALIDATION RESULT: PASSED ({len(passed_criteria)}/{len(passed_criteria)+len(failed_criteria)} criteria)")
                print("Strategy is validated for production use!")
            else:
                print(f"\n⚠️ VALIDATION RESULT: FAILED ({len(failed_criteria)}/{len(passed_criteria)+len(failed_criteria)} criteria)")
                print("Strategy needs improvement before production use.")
        
        # Print detailed results table
        print(f"\n📋 DETAILED RESULTS TABLE:")
        print("="*120)
        print(f"{'#':>3} | {'Start Date':>12} | {'End Date':>12} | {'Sharpe':>7} | {'Return %':>8} | {'P&L':>10} | {'Max DD %':>8} | {'Win %':>6} | {'Trades':>6} | {'PF':>5}")
        print("-"*120)
        
        for _, row in df_results.iterrows():
            start_str = row['start_date'].strftime('%d %b %Y')
            end_str = row['end_date'].strftime('%d %b %Y')
            print(f"{row['sim']:>3} | {start_str:>12} | {end_str:>12} | "
                  f"{row['sharpe']:>7.2f} | {row['return']:>8.1f} | "
                  f"${row['pnl']:>9,.0f} | {row['max_dd']:>8.1f} | "
                  f"{row['win_rate']:>6.1f} | {row['trades']:>6d} | "
                  f"{row['profit_factor']:>5.2f}")
        
        print("="*120)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'results/monte_carlo_{len(df_results)}_samples_{self.position_size_millions}M_{timestamp}.csv'
        os.makedirs('results', exist_ok=True)
        df_results.to_csv(results_file, index=False)
        print(f"\n💾 Detailed results saved to: {results_file}")
    
    def run_sequential_analysis(self, mode='yearly', start_year=None, end_year=None):
        """Run sequential analysis year-by-year or quarter-by-quarter
        
        Args:
            mode: 'yearly' or 'quarterly'
            start_year: Starting year (default: first year in data)
            end_year: Ending year (default: last year in data)
        """
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Determine year range
        data_start_year = self.df.index[0].year
        data_end_year = self.df.index[-1].year
        
        if start_year is None:
            start_year = data_start_year
        if end_year is None:
            end_year = data_end_year
            
        start_year = max(start_year, data_start_year)
        end_year = min(end_year, data_end_year)
        
        print(f"\n📊 SEQUENTIAL {mode.upper()} ANALYSIS")
        print(f"Period: {start_year} to {end_year}")
        print("="*80)
        
        all_results = []
        
        if mode == 'yearly':
            # Year-by-year analysis
            for year in range(start_year, end_year + 1):
                print(f"\n{'='*40}")
                print(f"YEAR {year}")
                print(f"{'='*40}")
                
                # Check if we have data for full year
                year_start = f"{year}-01-01"
                year_end = f"{year}-12-31"
                
                # Run backtest for this year
                try:
                    result, _ = self.run_backtest(year_start, year_end)
                    result['period'] = f"Year {year}"
                    result['start_date'] = year_start
                    result['end_date'] = year_end
                    all_results.append(result)
                    
                    # Display key metrics
                    print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                    print(f"Total Return: {result.get('total_return_pct', 0):.1f}%")
                    print(f"Max Drawdown: {result.get('max_drawdown_pct', 0):.1f}%")
                    print(f"Win Rate: {result.get('win_rate', 0):.1f}%")
                    print(f"Total Trades: {result.get('total_trades', 0)}")
                except Exception as e:
                    print(f"Error processing year {year}: {str(e)}")
                    
        elif mode == 'quarterly':
            # Quarter-by-quarter analysis
            quarters = ['Q1', 'Q2', 'Q3', 'Q4']
            quarter_months = {
                'Q1': ('01-01', '03-31'),
                'Q2': ('04-01', '06-30'),
                'Q3': ('07-01', '09-30'),
                'Q4': ('10-01', '12-31')
            }
            
            for year in range(start_year, end_year + 1):
                for quarter, (start_month, end_month) in quarter_months.items():
                    period_name = f"{year} {quarter}"
                    print(f"\n{'='*40}")
                    print(f"{period_name}")
                    print(f"{'='*40}")
                    
                    # Set dates
                    quarter_start = f"{year}-{start_month}"
                    quarter_end = f"{year}-{end_month}"
                    
                    # Run backtest for this quarter
                    try:
                        result, _ = self.run_backtest(quarter_start, quarter_end)
                        result['period'] = period_name
                        result['start_date'] = quarter_start
                        result['end_date'] = quarter_end
                        all_results.append(result)
                        
                        # Display key metrics
                        print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                        print(f"Total Return: {result.get('total_return_pct', 0):.1f}%")
                        print(f"Max Drawdown: {result.get('max_drawdown_pct', 0):.1f}%")
                        print(f"Win Rate: {result.get('win_rate', 0):.1f}%")
                        print(f"Total Trades: {result.get('total_trades', 0)}")
                    except Exception as e:
                        print(f"Error processing {period_name}: {str(e)}")
        
        # Summary statistics
        if all_results:
            print(f"\n{'='*80}")
            print(f"SUMMARY STATISTICS - {mode.upper()} ANALYSIS")
            print(f"{'='*80}")
            
            # Convert to DataFrame for easier analysis
            df_results = pd.DataFrame(all_results)
            
            # Calculate statistics
            avg_sharpe = df_results['sharpe_ratio'].mean()
            std_sharpe = df_results['sharpe_ratio'].std()
            avg_return = df_results['total_return_pct'].mean()
            avg_trades = df_results['total_trades'].mean()
            avg_win_rate = df_results['win_rate'].mean()
            
            print(f"\nAverage Sharpe Ratio: {avg_sharpe:.2f} (±{std_sharpe:.2f})")
            print(f"Average Return per {mode[:-2]}: {avg_return:.1f}%")
            print(f"Average Win Rate: {avg_win_rate:.1f}%")
            print(f"Average Trades per {mode[:-2]}: {avg_trades:.0f}")
            
            # Best and worst periods
            best_period = df_results.loc[df_results['sharpe_ratio'].idxmax()]
            worst_period = df_results.loc[df_results['sharpe_ratio'].idxmin()]
            
            print(f"\nBest Period: {best_period['period']} (Sharpe: {best_period['sharpe_ratio']:.2f})")
            print(f"Worst Period: {worst_period['period']} (Sharpe: {worst_period['sharpe_ratio']:.2f})")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f'results/sequential_{mode}_{start_year}_{end_year}_{timestamp}.csv'
            os.makedirs('results', exist_ok=True)
            df_results.to_csv(results_file, index=False)
            print(f"\n💾 Sequential analysis results saved to: {results_file}")
            
            # Plot if enabled
            if self.show_plots or self.save_plots:
                self._plot_sequential_results(df_results, mode)
                
        return all_results
    
    def _plot_sequential_results(self, df_results, mode):
        """Plot sequential analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Sequential {mode.capitalize()} Analysis Results', fontsize=16)
        
        # Sharpe Ratio over time
        ax = axes[0, 0]
        ax.plot(df_results.index, df_results['sharpe_ratio'], marker='o', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target (1.0)')
        ax.set_title('Sharpe Ratio Evolution')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Returns over time
        ax = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in df_results['total_return_pct']]
        ax.bar(df_results.index, df_results['total_return_pct'], color=colors, alpha=0.7)
        ax.set_title(f'Returns per {mode[:-2].capitalize()}')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        
        # Win Rate over time
        ax = axes[1, 0]
        ax.plot(df_results.index, df_results['win_rate'], marker='s', linewidth=2, color='orange')
        ax.axhline(y=65, color='r', linestyle='--', alpha=0.5, label='Target (65%)')
        ax.set_title('Win Rate Evolution')
        ax.set_ylabel('Win Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Trade count over time
        ax = axes[1, 1]
        ax.bar(df_results.index, df_results['total_trades'], alpha=0.7, color='purple')
        ax.set_title(f'Number of Trades per {mode[:-2].capitalize()}')
        ax.set_ylabel('Trade Count')
        ax.grid(True, alpha=0.3)
        
        # Add period labels
        for ax in axes.flatten():
            ax.set_xticks(df_results.index)
            ax.set_xticklabels([r['period'] for _, r in df_results.iterrows()], rotation=45, ha='right')
        
        plt.tight_layout()
        
        if self.save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'plots/sequential_{mode}_analysis_{timestamp}.png'
            os.makedirs('plots', exist_ok=True)
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 Plot saved to: {filename}")
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def run_monte_carlo_all_years(self, n_simulations=100, sample_size_days=90):
        """Run Monte Carlo simulation across all available years
        
        This method runs Monte Carlo on the entire dataset, sampling from all years
        """
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
            
        print(f"\n🎲 MONTE CARLO SIMULATION - ALL YEARS")
        print(f"Full dataset: {self.df.index[0].date()} to {self.df.index[-1].date()}")
        print(f"Simulations: {n_simulations}")
        print(f"Sample size: {sample_size_days} days per simulation")
        print("="*80)
        
        # Run standard Monte Carlo on full dataset
        self.run_monte_carlo(n_simulations=n_simulations, sample_size_days=sample_size_days)

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Run Validated Strategy")
    parser.add_argument('--show-plots', action='store_true', help='Display charts')
    parser.add_argument('--save-plots', action='store_true', help='Save charts to PNG files')
    parser.add_argument('--currency', default='AUDUSD', help='Currency pair to test')
    parser.add_argument('--capital', type=int, default=1000000, help='Initial capital')
    parser.add_argument('--position-size', type=int, choices=[1, 2], default=1, help='Position size in millions (1 or 2)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', choices=['2024', '2023', 'recent', 'last-quarter'], help='Predefined test period')
    parser.add_argument('--monte-carlo', type=int, help='Run Monte Carlo simulation with N random samples')
    parser.add_argument('--monte-carlo-all-years', action='store_true', help='Run Monte Carlo across all available years')
    parser.add_argument('--sequential', choices=['yearly', 'quarterly'], help='Run sequential analysis (yearly or quarterly)')
    parser.add_argument('--validate', action='store_true', help='Run Monte Carlo validation with strict criteria')
    parser.add_argument('--start-year', type=int, help='Start year for sequential analysis')
    parser.add_argument('--end-year', type=int, help='End year for sequential analysis')
    
    args = parser.parse_args()
    
    print("🚀 VALIDATED STRATEGY RUNNER")
    print("="*60)
    print(f"Currency: {args.currency}")
    print(f"Capital: ${args.capital:,}")
    print(f"Position Size: {args.position_size}M AUD")
    if args.show_plots:
        print("📊 Charts will be displayed")
    if args.save_plots:
        print("💾 Charts will be saved")
    
    # Create runner with plot settings
    runner = ValidatedStrategyRunner(
        currency_pair=args.currency, 
        initial_capital=args.capital,
        position_size_millions=float(args.position_size),
        show_plots=args.show_plots,
        save_plots=args.save_plots
    )
    
    # Check if validation mode
    if args.validate:
        # Run Monte Carlo validation with 50 samples
        runner.run_monte_carlo(n_simulations=50, sample_size_days=90, validate=True)
        return
    
    # Check if Monte Carlo mode
    if args.monte_carlo:
        # Run Monte Carlo simulation
        n_simulations = args.monte_carlo if args.monte_carlo > 0 else 25
        runner.run_monte_carlo(n_simulations=n_simulations, sample_size_days=90)
        return
    
    # Check if Monte Carlo all years mode
    if args.monte_carlo_all_years:
        # Run Monte Carlo across all available years
        runner.run_monte_carlo_all_years(n_simulations=100, sample_size_days=90)
        return
    
    # Check if sequential analysis mode
    if args.sequential:
        # Run sequential analysis
        runner.run_sequential_analysis(
            mode=args.sequential,
            start_year=args.start_year,
            end_year=args.end_year
        )
        return
    
    # Determine test period
    if args.period:
        if args.period == '2024':
            start_date, end_date = '2024-01-01', '2024-06-30'
            period_name = "2024 H1"
        elif args.period == '2023':
            start_date, end_date = '2023-01-01', '2023-12-31'
            period_name = "Full Year 2023"
        elif args.period == 'recent':
            start_date, end_date = '2024-04-01', '2024-06-30'
            period_name = "Recent Quarter"
        elif args.period == 'last-quarter':
            start_date, end_date = '2024-01-01', '2024-03-31'
            period_name = "Q1 2024"
    elif args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
        period_name = f"{start_date} to {end_date}"
    elif args.start_date:
        start_date, end_date = args.start_date, None
        period_name = f"From {start_date}"
    elif args.end_date:
        start_date, end_date = None, args.end_date
        period_name = f"Until {end_date}"
    else:
        # Default: test recent performance
        start_date, end_date = '2024-01-01', '2024-06-30'
        period_name = "Default: 2024 H1"
    
    # Run backtest
    print(f"\n🔄 Running backtest for: {period_name}")
    result, test_df = runner.run_backtest(start_date, end_date)
    
    # Display results with plots
    runner.print_results(result, test_df, period_name)
    
    # Save configuration for live trading
    config_data = {
        'strategy_name': 'Validated Aggressive Scalping',
        'currency_pair': args.currency,
        'timeframe': '15M',
        'risk_per_trade': 0.005,
        'sl_min_pips': 3.0,
        'sl_max_pips': 10.0,
        'tp_multipliers': [0.15, 0.25, 0.4],
        'tsl_activation_pips': 8.0,
        'relaxed_mode': True,
        'realistic_costs': True,
        'last_test_sharpe': result.get('sharpe_ratio', 0),
        'last_test_period': period_name,
        'last_test_date': datetime.now().isoformat()
    }
    
    with open('validated_strategy_config.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"\n💾 Strategy configuration saved to: validated_strategy_config.json")

def run_example_tests():
    """Run example tests without command line arguments"""
    print("🚀 VALIDATED STRATEGY RUNNER - EXAMPLE TESTS")
    print("="*60)
    
    # Create runner with plots enabled and 1M position size
    runner = ValidatedStrategyRunner('AUDUSD', initial_capital=1_000_000, position_size_millions=1.0, show_plots=True, save_plots=True)
    
    # Test recent performance
    print("\n1️⃣ Testing Recent Performance (2024 H1)...")
    result_2024, test_df_2024 = runner.run_backtest('2024-01-01', '2024-06-30')
    runner.print_results(result_2024, test_df_2024, "2024 H1")
    
    # Test full year
    print("\n2️⃣ Testing Full Year 2023...")
    result_2023, test_df_2023 = runner.run_backtest('2023-01-01', '2023-12-31')
    runner.print_results(result_2023, test_df_2023, "Full Year 2023")

if __name__ == "__main__":
    main()