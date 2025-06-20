"""
Institutional Trading Strategy - Investment Bank Grade
Implements professional position sizing and risk management
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

class InstitutionalStrategyRunner:
    """Run institutional-grade trading strategy with proper position sizing"""
    
    def __init__(self, currency_pair='AUDUSD', initial_capital=10_000_000, show_plots=False, save_plots=False):
        self.currency_pair = currency_pair
        self.initial_capital = initial_capital  # $10M for institutional trading
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
    
    def create_strategy(self):
        """Create institutional-grade strategy configuration"""
        config = OptimizedStrategyConfig(
            # Institutional capital settings
            initial_capital=self.initial_capital,
            risk_per_trade=0.002,  # 0.2% risk per trade (20K on 10M)
            
            # INSTITUTIONAL POSITION SIZING
            # Relaxed mode (1 indicator): 1M units
            # Standard mode (3 indicators): 2M units
            base_position_size_millions=2.0,  # Base size for high conviction
            relaxed_position_multiplier=0.5,  # 1M for relaxed entries
            
            # Professional stop loss settings
            sl_min_pips=5.0,   # Minimum 5 pips (not 3 - too tight)
            sl_max_pips=15.0,  # Maximum 15 pips (was 10)
            sl_atr_multiplier=1.0,
            
            # Institutional take profit settings
            tp_atr_multipliers=(0.3, 0.6, 1.0),  # More realistic targets
            max_tp_percent=0.01,  # 1% max move
            
            # Market regime adjustments
            tp_range_market_multiplier=0.5,
            tp_trend_market_multiplier=0.8,
            tp_chop_market_multiplier=0.4,
            
            # Professional trailing stop
            tsl_activation_pips=10.0,  # Activate after 10 pips (not 8)
            tsl_min_profit_pips=3.0,   # Guarantee 3 pips minimum
            trailing_atr_multiplier=1.0,
            tsl_initial_buffer_multiplier=1.5,
            
            # Exit logic
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=2.0,
            signal_flip_partial_exit_percent=0.5,  # Exit 50% on signal flip
            
            # IMPROVED PARTIAL PROFIT LOGIC
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.6,  # Take profit at 60% to TP1 (not 30% to SL)
            partial_profit_size_percent=0.4,  # Take 40% off (not 70% - too aggressive)
            
            # Enable intelligent sizing for confluence
            intelligent_sizing=True,
            confidence_thresholds=(40.0, 60.0, 80.0),
            size_multipliers=(0.5, 0.75, 1.0, 1.25),  # Scale with confidence
            tp_confidence_adjustment=True,
            
            # Volatility adjustments
            sl_volatility_adjustment=True,
            sl_range_market_multiplier=0.8,
            sl_trend_market_multiplier=1.2,
            
            # Entry modes
            relaxed_mode=True,  # Allow single indicator entries
            
            # Realistic execution costs
            realistic_costs=True,
            entry_slippage_pips=0.3,      # Tighter spreads for institutional
            stop_loss_slippage_pips=1.0,   # Better execution
            trailing_stop_slippage_pips=0.5,
            take_profit_slippage_pips=0.0,
            
            # Professional settings
            verbose=False,
            debug_decisions=False,
            use_daily_sharpe=True,
            intrabar_stop_on_touch=True
        )
        
        return OptimizedProdStrategy(config)
    
    def run_backtest(self, start_date=None, end_date=None, monte_carlo=False, n_simulations=100):
        """Run institutional backtest"""
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
        
        if monte_carlo:
            return self.run_monte_carlo(test_df, n_simulations)
        else:
            # Single run
            strategy = self.create_strategy()
            strategy.print_strategy_parameters()
            result = strategy.run_backtest(test_df)
            return result, test_df
    
    def run_monte_carlo(self, df, n_simulations=100):
        """Run Monte Carlo simulation"""
        print(f"\n🎲 MONTE CARLO SIMULATION - INSTITUTIONAL")
        print("="*80)
        print(f"Running {n_simulations} simulations with 90-day samples")
        print(f"Position Sizes: 1M (relaxed) / 2M (standard)")
        print("="*80)
        
        results = []
        sample_size = 8640  # 90 days of 15-min bars
        
        for i in range(n_simulations):
            # Random sample
            if len(df) > sample_size:
                start_idx = np.random.randint(0, len(df) - sample_size)
                sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
            else:
                sample_df = df.copy()
            
            # Run backtest
            strategy = self.create_strategy()
            if i == 0:  # Only print config once
                strategy.print_strategy_parameters()
            
            result = strategy.run_backtest(sample_df)
            
            # Store results
            results.append({
                'iteration': i + 1,
                'start_date': sample_df.index[0],
                'end_date': sample_df.index[-1],
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'total_return': result.get('total_return', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'win_rate': result.get('win_rate', 0),
                'total_trades': result.get('total_trades', 0),
                'profit_factor': result.get('profit_factor', 0),
                'total_pnl': result.get('total_pnl', 0)
            })
            
            # Progress update
            if (i + 1) % 10 == 0:
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
                avg_return = np.mean([r['total_return'] for r in results])
                print(f"Progress: {i+1}/{n_simulations} | Avg Sharpe: {avg_sharpe:.2f} | Avg Return: {avg_return:.1f}%")
        
        # Calculate summary statistics
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("📊 MONTE CARLO RESULTS - INSTITUTIONAL STRATEGY")
        print("="*80)
        
        print(f"\n🎯 Performance Metrics (n={n_simulations}):")
        print(f"  Sharpe Ratio:     {df_results['sharpe_ratio'].mean():.3f} ± {df_results['sharpe_ratio'].std():.3f}")
        print(f"  Total Return:     {df_results['total_return'].mean():.1f}% ± {df_results['total_return'].std():.1f}%")
        print(f"  Max Drawdown:     {df_results['max_drawdown'].mean():.1f}% ± {df_results['max_drawdown'].std():.1f}%")
        print(f"  Win Rate:         {df_results['win_rate'].mean():.1f}% ± {df_results['win_rate'].std():.1f}%")
        print(f"  Profit Factor:    {df_results['profit_factor'].mean():.2f} ± {df_results['profit_factor'].std():.2f}")
        
        print(f"\n📈 Profitability Analysis:")
        profitable = (df_results['total_pnl'] > 0).sum()
        print(f"  Profitable Runs:  {profitable}/{n_simulations} ({profitable/n_simulations*100:.1f}%)")
        print(f"  Average P&L:      ${df_results['total_pnl'].mean():,.0f}")
        print(f"  Best Run:         ${df_results['total_pnl'].max():,.0f}")
        print(f"  Worst Run:        ${df_results['total_pnl'].min():,.0f}")
        
        # Consistency check
        positive_sharpe = (df_results['sharpe_ratio'] > 0).sum()
        sharpe_above_1 = (df_results['sharpe_ratio'] > 1).sum()
        print(f"\n🎯 Consistency Metrics:")
        print(f"  Positive Sharpe:  {positive_sharpe}/{n_simulations} ({positive_sharpe/n_simulations*100:.1f}%)")
        print(f"  Sharpe > 1.0:     {sharpe_above_1}/{n_simulations} ({sharpe_above_1/n_simulations*100:.1f}%)")
        
        return df_results
    
    def print_results(self, result, test_df=None, period_name=""):
        """Print formatted institutional results"""
        print("\n" + "="*80)
        print("🏦 INSTITUTIONAL STRATEGY PERFORMANCE")
        if period_name:
            print(f"Period: {period_name}")
        print("="*80)
        
        # Key metrics
        print(f"\n📊 Institutional Performance Metrics:")
        print(f"  Sharpe Ratio:     {result.get('sharpe_ratio', 0):.3f}")
        print(f"  Sortino Ratio:    {result.get('sortino_ratio', 0):.3f}")
        print(f"  Total Return:     {result.get('total_return', 0):.2f}%")
        print(f"  Max Drawdown:     {result.get('max_drawdown', 0):.2f}%")
        print(f"  Calmar Ratio:     {abs(result.get('total_return', 0) / result.get('max_drawdown', 1)):.2f}")
        
        print(f"\n💰 Position Statistics:")
        print(f"  Total Trades:     {result.get('total_trades', 0):,}")
        print(f"  Win Rate:         {result.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor:    {result.get('profit_factor', 0):.2f}")
        print(f"  Average Trade:    ${result.get('avg_trade', 0):,.2f}")
        print(f"  Best Trade:       ${result.get('best_trade', 0):,.2f}")
        print(f"  Worst Trade:      ${result.get('worst_trade', 0):,.2f}")
        
        # Position sizing analysis
        if 'trades' in result and result['trades']:
            trades = result['trades']
            sizes_millions = [t.position_size / 1_000_000 for t in trades if hasattr(t, 'position_size')]
            if sizes_millions:
                avg_size = np.mean(sizes_millions)
                print(f"\n📏 Position Sizing:")
                print(f"  Average Size:     {avg_size:.1f}M units")
                print(f"  1M Positions:     {sum(1 for s in sizes_millions if s < 1.5):,} trades")
                print(f"  2M Positions:     {sum(1 for s in sizes_millions if s >= 1.5):,} trades")
        
        # Exit statistics
        if 'trades' in result and result['trades']:
            print(f"\n🎯 Exit Breakdown:")
            exit_counts = {}
            for trade in trades:
                if hasattr(trade, 'exit_reason'):
                    exit_reason = trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else str(trade.exit_reason)
                    exit_counts[exit_reason] = exit_counts.get(exit_reason, 0) + 1
            
            total_trades = len(trades)
            for exit_type, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {exit_type}: {count} ({count/total_trades*100:.1f}%)")
        
        # Final capital
        if 'total_pnl' in result:
            print(f"\n💵 Capital Performance:")
            print(f"  Starting Capital: ${self.initial_capital:,.0f}")
            print(f"  Ending Capital:   ${self.initial_capital + result['total_pnl']:,.0f}")
            print(f"  Total P&L:        ${result['total_pnl']:,.0f}")
            print(f"  ROI:              {result['total_pnl']/self.initial_capital*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Run Institutional Trading Strategy')
    parser.add_argument('--currency', default='AUDUSD', help='Currency pair')
    parser.add_argument('--capital', type=float, default=10_000_000, help='Initial capital')
    parser.add_argument('--monte-carlo', type=int, help='Run Monte Carlo with N simulations')
    parser.add_argument('--period', choices=['2024', '2023', 'recent', 'last-year'], help='Test period')
    parser.add_argument('--show-plots', action='store_true', help='Show charts')
    parser.add_argument('--save-plots', action='store_true', help='Save charts')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = InstitutionalStrategyRunner(
        currency_pair=args.currency,
        initial_capital=args.capital,
        show_plots=args.show_plots,
        save_plots=args.save_plots
    )
    
    # Define period
    start_date = None
    end_date = None
    period_name = ""
    
    if args.period == '2024':
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        period_name = "2024"
    elif args.period == '2023':
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        period_name = "2023"
    elif args.period == 'recent':
        start_date = '2024-04-01'
        end_date = '2024-06-30'
        period_name = "Recent Quarter"
    elif args.period == 'last-year':
        start_date = '2023-07-01'
        end_date = '2024-06-30'
        period_name = "Last 12 Months"
    
    print("🏦 INSTITUTIONAL TRADING STRATEGY")
    print("="*60)
    print(f"Currency: {args.currency}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Mode: {'Monte Carlo' if args.monte_carlo else 'Single Run'}")
    print("="*60)
    
    # Run strategy
    if args.monte_carlo:
        results = runner.run_backtest(
            start_date=start_date,
            end_date=end_date,
            monte_carlo=True,
            n_simulations=args.monte_carlo
        )
    else:
        result, test_df = runner.run_backtest(start_date=start_date, end_date=end_date)
        runner.print_results(result, test_df, period_name)


if __name__ == "__main__":
    main()