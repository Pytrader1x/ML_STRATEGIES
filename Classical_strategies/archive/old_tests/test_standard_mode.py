"""
Test Classical Trading Strategy in STANDARD MODE (Conservative)
This tests the strategy as it should be run - requiring all 3 indicators to align

Standard Mode Entry Requirements:
LONG:  NTI_Direction == 1 AND MB_Bias == 1 AND IC_Regime ∈ [1,2]
SHORT: NTI_Direction == -1 AND MB_Bias == -1 AND IC_Regime ∈ [1,2]
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_validated_strategy import ValidatedStrategyRunner
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC

class StandardModeValidator:
    """Validate strategy in Standard (Conservative) Mode"""
    
    def __init__(self):
        self.results = {}
        
    def create_standard_mode_strategy(self, initial_capital=1_000_000, position_size_millions=1.0):
        """Create strategy configuration with STANDARD MODE (all 3 indicators required)"""
        config = OptimizedStrategyConfig(
            # Capital settings
            initial_capital=initial_capital,
            risk_per_trade=0.001,  # Conservative 0.1% risk per trade
            base_position_size_millions=position_size_millions,
            
            # CRITICAL: Standard Mode - Require all 3 indicators
            relaxed_mode=False,  # THIS IS THE KEY SETTING
            relaxed_position_multiplier=0.5,  # Not used in standard mode
            
            # Stop loss settings
            sl_min_pips=5.0,  # More conservative min stop
            sl_max_pips=20.0,  # Standard max stop
            sl_atr_multiplier=1.5,  # Standard ATR multiplier
            
            # Take profit settings
            tp_atr_multipliers=(1.0, 2.0, 3.0),  # Standard TP levels
            max_tp_percent=0.01,
            
            # Market regime adjustments
            tp_range_market_multiplier=0.8,
            tp_trend_market_multiplier=1.2,
            tp_chop_market_multiplier=0.6,
            
            # Trailing stop settings
            tsl_activation_pips=10.0,  # More conservative activation
            tsl_min_profit_pips=2.0,
            trailing_atr_multiplier=1.0,
            
            # Exit logic
            exit_on_signal_flip=True,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=2.0,  # Longer hold time
            signal_flip_partial_exit_percent=1.0,
            
            # Partial profit taking
            partial_profit_before_sl=False,  # Disabled for cleaner analysis
            
            # CRITICAL: Realistic execution costs
            realistic_costs=True,
            entry_slippage_pips=0.1,  # Institutional slippage
            stop_loss_slippage_pips=0.5,
            trailing_stop_slippage_pips=0.3,
            take_profit_slippage_pips=0.0,
            
            # Other settings
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            verbose=False,
            debug_decisions=False,
            use_daily_sharpe=True
        )
        
        return OptimizedProdStrategy(config)
    
    def run_comprehensive_tests(self):
        """Run comprehensive validation tests in Standard Mode"""
        print("="*80)
        print("STANDARD MODE VALIDATION - CONSERVATIVE ENTRY LOGIC")
        print("="*80)
        print("Entry Requirements: ALL 3 indicators must align")
        print("- NeuroTrend (NTI) direction")
        print("- Market Bias (MB) alignment")
        print("- Intelligent Chop (IC) trending market")
        print("="*80)
        
        # Load data
        print("\n1. Loading data...")
        runner = ValidatedStrategyRunner('AUDUSD', initial_capital=1_000_000, position_size_millions=1.0)
        runner.load_data()
        
        # Test different time periods
        test_periods = [
            ('2020-01-01', '2020-12-31', '2020 - COVID Volatility'),
            ('2021-01-01', '2021-12-31', '2021 - Recovery Year'),
            ('2022-01-01', '2022-12-31', '2022 - Rate Hike Cycle'),
            ('2023-01-01', '2023-12-31', '2023 - Recent History'),
            ('2024-01-01', '2024-06-30', '2024 H1 - Current')
        ]
        
        print("\n2. Running backtests across multiple years...")
        
        all_results = []
        for start, end, period_name in test_periods:
            print(f"\n{'='*60}")
            print(f"Testing: {period_name}")
            print(f"{'='*60}")
            
            try:
                # Filter data
                test_df = runner.df.loc[start:end].copy()
                
                # Create strategy in STANDARD MODE
                strategy = self.create_standard_mode_strategy()
                
                # Run backtest
                result = strategy.run_backtest(test_df)
                
                # Store results
                result['period'] = period_name
                result['start_date'] = start
                result['end_date'] = end
                result['mode'] = 'STANDARD'
                all_results.append(result)
                
                # Display results
                print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
                print(f"Total Return: {result.get('total_return', 0):.2f}%")
                print(f"Max Drawdown: {result.get('max_drawdown', 0):.2f}%")
                print(f"Total Trades: {result.get('total_trades', 0)}")
                print(f"Win Rate: {result.get('win_rate', 0):.1f}%")
                print(f"Profit Factor: {result.get('profit_factor', 0):.2f}")
                print(f"Total P&L: ${result.get('total_pnl', 0):,.0f}")
                
                # Analyze entry frequency
                if result.get('total_trades', 0) > 0:
                    days = (pd.to_datetime(end) - pd.to_datetime(start)).days
                    trades_per_month = result['total_trades'] / (days / 30)
                    print(f"Trades per Month: {trades_per_month:.1f}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
                continue
        
        # Analyze results
        self._analyze_results(all_results)
        
        # Run Monte Carlo in Standard Mode
        print("\n3. Running Monte Carlo Simulation in STANDARD MODE...")
        self._run_monte_carlo_standard()
        
        # Compare with Relaxed Mode
        print("\n4. Comparing Standard vs Relaxed Mode...")
        self._compare_modes()
    
    def _analyze_results(self, results):
        """Analyze standard mode results"""
        if not results:
            print("\nNo results to analyze")
            return
            
        print("\n" + "="*80)
        print("STANDARD MODE ANALYSIS SUMMARY")
        print("="*80)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Overall statistics
        avg_sharpe = df['sharpe_ratio'].mean()
        avg_return = df['total_return'].mean()
        avg_trades = df['total_trades'].mean()
        avg_win_rate = df['win_rate'].mean()
        
        print(f"\nAverage Metrics Across All Periods:")
        print(f"  Average Sharpe Ratio: {avg_sharpe:.3f}")
        print(f"  Average Annual Return: {avg_return:.1f}%")
        print(f"  Average Win Rate: {avg_win_rate:.1f}%")
        print(f"  Average Trades per Year: {avg_trades:.0f}")
        
        # Consistency check
        positive_years = (df['sharpe_ratio'] > 0).sum()
        print(f"\nConsistency:")
        print(f"  Profitable Years: {positive_years}/{len(df)} ({positive_years/len(df)*100:.0f}%)")
        
        # Best and worst
        best_idx = df['sharpe_ratio'].idxmax()
        worst_idx = df['sharpe_ratio'].idxmin()
        
        print(f"\nBest Period: {df.loc[best_idx, 'period']}")
        print(f"  Sharpe: {df.loc[best_idx, 'sharpe_ratio']:.3f}")
        print(f"  Return: {df.loc[best_idx, 'total_return']:.1f}%")
        
        print(f"\nWorst Period: {df.loc[worst_idx, 'period']}")
        print(f"  Sharpe: {df.loc[worst_idx, 'sharpe_ratio']:.3f}")
        print(f"  Return: {df.loc[worst_idx, 'total_return']:.1f}%")
        
        # Entry analysis
        total_days = 0
        for _, row in df.iterrows():
            start = pd.to_datetime(row['start_date'])
            end = pd.to_datetime(row['end_date'])
            total_days += (end - start).days
        
        total_trades = df['total_trades'].sum()
        if total_days > 0:
            entries_per_day = total_trades / total_days
            print(f"\nEntry Frequency:")
            print(f"  Total Signals Generated: {total_trades}")
            print(f"  Signals per Day: {entries_per_day:.3f}")
            print(f"  Days per Signal: {1/entries_per_day:.1f}" if entries_per_day > 0 else "  Days per Signal: N/A")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results/standard_mode_validation_{timestamp}.csv'
        os.makedirs('results', exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"\n💾 Results saved to: {filename}")
    
    def _run_monte_carlo_standard(self):
        """Run Monte Carlo simulation in Standard Mode"""
        print("\n" + "="*60)
        print("MONTE CARLO - STANDARD MODE")
        print("="*60)
        
        # Create runner but override the strategy creation
        runner = ValidatedStrategyRunner('AUDUSD', initial_capital=1_000_000, position_size_millions=1.0)
        runner.load_data()
        
        # Run 50 random samples
        n_simulations = 50
        sample_days = 90
        sample_size = sample_days * 96  # 96 bars per day
        
        monte_carlo_results = []
        
        for sim in range(n_simulations):
            # Random sample
            max_start = len(runner.df) - sample_size - 1000
            start_idx = np.random.randint(1000, max_start)
            end_idx = start_idx + sample_size
            
            sample_df = runner.df.iloc[start_idx:end_idx].copy()
            start_date = sample_df.index[0]
            end_date = sample_df.index[-1]
            
            # Create STANDARD MODE strategy
            strategy = self.create_standard_mode_strategy()
            
            try:
                result = strategy.run_backtest(sample_df)
                
                monte_carlo_results.append({
                    'sim': sim + 1,
                    'start_date': start_date,
                    'end_date': end_date,
                    'sharpe': result.get('sharpe_ratio', 0),
                    'return': result.get('total_return', 0),
                    'trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0)
                })
                
                if (sim + 1) % 10 == 0:
                    print(f"Completed {sim + 1}/{n_simulations} simulations...")
                    
            except Exception as e:
                print(f"Simulation {sim + 1} failed: {str(e)}")
                continue
        
        # Analyze Monte Carlo results
        if monte_carlo_results:
            df_mc = pd.DataFrame(monte_carlo_results)
            
            print(f"\nMonte Carlo Results ({len(df_mc)} successful simulations):")
            print(f"  Average Sharpe: {df_mc['sharpe'].mean():.3f} (±{df_mc['sharpe'].std():.3f})")
            print(f"  Average Return: {df_mc['return'].mean():.1f}% (±{df_mc['return'].std():.1f})")
            print(f"  Average Trades: {df_mc['trades'].mean():.0f}")
            print(f"  Success Rate: {(df_mc['sharpe'] > 0).sum() / len(df_mc) * 100:.1f}%")
            
            # Performance distribution
            print(f"\nPerformance Distribution:")
            print(f"  Sharpe > 1.0: {(df_mc['sharpe'] > 1.0).sum()} ({(df_mc['sharpe'] > 1.0).sum()/len(df_mc)*100:.1f}%)")
            print(f"  Sharpe > 0.5: {(df_mc['sharpe'] > 0.5).sum()} ({(df_mc['sharpe'] > 0.5).sum()/len(df_mc)*100:.1f}%)")
            print(f"  Sharpe > 0.0: {(df_mc['sharpe'] > 0.0).sum()} ({(df_mc['sharpe'] > 0.0).sum()/len(df_mc)*100:.1f}%)")
    
    def _compare_modes(self):
        """Compare Standard Mode vs Relaxed Mode on same period"""
        print("\n" + "="*80)
        print("STANDARD vs RELAXED MODE COMPARISON")
        print("="*80)
        
        # Test on recent data
        start_date = '2023-01-01'
        end_date = '2023-12-31'
        
        runner = ValidatedStrategyRunner('AUDUSD', initial_capital=1_000_000, position_size_millions=1.0)
        runner.load_data()
        test_df = runner.df.loc[start_date:end_date].copy()
        
        print(f"Test Period: {start_date} to {end_date}")
        print(f"Test Data: {len(test_df)} bars")
        
        # Test Standard Mode
        print("\n1. STANDARD MODE (All 3 indicators required):")
        strategy_standard = self.create_standard_mode_strategy()
        result_standard = strategy_standard.run_backtest(test_df)
        
        print(f"   Sharpe Ratio: {result_standard.get('sharpe_ratio', 0):.3f}")
        print(f"   Total Return: {result_standard.get('total_return', 0):.2f}%")
        print(f"   Total Trades: {result_standard.get('total_trades', 0)}")
        print(f"   Win Rate: {result_standard.get('win_rate', 0):.1f}%")
        print(f"   Max Drawdown: {result_standard.get('max_drawdown', 0):.2f}%")
        
        # Test Relaxed Mode  
        print("\n2. RELAXED MODE (Only NTI required):")
        # Create relaxed mode strategy
        config_relaxed = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            base_position_size_millions=1.0,
            relaxed_mode=True,  # RELAXED MODE
            relaxed_position_multiplier=0.5,
            realistic_costs=True,
            use_daily_sharpe=True
        )
        strategy_relaxed = OptimizedProdStrategy(config_relaxed)
        result_relaxed = strategy_relaxed.run_backtest(test_df)
        
        print(f"   Sharpe Ratio: {result_relaxed.get('sharpe_ratio', 0):.3f}")
        print(f"   Total Return: {result_relaxed.get('total_return', 0):.2f}%")
        print(f"   Total Trades: {result_relaxed.get('total_trades', 0)}")
        print(f"   Win Rate: {result_relaxed.get('win_rate', 0):.1f}%")
        print(f"   Max Drawdown: {result_relaxed.get('max_drawdown', 0):.2f}%")
        
        # Analysis
        print("\n3. COMPARATIVE ANALYSIS:")
        
        trade_ratio = result_relaxed['total_trades'] / result_standard['total_trades'] if result_standard['total_trades'] > 0 else float('inf')
        print(f"   Trade Frequency Ratio: {trade_ratio:.1f}x more trades in Relaxed Mode")
        
        print(f"   Sharpe Difference: {result_relaxed['sharpe_ratio'] - result_standard['sharpe_ratio']:.3f}")
        print(f"   Return Difference: {result_relaxed['total_return'] - result_standard['total_return']:.1f}%")
        
        # Quality assessment
        print("\n4. QUALITY ASSESSMENT:")
        
        if result_standard['sharpe_ratio'] > result_relaxed['sharpe_ratio']:
            print("   ✅ STANDARD MODE has better risk-adjusted returns")
        else:
            print("   ⚠️  RELAXED MODE has better risk-adjusted returns (concerning)")
            
        if result_standard['max_drawdown'] < result_relaxed['max_drawdown']:
            print("   ✅ STANDARD MODE has lower drawdown")
        else:
            print("   ⚠️  RELAXED MODE has lower drawdown")
            
        if result_standard['win_rate'] > 60:
            print("   ✅ STANDARD MODE maintains good win rate")
        else:
            print("   ⚠️  STANDARD MODE win rate below 60%")
        
        # Final verdict
        print("\n5. RECOMMENDATION:")
        
        if result_standard['sharpe_ratio'] > 0.5 and result_standard['total_trades'] > 50:
            print("   ✅ STANDARD MODE is viable for production")
            print("   - Generates sufficient trades")
            print("   - Maintains positive risk-adjusted returns")
            print("   - More selective entry reduces false signals")
        else:
            print("   ⚠️  STANDARD MODE may be too restrictive")
            print("   - Consider adjusting indicator thresholds")
            print("   - May need different confirmation logic")

def main():
    """Run Standard Mode validation"""
    validator = StandardModeValidator()
    validator.run_comprehensive_tests()

if __name__ == "__main__":
    main()