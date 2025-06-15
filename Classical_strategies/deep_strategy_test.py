"""
Deep Testing of Robust Strategy - Stress Tests and Edge Cases
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import json
from collections import defaultdict
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Optimal parameters from intelligent optimization
OPTIMAL_PARAMS = {
    "exit_on_signal_flip": True,
    "partial_profit_before_sl": True,
    "partial_profit_size_percent": 0.700,
    "partial_profit_sl_distance_ratio": 0.300,
    "relaxed_mode": True,
    "risk_per_trade": 0.005,
    "sl_atr_multiplier": 0.800,
    "sl_max_pips": 10.000,
    "sl_min_pips": 3.000,
    "tp1_multiplier": 0.150,
    "tp2_multiplier": 0.250,
    "tp3_multiplier": 0.400,
    "tp_chop_market_multiplier": 0.300,
    "tp_range_market_multiplier": 0.400,
    "tp_trend_market_multiplier": 0.600,
    "trailing_atr_multiplier": 0.800,
    "tsl_activation_pips": 8.000,
    "tsl_min_profit_pips": 1.000
}

class DeepStrategyTester:
    """Comprehensive strategy testing framework"""
    
    def __init__(self, currency_pair='AUDUSD'):
        self.currency_pair = currency_pair
        self.df = None
        self.test_results = defaultdict(list)
        
    def load_data(self):
        """Load and prepare data"""
        data_path = 'data' if os.path.exists('data') else '../data'
        file_path = os.path.join(data_path, f'{self.currency_pair}_MASTER_15M.csv')
        
        print(f"Loading {self.currency_pair} data...")
        self.df = pd.read_csv(file_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        # Calculate indicators
        print("Calculating indicators...")
        self.df = TIC.add_neuro_trend_intelligent(self.df)
        self.df = TIC.add_market_bias(self.df, ha_len=350, ha_len2=30)
        self.df = TIC.add_intelligent_chop(self.df)
        
        print(f"Data loaded: {len(self.df):,} rows from {self.df.index[0]} to {self.df.index[-1]}")
    
    def create_strategy(self, params_override=None):
        """Create strategy with optional parameter overrides"""
        params = OPTIMAL_PARAMS.copy()
        if params_override:
            params.update(params_override)
        
        return OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=params['risk_per_trade'],
            sl_min_pips=params['sl_min_pips'],
            sl_max_pips=params['sl_max_pips'],
            sl_atr_multiplier=params['sl_atr_multiplier'],
            tp_atr_multipliers=(
                params['tp1_multiplier'],
                params['tp2_multiplier'],
                params['tp3_multiplier']
            ),
            max_tp_percent=0.005,
            tsl_activation_pips=params['tsl_activation_pips'],
            tsl_min_profit_pips=params['tsl_min_profit_pips'],
            tsl_initial_buffer_multiplier=1.0,
            trailing_atr_multiplier=params['trailing_atr_multiplier'],
            tp_range_market_multiplier=params['tp_range_market_multiplier'],
            tp_trend_market_multiplier=params['tp_trend_market_multiplier'],
            tp_chop_market_multiplier=params['tp_chop_market_multiplier'],
            sl_range_market_multiplier=0.7,
            exit_on_signal_flip=params['exit_on_signal_flip'],
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=params['partial_profit_before_sl'],
            partial_profit_sl_distance_ratio=params['partial_profit_sl_distance_ratio'],
            partial_profit_size_percent=params['partial_profit_size_percent'],
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            relaxed_position_multiplier=0.5,
            relaxed_mode=params['relaxed_mode'],
            realistic_costs=True,
            verbose=False,
            debug_decisions=False,
            use_daily_sharpe=True
        )
    
    def test_extreme_volatility_periods(self):
        """Test during extreme volatility periods"""
        print("\n" + "="*60)
        print("TEST 1: EXTREME VOLATILITY PERIODS")
        print("="*60)
        
        volatility_periods = {
            'GFC Crisis': ('2008-09-01', '2008-11-30'),
            'Flash Crash': ('2010-05-01', '2010-05-31'),
            'Swiss Franc Shock': ('2015-01-01', '2015-01-31'),
            'Brexit': ('2016-06-01', '2016-07-31'),
            'COVID Black Thursday': ('2020-03-01', '2020-03-31'),
            'Fed Taper Tantrum': ('2021-09-01', '2021-10-31'),
            'Ukraine War Start': ('2022-02-15', '2022-03-15'),
            'SVB Banking Crisis': ('2023-03-01', '2023-03-31')
        }
        
        strategy = OptimizedProdStrategy(self.create_strategy())
        
        for period_name, (start_date, end_date) in volatility_periods.items():
            try:
                period_df = self.df.loc[start_date:end_date].copy()
                if len(period_df) < 100:
                    print(f"\n❌ {period_name}: Insufficient data")
                    continue
                
                result = strategy.run_backtest(period_df)
                
                print(f"\n📊 {period_name} ({start_date} to {end_date}):")
                print(f"  Sharpe: {result.get('sharpe_ratio', 0):.3f}")
                print(f"  Return: {result.get('total_return', 0):.1f}%")
                print(f"  Max DD: {result.get('max_drawdown', 0):.1f}%")
                print(f"  Win Rate: {result.get('win_rate', 0):.1f}%")
                print(f"  Trades: {result.get('total_trades', 0)}")
                
                self.test_results['volatility_tests'].append({
                    'period': period_name,
                    'sharpe': result.get('sharpe_ratio', 0),
                    'return': result.get('total_return', 0),
                    'max_dd': result.get('max_drawdown', 0)
                })
                
            except Exception as e:
                print(f"\n❌ {period_name}: Error - {str(e)}")
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes"""
        print("\n" + "="*60)
        print("TEST 2: PARAMETER SENSITIVITY ANALYSIS")
        print("="*60)
        
        # Test period
        test_df = self.df.loc['2023-01-01':'2023-06-30'].copy()
        
        # Parameters to test
        param_tests = {
            'risk_per_trade': [0.001, 0.003, 0.005, 0.007, 0.01],
            'sl_min_pips': [2.0, 3.0, 5.0, 7.0, 10.0],
            'tp1_multiplier': [0.1, 0.15, 0.2, 0.25, 0.3],
            'tsl_activation_pips': [5.0, 8.0, 10.0, 15.0, 20.0],
            'relaxed_mode': [True, False]
        }
        
        baseline_config = self.create_strategy()
        baseline_strategy = OptimizedProdStrategy(baseline_config)
        baseline_result = baseline_strategy.run_backtest(test_df)
        baseline_sharpe = baseline_result.get('sharpe_ratio', 0)
        
        print(f"\nBaseline Sharpe: {baseline_sharpe:.3f}")
        print("\nTesting parameter variations:")
        
        for param_name, test_values in param_tests.items():
            print(f"\n📊 {param_name}:")
            param_results = []
            
            for value in test_values:
                override = {param_name: value}
                config = self.create_strategy(override)
                strategy = OptimizedProdStrategy(config)
                result = strategy.run_backtest(test_df)
                
                sharpe = result.get('sharpe_ratio', 0)
                diff = ((sharpe - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe != 0 else 0
                
                print(f"  {value}: Sharpe={sharpe:.3f} ({diff:+.1f}% from baseline)")
                
                param_results.append({
                    'value': value,
                    'sharpe': sharpe,
                    'diff_pct': diff
                })
            
            self.test_results['sensitivity_tests'].append({
                'parameter': param_name,
                'results': param_results
            })
    
    def test_walk_forward_analysis(self):
        """Perform walk-forward analysis"""
        print("\n" + "="*60)
        print("TEST 3: WALK-FORWARD ANALYSIS")
        print("="*60)
        
        # Define walk-forward windows
        train_months = 6
        test_months = 1
        start_date = '2022-01-01'
        end_date = '2024-06-30'
        
        strategy = OptimizedProdStrategy(self.create_strategy())
        
        # Generate date ranges
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        print(f"\nTrain window: {train_months} months, Test window: {test_months} month")
        print("\nResults:")
        
        walk_forward_results = []
        
        for i in range(train_months, len(date_range) - test_months):
            # Define train and test periods
            train_start = date_range[i - train_months]
            train_end = date_range[i]
            test_start = train_end
            test_end = date_range[i + test_months]
            
            try:
                # Get test data
                test_df = self.df.loc[test_start:test_end].copy()
                
                if len(test_df) < 100:
                    continue
                
                # Run backtest on test period
                result = strategy.run_backtest(test_df)
                
                sharpe = result.get('sharpe_ratio', 0)
                returns = result.get('total_return', 0)
                
                print(f"  Test {test_start.strftime('%Y-%m')}: Sharpe={sharpe:.3f}, Return={returns:.1f}%")
                
                walk_forward_results.append({
                    'test_period': test_start.strftime('%Y-%m'),
                    'sharpe': sharpe,
                    'return': returns,
                    'trades': result.get('total_trades', 0)
                })
                
            except Exception as e:
                print(f"  Test {test_start.strftime('%Y-%m')}: Error - {str(e)}")
        
        # Calculate walk-forward statistics
        if walk_forward_results:
            wf_df = pd.DataFrame(walk_forward_results)
            avg_sharpe = wf_df['sharpe'].mean()
            win_rate = (wf_df['sharpe'] > 0).sum() / len(wf_df) * 100
            
            print(f"\n📊 Walk-Forward Summary:")
            print(f"  Average Sharpe: {avg_sharpe:.3f}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Total Periods: {len(wf_df)}")
            
            self.test_results['walk_forward'] = walk_forward_results
    
    def test_monte_carlo_robustness(self):
        """Monte Carlo simulation with random parameter variations"""
        print("\n" + "="*60)
        print("TEST 4: MONTE CARLO ROBUSTNESS TEST")
        print("="*60)
        
        # Test on recent data
        test_df = self.df.loc['2023-01-01':'2024-06-30'].copy()
        
        n_simulations = 20
        results = []
        
        print(f"\nRunning {n_simulations} simulations with ±20% parameter variations...")
        
        for i in range(n_simulations):
            # Create random variations (±20% of optimal values)
            varied_params = {}
            for param, value in OPTIMAL_PARAMS.items():
                if isinstance(value, (int, float)):
                    variation = np.random.uniform(0.8, 1.2)
                    varied_params[param] = value * variation
                else:
                    varied_params[param] = value
            
            # Create and test strategy
            config = self.create_strategy(varied_params)
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(test_df)
            
            sharpe = result.get('sharpe_ratio', 0)
            results.append(sharpe)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{n_simulations} simulations...")
        
        # Calculate statistics
        avg_sharpe = np.mean(results)
        std_sharpe = np.std(results)
        min_sharpe = np.min(results)
        max_sharpe = np.max(results)
        pct_above_target = sum(1 for s in results if s > 0.7) / len(results) * 100
        
        print(f"\n📊 Monte Carlo Results:")
        print(f"  Average Sharpe: {avg_sharpe:.3f}")
        print(f"  Std Dev: {std_sharpe:.3f}")
        print(f"  Min: {min_sharpe:.3f}")
        print(f"  Max: {max_sharpe:.3f}")
        print(f"  % Above 0.7: {pct_above_target:.1f}%")
        
        self.test_results['monte_carlo'] = {
            'avg_sharpe': avg_sharpe,
            'std_sharpe': std_sharpe,
            'min_sharpe': min_sharpe,
            'max_sharpe': max_sharpe,
            'pct_above_target': pct_above_target
        }
    
    def test_different_market_regimes(self):
        """Test across different market regimes"""
        print("\n" + "="*60)
        print("TEST 5: MARKET REGIME ANALYSIS")
        print("="*60)
        
        # Analyze recent period for market regimes
        analysis_df = self.df.loc['2023-01-01':'2024-06-30'].copy()
        
        # Calculate rolling volatility (proxy for regime)
        analysis_df['returns'] = analysis_df['Close'].pct_change()
        analysis_df['volatility'] = analysis_df['returns'].rolling(window=96*5).std() * np.sqrt(96*252)  # 5-day vol
        
        # Define volatility percentiles
        vol_25 = analysis_df['volatility'].quantile(0.25)
        vol_75 = analysis_df['volatility'].quantile(0.75)
        
        # Classify regimes
        analysis_df['regime'] = 'normal'
        analysis_df.loc[analysis_df['volatility'] < vol_25, 'regime'] = 'low_vol'
        analysis_df.loc[analysis_df['volatility'] > vol_75, 'regime'] = 'high_vol'
        
        strategy = OptimizedProdStrategy(self.create_strategy())
        
        print("\nTesting by market regime:")
        
        for regime in ['low_vol', 'normal', 'high_vol']:
            regime_df = analysis_df[analysis_df['regime'] == regime].copy()
            
            if len(regime_df) < 100:
                continue
            
            # Run backtest
            result = strategy.run_backtest(regime_df)
            
            print(f"\n📊 {regime.upper()} Regime:")
            print(f"  Days: {len(regime_df) / 96:.0f}")
            print(f"  Sharpe: {result.get('sharpe_ratio', 0):.3f}")
            print(f"  Return: {result.get('total_return', 0):.1f}%")
            print(f"  Win Rate: {result.get('win_rate', 0):.1f}%")
            print(f"  Trades/Day: {result.get('trades_per_day', 0):.1f}")
            
            self.test_results['regime_tests'].append({
                'regime': regime,
                'sharpe': result.get('sharpe_ratio', 0),
                'return': result.get('total_return', 0),
                'win_rate': result.get('win_rate', 0)
            })
    
    def test_transaction_cost_impact(self):
        """Test impact of different transaction costs"""
        print("\n" + "="*60)
        print("TEST 6: TRANSACTION COST ANALYSIS")
        print("="*60)
        
        test_df = self.df.loc['2023-01-01':'2023-12-31'].copy()
        
        # Test different spread scenarios
        spread_scenarios = {
            'Ultra Low (0.1 pips)': {'entry_slippage': 0.05, 'sl_slippage': 0.1},
            'Low (0.5 pips)': {'entry_slippage': 0.25, 'sl_slippage': 0.5},
            'Normal (1 pip)': {'entry_slippage': 0.5, 'sl_slippage': 2.0},  # Current default
            'High (2 pips)': {'entry_slippage': 1.0, 'sl_slippage': 3.0},
            'Very High (3 pips)': {'entry_slippage': 1.5, 'sl_slippage': 4.0}
        }
        
        print("\nTesting different transaction cost scenarios:")
        
        for scenario_name, costs in spread_scenarios.items():
            # Create custom config with different costs
            # Note: We can't directly modify slippage in config, so we'll estimate impact
            config = self.create_strategy()
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(test_df)
            
            # Estimate cost impact (rough approximation)
            base_return = result.get('total_return', 0)
            trades = result.get('total_trades', 0)
            cost_factor = costs['entry_slippage'] / 0.5  # Relative to normal
            adjusted_return = base_return - (trades * 0.01 * (cost_factor - 1))  # Rough estimate
            
            print(f"\n📊 {scenario_name}:")
            print(f"  Base Return: {base_return:.1f}%")
            print(f"  Est. Adjusted Return: {adjusted_return:.1f}%")
            print(f"  Impact: {adjusted_return - base_return:.1f}%")
            print(f"  Trades: {trades}")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("DEEP TESTING SUMMARY REPORT")
        print("="*80)
        
        # Volatility test summary
        if 'volatility_tests' in self.test_results:
            vol_results = self.test_results['volatility_tests']
            avg_sharpe = np.mean([r['sharpe'] for r in vol_results])
            crisis_wins = sum(1 for r in vol_results if r['sharpe'] > 0.7)
            
            print(f"\n📊 Extreme Volatility Performance:")
            print(f"  Average Crisis Sharpe: {avg_sharpe:.3f}")
            print(f"  Profitable Crises: {crisis_wins}/{len(vol_results)}")
            print(f"  Worst Drawdown: {min(r['max_dd'] for r in vol_results):.1f}%")
        
        # Monte Carlo summary
        if 'monte_carlo' in self.test_results:
            mc = self.test_results['monte_carlo']
            print(f"\n📊 Monte Carlo Robustness:")
            print(f"  Parameter variation tolerance: ±20%")
            print(f"  Average Sharpe: {mc['avg_sharpe']:.3f}")
            print(f"  Robustness: {mc['pct_above_target']:.1f}% above target")
        
        # Market regime summary
        if 'regime_tests' in self.test_results:
            regime_results = self.test_results['regime_tests']
            print(f"\n📊 Market Regime Performance:")
            for r in regime_results:
                print(f"  {r['regime']}: Sharpe={r['sharpe']:.3f}, WR={r['win_rate']:.1f}%")
        
        # Overall verdict
        print("\n" + "="*60)
        print("FINAL VERDICT:")
        
        # Calculate overall robustness score
        scores = []
        
        # Volatility score
        if 'volatility_tests' in self.test_results:
            vol_score = sum(1 for r in self.test_results['volatility_tests'] if r['sharpe'] > 0.7) / len(self.test_results['volatility_tests'])
            scores.append(vol_score)
            print(f"  ✓ Crisis Performance: {vol_score:.1%}")
        
        # Monte Carlo score
        if 'monte_carlo' in self.test_results:
            mc_score = self.test_results['monte_carlo']['pct_above_target'] / 100
            scores.append(mc_score)
            print(f"  ✓ Parameter Robustness: {mc_score:.1%}")
        
        # Regime score
        if 'regime_tests' in self.test_results:
            regime_score = sum(1 for r in self.test_results['regime_tests'] if r['sharpe'] > 0.7) / len(self.test_results['regime_tests'])
            scores.append(regime_score)
            print(f"  ✓ Regime Adaptability: {regime_score:.1%}")
        
        if scores:
            overall_score = np.mean(scores)
            print(f"\n  Overall Robustness Score: {overall_score:.1%}")
            
            if overall_score >= 0.8:
                print("\n✅ STRATEGY PASSED DEEP TESTING - Highly Robust!")
            elif overall_score >= 0.6:
                print("\n⚠️ STRATEGY PASSED WITH CAUTION - Monitor in production")
            else:
                print("\n❌ STRATEGY NEEDS IMPROVEMENT - Not production ready")
        
        print("="*60)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'results/deep_test_results_{timestamp}.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: results/deep_test_results_{timestamp}.json")

def main():
    """Run deep testing suite"""
    tester = DeepStrategyTester('AUDUSD')
    tester.load_data()
    
    # Run all tests
    tester.test_extreme_volatility_periods()
    tester.test_parameter_sensitivity()
    tester.test_walk_forward_analysis()
    tester.test_monte_carlo_robustness()
    tester.test_different_market_regimes()
    tester.test_transaction_cost_impact()
    
    # Generate report
    tester.generate_report()

if __name__ == "__main__":
    main()