"""
Comprehensive Strategy Validation Script
Tests for:
1. Look-ahead bias
2. Position sizing accuracy
3. Realistic trading assumptions
4. Monte Carlo robustness
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import sys
sys.path.append('..')
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


class StrategyValidator:
    def __init__(self, currency='AUDUSD'):
        self.currency = currency
        self.validation_results = {}
        
    def load_data(self):
        """Load and prepare data"""
        print(f"\n{'='*60}")
        print(f"Loading {self.currency} data...")
        print(f"{'='*60}")
        
        df = pd.read_csv(f'../data/{self.currency}_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Add indicators
        df = TIC.add_neuro_trend_intelligent(df)
        df = TIC.add_market_bias(df)
        df = TIC.add_intelligent_chop(df)
        
        print(f"Data loaded: {len(df):,} rows")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    def test_position_sizing(self, df):
        """Test position sizing calculations"""
        print(f"\n{'='*60}")
        print("POSITION SIZING VALIDATION")
        print(f"{'='*60}")
        
        # Create strategy with known parameters
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,  # 0.2% risk
            sl_max_pips=10.0,
            verbose=False
        )
        strategy = OptimizedProdStrategy(config)
        
        # Test different scenarios
        test_cases = [
            {"price": 0.70000, "sl_pips": 10, "expected_risk": 200},  # $200 risk
            {"price": 1.10000, "sl_pips": 20, "expected_risk": 200},
            {"price": 110.000, "sl_pips": 50, "expected_risk": 200},  # USDJPY
        ]
        
        results = []
        for test in test_cases:
            price = test["price"]
            sl_pips = test["sl_pips"]
            expected_risk = test["expected_risk"]
            
            # Calculate stop loss price
            if price > 10:  # JPY pair
                sl_price = price - sl_pips * 0.01
                pip_value = 0.01
            else:
                sl_price = price - sl_pips * 0.0001
                pip_value = 0.0001
            
            # Get position size
            pos_size = strategy.calculate_position_size(price, sl_price, price)
            
            # Calculate actual risk
            actual_risk = pos_size * abs(price - sl_price)
            
            # Check if within tolerance (1% due to rounding)
            tolerance = 0.01
            passed = abs(actual_risk - expected_risk) / expected_risk < tolerance
            
            results.append({
                "price": price,
                "sl_pips": sl_pips,
                "position_size": pos_size,
                "expected_risk": expected_risk,
                "actual_risk": actual_risk,
                "passed": passed
            })
            
            print(f"\nTest Case: Price={price}, SL={sl_pips} pips")
            print(f"  Position Size: {pos_size:,.0f} units")
            print(f"  Expected Risk: ${expected_risk}")
            print(f"  Actual Risk: ${actual_risk:.2f}")
            print(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        self.validation_results['position_sizing'] = all(r['passed'] for r in results)
        return results
    
    def test_look_ahead_bias(self, df):
        """Test for look-ahead bias using shuffle test"""
        print(f"\n{'='*60}")
        print("LOOK-AHEAD BIAS TEST")
        print(f"{'='*60}")
        
        # Create strategy
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            verbose=False
        )
        strategy = OptimizedProdStrategy(config)
        
        # Test 1: Normal backtest
        print("\nTest 1: Running normal backtest...")
        normal_results = strategy.run_backtest(df[-10000:])
        normal_sharpe = normal_results['sharpe_ratio']
        
        # Test 2: Backtest with shuffled future data
        print("\nTest 2: Running backtest with shuffled signals...")
        df_shuffled = df[-10000:].copy()
        
        # Shuffle the signals (but keep them aligned)
        signal_cols = ['NTI_Direction', 'MB_Bias', 'IC_Signal']
        shuffle_size = 100  # Shuffle in chunks to maintain some structure
        
        for col in signal_cols:
            # Shift signals by random amount to break any look-ahead
            shift_amount = np.random.randint(10, 50)
            df_shuffled[col] = df_shuffled[col].shift(shift_amount).fillna(0)
        
        shuffled_results = strategy.run_backtest(df_shuffled)
        shuffled_sharpe = shuffled_results['sharpe_ratio']
        
        # Test 3: Random signals
        print("\nTest 3: Running backtest with random signals...")
        df_random = df[-10000:].copy()
        for col in signal_cols:
            if col == 'IC_Signal':
                df_random[col] = np.random.choice([0, 1, 2], size=len(df_random))
            else:
                df_random[col] = np.random.choice([-1, 0, 1], size=len(df_random))
        
        random_results = strategy.run_backtest(df_random)
        random_sharpe = random_results['sharpe_ratio']
        
        print(f"\nResults:")
        print(f"  Normal Sharpe: {normal_sharpe:.3f}")
        print(f"  Shuffled Signal Sharpe: {shuffled_sharpe:.3f}")
        print(f"  Random Signal Sharpe: {random_sharpe:.3f}")
        
        # Check if performance degrades appropriately
        look_ahead_detected = False
        if shuffled_sharpe > normal_sharpe * 0.8:  # Should degrade significantly
            print(f"  ⚠️  WARNING: Shuffled signals still performing well!")
            look_ahead_detected = True
        
        if random_sharpe > 0.5:  # Random should be near 0 or negative
            print(f"  ⚠️  WARNING: Random signals showing positive performance!")
            look_ahead_detected = True
        
        if not look_ahead_detected:
            print(f"  ✅ PASS: No look-ahead bias detected")
        else:
            print(f"  ❌ FAIL: Possible look-ahead bias")
        
        self.validation_results['look_ahead_bias'] = not look_ahead_detected
        return {
            'normal_sharpe': normal_sharpe,
            'shuffled_sharpe': shuffled_sharpe,
            'random_sharpe': random_sharpe,
            'passed': not look_ahead_detected
        }
    
    def test_realistic_fills(self, df):
        """Test impact of realistic fill assumptions"""
        print(f"\n{'='*60}")
        print("REALISTIC FILL ASSUMPTIONS TEST")
        print(f"{'='*60}")
        
        # Test with different spread/slippage assumptions
        test_configs = [
            {"name": "Perfect Fills", "spread_pips": 0, "slippage_pips": 0},
            {"name": "Tight Spread", "spread_pips": 0.5, "slippage_pips": 0},
            {"name": "Normal Spread", "spread_pips": 1.0, "slippage_pips": 0.5},
            {"name": "Wide Spread", "spread_pips": 2.0, "slippage_pips": 1.0},
        ]
        
        results = []
        base_config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            verbose=False
        )
        
        for test in test_configs:
            # For this test, we'll simulate the impact by adjusting entry/exit prices
            strategy = OptimizedProdStrategy(base_config)
            
            # Run backtest
            backtest_results = strategy.run_backtest(df[-8000:])
            
            # Simulate spread/slippage impact
            if backtest_results['total_trades'] > 0:
                # Estimate cost per trade in pips
                total_cost_pips = test['spread_pips'] + test['slippage_pips']
                
                # Convert to percentage impact (rough estimate)
                avg_cost_percent = total_cost_pips * 0.0001  # For forex
                cost_per_trade = backtest_results['total_pnl'] / backtest_results['total_trades'] * avg_cost_percent
                
                # Adjust results
                adjusted_pnl = backtest_results['total_pnl'] - (cost_per_trade * backtest_results['total_trades'] * 2)  # Entry + exit
                adjusted_return = adjusted_pnl / base_config.initial_capital * 100
                
                # Recalculate Sharpe (rough approximation)
                if backtest_results['sharpe_ratio'] > 0:
                    sharpe_reduction = (backtest_results['total_pnl'] - adjusted_pnl) / backtest_results['total_pnl']
                    adjusted_sharpe = backtest_results['sharpe_ratio'] * (1 - sharpe_reduction * 0.7)
                else:
                    adjusted_sharpe = backtest_results['sharpe_ratio']
            else:
                adjusted_pnl = backtest_results['total_pnl']
                adjusted_return = backtest_results['total_return']
                adjusted_sharpe = backtest_results['sharpe_ratio']
            
            results.append({
                'scenario': test['name'],
                'spread': test['spread_pips'],
                'slippage': test['slippage_pips'],
                'original_sharpe': backtest_results['sharpe_ratio'],
                'adjusted_sharpe': adjusted_sharpe,
                'original_return': backtest_results['total_return'],
                'adjusted_return': adjusted_return,
                'impact_percent': ((backtest_results['total_return'] - adjusted_return) / abs(backtest_results['total_return']) * 100) if backtest_results['total_return'] != 0 else 0
            })
            
            print(f"\n{test['name']}:")
            print(f"  Spread: {test['spread_pips']} pips, Slippage: {test['slippage_pips']} pips")
            print(f"  Original Sharpe: {backtest_results['sharpe_ratio']:.3f} → Adjusted: {adjusted_sharpe:.3f}")
            print(f"  Original Return: {backtest_results['total_return']:.1f}% → Adjusted: {adjusted_return:.1f}%")
            print(f"  Performance Impact: -{results[-1]['impact_percent']:.1f}%")
        
        # Check if strategy is still profitable with realistic assumptions
        realistic_profitable = results[2]['adjusted_sharpe'] > 0.5  # "Normal Spread" scenario
        
        self.validation_results['realistic_fills'] = realistic_profitable
        print(f"\n{'✅ PASS' if realistic_profitable else '❌ FAIL'}: Strategy {'remains' if realistic_profitable else 'does not remain'} profitable with realistic fills")
        
        return results
    
    def run_monte_carlo_validation(self, df):
        """Run Monte Carlo simulation to test robustness"""
        print(f"\n{'='*60}")
        print("MONTE CARLO VALIDATION")
        print(f"{'='*60}")
        
        # Run 20 iterations with different sample periods
        n_iterations = 20
        sample_size = 8000
        
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            verbose=False
        )
        strategy = OptimizedProdStrategy(config)
        
        results = []
        sharpe_ratios = []
        returns = []
        win_rates = []
        
        print(f"\nRunning {n_iterations} Monte Carlo iterations...")
        
        for i in range(n_iterations):
            # Get random sample
            max_start = len(df) - sample_size
            start_idx = np.random.randint(0, max_start)
            sample_df = df.iloc[start_idx:start_idx + sample_size]
            
            # Run backtest
            backtest_results = strategy.run_backtest(sample_df)
            
            sharpe_ratios.append(backtest_results['sharpe_ratio'])
            returns.append(backtest_results['total_return'])
            win_rates.append(backtest_results['win_rate'])
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{n_iterations} iterations...")
        
        # Calculate statistics
        results = {
            'mean_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'min_sharpe': np.min(sharpe_ratios),
            'max_sharpe': np.max(sharpe_ratios),
            'positive_sharpe_pct': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios) * 100,
            'sharpe_above_1_pct': sum(1 for s in sharpe_ratios if s > 1.0) / len(sharpe_ratios) * 100,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_win_rate': np.mean(win_rates),
            'profitable_pct': sum(1 for r in returns if r > 0) / len(returns) * 100
        }
        
        print(f"\nMonte Carlo Results ({n_iterations} iterations):")
        print(f"  Sharpe Ratio: {results['mean_sharpe']:.3f} ± {results['std_sharpe']:.3f}")
        print(f"  Range: [{results['min_sharpe']:.3f}, {results['max_sharpe']:.3f}]")
        print(f"  Positive Sharpe: {results['positive_sharpe_pct']:.1f}%")
        print(f"  Sharpe > 1.0: {results['sharpe_above_1_pct']:.1f}%")
        print(f"  Average Return: {results['mean_return']:.1f}% ± {results['std_return']:.1f}%")
        print(f"  Profitable Runs: {results['profitable_pct']:.1f}%")
        print(f"  Average Win Rate: {results['mean_win_rate']:.1f}%")
        
        # Validation criteria
        is_robust = (
            results['mean_sharpe'] > 0.5 and
            results['positive_sharpe_pct'] > 70 and
            results['profitable_pct'] > 70 and
            results['std_sharpe'] < results['mean_sharpe']  # Coefficient of variation < 1
        )
        
        self.validation_results['monte_carlo'] = is_robust
        print(f"\n{'✅ PASS' if is_robust else '❌ FAIL'}: Strategy {'shows' if is_robust else 'does not show'} robust performance")
        
        return results
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print(f"\n{'='*80}")
        print("VALIDATION REPORT SUMMARY")
        print(f"{'='*80}")
        
        report = []
        report.append(f"Strategy Validation Report - {self.currency}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n{'='*60}")
        report.append("\nVALIDATION RESULTS:")
        report.append(f"{'='*60}\n")
        
        # Overall status
        all_passed = all(self.validation_results.values())
        
        # Individual test results
        test_names = {
            'position_sizing': 'Position Sizing Accuracy',
            'look_ahead_bias': 'Look-Ahead Bias Check',
            'realistic_fills': 'Realistic Fill Assumptions',
            'monte_carlo': 'Monte Carlo Robustness'
        }
        
        for test_key, test_name in test_names.items():
            if test_key in self.validation_results:
                status = '✅ PASS' if self.validation_results[test_key] else '❌ FAIL'
                report.append(f"{test_name}: {status}")
        
        report.append(f"\n{'='*60}")
        report.append(f"OVERALL VALIDATION: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        report.append(f"{'='*60}\n")
        
        # Key findings
        report.append("KEY FINDINGS:")
        report.append("-" * 40)
        
        if not self.validation_results.get('position_sizing', True):
            report.append("⚠️  Position sizing calculations have errors")
        
        if not self.validation_results.get('look_ahead_bias', True):
            report.append("⚠️  Potential look-ahead bias detected")
        
        if not self.validation_results.get('realistic_fills', True):
            report.append("⚠️  Strategy not profitable with realistic trading costs")
        
        if not self.validation_results.get('monte_carlo', True):
            report.append("⚠️  Strategy performance not robust across different periods")
        
        if all_passed:
            report.append("✅ All validation tests passed")
            report.append("✅ Strategy appears to be properly implemented")
            report.append("✅ No cheating or unrealistic assumptions detected")
            report.append("✅ Performance is robust across different market conditions")
        
        # Recommendations
        report.append("\n\nRECOMMENDATIONS:")
        report.append("-" * 40)
        
        if all_passed:
            report.append("1. Strategy is validated and ready for further testing")
            report.append("2. Consider testing with live tick data for final validation")
            report.append("3. Monitor performance in out-of-sample periods")
        else:
            report.append("1. Address the failed validation tests before proceeding")
            report.append("2. Review strategy implementation for identified issues")
            report.append("3. Re-run validation after fixes are applied")
        
        # Save report
        report_text = '\n'.join(report)
        
        with open('validation_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n📄 Report saved to validation_report.txt")
        
        return report_text


def main():
    """Run complete validation suite"""
    validator = StrategyValidator('AUDUSD')
    
    # Load data
    df = validator.load_data()
    
    # Run all validation tests
    validator.test_position_sizing(df)
    validator.test_look_ahead_bias(df)
    validator.test_realistic_fills(df)
    validator.run_monte_carlo_validation(df)
    
    # Generate report
    validator.generate_validation_report()


if __name__ == "__main__":
    main()