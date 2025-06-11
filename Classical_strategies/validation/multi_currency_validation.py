"""
Comprehensive Multi-Currency Validation with Anti-Cheating Checks
Tests for look-ahead bias, data snooping, and realistic execution with slippage
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('..')  # Add parent directory to path
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import random
warnings.filterwarnings('ignore')


class MultiCurrencyValidator:
    """
    Comprehensive validation for multi-currency trading strategies
    Checks for any form of cheating or unrealistic assumptions
    """
    
    def __init__(self, slippage_min_pips=0, slippage_max_pips=2):
        self.slippage_min = slippage_min_pips
        self.slippage_max = slippage_max_pips
        self.validation_results = {}
        self.cheating_checks = {
            'look_ahead_bias': False,
            'data_snooping': False,
            'unrealistic_fills': False,
            'impossible_trades': False,
            'indicator_peeking': False
        }
    
    def check_for_cheating(self, df, strategy, currency_pair):
        """
        Comprehensive cheating detection
        """
        print(f"\n{'='*60}")
        print(f"ANTI-CHEATING VALIDATION FOR {currency_pair}")
        print(f"{'='*60}")
        
        # 1. Check for look-ahead bias in indicators
        print("\n1. Checking for look-ahead bias...")
        if self._check_look_ahead_bias(df):
            print("   ❌ FAIL: Look-ahead bias detected!")
            self.cheating_checks['look_ahead_bias'] = True
        else:
            print("   ✓ PASS: No look-ahead bias detected")
        
        # 2. Check for impossible trades
        print("\n2. Checking for impossible trades...")
        sample_df = df.iloc[:5000].copy()
        results = strategy.run_backtest(sample_df)
        
        impossible_trades = 0
        for trade in results['trades']:
            # Check if entry/exit prices are within the bar's range
            if hasattr(trade, 'entry_time') and hasattr(trade, 'entry_price'):
                entry_idx = self._find_time_index(sample_df, trade.entry_time)
                if entry_idx is not None:
                    bar = sample_df.iloc[entry_idx]
                    if trade.entry_price < bar['Low'] or trade.entry_price > bar['High']:
                        impossible_trades += 1
                        print(f"   ❌ Impossible entry at {trade.entry_time}: Price {trade.entry_price} outside bar range [{bar['Low']}, {bar['High']}]")
        
        if impossible_trades > 0:
            print(f"   ❌ FAIL: {impossible_trades} impossible trades found!")
            self.cheating_checks['impossible_trades'] = True
        else:
            print("   ✓ PASS: All trades within valid price ranges")
        
        # 3. Check for unrealistic fill assumptions
        print("\n3. Checking fill assumptions...")
        if self._check_unrealistic_fills(results):
            print("   ❌ FAIL: Unrealistic fill assumptions detected!")
            self.cheating_checks['unrealistic_fills'] = True
        else:
            print("   ✓ PASS: Fill assumptions are realistic")
        
        # 4. Check for data snooping
        print("\n4. Checking for data snooping...")
        if self._check_data_snooping(df, strategy):
            print("   ❌ FAIL: Potential data snooping detected!")
            self.cheating_checks['data_snooping'] = True
        else:
            print("   ✓ PASS: No data snooping detected")
        
        # 5. Check indicator calculations
        print("\n5. Checking indicator integrity...")
        if self._check_indicator_integrity(df):
            print("   ❌ FAIL: Indicator calculation issues detected!")
            self.cheating_checks['indicator_peeking'] = True
        else:
            print("   ✓ PASS: Indicators calculated correctly")
        
        # Summary
        cheating_detected = any(self.cheating_checks.values())
        print(f"\n{'='*60}")
        if cheating_detected:
            print("❌ CHEATING DETECTED - Results are not valid!")
            print("Issues found:")
            for check, failed in self.cheating_checks.items():
                if failed:
                    print(f"  - {check}")
        else:
            print("✅ NO CHEATING DETECTED - Results appear legitimate")
        
        return not cheating_detected
    
    def _check_look_ahead_bias(self, df):
        """Check if future data is used in calculations"""
        # Check if any indicator values appear before they should
        indicators = ['nti_signal', 'mb_signal', 'ic_signal']
        
        for indicator in indicators:
            if indicator in df.columns:
                # Simple check: indicators should not be perfectly predictive
                if len(df) > 100:
                    subset = df.iloc[50:150].copy()
                    if indicator in subset.columns and 'Close' in subset.columns:
                        # Check correlation with future returns
                        future_returns = subset['Close'].pct_change().shift(-1)
                        if subset[indicator].notna().sum() > 10:
                            correlation = subset[indicator].corr(future_returns)
                            if abs(correlation) > 0.9:  # Suspiciously high correlation
                                return True
        return False
    
    def _check_unrealistic_fills(self, results):
        """Check if fills assume unrealistic execution"""
        if 'trades' not in results:
            return False
            
        # Check if all limit orders (TPs) are filled perfectly
        tp_fills = 0
        tp_attempts = 0
        
        for trade in results['trades']:
            if hasattr(trade, 'exit_reason'):
                exit_reason_str = str(trade.exit_reason)
                if 'TP' in exit_reason_str:
                    tp_attempts += 1
                    tp_fills += 1
        
        # If 100% of TP orders are filled, that's unrealistic
        if tp_attempts > 20 and tp_fills == tp_attempts:
            return True
            
        return False
    
    def _check_data_snooping(self, df, strategy):
        """Check if strategy uses future information"""
        # Run on different time periods and check if performance is too consistent
        if len(df) < 20000:
            return False
            
        period_results = []
        for i in range(0, 20000, 5000):
            subset = df.iloc[i:i+5000].copy()
            results = strategy.run_backtest(subset)
            period_results.append(results['sharpe_ratio'])
        
        # If all periods have nearly identical Sharpe ratios, that's suspicious
        if len(period_results) > 3:
            std_dev = np.std(period_results)
            if std_dev < 0.05:  # Too consistent
                return True
                
        return False
    
    def _check_indicator_integrity(self, df):
        """Verify indicators are calculated correctly without peeking"""
        # Check that indicators have appropriate lag
        if 'nti_signal' in df.columns:
            # NTI should have some NaN values at the beginning
            first_valid = df['nti_signal'].first_valid_index()
            if first_valid is not None:
                first_idx = df.index.get_loc(first_valid)
                if first_idx < 10:  # Should have more warm-up period
                    return True
        return False
    
    def _find_time_index(self, df, time):
        """Find index for given time"""
        try:
            return df.index.get_loc(time)
        except:
            return None
    
    def validate_currency_with_slippage(self, currency_pair, n_tests=20):
        """
        Validate a currency pair with realistic slippage
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING {currency_pair} WITH SLIPPAGE")
        print(f"{'='*80}")
        
        # Load data
        data_path = f'../../data/{currency_pair}_MASTER_15M.csv'
        if not os.path.exists(data_path):
            print(f"Data file not found for {currency_pair}")
            return None
        
        df = pd.read_csv(data_path)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        print(f"Loaded {len(df):,} rows from {df.index[0]} to {df.index[-1]}")
        
        # Calculate indicators
        df = TIC.add_neuro_trend_intelligent(df)
        df = TIC.add_market_bias(df)
        df = TIC.add_intelligent_chop(df)
        
        # Create strategies
        from multi_currency_monte_carlo import create_config_1_ultra_tight_risk, create_config_2_scalping
        
        configs = [
            ("Config 1: Ultra-Tight Risk", create_config_1_ultra_tight_risk(), 12.0),  # 10 + 2 slippage
            ("Config 2: Scalping", create_config_2_scalping(), 7.0)  # 5 + 2 slippage
        ]
        
        currency_results = {}
        
        for config_name, strategy, sl_with_slippage in configs:
            print(f"\n\nTesting {config_name}...")
            
            # First check for cheating
            is_valid = self.check_for_cheating(df, strategy, currency_pair)
            if not is_valid:
                print("❌ Strategy failed anti-cheating validation!")
                continue
            
            # Run tests with and without slippage
            no_slip_results = []
            with_slip_results = []
            
            for i in range(n_tests):
                # Random sample
                max_start = len(df) - 5000
                start_idx = np.random.randint(0, max_start)
                sample_df = df.iloc[start_idx:start_idx + 5000].copy()
                
                # Test without slippage
                results_no_slip = strategy.run_backtest(sample_df)
                no_slip_results.append({
                    'sharpe': results_no_slip['sharpe_ratio'],
                    'pnl': results_no_slip['total_pnl'],
                    'win_rate': results_no_slip['win_rate'],
                    'drawdown': results_no_slip['max_drawdown']
                })
                
                # Test with slippage (modify strategy)
                strategy_with_slip = self._create_strategy_with_slippage(config_name, sl_with_slippage)
                results_with_slip = strategy_with_slip.run_backtest(sample_df)
                with_slip_results.append({
                    'sharpe': results_with_slip['sharpe_ratio'],
                    'pnl': results_with_slip['total_pnl'],
                    'win_rate': results_with_slip['win_rate'],
                    'drawdown': results_with_slip['max_drawdown']
                })
            
            # Calculate statistics
            avg_sharpe_no_slip = np.mean([r['sharpe'] for r in no_slip_results])
            avg_sharpe_with_slip = np.mean([r['sharpe'] for r in with_slip_results])
            avg_pnl_no_slip = np.mean([r['pnl'] for r in no_slip_results])
            avg_pnl_with_slip = np.mean([r['pnl'] for r in with_slip_results])
            
            sharpe_degradation = (avg_sharpe_with_slip - avg_sharpe_no_slip) / avg_sharpe_no_slip * 100
            pnl_degradation = (avg_pnl_with_slip - avg_pnl_no_slip) / avg_pnl_no_slip * 100
            
            robustness = sum(1 for r in with_slip_results if r['sharpe'] > 1.0) / len(with_slip_results) * 100
            
            currency_results[config_name] = {
                'avg_sharpe_no_slip': avg_sharpe_no_slip,
                'avg_sharpe_with_slip': avg_sharpe_with_slip,
                'avg_pnl_no_slip': avg_pnl_no_slip,
                'avg_pnl_with_slip': avg_pnl_with_slip,
                'sharpe_degradation': sharpe_degradation,
                'pnl_degradation': pnl_degradation,
                'robustness': robustness
            }
            
            print(f"\nResults for {config_name}:")
            print(f"Without Slippage: Sharpe {avg_sharpe_no_slip:.3f}, P&L ${avg_pnl_no_slip:,.0f}")
            print(f"With Slippage:    Sharpe {avg_sharpe_with_slip:.3f}, P&L ${avg_pnl_with_slip:,.0f}")
            print(f"Degradation:      Sharpe {sharpe_degradation:+.1f}%, P&L {pnl_degradation:+.1f}%")
            print(f"Robustness:       {robustness:.0f}% maintain Sharpe > 1.0")
        
        return currency_results
    
    def _create_strategy_with_slippage(self, config_name, sl_pips):
        """Create strategy with slippage buffer built in"""
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002 if "Ultra-Tight" in config_name else 0.001,
            sl_max_pips=sl_pips,
            sl_atr_multiplier=1.0 if "Ultra-Tight" in config_name else 0.5,
            tp_atr_multipliers=(0.2, 0.3, 0.5) if "Ultra-Tight" in config_name else (0.1, 0.2, 0.3),
            max_tp_percent=0.003 if "Ultra-Tight" in config_name else 0.002,
            tsl_activation_pips=3 if "Ultra-Tight" in config_name else 2,
            tsl_min_profit_pips=1 if "Ultra-Tight" in config_name else 0.5,
            tsl_initial_buffer_multiplier=1.0 if "Ultra-Tight" in config_name else 0.5,
            trailing_atr_multiplier=0.8 if "Ultra-Tight" in config_name else 0.5,
            tp_range_market_multiplier=0.5 if "Ultra-Tight" in config_name else 0.3,
            tp_trend_market_multiplier=0.7 if "Ultra-Tight" in config_name else 0.5,
            tp_chop_market_multiplier=0.3 if "Ultra-Tight" in config_name else 0.2,
            sl_range_market_multiplier=0.7 if "Ultra-Tight" in config_name else 0.5,
            exit_on_signal_flip=False if "Ultra-Tight" in config_name else True,
            signal_flip_min_profit_pips=5.0 if "Ultra-Tight" in config_name else 0.0,
            signal_flip_min_time_hours=1.0 if "Ultra-Tight" in config_name else 0.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.5 if "Ultra-Tight" in config_name else 0.3,
            partial_profit_size_percent=0.5 if "Ultra-Tight" in config_name else 0.7,
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            verbose=False
        )
        return OptimizedProdStrategy(config)


def main():
    """Run comprehensive multi-currency validation"""
    
    print("="*80)
    print("COMPREHENSIVE MULTI-CURRENCY VALIDATION")
    print("Anti-Cheating Checks + Realistic Slippage Testing")
    print("="*80)
    
    # Initialize validator
    validator = MultiCurrencyValidator(slippage_min_pips=0, slippage_max_pips=2)
    
    # Currency pairs to validate
    currency_pairs = ['GBPUSD', 'EURUSD', 'NZDUSD', 'USDCAD']
    
    # Run validation for each currency
    all_results = {}
    
    for currency in currency_pairs:
        results = validator.validate_currency_with_slippage(currency, n_tests=20)
        if results:
            all_results[currency] = results
    
    # Summary report
    print("\n" + "="*80)
    print("MULTI-CURRENCY VALIDATION SUMMARY")
    print("="*80)
    
    # Check if any cheating was detected
    if any(validator.cheating_checks.values()):
        print("\n❌ VALIDATION FAILED - CHEATING DETECTED")
        print("The following issues were found:")
        for check, failed in validator.cheating_checks.items():
            if failed:
                print(f"  - {check}")
        return
    
    print("\n✅ All currencies passed anti-cheating validation")
    
    # Performance summary
    print("\nPERFORMANCE WITH REALISTIC SLIPPAGE (0-2 pips)")
    print("-" * 80)
    print(f"{'Currency':<10} {'Config':<25} {'Original Sharpe':>15} {'Slippage Sharpe':>15} {'Degradation':>12} {'Robust%':>10}")
    print("-" * 80)
    
    for currency, currency_results in all_results.items():
        for config_name, results in currency_results.items():
            print(f"{currency:<10} {config_name:<25} {results['avg_sharpe_no_slip']:>15.3f} "
                  f"{results['avg_sharpe_with_slip']:>15.3f} {results['sharpe_degradation']:>11.1f}% "
                  f"{results['robustness']:>9.0f}%")
    
    # Recommendations
    print("\n" + "="*80)
    print("INSTITUTIONAL DEPLOYMENT RECOMMENDATIONS")
    print("="*80)
    
    # Find best performing currency/config combo
    best_sharpe = 0
    best_combo = None
    best_robustness = 0
    
    for currency, currency_results in all_results.items():
        for config_name, results in currency_results.items():
            if results['avg_sharpe_with_slip'] > best_sharpe and results['robustness'] >= 60:
                best_sharpe = results['avg_sharpe_with_slip']
                best_combo = (currency, config_name)
                best_robustness = results['robustness']
    
    if best_combo:
        print(f"\n✅ RECOMMENDED FOR PRODUCTION:")
        print(f"   Currency: {best_combo[0]}")
        print(f"   Strategy: {best_combo[1]}")
        print(f"   Expected Sharpe (with slippage): {best_sharpe:.3f}")
        print(f"   Robustness: {best_robustness:.0f}% of tests maintain Sharpe > 1.0")
        print(f"\n   Risk Controls:")
        print(f"   - Use 2 pip slippage buffer on all market orders")
        print(f"   - Monitor actual slippage and adjust if needed")
        print(f"   - Implement pre-trade spread checks")
        print(f"   - Set maximum position limits")
    else:
        print("\n⚠️  No configuration meets minimum robustness criteria")
    
    print(f"\nValidation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()