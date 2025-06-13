"""
Look-Ahead Bias Detection Tests

This module tests for look-ahead bias in indicators and strategy execution,
which is a critical source of false positive results in backtesting.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Import indicators from clone_indicators
from clone_indicators.indicators import (
    supertrend_indicator, 
    market_bias_indicator, 
    support_resistance_indicator_fractal,
    neurotrend_indicator
)


class LookAheadBiasDetector:
    """Detect and validate look-ahead bias in trading indicators."""
    
    def __init__(self):
        self.test_results = []
        
    def create_test_data(self, pattern='monotonic', n_bars=1000):
        """
        Create test data with specific patterns to detect look-ahead bias.
        
        Args:
            pattern: Type of pattern ('monotonic', 'spike', 'reversal')
            n_bars: Number of bars to generate
        """
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='1H')
        
        if pattern == 'monotonic':
            # Strictly increasing prices - perfect for detecting future peeking
            prices = np.arange(100, 100 + n_bars * 0.1, 0.1)
        elif pattern == 'spike':
            # Sudden spike pattern
            prices = np.ones(n_bars) * 100
            spike_idx = n_bars // 2
            prices[spike_idx] = 150  # Large spike
        elif pattern == 'reversal':
            # V-shaped reversal
            mid = n_bars // 2
            prices = np.concatenate([
                np.linspace(100, 80, mid),
                np.linspace(80, 100, n_bars - mid)
            ])
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = np.roll(prices, 1)
        df['Open'][0] = prices[0]
        df['High'] = prices + np.random.uniform(0, 0.5, n_bars)
        df['Low'] = prices - np.random.uniform(0, 0.5, n_bars)
        df['Volume'] = np.random.randint(1000, 10000, n_bars)
        
        return df
    
    def test_indicator_causality(self, indicator_func, indicator_name, **kwargs):
        """
        Test if an indicator is causal (doesn't use future data).
        
        The test works by:
        1. Running indicator on monotonically increasing data
        2. Modifying future data points
        3. Checking if past indicator values change
        """
        print(f"\nTesting {indicator_name} for look-ahead bias...")
        
        # Create monotonic test data
        df_original = self.create_test_data('monotonic', n_bars=100)
        
        # Calculate indicator on original data
        try:
            result_original = indicator_func(df_original, **kwargs)
        except Exception as e:
            print(f"Error calculating {indicator_name}: {e}")
            return False
        
        # Test at multiple points
        test_points = [20, 40, 60, 80]
        bias_detected = False
        
        for test_idx in test_points:
            # Create modified data where future values are changed
            df_modified = df_original.copy()
            
            # Dramatically change all future values
            future_mask = df_modified.index > df_modified.index[test_idx]
            df_modified.loc[future_mask, 'Close'] *= 2.0
            df_modified.loc[future_mask, 'High'] *= 2.0
            df_modified.loc[future_mask, 'Low'] *= 2.0
            df_modified.loc[future_mask, 'Open'] *= 2.0
            
            # Recalculate indicator
            try:
                result_modified = indicator_func(df_modified, **kwargs)
            except Exception as e:
                print(f"Error recalculating {indicator_name}: {e}")
                continue
            
            # Check if any values before test_idx changed
            for col in result_original.columns:
                if col in result_modified.columns:
                    past_original = result_original[col].iloc[:test_idx]
                    past_modified = result_modified[col].iloc[:test_idx]
                    
                    # Compare, handling NaN values
                    mask = ~(past_original.isna() | past_modified.isna())
                    if mask.any():
                        diff = np.abs(past_original[mask] - past_modified[mask])
                        if (diff > 1e-10).any():
                            bias_detected = True
                            max_diff_idx = diff.idxmax()
                            print(f"  ⚠️  Look-ahead bias detected in column '{col}'!")
                            print(f"     Value at index {max_diff_idx} changed when future data modified")
                            print(f"     Original: {past_original[max_diff_idx]:.6f}")
                            print(f"     Modified: {past_modified[max_diff_idx]:.6f}")
                            break
            
            if bias_detected:
                break
        
        if not bias_detected:
            print(f"  ✅ {indicator_name} appears to be causal (no look-ahead bias detected)")
        
        return not bias_detected
    
    def test_fractal_sr_shift_fix(self):
        """Test if the fractal SR indicator properly shifts its signals."""
        print("\n" + "="*60)
        print("TEST: Fractal Support/Resistance Shift Fix")
        print("="*60)
        
        # Create test data with clear fractal patterns
        n_bars = 50
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='1H')
        
        # Create V-shaped pattern for clear fractal
        prices = np.concatenate([
            np.linspace(100, 90, 20),  # Down
            np.linspace(90, 100, 20),  # Up
            np.ones(10) * 100          # Flat
        ])
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['High'] = prices + 0.5
        df['Low'] = prices - 0.5
        df['Open'] = prices
        df['Volume'] = 1000
        
        # Run fractal indicator
        result = support_resistance_indicator_fractal(df, noise_filter=False)
        
        # Check for the fractal at the bottom (should be at index 19)
        fractal_low_idx = 19
        
        # The fractal should NOT be available at the fractal point itself
        # It should only be confirmed 2 bars later
        print(f"Fractal low at index {fractal_low_idx} (price={prices[fractal_low_idx]:.2f})")
        
        # Check if signal appears immediately (bad) or is delayed (good)
        for i in range(fractal_low_idx - 2, min(fractal_low_idx + 5, len(result))):
            has_signal = not pd.isna(result['SR_FractalLows'].iloc[i])
            print(f"  Index {i}: Signal present = {has_signal}")
        
        # The signal should NOT be present at fractal_low_idx or fractal_low_idx + 1
        immediate_signal = not pd.isna(result['SR_FractalLows'].iloc[fractal_low_idx])
        
        test_passed = immediate_signal  # Currently expecting it to fail (look-ahead bias)
        
        self.test_results.append({
            'test': 'Fractal SR Look-Ahead Detection',
            'passed': not test_passed,  # We expect to find the bias
            'details': f"Look-ahead bias {'detected' if immediate_signal else 'not detected'}"
        })
        
        return not immediate_signal
    
    def test_all_indicators(self):
        """Test all major indicators for look-ahead bias."""
        print("\n" + "="*60)
        print("TESTING ALL INDICATORS FOR LOOK-AHEAD BIAS")
        print("="*60)
        
        indicators_to_test = [
            (supertrend_indicator, 'SuperTrend', {'atr_period': 10, 'multiplier': 3.0}),
            (market_bias_indicator, 'Market Bias', {'ha_len': 50, 'ha_len2': 10}),
            (support_resistance_indicator_fractal, 'Fractal S/R', {'noise_filter': False}),
            (neurotrend_indicator, 'NeuroTrend', {'base_fast_len': 10, 'base_slow_len': 21}),
        ]
        
        all_passed = True
        for func, name, params in indicators_to_test:
            passed = self.test_indicator_causality(func, name, **params)
            all_passed &= passed
            
            self.test_results.append({
                'test': f'{name} Causality',
                'passed': passed,
                'details': 'No look-ahead bias' if passed else 'Look-ahead bias detected'
            })
        
        return all_passed
    
    def test_execution_timing(self):
        """Test if trade execution happens at the correct time."""
        print("\n" + "="*60)
        print("TEST: Trade Execution Timing")
        print("="*60)
        
        # This test would require access to the strategy execution logic
        # For now, we'll document what should be tested
        
        print("Trade execution timing checks needed:")
        print("1. Entry price should use Close[i] or Open[i+1], never Close[i+1]")
        print("2. Stop loss placement should use data available at entry time")
        print("3. Indicator signals should be from completed bars only")
        print("4. No position sizing based on future performance")
        
        # Placeholder test result
        self.test_results.append({
            'test': 'Execution Timing',
            'passed': None,
            'details': 'Manual review required'
        })
        
        return True
    
    def run_all_tests(self):
        """Run all look-ahead bias tests."""
        print("\n" + "="*80)
        print("LOOK-AHEAD BIAS DETECTION TEST SUITE")
        print("="*80)
        
        # Run tests
        self.test_all_indicators()
        self.test_fractal_sr_shift_fix()
        self.test_execution_timing()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed_tests = sum(1 for result in self.test_results if result['passed'] == True)
        failed_tests = sum(1 for result in self.test_results if result['passed'] == False)
        manual_tests = sum(1 for result in self.test_results if result['passed'] is None)
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            if result['passed'] is None:
                status = "MANUAL"
            elif result['passed']:
                status = "PASS"
            else:
                status = "FAIL"
            print(f"{result['test']:.<40} {status}")
        
        print(f"\nTotal: {passed_tests} passed, {failed_tests} failed, {manual_tests} need manual review")
        
        # Focus on the critical issue
        if any(r['test'] == 'Fractal S/R Causality' and not r['passed'] for r in self.test_results):
            print("\n" + "="*60)
            print("⚠️  CRITICAL ISSUE: Fractal S/R indicator has look-ahead bias!")
            print("This indicator looks at future bars (i+1, i+2) to identify fractals.")
            print("This will cause unrealistic backtest results.")
            print("="*60)
        
        return failed_tests == 0


if __name__ == "__main__":
    detector = LookAheadBiasDetector()
    all_passed = detector.run_all_tests()
    
    if not all_passed:
        sys.exit(1)
    else:
        sys.exit(0)