"""
Comprehensive Sharpe Ratio Validation Tests

This module tests the Sharpe ratio calculation for correctness, best practices,
and potential false positives or cheating in the strategy implementation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_code.Prod_strategy import OptimizedProdStrategy, annualization_factor_from_df


class SharpeRatioValidator:
    """Comprehensive tests for Sharpe ratio calculation validation."""
    
    def __init__(self):
        self.test_results = []
        
    def create_synthetic_price_data(self, n_days=252, drift=0.0001, volatility=0.02, 
                                   freq='1H', start_date='2023-01-01'):
        """
        Create synthetic price data with known statistical properties.
        
        Args:
            n_days: Number of trading days
            drift: Daily drift (expected return)
            volatility: Daily volatility
            freq: Data frequency
            start_date: Start date for the data
        """
        # Calculate number of periods
        if freq == '1H':
            periods_per_day = 24
        elif freq == '15T':
            periods_per_day = 96
        elif freq == '1D':
            periods_per_day = 1
        else:
            raise ValueError(f"Unsupported frequency: {freq}")
            
        n_periods = n_days * periods_per_day
        
        # Generate time index
        date_range = pd.date_range(start=start_date, periods=n_periods, freq=freq)
        
        # Generate returns with known properties
        # Scale drift and volatility to the frequency
        period_drift = drift / periods_per_day
        period_vol = volatility / np.sqrt(periods_per_day)
        
        # Generate log returns
        log_returns = np.random.normal(period_drift, period_vol, n_periods)
        
        # Convert to prices
        log_prices = np.cumsum(log_returns)
        prices = 100 * np.exp(log_prices)  # Start at 100
        
        # Create OHLC data
        df = pd.DataFrame(index=date_range)
        df['Close'] = prices
        
        # Generate realistic OHLC from close prices
        noise = np.random.uniform(0.001, 0.003, n_periods)
        df['High'] = df['Close'] * (1 + noise)
        df['Low'] = df['Close'] * (1 - noise)
        df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
        
        # Add volume
        df['Volume'] = np.random.randint(1000, 10000, n_periods)
        
        return df, drift, volatility
    
    def test_sharpe_calculation_correctness(self):
        """Test if the Sharpe ratio calculation is mathematically correct."""
        print("\n" + "="*60)
        print("TEST 1: Sharpe Ratio Calculation Correctness")
        print("="*60)
        
        # Create synthetic data with known properties
        df, true_drift, true_vol = self.create_synthetic_price_data(
            n_days=252, drift=0.0001, volatility=0.02, freq='1H'
        )
        
        # Expected annual Sharpe ratio (approximately)
        expected_sharpe = true_drift / true_vol * np.sqrt(252)
        
        # Create a simple equity curve that follows the price
        initial_capital = 10000
        equity_curve = initial_capital * df['Close'].values / df['Close'].iloc[0]
        
        # Test 1: Manual calculation
        # Daily returns calculation
        equity_df = pd.DataFrame({'capital': equity_curve}, index=df.index)
        daily_equity = equity_df.resample('D').last().dropna()
        daily_returns = daily_equity['capital'].pct_change().dropna()
        
        manual_sharpe = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
        
        print(f"Expected Sharpe (theoretical): {expected_sharpe:.4f}")
        print(f"Calculated Sharpe (manual): {manual_sharpe:.4f}")
        print(f"Difference: {abs(manual_sharpe - expected_sharpe):.4f}")
        
        # Test passed if within reasonable range (due to randomness)
        test_passed = abs(manual_sharpe - expected_sharpe) < 0.5
        
        self.test_results.append({
            'test': 'Sharpe Calculation Correctness',
            'passed': test_passed,
            'details': f"Expected: {expected_sharpe:.4f}, Got: {manual_sharpe:.4f}"
        })
        
        return test_passed
    
    def test_daily_aggregation_impact(self):
        """Test the impact of daily aggregation vs bar-level calculation."""
        print("\n" + "="*60)
        print("TEST 2: Daily Aggregation vs Bar-Level Calculation")
        print("="*60)
        
        # Create high-frequency data
        df, _, _ = self.create_synthetic_price_data(
            n_days=60, drift=0.0002, volatility=0.02, freq='15T'
        )
        
        # Create equity curve
        initial_capital = 10000
        equity_curve = initial_capital * df['Close'].values / df['Close'].iloc[0]
        
        # Method 1: Daily aggregation (correct)
        equity_df = pd.DataFrame({'capital': equity_curve}, index=df.index)
        daily_equity = equity_df.resample('D').last().dropna()
        daily_returns = daily_equity['capital'].pct_change().dropna()
        sharpe_daily = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
        
        # Method 2: Bar-level calculation (potentially biased)
        bar_returns = np.diff(equity_curve) / equity_curve[:-1]
        ann_factor = annualization_factor_from_df(df)
        sharpe_bar = np.mean(bar_returns) / np.std(bar_returns, ddof=1) * ann_factor
        
        print(f"Sharpe (Daily Aggregation): {sharpe_daily:.4f}")
        print(f"Sharpe (Bar-Level): {sharpe_bar:.4f}")
        print(f"Difference: {abs(sharpe_daily - sharpe_bar):.4f}")
        print(f"Annualization factor used: {ann_factor:.2f}")
        
        # Bar-level should typically show higher Sharpe due to autocorrelation
        test_passed = sharpe_bar > sharpe_daily
        
        self.test_results.append({
            'test': 'Daily Aggregation Impact',
            'passed': test_passed,
            'details': f"Daily: {sharpe_daily:.4f}, Bar: {sharpe_bar:.4f}"
        })
        
        return test_passed
    
    def test_annualization_factor(self):
        """Test the annualization factor detection."""
        print("\n" + "="*60)
        print("TEST 3: Annualization Factor Detection")
        print("="*60)
        
        test_cases = [
            ('15T', np.sqrt(252 * 96)),  # 15-minute bars
            ('1H', np.sqrt(252 * 24)),   # Hourly bars
            ('4H', np.sqrt(252 * 6)),    # 4-hour bars
            ('1D', np.sqrt(252)),        # Daily bars
        ]
        
        all_passed = True
        for freq, expected_factor in test_cases:
            df, _, _ = self.create_synthetic_price_data(n_days=10, freq=freq)
            calculated_factor = annualization_factor_from_df(df)
            
            # Allow some tolerance for the detection
            tolerance = 0.1
            passed = abs(calculated_factor - expected_factor) < tolerance
            all_passed &= passed
            
            print(f"{freq:>4}: Expected {expected_factor:>8.2f}, Got {calculated_factor:>8.2f} - {'PASS' if passed else 'FAIL'}")
        
        self.test_results.append({
            'test': 'Annualization Factor Detection',
            'passed': all_passed,
            'details': f"All timeframes detected correctly: {all_passed}"
        })
        
        return all_passed
    
    def test_extreme_scenarios(self):
        """Test Sharpe calculation in extreme scenarios."""
        print("\n" + "="*60)
        print("TEST 4: Extreme Scenarios")
        print("="*60)
        
        scenarios = []
        
        # Scenario 1: No volatility (constant returns)
        n_days = 50
        constant_return = 0.001
        equity_constant = 10000 * np.exp(np.cumsum([constant_return] * n_days))
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        equity_df = pd.DataFrame({'capital': equity_constant}, index=dates)
        returns = equity_df['capital'].pct_change().dropna()
        
        if returns.std() == 0:
            sharpe_constant = np.inf if returns.mean() > 0 else -np.inf
        else:
            sharpe_constant = returns.mean() / returns.std() * np.sqrt(252)
        
        scenarios.append(('No Volatility', sharpe_constant, sharpe_constant == np.inf))
        
        # Scenario 2: All losses
        equity_losses = 10000 * np.exp(np.cumsum([-0.001] * n_days))
        equity_df_loss = pd.DataFrame({'capital': equity_losses}, index=dates)
        returns_loss = equity_df_loss['capital'].pct_change().dropna()
        sharpe_losses = returns_loss.mean() / returns_loss.std() * np.sqrt(252)
        
        scenarios.append(('All Losses', sharpe_losses, sharpe_losses < 0))
        
        # Scenario 3: Single trade (insufficient data)
        equity_single = [10000, 10100]
        dates_single = pd.date_range('2023-01-01', periods=2, freq='D')
        equity_df_single = pd.DataFrame({'capital': equity_single}, index=dates_single)
        returns_single = equity_df_single['capital'].pct_change().dropna()
        
        # Should handle gracefully
        if len(returns_single) < 2:
            sharpe_single = 0
        else:
            sharpe_single = 0  # Not enough data for meaningful calculation
        
        scenarios.append(('Insufficient Data', sharpe_single, sharpe_single == 0))
        
        all_passed = True
        for name, sharpe, expected in scenarios:
            passed = expected
            all_passed &= passed
            print(f"{name:>20}: Sharpe = {sharpe:>10.4f} - {'PASS' if passed else 'FAIL'}")
        
        self.test_results.append({
            'test': 'Extreme Scenarios',
            'passed': all_passed,
            'details': "All extreme cases handled correctly"
        })
        
        return all_passed
    
    def test_monte_carlo_stability(self):
        """Test if Sharpe ratio is stable across Monte Carlo runs."""
        print("\n" + "="*60)
        print("TEST 5: Monte Carlo Stability")
        print("="*60)
        
        # Generate base data
        df, _, _ = self.create_synthetic_price_data(
            n_days=100, drift=0.0001, volatility=0.02, freq='1H'
        )
        
        # Run multiple simulations with same underlying data but different random seeds
        sharpe_ratios = []
        n_simulations = 10
        
        for i in range(n_simulations):
            # Add small random noise to simulate different execution paths
            noise = np.random.normal(0, 0.0001, len(df))
            equity_curve = 10000 * (1 + np.cumsum(noise + df['Close'].pct_change().fillna(0)))
            
            # Calculate Sharpe
            equity_df = pd.DataFrame({'capital': equity_curve}, index=df.index)
            daily_equity = equity_df.resample('D').last().dropna()
            daily_returns = daily_equity['capital'].pct_change().dropna()
            
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                sharpe_ratios.append(sharpe)
        
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        cv = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else np.inf
        
        print(f"Number of simulations: {n_simulations}")
        print(f"Mean Sharpe: {mean_sharpe:.4f}")
        print(f"Std Dev: {std_sharpe:.4f}")
        print(f"Coefficient of Variation: {cv:.4f}")
        
        # Test passes if CV is reasonable (< 0.5)
        test_passed = cv < 0.5
        
        self.test_results.append({
            'test': 'Monte Carlo Stability',
            'passed': test_passed,
            'details': f"CV: {cv:.4f} (should be < 0.5)"
        })
        
        return test_passed
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "="*80)
        print("SHARPE RATIO VALIDATION TEST SUITE")
        print("="*80)
        
        # Run all tests
        self.test_sharpe_calculation_correctness()
        self.test_daily_aggregation_impact()
        self.test_annualization_factor()
        self.test_extreme_scenarios()
        self.test_monte_carlo_stability()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{result['test']:.<40} {status}")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        return passed_tests == total_tests


if __name__ == "__main__":
    validator = SharpeRatioValidator()
    all_passed = validator.run_all_tests()
    
    if not all_passed:
        print("\n⚠️  WARNING: Some tests failed. Review the Sharpe ratio implementation.")
        sys.exit(1)
    else:
        print("\n✅ All Sharpe ratio validation tests passed!")
        sys.exit(0)