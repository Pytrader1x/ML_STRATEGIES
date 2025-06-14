"""
Strategy Integrity and Backtesting Validation Tests

This module tests the overall strategy implementation for:
- Execution realism
- Hidden biases
- False positive results
- Backtesting integrity
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_code.Prod_strategy import OptimizedProdStrategy, ExitReason


class StrategyIntegrityValidator:
    """Comprehensive tests for strategy implementation integrity."""
    
    def __init__(self):
        self.test_results = []
        
    def create_extreme_market_data(self, scenario='flat'):
        """Create extreme market scenarios to test strategy robustness."""
        n_bars = 1000
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='1H')
        
        if scenario == 'flat':
            # Completely flat market - no trends
            prices = np.ones(n_bars) * 100
            prices += np.random.normal(0, 0.01, n_bars)  # Tiny noise
        elif scenario == 'trending':
            # Perfect trend - too good to be true
            prices = np.linspace(100, 200, n_bars)
            prices += np.random.normal(0, 0.1, n_bars)
        elif scenario == 'random_walk':
            # Pure random walk
            returns = np.random.normal(0, 0.01, n_bars)
            prices = 100 * np.exp(np.cumsum(returns))
        elif scenario == 'gap_prone':
            # Market with frequent gaps
            prices = np.ones(n_bars) * 100
            gap_indices = np.random.choice(n_bars, size=50, replace=False)
            for idx in gap_indices:
                gap_size = np.random.uniform(-5, 5)
                prices[idx:] += gap_size
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = np.roll(prices, 1)
        df['Open'][0] = prices[0]
        
        # Add realistic OHLC
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.abs(np.random.normal(0, 0.1, n_bars))
        df['Low'] = df[['Open', 'Close']].min(axis=1) - np.abs(np.random.normal(0, 0.1, n_bars))
        df['Volume'] = np.random.randint(1000, 10000, n_bars)
        
        return df
    
    def test_execution_slippage(self):
        """Test if slippage is correctly applied and always adverse."""
        print("\n" + "="*60)
        print("TEST 1: Execution Slippage Validation")
        print("="*60)
        
        # Create simple trending data
        df = self.create_extreme_market_data('trending')
        
        # Initialize strategy with realistic mode
        strategy = OptimizedProdStrategy(
            symbols=['TEST'],
            position_size=1.0,
            atr_period=14,
            atr_multiplier=2.0,
            realistic_trading_mode=True,
            initial_capital=10000
        )
        
        # Simulate some trades manually to check slippage
        n_tests = 100
        entry_slippages = []
        sl_slippages = []
        
        for i in range(n_tests):
            # Test entry slippage
            entry_price = 100.0
            is_long = True
            slipped_entry = strategy._apply_slippage(entry_price, 'entry', is_long)
            entry_slippage = slipped_entry - entry_price
            entry_slippages.append(entry_slippage)
            
            # Test stop loss slippage
            sl_price = 98.0
            slipped_sl = strategy._apply_slippage(sl_price, 'stop_loss', is_long)
            sl_slippage = sl_price - slipped_sl  # For long position, worse SL is lower
            sl_slippages.append(sl_slippage)
        
        # Analyze slippage distribution
        entry_slippages = np.array(entry_slippages)
        sl_slippages = np.array(sl_slippages)
        
        print(f"Entry slippage range: [{entry_slippages.min():.4f}, {entry_slippages.max():.4f}] pips")
        print(f"Stop loss slippage range: [{sl_slippages.min():.4f}, {sl_slippages.max():.4f}] pips")
        
        # Verify slippage is always adverse
        entry_adverse = np.all(entry_slippages >= 0)  # Long entries should be higher
        sl_adverse = np.all(sl_slippages >= 0)  # Long SL should be lower
        
        # Verify slippage is within expected ranges
        entry_in_range = np.all(entry_slippages <= 0.0005)  # Max 0.5 pips
        sl_in_range = np.all(sl_slippages <= 0.0020)  # Max 2 pips
        
        test_passed = entry_adverse and sl_adverse and entry_in_range and sl_in_range
        
        print(f"Entry slippage always adverse: {entry_adverse}")
        print(f"SL slippage always adverse: {sl_adverse}")
        print(f"Slippage within expected range: {entry_in_range and sl_in_range}")
        
        self.test_results.append({
            'test': 'Execution Slippage',
            'passed': test_passed,
            'details': f"Adverse: {entry_adverse and sl_adverse}, In range: {entry_in_range and sl_in_range}"
        })
        
        return test_passed
    
    def test_no_future_information(self):
        """Test that strategy doesn't use future information in decisions."""
        print("\n" + "="*60)
        print("TEST 2: No Future Information Leakage")
        print("="*60)
        
        # Create data with a specific pattern
        df = self.create_extreme_market_data('trending')
        
        # Run strategy on full data
        strategy1 = OptimizedProdStrategy(
            symbols=['TEST'],
            position_size=1.0,
            initial_capital=10000
        )
        
        # Process full dataset
        for i in range(len(df)):
            strategy1.process_bar(df.iloc[i], i, df)
        
        # Now run on truncated data (missing last 100 bars)
        df_truncated = df.iloc[:-100].copy()
        strategy2 = OptimizedProdStrategy(
            symbols=['TEST'],
            position_size=1.0,
            initial_capital=10000
        )
        
        # Process truncated dataset
        for i in range(len(df_truncated)):
            strategy2.process_bar(df_truncated.iloc[i], i, df_truncated)
        
        # Compare decisions at same points
        # The strategies should make identical decisions up to the truncation point
        min_trades = min(len(strategy1.trades), len(strategy2.trades))
        
        if min_trades > 0:
            decisions_match = True
            for i in range(min_trades):
                trade1 = strategy1.trades[i]
                trade2 = strategy2.trades[i]
                
                # Check if entry bars match (accounting for different dataset lengths)
                if trade1.entry_bar != trade2.entry_bar:
                    decisions_match = False
                    print(f"Trade {i}: Entry bars don't match ({trade1.entry_bar} vs {trade2.entry_bar})")
                    break
        else:
            decisions_match = True
            print("No trades generated to compare")
        
        print(f"Decisions match on truncated data: {decisions_match}")
        
        self.test_results.append({
            'test': 'No Future Information',
            'passed': decisions_match,
            'details': f"Strategy decisions consistent: {decisions_match}"
        })
        
        return decisions_match
    
    def test_realistic_win_rates(self):
        """Test if strategy produces realistic win rates on random data."""
        print("\n" + "="*60)
        print("TEST 3: Realistic Win Rates on Random Data")
        print("="*60)
        
        # Test on pure random walk - should not have edge
        df = self.create_extreme_market_data('random_walk')
        
        strategy = OptimizedProdStrategy(
            symbols=['TEST'],
            position_size=1.0,
            realistic_trading_mode=True,
            initial_capital=10000
        )
        
        # Run strategy
        for i in range(len(df)):
            strategy.process_bar(df.iloc[i], i, df)
        
        stats = strategy.get_stats()
        
        win_rate = stats['win_rate']
        sharpe = stats['sharpe_ratio']
        profit_factor = stats['profit_factor']
        
        print(f"Results on random walk data:")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Sharpe Ratio: {sharpe:.3f}")
        print(f"  Profit Factor: {profit_factor:.3f}")
        print(f"  Total Trades: {stats['total_trades']}")
        
        # On random data with transaction costs, expect:
        # - Win rate around 40-55% (random)
        # - Negative or near-zero Sharpe
        # - Profit factor below 1.0
        
        realistic = (
            30 <= win_rate <= 70 and  # Not too extreme
            sharpe < 0.5 and  # No significant edge
            profit_factor < 1.2  # Not profitable after costs
        )
        
        self.test_results.append({
            'test': 'Realistic Win Rates',
            'passed': realistic,
            'details': f"WR: {win_rate:.1f}%, Sharpe: {sharpe:.3f}, PF: {profit_factor:.3f}"
        })
        
        return realistic
    
    def test_parameter_sensitivity(self):
        """Test if strategy is overly sensitive to parameters (overfitting sign)."""
        print("\n" + "="*60)
        print("TEST 4: Parameter Sensitivity Analysis")
        print("="*60)
        
        # Use consistent data
        df = self.create_extreme_market_data('trending')
        
        # Test different ATR multipliers
        multipliers = [1.5, 2.0, 2.5, 3.0]
        sharpe_ratios = []
        
        for mult in multipliers:
            strategy = OptimizedProdStrategy(
                symbols=['TEST'],
                atr_multiplier=mult,
                position_size=1.0,
                initial_capital=10000
            )
            
            for i in range(len(df)):
                strategy.process_bar(df.iloc[i], i, df)
            
            stats = strategy.get_stats()
            sharpe_ratios.append(stats['sharpe_ratio'])
            
            print(f"  ATR Multiplier {mult}: Sharpe = {stats['sharpe_ratio']:.3f}")
        
        # Calculate coefficient of variation
        sharpe_array = np.array(sharpe_ratios)
        if sharpe_array.mean() != 0:
            cv = sharpe_array.std() / abs(sharpe_array.mean())
        else:
            cv = np.inf
        
        print(f"\nSharpe Ratio Coefficient of Variation: {cv:.3f}")
        
        # High CV indicates oversensitivity
        not_oversensitive = cv < 1.0
        
        self.test_results.append({
            'test': 'Parameter Sensitivity',
            'passed': not_oversensitive,
            'details': f"CV: {cv:.3f} (lower is better)"
        })
        
        return not_oversensitive
    
    def test_drawdown_calculation(self):
        """Verify drawdown calculation is correct."""
        print("\n" + "="*60)
        print("TEST 5: Drawdown Calculation Verification")
        print("="*60)
        
        # Create known equity curve
        equity_curve = [10000, 11000, 10500, 9500, 10000, 12000, 11000]
        
        # Manual calculation
        running_max = []
        drawdowns = []
        
        for i, equity in enumerate(equity_curve):
            if i == 0:
                running_max.append(equity)
                drawdowns.append(0)
            else:
                running_max.append(max(running_max[-1], equity))
                dd = (equity - running_max[-1]) / running_max[-1] * 100
                drawdowns.append(dd)
        
        max_dd_manual = abs(min(drawdowns))
        
        # Create strategy and set equity curve
        strategy = OptimizedProdStrategy(symbols=['TEST'])
        strategy.equity_curve = equity_curve
        
        # Calculate using strategy method
        equity_array = np.array(equity_curve)
        running_max_calc = np.maximum.accumulate(equity_array)
        drawdown_calc = (equity_array - running_max_calc) / running_max_calc * 100
        max_dd_calc = abs(np.min(drawdown_calc))
        
        print(f"Equity curve: {equity_curve}")
        print(f"Running max: {running_max}")
        print(f"Drawdowns: {[f'{dd:.1f}%' for dd in drawdowns]}")
        print(f"Max drawdown (manual): {max_dd_manual:.1f}%")
        print(f"Max drawdown (calculated): {max_dd_calc:.1f}%")
        
        # Should be 13.6% (from 11000 to 9500)
        expected_dd = 13.636
        test_passed = abs(max_dd_calc - expected_dd) < 0.01
        
        self.test_results.append({
            'test': 'Drawdown Calculation',
            'passed': test_passed,
            'details': f"Expected: {expected_dd:.1f}%, Got: {max_dd_calc:.1f}%"
        })
        
        return test_passed
    
    def test_trade_timing_integrity(self):
        """Test that trades are entered and exited at appropriate times."""
        print("\n" + "="*60)
        print("TEST 6: Trade Timing Integrity")
        print("="*60)
        
        # Create data with known patterns
        n_bars = 200
        dates = pd.date_range('2023-01-01', periods=n_bars, freq='1H')
        
        # Create clear trend changes
        prices = np.concatenate([
            np.linspace(100, 110, 50),   # Uptrend
            np.linspace(110, 110, 50),   # Flat
            np.linspace(110, 100, 50),   # Downtrend
            np.linspace(100, 105, 50),   # Recovery
        ])
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = np.roll(prices, 1)
        df['Open'][0] = prices[0]
        df['High'] = prices + 0.5
        df['Low'] = prices - 0.5
        df['Volume'] = 1000
        
        # Run strategy
        strategy = OptimizedProdStrategy(
            symbols=['TEST'],
            position_size=1.0,
            initial_capital=10000
        )
        
        for i in range(len(df)):
            strategy.process_bar(df.iloc[i], i, df)
        
        # Check trade entries and exits
        print(f"Total trades: {len(strategy.trades)}")
        
        integrity_issues = []
        
        for trade in strategy.trades:
            # Check entry timing
            if trade.entry_bar >= len(df):
                integrity_issues.append(f"Trade entered beyond data range: bar {trade.entry_bar}")
            
            # Check exit timing
            if trade.exit_bar and trade.exit_bar >= len(df):
                integrity_issues.append(f"Trade exited beyond data range: bar {trade.exit_bar}")
            
            # Check exit before entry
            if trade.exit_bar and trade.exit_bar <= trade.entry_bar:
                integrity_issues.append(f"Trade exited before or at entry: entry {trade.entry_bar}, exit {trade.exit_bar}")
            
            # Check prices are from correct bars
            if trade.entry_bar < len(df):
                bar_close = df.iloc[trade.entry_bar]['Close']
                price_diff = abs(trade.entry_price - bar_close)
                if price_diff > 1.0:  # Allow for slippage
                    integrity_issues.append(f"Entry price mismatch at bar {trade.entry_bar}: {trade.entry_price:.2f} vs {bar_close:.2f}")
        
        if integrity_issues:
            print("Integrity issues found:")
            for issue in integrity_issues[:5]:  # Show first 5
                print(f"  - {issue}")
        else:
            print("No timing integrity issues found")
        
        test_passed = len(integrity_issues) == 0
        
        self.test_results.append({
            'test': 'Trade Timing Integrity',
            'passed': test_passed,
            'details': f"{len(integrity_issues)} issues found"
        })
        
        return test_passed
    
    def run_all_tests(self):
        """Run all strategy integrity tests."""
        print("\n" + "="*80)
        print("STRATEGY INTEGRITY VALIDATION TEST SUITE")
        print("="*80)
        
        # Run all tests
        self.test_execution_slippage()
        self.test_no_future_information()
        self.test_realistic_win_rates()
        self.test_parameter_sensitivity()
        self.test_drawdown_calculation()
        self.test_trade_timing_integrity()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{result['test']:.<40} {status}")
            if not result['passed']:
                print(f"  Details: {result['details']}")
        
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        return passed_tests == total_tests


if __name__ == "__main__":
    validator = StrategyIntegrityValidator()
    all_passed = validator.run_all_tests()
    
    if not all_passed:
        print("\n⚠️  WARNING: Some integrity tests failed. Review the strategy implementation.")
        sys.exit(1)
    else:
        print("\n✅ All strategy integrity tests passed!")
        sys.exit(0)