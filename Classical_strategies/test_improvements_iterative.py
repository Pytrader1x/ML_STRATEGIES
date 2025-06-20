"""
Iterative Strategy Improvement Tester
Tests individual improvements one by one to measure impact
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

warnings.filterwarnings('ignore')

class StrategyTester:
    def __init__(self, currency='AUDUSD', test_days=30):
        self.currency = currency
        self.test_days = test_days
        self.results = []
        self.load_data()
        
    def load_data(self):
        """Load and prepare test data"""
        print(f"Loading {self.currency} data...")
        data_path = '../data' if os.path.exists('../data') else 'data'
        self.df = pd.read_csv(f'{data_path}/{self.currency}_MASTER_15M.csv')
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        # Use recent data for faster testing
        self.df = self.df.tail(self.test_days * 96)  # 96 bars per day
        
        # Calculate indicators
        print("Calculating indicators...")
        self.df = TIC.add_neuro_trend_intelligent(self.df)
        self.df = TIC.add_market_bias(self.df)
        self.df = TIC.add_intelligent_chop(self.df)
        print(f"Test data ready: {len(self.df)} rows\n")
    
    def test_baseline(self):
        """Test baseline validated strategy"""
        print("="*60)
        print("TEST 1: BASELINE (Current Validated Strategy)")
        print("="*60)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            base_position_size_millions=1.0,
            
            # Original settings
            sl_min_pips=3.0,
            sl_max_pips=10.0,
            sl_atr_multiplier=0.8,
            
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.3,  # 30% to SL
            partial_profit_size_percent=0.7,  # 70% off
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            intelligent_sizing=False,  # OFF
            realistic_costs=True,
            
            verbose=False
        )
        
        strategy = OptimizedProdStrategy(config)
        result = strategy.run_backtest(self.df)
        
        self.results.append({
            'Test': 'Baseline',
            'Changes': 'Original validated settings',
            'Sharpe': result['sharpe_ratio'],
            'Return': result['total_return'],
            'Win Rate': result['win_rate'],
            'Trades': result['total_trades'],
            'PnL': result['total_pnl']
        })
        
        print(f"Results: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.1f}%, Win Rate={result['win_rate']:.1f}%")
        return result
    
    def test_institutional_sizing(self):
        """Test with institutional position sizing (1M relaxed, 2M standard)"""
        print("\n" + "="*60)
        print("TEST 2: INSTITUTIONAL SIZING")
        print("Changes: 1M for relaxed entries, 2M for standard entries")
        print("="*60)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            base_position_size_millions=2.0,  # CHANGED: 2M base
            
            sl_min_pips=3.0,
            sl_max_pips=10.0,
            sl_atr_multiplier=0.8,
            
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.3,
            partial_profit_size_percent=0.7,
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,  # 2M * 0.5 = 1M
            
            intelligent_sizing=False,
            realistic_costs=True,
            
            verbose=False
        )
        
        strategy = OptimizedProdStrategy(config)
        result = strategy.run_backtest(self.df)
        
        self.results.append({
            'Test': 'Institutional Sizing',
            'Changes': '1M relaxed, 2M standard',
            'Sharpe': result['sharpe_ratio'],
            'Return': result['total_return'],
            'Win Rate': result['win_rate'],
            'Trades': result['total_trades'],
            'PnL': result['total_pnl']
        })
        
        print(f"Results: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.1f}%, Win Rate={result['win_rate']:.1f}%")
        return result
    
    def test_improved_partial_profit(self):
        """Test with better partial profit logic"""
        print("\n" + "="*60)
        print("TEST 3: IMPROVED PARTIAL PROFIT")
        print("Changes: Take 40% at 60% to TP1 (not 70% at 30% to SL)")
        print("="*60)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            base_position_size_millions=2.0,
            
            sl_min_pips=3.0,
            sl_max_pips=10.0,
            sl_atr_multiplier=0.8,
            
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.6,  # CHANGED: 60% to TP1
            partial_profit_size_percent=0.4,  # CHANGED: 40% off
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            intelligent_sizing=False,
            realistic_costs=True,
            
            verbose=False
        )
        
        strategy = OptimizedProdStrategy(config)
        result = strategy.run_backtest(self.df)
        
        self.results.append({
            'Test': 'Improved Partial',
            'Changes': '40% off at 60% to TP1',
            'Sharpe': result['sharpe_ratio'],
            'Return': result['total_return'],
            'Win Rate': result['win_rate'],
            'Trades': result['total_trades'],
            'PnL': result['total_pnl']
        })
        
        print(f"Results: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.1f}%, Win Rate={result['win_rate']:.1f}%")
        return result
    
    def test_wider_stops(self):
        """Test with slightly wider stops"""
        print("\n" + "="*60)
        print("TEST 4: WIDER STOPS")
        print("Changes: 5-15 pip stops (was 3-10)")
        print("="*60)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            base_position_size_millions=2.0,
            
            sl_min_pips=5.0,  # CHANGED: was 3
            sl_max_pips=15.0,  # CHANGED: was 10
            sl_atr_multiplier=0.8,
            
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.6,
            partial_profit_size_percent=0.4,
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            intelligent_sizing=False,
            realistic_costs=True,
            
            verbose=False
        )
        
        strategy = OptimizedProdStrategy(config)
        result = strategy.run_backtest(self.df)
        
        self.results.append({
            'Test': 'Wider Stops',
            'Changes': '5-15 pip stops',
            'Sharpe': result['sharpe_ratio'],
            'Return': result['total_return'],
            'Win Rate': result['win_rate'],
            'Trades': result['total_trades'],
            'PnL': result['total_pnl']
        })
        
        print(f"Results: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.1f}%, Win Rate={result['win_rate']:.1f}%")
        return result
    
    def test_intelligent_sizing(self):
        """Test with intelligent sizing enabled"""
        print("\n" + "="*60)
        print("TEST 5: INTELLIGENT SIZING")
        print("Changes: Scale position with NTI confidence")
        print("="*60)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.005,
            base_position_size_millions=2.0,
            
            sl_min_pips=5.0,
            sl_max_pips=15.0,
            sl_atr_multiplier=0.8,
            
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.6,
            partial_profit_size_percent=0.4,
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            intelligent_sizing=True,  # CHANGED: ON
            confidence_thresholds=(40.0, 60.0, 80.0),
            size_multipliers=(0.5, 0.75, 1.0, 1.25),  # CHANGED
            
            realistic_costs=True,
            
            verbose=False
        )
        
        strategy = OptimizedProdStrategy(config)
        result = strategy.run_backtest(self.df)
        
        self.results.append({
            'Test': 'Intelligent Sizing',
            'Changes': 'Confidence-based sizing',
            'Sharpe': result['sharpe_ratio'],
            'Return': result['total_return'],
            'Win Rate': result['win_rate'],
            'Trades': result['total_trades'],
            'PnL': result['total_pnl']
        })
        
        print(f"Results: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.1f}%, Win Rate={result['win_rate']:.1f}%")
        return result
    
    def test_combined_best(self):
        """Test with all best improvements combined"""
        print("\n" + "="*60)
        print("TEST 6: COMBINED BEST FEATURES")
        print("Changes: All improvements that showed positive impact")
        print("="*60)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.003,  # Slightly more conservative
            base_position_size_millions=2.0,
            
            sl_min_pips=5.0,
            sl_max_pips=15.0,
            sl_atr_multiplier=1.0,  # Less tight
            
            tp_atr_multipliers=(0.3, 0.5, 0.8),  # Wider TPs
            max_tp_percent=0.008,
            
            tsl_activation_pips=10.0,  # Later activation
            tsl_min_profit_pips=2.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.5,  # Middle ground
            partial_profit_size_percent=0.5,  # 50-50 split
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            intelligent_sizing=True,
            confidence_thresholds=(40.0, 60.0, 80.0),
            size_multipliers=(0.5, 0.75, 1.0, 1.5),
            
            # Market adaptations
            tp_range_market_multiplier=0.5,
            tp_trend_market_multiplier=1.0,
            tp_chop_market_multiplier=0.3,
            
            sl_volatility_adjustment=True,
            
            realistic_costs=True,
            
            verbose=False
        )
        
        strategy = OptimizedProdStrategy(config)
        result = strategy.run_backtest(self.df)
        
        self.results.append({
            'Test': 'Combined Best',
            'Changes': 'All improvements',
            'Sharpe': result['sharpe_ratio'],
            'Return': result['total_return'],
            'Win Rate': result['win_rate'],
            'Trades': result['total_trades'],
            'PnL': result['total_pnl']
        })
        
        print(f"Results: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.1f}%, Win Rate={result['win_rate']:.1f}%")
        return result
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("ITERATIVE IMPROVEMENT SUMMARY")
        print("="*80)
        
        df_results = pd.DataFrame(self.results)
        print(tabulate(df_results, headers='keys', tablefmt='grid', floatfmt='.2f'))
        
        # Find best configuration
        best_idx = df_results['Sharpe'].idxmax()
        best_test = df_results.iloc[best_idx]
        
        print(f"\n🏆 BEST CONFIGURATION: {best_test['Test']}")
        print(f"   Sharpe: {best_test['Sharpe']:.3f}")
        print(f"   Return: {best_test['Return']:.1f}%")
        print(f"   Win Rate: {best_test['Win Rate']:.1f}%")
        
        # Calculate improvements
        baseline_sharpe = df_results.iloc[0]['Sharpe']
        improvement = ((best_test['Sharpe'] - baseline_sharpe) / abs(baseline_sharpe)) * 100
        print(f"\n📈 Improvement over baseline: {improvement:.1f}%")


def main():
    import os
    print("🔬 ITERATIVE STRATEGY IMPROVEMENT TESTER")
    print("Testing each improvement individually to measure impact")
    print("="*60)
    
    tester = StrategyTester(currency='AUDUSD', test_days=30)
    
    # Run tests sequentially
    tester.test_baseline()
    tester.test_institutional_sizing()
    tester.test_improved_partial_profit()
    tester.test_wider_stops()
    tester.test_intelligent_sizing()
    tester.test_combined_best()
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    main()