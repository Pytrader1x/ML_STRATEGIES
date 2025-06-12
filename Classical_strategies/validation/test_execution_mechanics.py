"""
Test Execution Mechanics - Pinpoint the exact issue
Focus on entry/exit prices and spread implementation
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, Trade, TradeDirection
from technical_indicators_custom import TIC
import warnings

warnings.filterwarnings('ignore')

class ExecutionTester:
    
    def test_entry_price_mechanics(self):
        """Test exactly how entry prices are determined"""
        print("="*80)
        print("ENTRY PRICE MECHANICS TEST")
        print("="*80)
        
        # Create minimal test data (need more for indicators)
        dates = pd.date_range('2023-01-01', periods=500, freq='15min')
        test_df = pd.DataFrame({
            'Open': [1.0000 + i*0.0001 for i in range(500)],
            'High': [1.0010 + i*0.0001 for i in range(500)],
            'Low': [0.9990 + i*0.0001 for i in range(500)],
            'Close': [1.0000 + i*0.0001 for i in range(500)],
            'Volume': [1000] * 500
        }, index=dates)
        
        # Add indicators
        test_df = TIC.add_neuro_trend_intelligent(test_df)
        test_df = TIC.add_market_bias(test_df)
        test_df = TIC.add_intelligent_chop(test_df)
        
        # Force a trade signal
        test_idx = 250  # Middle of data
        test_df.loc[test_df.index[test_idx], 'NTI_Direction'] = 1
        test_df.loc[test_df.index[test_idx], 'MB_Bias'] = 1
        test_df.loc[test_df.index[test_idx], 'IC_Signal'] = 1
        
        print(f"\nTest bar at index {test_idx}:")
        print(f"Open:  {test_df.iloc[test_idx]['Open']:.5f}")
        print(f"High:  {test_df.iloc[test_idx]['High']:.5f}")
        print(f"Low:   {test_df.iloc[test_idx]['Low']:.5f}")
        print(f"Close: {test_df.iloc[test_idx]['Close']:.5f}")
        
        # Run strategy
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            verbose=True
        )
        strategy = OptimizedProdStrategy(config)
        results = strategy.run_backtest(test_df)
        
        # Analyze trades
        if results['trades']:
            trade = results['trades'][0]
            print(f"\n✅ Trade executed:")
            print(f"Entry price: {trade.entry_price:.5f}")
            print(f"Entry used: {'CLOSE' if trade.entry_price == test_df.iloc[test_idx]['Close'] else 'OTHER'}")
            
            # Check if using next bar's open
            if len(test_df) > test_idx+1 and abs(trade.entry_price - test_df.iloc[test_idx+1]['Open']) < 0.00001:
                print("⚠️  Entry at NEXT bar's open!")
        else:
            print("\n❌ No trades executed")
    
    def test_spread_implementation(self):
        """Test if spread is being applied"""
        print("\n" + "="*80)
        print("SPREAD IMPLEMENTATION TEST")
        print("="*80)
        
        # Create flat price data
        dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
        
        # Scenario 1: Perfectly flat market
        flat_df = pd.DataFrame({
            'Open': [1.0000] * 1000,
            'High': [1.0001] * 1000,
            'Low': [0.9999] * 1000,
            'Close': [1.0000] * 1000,
            'Volume': [1000] * 1000
        }, index=dates)
        
        flat_df = TIC.add_neuro_trend_intelligent(flat_df)
        flat_df = TIC.add_market_bias(flat_df)
        flat_df = TIC.add_intelligent_chop(flat_df)
        
        # Test with no spread
        config_no_spread = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            verbose=False
        )
        
        # Create random entry strategy
        class RandomFlatStrategy(OptimizedProdStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.trade_count = 0
                
            def generate_signal(self, row, prev_row=None):
                # Trade every 50 bars, alternating direction
                self.trade_count += 1
                if self.trade_count % 50 == 0:
                    return 1 if (self.trade_count // 50) % 2 == 1 else -1
                return 0
        
        strategy = RandomFlatStrategy(config_no_spread)
        results = strategy.run_backtest(flat_df)
        
        print(f"\nFlat market test (no movement):")
        print(f"Total trades: {results['total_trades']}")
        print(f"Total P&L: ${results['total_pnl']:.2f}")
        print(f"Expected P&L: $0 (flat market, no spread)")
        
        if abs(results['total_pnl']) > 100:
            print("⚠️  WARNING: P&L in flat market indicates spread/commission issues!")
        
        # Analyze individual trades
        if results['trades']:
            print(f"\nFirst 5 trades:")
            for i, trade in enumerate(results['trades'][:5]):
                print(f"Trade {i+1}: Entry={trade.entry_price:.5f}, Exit={trade.exit_price:.5f}, P&L=${trade.pnl:.2f}")
    
    def test_pnl_calculation(self):
        """Test P&L calculation logic"""
        print("\n" + "="*80)
        print("P&L CALCULATION TEST")
        print("="*80)
        
        # Direct P&L calculation test
        from strategy_code.Prod_strategy import PnLCalculator, OptimizedStrategyConfig, TradeDirection
        
        config = OptimizedStrategyConfig()
        calc = PnLCalculator(config)
        
        # Test cases
        test_cases = [
            # (entry, exit, size, direction, expected_pnl)
            (1.0000, 1.0010, 1_000_000, TradeDirection.LONG, 1000),   # 10 pip win long
            (1.0000, 0.9990, 1_000_000, TradeDirection.LONG, -1000),  # 10 pip loss long
            (1.0000, 0.9990, 1_000_000, TradeDirection.SHORT, 1000),  # 10 pip win short
            (1.0000, 1.0010, 1_000_000, TradeDirection.SHORT, -1000), # 10 pip loss short
        ]
        
        print("Testing P&L calculations:")
        for entry, exit, size, direction, expected in test_cases:
            pnl, pips = calc.calculate_pnl(entry, exit, size, direction)
            print(f"\n{direction.value.upper()}: Entry={entry:.4f}, Exit={exit:.4f}")
            print(f"  Pips: {pips:.1f}")
            print(f"  P&L: ${pnl:.2f}")
            print(f"  Expected: ${expected:.2f}")
            print(f"  ✅ PASS" if abs(pnl - expected) < 0.01 else f"  ❌ FAIL")
    
    def test_weekend_filter(self):
        """Test if weekend bars are being filtered"""
        print("\n" + "="*80)
        print("WEEKEND FILTER TEST")
        print("="*80)
        
        # Load real data
        df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Check weekend bars
        weekend_bars = df[df.index.dayofweek >= 5]
        print(f"Weekend bars in data: {len(weekend_bars)}")
        
        if len(weekend_bars) > 0:
            print("\nSample weekend bars:")
            print(weekend_bars.head())
            
            # Check if strategy trades on weekends
            weekend_sample = weekend_bars.iloc[:1000].copy()
            weekend_sample = TIC.add_neuro_trend_intelligent(weekend_sample)
            weekend_sample = TIC.add_market_bias(weekend_sample)
            weekend_sample = TIC.add_intelligent_chop(weekend_sample)
            
            config = OptimizedStrategyConfig(
                initial_capital=100_000,
                risk_per_trade=0.002,
                verbose=False
            )
            strategy = OptimizedProdStrategy(config)
            results = strategy.run_backtest(weekend_sample)
            
            print(f"\nTrades on weekend data: {results['total_trades']}")
            if results['total_trades'] > 0:
                print("⚠️  WARNING: Strategy trades on weekends when markets are closed!")
    
    def test_with_synthetic_data(self):
        """Test with perfectly controlled synthetic data"""
        print("\n" + "="*80)
        print("SYNTHETIC DATA TEST")
        print("="*80)
        
        # Create trending data with known outcome
        dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
        
        # Uptrend: price increases by 1 pip per bar
        prices = [1.0000 + i*0.0001 for i in range(1000)]
        
        trend_df = pd.DataFrame({
            'Open': prices,
            'High': [p + 0.0002 for p in prices],
            'Low': [p - 0.0002 for p in prices],
            'Close': prices,
            'Volume': [1000] * 1000
        }, index=dates)
        
        trend_df = TIC.add_neuro_trend_intelligent(trend_df)
        trend_df = TIC.add_market_bias(trend_df)
        trend_df = TIC.add_intelligent_chop(trend_df)
        
        # Buy and hold calculation
        bh_return = (prices[-1] - prices[0]) / prices[0] * 100
        print(f"Buy & Hold return: {bh_return:.2f}%")
        print(f"Total price movement: {(prices[-1] - prices[0]) * 10000:.0f} pips")
        
        # Test strategy
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            verbose=False
        )
        
        # Force long-only trades
        class LongOnlyStrategy(OptimizedProdStrategy):
            def generate_signal(self, row, prev_row=None):
                # Only long signals
                if hasattr(self, 'last_signal_bar'):
                    if self.current_bar_index - self.last_signal_bar < 100:
                        return 0
                
                if row.get('NTI_Direction', 0) > 0:
                    self.last_signal_bar = getattr(self, 'current_bar_index', 0)
                    return 1
                return 0
        
        strategy = LongOnlyStrategy(config)
        results = strategy.run_backtest(trend_df)
        
        print(f"\nStrategy results:")
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.1f}%")
        print(f"Total return: {results['total_return']:.1f}%")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        
        # In perfect uptrend, long-only should be profitable
        if results['total_return'] < 0:
            print("⚠️  WARNING: Strategy loses money in perfect uptrend!")
    
    def run_all_tests(self):
        """Run all execution tests"""
        self.test_entry_price_mechanics()
        self.test_spread_implementation()
        self.test_pnl_calculation()
        self.test_weekend_filter()
        self.test_with_synthetic_data()
        
        print("\n" + "="*80)
        print("EXECUTION MECHANICS TEST COMPLETE")
        print("="*80)


def main():
    tester = ExecutionTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()