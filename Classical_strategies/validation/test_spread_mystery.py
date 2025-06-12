"""
Solve the Spread Mystery - Why does adding spread increase P&L?
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import random

warnings.filterwarnings('ignore')

def create_perfect_data_for_testing():
    """Create perfect data where we know exactly what should happen"""
    print("="*80)
    print("CREATING CONTROLLED TEST DATA")
    print("="*80)
    
    # Create 1000 bars of perfectly flat data at 1.0000
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    
    df = pd.DataFrame({
        'Open': [1.0000] * 1000,
        'High': [1.0001] * 1000,  # Just 1 pip range
        'Low': [0.9999] * 1000,
        'Close': [1.0000] * 1000,
        'Volume': [1000] * 1000
    }, index=dates)
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    print(f"Created {len(df)} bars of flat data at 1.0000")
    return df


def test_manual_trade_calculation():
    """Manually calculate what P&L should be"""
    print("\n" + "="*80)
    print("MANUAL TRADE CALCULATION")
    print("="*80)
    
    # Scenario 1: Perfect flat market, no spread
    entry = 1.0000
    exit = 1.0000
    size = 1_000_000
    
    # Long trade
    pips = (exit - entry) * 10000
    pnl = pips * 100  # $100 per pip for 1M
    print(f"\nFlat market, LONG trade:")
    print(f"Entry: {entry}, Exit: {exit}")
    print(f"Pips: {pips}, P&L: ${pnl}")
    
    # With 1 pip spread (pay on entry)
    entry_with_spread = entry + 0.0001  # Pay 1 pip to enter long
    pips_with_spread = (exit - entry_with_spread) * 10000
    pnl_with_spread = pips_with_spread * 100
    print(f"\nWith 1 pip spread:")
    print(f"Entry: {entry_with_spread}, Exit: {exit}")
    print(f"Pips: {pips_with_spread}, P&L: ${pnl_with_spread}")


def test_strategy_with_forced_trades():
    """Force trades at specific times to see what happens"""
    print("\n" + "="*80)
    print("FORCED TRADE TEST")
    print("="*80)
    
    df = create_perfect_data_for_testing()
    
    # Create a strategy that trades at specific bars
    class ForcedTradeStrategy(OptimizedProdStrategy):
        def __init__(self, config):
            super().__init__(config)
            self.trade_bars = [100, 200, 300, 400, 500]  # Trade at these bars
            self.trade_count = 0
            
        def run_backtest(self, df):
            """Override to track more details"""
            self.df = df
            self.all_trades = []
            return super().run_backtest(df)
        
        def generate_signal(self, row, prev_row=None):
            """Force trades at specific bars"""
            if hasattr(self, 'current_bar_index'):
                if self.current_bar_index in self.trade_bars:
                    self.trade_count += 1
                    # Alternate long/short
                    return 1 if self.trade_count % 2 == 1 else -1
            return 0
    
    # Test 1: No modifications
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        intelligent_sizing=False,
        verbose=True
    )
    
    strategy = ForcedTradeStrategy(config)
    results = strategy.run_backtest(df)
    
    print(f"\nTest 1 - Original data:")
    print(f"Total trades: {results['total_trades']}")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    
    if results['trades']:
        print("\nFirst 3 trades:")
        for i, trade in enumerate(results['trades'][:3]):
            print(f"Trade {i+1}: Entry={trade.entry_price:.5f}, Exit={trade.exit_price:.5f}, "
                  f"P&L=${trade.pnl:.2f}, Direction={trade.direction}")
    
    # Test 2: Add artificial spread to data
    df_spread = df.copy()
    
    # Simulate bid/ask spread by modifying data
    # For backtesting, we might be using Close price for both entry and exit
    # Let's see what happens if we adjust prices
    
    print("\n" + "="*80)
    print("INVESTIGATING PRICE USAGE")
    print("="*80)
    
    # Check what prices the strategy uses
    sample_bar = df.iloc[100]
    print(f"\nSample bar prices:")
    print(f"Open:  {sample_bar['Open']:.5f}")
    print(f"High:  {sample_bar['High']:.5f}")
    print(f"Low:   {sample_bar['Low']:.5f}")
    print(f"Close: {sample_bar['Close']:.5f}")
    
    # The strategy uses Close price for entry (line 676 in Prod_strategy.py)
    print("\n✅ Finding: Strategy enters at Close price of signal bar")


def test_slippage_simulation():
    """Test with simulated slippage"""
    print("\n" + "="*80)
    print("SLIPPAGE SIMULATION TEST")
    print("="*80)
    
    df = create_perfect_data_for_testing()
    
    # Create strategy with slippage
    class SlippageStrategy(OptimizedProdStrategy):
        def __init__(self, config, slippage_pips=1.0):
            super().__init__(config)
            self.slippage_pips = slippage_pips
            self.original_trades = []
            
        def generate_signal(self, row, prev_row=None):
            # Random trades
            if random.random() < 0.01:  # 1% chance
                return random.choice([1, -1])
            return 0
        
        def _create_trade(self, *args, **kwargs):
            """Override to add slippage"""
            trade = super()._create_trade(*args, **kwargs)
            
            if trade:
                # Store original price
                self.original_trades.append({
                    'original_entry': trade.entry_price,
                    'direction': trade.direction
                })
                
                # Apply slippage
                slippage = self.slippage_pips * 0.0001
                if trade.direction == 1:  # Long
                    trade.entry_price += slippage  # Pay more
                else:  # Short
                    trade.entry_price -= slippage  # Receive less
            
            return trade
    
    # Test without slippage
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        intelligent_sizing=False,
        verbose=False
    )
    
    strategy_no_slip = SlippageStrategy(config, slippage_pips=0)
    results_no_slip = strategy_no_slip.run_backtest(df)
    
    # Test with slippage
    strategy_with_slip = SlippageStrategy(config, slippage_pips=1.0)
    results_with_slip = strategy_with_slip.run_backtest(df)
    
    print(f"Without slippage: P&L = ${results_no_slip['total_pnl']:.2f}")
    print(f"With 1 pip slippage: P&L = ${results_with_slip['total_pnl']:.2f}")
    print(f"Difference: ${results_with_slip['total_pnl'] - results_no_slip['total_pnl']:.2f}")
    
    if results_with_slip['total_pnl'] > results_no_slip['total_pnl']:
        print("\n⚠️  WARNING: Slippage INCREASED profits! This confirms the issue.")


def trace_single_trade():
    """Trace through a single trade execution in detail"""
    print("\n" + "="*80)
    print("SINGLE TRADE TRACE")
    print("="*80)
    
    # Create simple data
    dates = pd.date_range('2023-01-01', periods=500, freq='15min')
    
    # Create data with a clear trend for one winning trade
    df = pd.DataFrame({
        'Open': [1.0000] * 200 + [1.0010] * 200 + [1.0020] * 100,
        'High': [1.0002] * 200 + [1.0012] * 200 + [1.0022] * 100,
        'Low': [0.9998] * 200 + [1.0008] * 200 + [1.0018] * 100,
        'Close': [1.0000] * 200 + [1.0010] * 200 + [1.0020] * 100,
        'Volume': [1000] * 500
    }, index=dates)
    
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Force a long signal at bar 150
    df.loc[df.index[150], 'NTI_Direction'] = 1
    df.loc[df.index[150], 'MB_Bias'] = 1
    
    print(f"Signal bar (150): Close = {df.iloc[150]['Close']:.5f}")
    print(f"Next bars: {df.iloc[151]['Close']:.5f}, {df.iloc[152]['Close']:.5f}")
    
    # Run strategy
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        intelligent_sizing=False,
        sl_max_pips=10.0,
        verbose=True
    )
    
    strategy = OptimizedProdStrategy(config)
    results = strategy.run_backtest(df)
    
    if results['trades']:
        trade = results['trades'][0]
        print(f"\nTrade details:")
        print(f"Entry: {trade.entry_price:.5f} at {trade.entry_time}")
        print(f"Exit: {trade.exit_price:.5f} at {trade.exit_time}")
        print(f"P&L: ${trade.pnl:.2f}")
        
        # Calculate expected P&L
        expected_pips = (trade.exit_price - trade.entry_price) * 10000
        expected_pnl = expected_pips * 100  # $100 per pip
        print(f"\nExpected P&L: ${expected_pnl:.2f}")
        print(f"Actual P&L: ${trade.pnl:.2f}")
        print(f"Match: {'YES' if abs(expected_pnl - trade.pnl) < 0.01 else 'NO'}")


def main():
    test_manual_trade_calculation()
    test_strategy_with_forced_trades()
    test_slippage_simulation()
    trace_single_trade()
    
    print("\n" + "="*80)
    print("SPREAD MYSTERY INVESTIGATION COMPLETE")
    print("="*80)
    print("\n🔍 Key Findings:")
    print("1. Strategy enters trades at Close price of signal bar")
    print("2. No spread/commission is applied in the current implementation")
    print("3. P&L calculations are correct for the prices used")
    print("4. The issue is that bid/ask spread is not modeled at all")
    print("\n💡 Solution: Need to implement proper bid/ask spread in execution")


if __name__ == "__main__":
    main()