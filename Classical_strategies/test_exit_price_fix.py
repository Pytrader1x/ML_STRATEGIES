"""
Test script to verify that exit prices respect candle boundaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from datetime import datetime, timedelta

def create_test_data():
    """Create synthetic test data with specific scenarios"""
    # Create dates
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
    
    # Create base price series
    base_price = 1.0800
    prices = []
    
    for i in range(len(dates)):
        if i == 50:  # Create a scenario where stop loss would be hit
            # For a short trade with SL at 1.0820, create a candle that touches it
            open_price = 1.0810
            high_price = 1.0825  # High touches/exceeds stop loss
            low_price = 1.0805
            close_price = 1.0815
        else:
            # Normal candle
            open_price = base_price + np.random.uniform(-0.0010, 0.0010)
            close_price = open_price + np.random.uniform(-0.0005, 0.0005)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.0003)
            low_price = min(open_price, close_price) - np.random.uniform(0, 0.0003)
        
        prices.append({
            'DateTime': dates[i],
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price
        })
        
        # Add some trend
        base_price += np.random.uniform(-0.0002, 0.0002)
    
    df = pd.DataFrame(prices)
    df.set_index('DateTime', inplace=True)
    
    # Add required indicators (simplified)
    df['NTI_Direction'] = -1  # Short bias
    df['MB_Bias'] = -1
    df['IC_Regime'] = 1  # Trending
    df['IC_RegimeName'] = 'Strong Trend'
    df['IC_ATR_Normalized'] = 0.0010
    df['NTI_Confidence'] = 60.0
    
    # Add other required columns
    df['NTI_FastEMA'] = df['Close'].ewm(span=12).mean()
    df['NTI_SlowEMA'] = df['Close'].ewm(span=26).mean()
    df['MB_o2'] = df['Open']
    df['MB_c2'] = df['Close']
    df['MB_h2'] = df['High']
    df['MB_l2'] = df['Low']
    
    return df

def run_test():
    """Run test to verify exit prices respect candle boundaries"""
    print("="*80)
    print("TESTING EXIT PRICE BOUNDARY FIX")
    print("="*80)
    
    # Create test data
    df = create_test_data()
    print(f"\nCreated test data with {len(df)} candles")
    
    # Create strategy with debug mode enabled
    config = OptimizedStrategyConfig(
        initial_capital=100000,
        risk_per_trade=0.01,
        sl_min_pips=10,
        sl_max_pips=20,
        realistic_costs=True,
        stop_loss_slippage_pips=2.0,  # 2 pip slippage on stop loss
        intrabar_stop_on_touch=True,  # Enable intrabar stop detection
        debug_decisions=True,  # Enable debug output
        relaxed_mode=True  # Allow easier entry
    )
    
    strategy = OptimizedProdStrategy(config)
    
    print("\nRunning backtest...")
    print("Looking for stop loss exits that might exceed candle boundaries...\n")
    
    # Run backtest
    result = strategy.run_backtest(df)
    
    # Analyze trades
    print(f"\n{'='*80}")
    print("TRADE ANALYSIS")
    print(f"{'='*80}")
    print(f"Total trades: {result['total_trades']}")
    
    # Check each trade for boundary violations
    violations = 0
    for i, trade in enumerate(result['trades']):
        if hasattr(trade, 'exit_price') and trade.exit_price is not None:
            # Find the exit candle
            exit_idx = None
            for idx, time in enumerate(df.index):
                if time == trade.exit_time:
                    exit_idx = idx
                    break
            
            if exit_idx is not None:
                exit_candle = df.iloc[exit_idx]
                
                # Check if exit price is within candle bounds
                if trade.exit_price > exit_candle['High'] or trade.exit_price < exit_candle['Low']:
                    violations += 1
                    print(f"\n❌ BOUNDARY VIOLATION in Trade {i+1}:")
                    print(f"   Exit Time: {trade.exit_time}")
                    print(f"   Exit Price: {trade.exit_price:.5f}")
                    print(f"   Candle High: {exit_candle['High']:.5f}")
                    print(f"   Candle Low: {exit_candle['Low']:.5f}")
                    print(f"   Exit Reason: {trade.exit_reason}")
                    print(f"   Direction: {trade.direction}")
                else:
                    # Good exit
                    margin_from_high = exit_candle['High'] - trade.exit_price
                    margin_from_low = trade.exit_price - exit_candle['Low']
                    
                    if trade.exit_reason.value in ['stop_loss', 'trailing_stop']:
                        print(f"\n✅ Trade {i+1} - Stop Loss Exit Within Bounds:")
                        print(f"   Exit Price: {trade.exit_price:.5f}")
                        print(f"   Candle Range: {exit_candle['Low']:.5f} - {exit_candle['High']:.5f}")
                        print(f"   Margin from High: {margin_from_high:.5f}")
                        print(f"   Margin from Low: {margin_from_low:.5f}")
                        print(f"   Direction: {trade.direction.value}")
    
    print(f"\n{'='*80}")
    print("TEST RESULTS")
    print(f"{'='*80}")
    if violations == 0:
        print("✅ SUCCESS: All exit prices are within candle boundaries!")
    else:
        print(f"❌ FAILED: Found {violations} trades with exit prices outside candle boundaries")
    
    print(f"\nExit Reason Breakdown:")
    for reason, count in result['exit_reasons'].items():
        print(f"  {reason}: {count}")
    
    return result, df

if __name__ == "__main__":
    result, df = run_test()