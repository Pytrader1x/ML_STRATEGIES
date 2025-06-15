#!/usr/bin/env python3
"""
Debug TP exit execution with detailed tracing
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings

warnings.filterwarnings('ignore')

def debug_tp_execution():
    """Run a debug test to trace TP exit execution"""
    
    print("🔍 DEBUGGING TP EXIT EXECUTION")
    print("="*60)
    
    # Create a simple test strategy with debug enabled
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,
        sl_max_pips=50.0,          # Very wide SL
        sl_atr_multiplier=3.0,     # Very wide SL
        tp_atr_multipliers=(0.05, 0.1, 0.15),  # VERY tight TPs
        max_tp_percent=0.01,       # High TP constraint
        tsl_activation_pips=100,   # TSL activates way later
        tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=3.0,  # Wide trailing distance
        tp_range_market_multiplier=1.0,
        tp_trend_market_multiplier=1.0,
        tp_chop_market_multiplier=1.0,
        sl_range_market_multiplier=1.0,
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=False,  # Disable partial profit
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        relaxed_position_multiplier=0.5,
        relaxed_mode=True,  # Easier entry conditions
        realistic_costs=False,  # No costs for debug
        verbose=True,
        debug_decisions=True,  # ENABLE DEBUG OUTPUT
        use_daily_sharpe=True
    )
    
    strategy = OptimizedProdStrategy(config)
    
    # Load a small sample of data
    print("\n📊 Loading test data...")
    if True:  # Simple mock data
        # Create mock data with obvious TP opportunity
        dates = pd.date_range('2023-01-01', periods=100, freq='15min')
        
        # Create price action that should hit TPs
        base_price = 0.6500
        prices = []
        
        # Price moves up gradually then clearly hits TP1
        for i in range(100):
            if i < 20:
                price = base_price + (i * 0.0001)  # Gradual rise to 0.6519
            elif i == 20:
                price = 0.65220  # Clearly hit TP1 at 0.65215
            elif i < 40:
                price = base_price + 0.003 + (i-20) * 0.0002  # Continue rising for TP2/3
            else:
                price = base_price + 0.007 - (i-40) * 0.00005  # Slight decline
            prices.append(price)
        
        # Create OHLC data
        df_data = []
        for i, date in enumerate(dates):
            price = prices[i]
            high = price + 0.0005
            low = price - 0.0005
            df_data.append({
                'DateTime': date,
                'Open': price,
                'High': high,
                'Low': low,
                'Close': price + 0.0001,
                'Volume': 1000
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('DateTime', inplace=True)
        
        # Add required indicators with proper column names
        df['ATR'] = 0.006  # Fixed ATR
        df['NTI_Direction'] = 1  # Always long signal
        df['NTI_Confidence'] = 0.8
        df['MB_Bias'] = 1  # Always bullish bias
        df['IC_Regime'] = 1  # Trending regime
        df['IC_ATR_Normalized'] = 0.5  # Moderate volatility
        df['IC_RegimeName'] = 'Trending'  # Regime name
        
        print(f"Created mock data: {len(df)} bars")
        print(f"Price range: {df['Low'].min():.4f} to {df['High'].max():.4f}")
        
        # Run strategy
        print(f"\n🚀 Running strategy with VERY tight TPs...")
        print(f"Expected TP levels around:")
        atr = 0.006
        tp1_dist = 0.05 * atr
        tp2_dist = 0.1 * atr  
        tp3_dist = 0.15 * atr
        print(f"  TP1: ~{tp1_dist/0.0001:.1f} pips")
        print(f"  TP2: ~{tp2_dist/0.0001:.1f} pips") 
        print(f"  TP3: ~{tp3_dist/0.0001:.1f} pips")
        
        results = strategy.run_backtest(df)
        
        print(f"\n📈 Results:")
        print(f"  Total trades: {results.get('total_trades', 0)}")
        print(f"  Total P&L: ${results.get('total_pnl', 0):,.2f}")
        print(f"  Win rate: {results.get('win_rate', 0):.1f}%")
        
        # Check for TP exits
        trades = results.get('trades', [])
        tp_exits = 0
        for trade in trades:
            if hasattr(trade, 'exit_reason'):
                exit_reason = str(trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason)
                if 'take_profit' in exit_reason:
                    tp_exits += 1
                    print(f"  ✅ Found TP exit: {exit_reason}")
        
        print(f"\n🎯 TP Exits Found: {tp_exits} out of {len(trades)} trades")
        
        if tp_exits == 0:
            print("\n❌ NO TP EXITS FOUND!")
            print("This indicates the TP exit logic is still not working correctly.")
            
            # Print trade details
            for i, trade in enumerate(trades[:3]):  # First 3 trades
                print(f"\nTrade {i+1} details:")
                print(f"  Entry: {trade.entry_price:.4f}")
                print(f"  Exit: {trade.exit_price:.4f}")
                print(f"  Exit reason: {trade.exit_reason}")
                print(f"  TP levels: {[f'{tp:.4f}' for tp in trade.take_profits]}")
                print(f"  TP hits: {trade.tp_hits}")
                
        else:
            print(f"\n✅ SUCCESS! TP exits are working!")

if __name__ == "__main__":
    debug_tp_execution()