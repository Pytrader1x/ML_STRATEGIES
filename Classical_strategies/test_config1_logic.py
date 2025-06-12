"""
Test file to understand Config 1 (Ultra-Tight Risk Management) logic
This will help us trace through exactly how the strategy makes decisions
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import sys
sys.path.append('..')
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


def create_config_1_ultra_tight_risk():
    """Config 1: Ultra-Tight Risk Management"""
    config = OptimizedStrategyConfig(
        # Risk management
        initial_capital=100_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
        
        # Stop loss settings
        sl_max_pips=10.0,  # Maximum 10 pip stop loss
        sl_atr_multiplier=1.0,
        
        # Take profit levels (KEY PART)
        tp_atr_multipliers=(0.2, 0.3, 0.5),  # TP1=0.2xATR, TP2=0.3xATR, TP3=0.5xATR
        max_tp_percent=0.003,  # Max 0.3% TP
        
        # Trailing stop settings
        tsl_activation_pips=3,  # TSL activates at 3 pips profit
        tsl_min_profit_pips=1,  # Minimum 1 pip profit when TSL triggers
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=0.8,
        
        # Market condition adjustments
        tp_range_market_multiplier=0.5,  # Reduce TP in ranging market
        tp_trend_market_multiplier=0.7,  # Moderate TP in trending market
        tp_chop_market_multiplier=0.3,   # Very small TP in choppy market
        sl_range_market_multiplier=0.7,
        
        # Exit strategies
        exit_on_signal_flip=False,  # Don't exit just on signal flip
        signal_flip_min_profit_pips=5.0,  # Need 5 pips to exit on flip
        signal_flip_min_time_hours=1.0,  # Need 1 hour before flip exit
        signal_flip_partial_exit_percent=1.0,  # Exit full position
        
        # Partial profits
        partial_profit_before_sl=True,  # Take partial profits
        partial_profit_sl_distance_ratio=0.5,  # At 50% to SL
        partial_profit_size_percent=0.5,  # Take 50% off
        
        # Other
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=True  # IMPORTANT: Set to True to see decision logic
    )
    
    return OptimizedProdStrategy(config)


def analyze_exit_logic():
    """Analyze how Config 1 decides to exit trades"""
    
    print("="*80)
    print("CONFIG 1 EXIT LOGIC ANALYSIS")
    print("="*80)
    
    # Load some data
    print("\n1. Loading test data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use a small sample for detailed analysis
    # Pick a volatile period for interesting trades
    test_df = df['2024-01-01':'2024-01-10'].copy()
    
    print(f"   Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"   Total bars: {len(test_df)}")
    
    # Add indicators
    print("\n2. Adding indicators...")
    test_df = TIC.add_neuro_trend_intelligent(test_df)
    test_df = TIC.add_market_bias(test_df)
    test_df = TIC.add_intelligent_chop(test_df)
    
    # Create strategy
    print("\n3. Creating Config 1 strategy...")
    strategy = create_config_1_ultra_tight_risk()
    
    # Run backtest with verbose output
    print("\n4. Running backtest (watch for exit decisions)...")
    print("-"*80)
    results = strategy.run_backtest(test_df)
    print("-"*80)
    
    # Analyze results
    print("\n5. RESULTS ANALYSIS")
    print(f"   Total trades: {results['total_trades']}")
    print(f"   Win rate: {results['win_rate']:.1f}%")
    print(f"   Average win: ${results['avg_win']:.2f}")
    print(f"   Average loss: ${abs(results['avg_loss']):.2f}")
    
    # Extract trade details if available
    if 'trades' in results:
        print("\n6. INDIVIDUAL TRADE ANALYSIS")
        trades = results['trades']
        
        for i, trade in enumerate(trades[:10]):  # First 10 trades
            print(f"\n   Trade {i+1}:")
            # Handle Trade objects
            if hasattr(trade, 'direction'):
                print(f"   - Direction: {trade.direction.value if hasattr(trade.direction, 'value') else trade.direction}")
                print(f"   - Entry price: {trade.entry_price:.5f}")
                print(f"   - Stop loss: {trade.stop_loss:.5f} ({abs(trade.entry_price - trade.stop_loss) * 10000:.1f} pips)")
                if hasattr(trade, 'take_profits') and trade.take_profits:
                    print(f"   - TP1: {trade.take_profits[0]:.5f} ({abs(trade.entry_price - trade.take_profits[0]) * 10000:.1f} pips)")
                print(f"   - Exit price: {trade.exit_price:.5f}")
                print(f"   - Exit reason: {trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason}")
                print(f"   - P&L: ${trade.pnl:.2f}")
            # Handle dictionaries
            else:
                print(f"   - Direction: {'LONG' if trade.get('direction', 0) > 0 else 'SHORT'}")
                print(f"   - Entry price: {trade.get('entry_price', 0):.5f}")
                print(f"   - Stop loss: {trade.get('stop_loss', 0):.5f}")
                print(f"   - Exit price: {trade.get('exit_price', 0):.5f}")
                print(f"   - Exit reason: {trade.get('exit_reason', 'Unknown')}")
                print(f"   - P&L: ${trade.get('pnl', 0):.2f}")
    
    # Calculate ATR to understand TP levels
    print("\n7. ATR ANALYSIS (for TP calculation)")
    test_df['ATR'] = calculate_atr(test_df, period=14)
    avg_atr_pips = test_df['ATR'].mean() * 10000  # Convert to pips
    
    print(f"   Average ATR: {avg_atr_pips:.1f} pips")
    print(f"   Expected TP1 (0.2 x ATR): {avg_atr_pips * 0.2:.1f} pips")
    print(f"   Expected TP2 (0.3 x ATR): {avg_atr_pips * 0.3:.1f} pips")
    print(f"   Expected TP3 (0.5 x ATR): {avg_atr_pips * 0.5:.1f} pips")
    
    return results


def calculate_atr(df, period=14):
    """Calculate ATR for analysis"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(period).mean()


def trace_single_trade():
    """Trace through a single trade to see exact decision making"""
    print("\n" + "="*80)
    print("SINGLE TRADE TRACE ANALYSIS")
    print("="*80)
    
    # This would need access to internal strategy state
    # For now, let's understand the exit priority:
    
    print("\nEXIT DECISION PRIORITY (Config 1):")
    print("1. Stop Loss Hit (max 10 pips)")
    print("2. Take Profit Hit (TP1 @ 0.2xATR, TP2 @ 0.3xATR, TP3 @ 0.5xATR)")
    print("3. Trailing Stop Hit (activates at +3 pips, trails by 0.8xATR)")
    print("4. Signal Flip (only if +5 pips profit AND 1 hour passed)")
    print("5. Partial Profit (50% position when 50% to SL)")
    
    print("\nKEY INSIGHTS:")
    print("- TP1 is usually 10-20 pips (depending on ATR)")
    print("- Trailing stop activates early (3 pips) to protect profits")
    print("- Signal flips need minimum 5 pips to exit")
    print("- Market conditions affect TP levels:")
    print("  * Range market: TP × 0.5 (smaller targets)")
    print("  * Trend market: TP × 0.7 (moderate targets)")
    print("  * Chop market: TP × 0.3 (very small targets)")


if __name__ == "__main__":
    # Run the analysis
    results = analyze_exit_logic()
    
    # Trace single trade logic
    trace_single_trade()
    
    print("\n" + "="*80)
    print("SUMMARY: Config 1 Exit Logic")
    print("="*80)
    print("1. Prioritizes small, consistent profits (5-20 pips)")
    print("2. Uses dynamic TP based on ATR and market conditions")
    print("3. Aggressive trailing stop (activates at +3 pips)")
    print("4. Will exit on signal reversal if profitable")
    print("5. Takes partial profits to reduce risk")
    print("\nThis explains why you see exits at:")
    print("- 5-6 pips: Signal flip with minimum profit")
    print("- 10-15 pips: TP1 level (most common)")
    print("- 15-20 pips: TP1 in higher volatility")
    print("- Various levels: Trailing stop or partial profit exits")