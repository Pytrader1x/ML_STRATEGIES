"""
Test script for intrabar stop-loss feature
Demonstrates the difference between close-only and intrabar touch stop losses
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')

def compare_stop_loss_modes():
    """Compare results with and without intrabar stop loss"""
    
    # Load data
    df_full = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
    df_full.set_index('DateTime', inplace=True)
    df = df_full.iloc[-5000:].copy()
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Test configuration
    base_config = {
        'initial_capital': 1_000_000,
        'risk_per_trade': 0.002,
        'sl_max_pips': 10.0,
        'sl_atr_multiplier': 1.0,
        'tp_atr_multipliers': (0.2, 0.3, 0.5),
        'max_tp_percent': 0.003,
        'tsl_activation_pips': 15,
        'tsl_min_profit_pips': 1,
        'tsl_initial_buffer_multiplier': 1.0,
        'trailing_atr_multiplier': 1.2,
        'tp_range_market_multiplier': 0.5,
        'tp_trend_market_multiplier': 0.7,
        'tp_chop_market_multiplier': 0.3,
        'sl_range_market_multiplier': 0.7,
        'exit_on_signal_flip': False,
        'partial_profit_before_sl': False,
        'debug_decisions': False,
        'use_daily_sharpe': True
    }
    
    # Run with close-only stop loss (default)
    print("="*80)
    print("TESTING WITH CLOSE-ONLY STOP LOSS (Default)")
    print("="*80)
    
    config_close_only = OptimizedStrategyConfig(**base_config, intrabar_stop_on_touch=False)
    strategy_close_only = OptimizedProdStrategy(config_close_only)
    results_close_only = strategy_close_only.run_backtest(df)
    
    # Run with intrabar touch stop loss
    print("\n" + "="*80)
    print("TESTING WITH INTRABAR TOUCH STOP LOSS")
    print("="*80)
    
    config_intrabar = OptimizedStrategyConfig(**base_config, intrabar_stop_on_touch=True)
    strategy_intrabar = OptimizedProdStrategy(config_intrabar)
    results_intrabar = strategy_intrabar.run_backtest(df)
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON OF RESULTS")
    print("="*80)
    
    metrics_close = results_close_only
    metrics_intrabar = results_intrabar
    
    # Count stop loss exits
    sl_exits_close = len([t for t in results_close_only['trades'] if str(t.exit_reason) == 'ExitReason.STOP_LOSS'])
    sl_exits_intrabar = len([t for t in results_intrabar['trades'] if str(t.exit_reason) == 'ExitReason.STOP_LOSS'])
    
    print(f"\nClose-Only Stop Loss:")
    print(f"  Total Trades: {metrics_close['total_trades']}")
    print(f"  Win Rate: {metrics_close['win_rate']:.1f}%")
    print(f"  Stop Loss Count: {sl_exits_close}")
    print(f"  Total Return: {metrics_close['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics_close['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics_close['max_drawdown']:.2f}%")
    
    print(f"\nIntrabar Touch Stop Loss:")
    print(f"  Total Trades: {metrics_intrabar['total_trades']}")
    print(f"  Win Rate: {metrics_intrabar['win_rate']:.1f}%")
    print(f"  Stop Loss Count: {sl_exits_intrabar}")
    print(f"  Total Return: {metrics_intrabar['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {metrics_intrabar['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics_intrabar['max_drawdown']:.2f}%")
    
    print(f"\nDifferences:")
    print(f"  Additional SL exits with intrabar: {sl_exits_intrabar - sl_exits_close}")
    print(f"  Win Rate Change: {metrics_intrabar['win_rate'] - metrics_close['win_rate']:+.1f}%")
    print(f"  Return Change: {metrics_intrabar['total_return'] - metrics_close['total_return']:+.2f}%")
    
    # Find examples of trades that exited differently
    print("\n" + "="*80)
    print("EXAMPLE TRADES WITH DIFFERENT EXITS")
    print("="*80)
    
    trades_close = results_close_only['trades']
    trades_intrabar = results_intrabar['trades']
    
    # Create sets of trade identifiers
    close_sl_trades = {(t.entry_time, t.entry_price) for t in trades_close 
                       if str(t.exit_reason) == 'ExitReason.STOP_LOSS'}
    intrabar_sl_trades = {(t.entry_time, t.entry_price) for t in trades_intrabar 
                          if str(t.exit_reason) == 'ExitReason.STOP_LOSS'}
    
    # Find trades that hit SL with intrabar but not with close-only
    new_sl_trades = intrabar_sl_trades - close_sl_trades
    
    if new_sl_trades:
        print(f"\nFound {len(new_sl_trades)} trades that hit SL with intrabar touch but not close-only:")
        
        # Show first 3 examples
        for i, (entry_time, entry_price) in enumerate(list(new_sl_trades)[:3]):
            # Find the trade in both results
            close_trade = next((t for t in trades_close if t.entry_time == entry_time), None)
            intrabar_trade = next((t for t in trades_intrabar if t.entry_time == entry_time), None)
            
            if close_trade and intrabar_trade:
                print(f"\n  Example {i+1}:")
                print(f"    Entry: {entry_time} @ {entry_price:.5f}")
                print(f"    Direction: {close_trade.direction.value}")
                print(f"    Stop Loss: {close_trade.stop_loss:.5f}")
                
                # Get the candle where they differ
                exit_candle_time = intrabar_trade.exit_time
                if exit_candle_time in df.index:
                    candle = df.loc[exit_candle_time]
                    print(f"    \n    Exit Candle ({exit_candle_time}):")
                    print(f"      Open:  {candle['Open']:.5f}")
                    print(f"      High:  {candle['High']:.5f}")
                    print(f"      Low:   {candle['Low']:.5f}")
                    print(f"      Close: {candle['Close']:.5f}")
                    
                    if close_trade.direction.value == 'long':
                        print(f"      Low vs SL: {candle['Low']:.5f} {'<=' if candle['Low'] <= close_trade.stop_loss else '>'} {close_trade.stop_loss:.5f}")
                        print(f"      Close vs SL: {candle['Close']:.5f} {'<=' if candle['Close'] <= close_trade.stop_loss else '>'} {close_trade.stop_loss:.5f}")
                    else:
                        print(f"      High vs SL: {candle['High']:.5f} {'>=' if candle['High'] >= close_trade.stop_loss else '<'} {close_trade.stop_loss:.5f}")
                        print(f"      Close vs SL: {candle['Close']:.5f} {'>=' if candle['Close'] >= close_trade.stop_loss else '<'} {close_trade.stop_loss:.5f}")
                
                print(f"    \n    Close-only exit: {close_trade.exit_reason} @ {close_trade.exit_time}")
                print(f"    Intrabar exit: {intrabar_trade.exit_reason} @ {intrabar_trade.exit_time}")
                print(f"    P&L difference: ${intrabar_trade.pnl - close_trade.pnl:.2f}")
    else:
        print("\nNo trades found with different stop loss behavior.")

if __name__ == "__main__":
    compare_stop_loss_modes()