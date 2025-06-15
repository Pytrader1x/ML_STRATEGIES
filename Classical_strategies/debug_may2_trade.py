"""
Debug May 2 06:00 trade - TP2 should have been hit before TP1 pullback
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')

# Load data
df_full = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
df_full.set_index('DateTime', inplace=True)
df = df_full.iloc[-5000:].copy()

# Add indicators
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df)
df = TIC.add_intelligent_chop(df)

# Create strategy
strategy_config = OptimizedStrategyConfig(
    initial_capital=1_000_000, risk_per_trade=0.002, sl_max_pips=10.0,
    sl_atr_multiplier=1.0, tp_atr_multipliers=(0.2, 0.3, 0.5),
    max_tp_percent=0.003, tsl_activation_pips=15, tsl_min_profit_pips=1,
    tsl_initial_buffer_multiplier=1.0, trailing_atr_multiplier=1.2,
    tp_range_market_multiplier=0.5, tp_trend_market_multiplier=0.7,
    tp_chop_market_multiplier=0.3, sl_range_market_multiplier=0.7,
    exit_on_signal_flip=False, partial_profit_before_sl=False,
    debug_decisions=False, use_daily_sharpe=True
)

strategy = OptimizedProdStrategy(strategy_config)
results = strategy.run_backtest(df)

# Find the May 2 06:00 trade
target_trade = None
for trade in results['trades']:
    if str(trade.entry_time) == '2025-05-02 06:00:00':
        target_trade = trade
        break

if target_trade:
    print("="*80)
    print("MAY 2 06:00 TRADE ANALYSIS")
    print("="*80)
    
    print(f"\nTrade Details:")
    print(f"Entry Time: {target_trade.entry_time}")
    print(f"Entry Price: {target_trade.entry_price:.5f}")
    print(f"Direction: {target_trade.direction.value}")
    print(f"Position Size: {target_trade.position_size/1e6:.2f}M")
    
    print(f"\nTP Levels:")
    for i, tp in enumerate(target_trade.take_profits):
        print(f"  TP{i+1}: {tp:.5f}")
    print(f"  SL: {target_trade.stop_loss:.5f}")
    
    print(f"\nExit Details:")
    print(f"Exit Time: {target_trade.exit_time}")
    print(f"Exit Price: {target_trade.exit_price:.5f}")
    print(f"Exit Reason: {target_trade.exit_reason}")
    print(f"Total P&L: ${target_trade.pnl:.2f}")
    
    print(f"\nPartial Exits:")
    for i, pe in enumerate(target_trade.partial_exits):
        print(f"  {i+1}. Time: {pe.time}, TP{pe.tp_level}, Size: {pe.size/1e6:.2f}M, P&L: ${pe.pnl:.2f}")
    
    # Now check the candle data around the exit
    print("\n" + "="*60)
    print("CANDLE DATA ANALYSIS")
    print("="*60)
    
    # Get candles from TP1 hit to final exit
    tp1_time = target_trade.partial_exits[0].time if target_trade.partial_exits else None
    if tp1_time:
        # Get 5 candles after TP1
        start_idx = df.index.get_loc(tp1_time)
        candle_data = df.iloc[start_idx:start_idx+10][['Open', 'High', 'Low', 'Close']]
        
        print(f"\nCandles after TP1 hit at {tp1_time}:")
        print("-"*60)
        
        for idx, (time, row) in enumerate(candle_data.iterrows()):
            print(f"\n{time}:")
            print(f"  Open:  {row['Open']:.5f}")
            print(f"  High:  {row['High']:.5f}")
            print(f"  Low:   {row['Low']:.5f}")
            print(f"  Close: {row['Close']:.5f}")
            
            # Check if this candle hit any TP levels
            if target_trade.direction.value == 'long':
                for i, tp in enumerate(target_trade.take_profits):
                    if row['High'] >= tp:
                        print(f"  >>> HIGH REACHED TP{i+1} at {tp:.5f}!")
                if row['Low'] <= target_trade.stop_loss:
                    print(f"  >>> LOW HIT SL at {target_trade.stop_loss:.5f}!")
            
            # Check for exits at this time
            for pe in target_trade.partial_exits:
                if pe.time == time:
                    print(f"  *** EXIT HERE: TP{pe.tp_level} at {pe.price:.5f}")
    
    # Specific check for the TP1 pullback candle
    print("\n" + "="*60)
    print("TP1 PULLBACK CANDLE ANALYSIS")
    print("="*60)
    
    pb_time = target_trade.exit_time
    if pb_time in df.index:
        pb_candle = df.loc[pb_time]
        print(f"\nCandle at {pb_time} (TP1 Pullback exit):")
        print(f"  Open:  {pb_candle['Open']:.5f}")
        print(f"  High:  {pb_candle['High']:.5f}")
        print(f"  Low:   {pb_candle['Low']:.5f}")
        print(f"  Close: {pb_candle['Close']:.5f}")
        
        print(f"\nTP Level Analysis:")
        print(f"  TP1: {target_trade.take_profits[0]:.5f}")
        print(f"  TP2: {target_trade.take_profits[1]:.5f}")
        print(f"  High - TP2 = {pb_candle['High'] - target_trade.take_profits[1]:.5f}")
        
        if pb_candle['High'] >= target_trade.take_profits[1]:
            print(f"\n❌ ISSUE: Candle high ({pb_candle['High']:.5f}) reached TP2 ({target_trade.take_profits[1]:.5f})")
            print(f"   But trade exited at TP1 pullback ({target_trade.exit_price:.5f})")
            print(f"   This suggests TP2 should have been hit FIRST!")
else:
    print("Trade not found for 2025-05-02 06:00:00")