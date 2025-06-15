"""
Test if the May 2 trade now correctly exits at TP2 instead of TP1 pullback
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
for trade in results['trades']:
    if str(trade.entry_time) == '2025-05-02 06:00:00':
        print("="*80)
        print("MAY 2 06:00 TRADE - AFTER FIX")
        print("="*80)
        
        print(f"\nTrade Details:")
        print(f"Entry: {trade.entry_time} @ {trade.entry_price:.5f}")
        print(f"Exit: {trade.exit_time} @ {trade.exit_price:.5f}")
        print(f"Exit Reason: {trade.exit_reason}")
        
        print(f"\nPartial Exits:")
        for i, pe in enumerate(trade.partial_exits):
            print(f"  {i+1}. {pe.time} - TP{pe.tp_level} - {pe.size/1e6:.2f}M @ {pe.price:.5f} = ${pe.pnl:.2f}")
        
        # Check the specific candle at 09:15
        candle_time = pd.Timestamp('2025-05-02 09:15:00')
        if candle_time in df.index:
            candle = df.loc[candle_time]
            print(f"\nCandle at {candle_time}:")
            print(f"  High: {candle['High']:.5f} (TP2: {trade.take_profits[1]:.5f})")
            print(f"  Low:  {candle['Low']:.5f} (TP1: {trade.take_profits[0]:.5f})")
            
            # Check which exits happened at this time
            exits_at_0915 = [pe for pe in trade.partial_exits if pe.time == candle_time]
            if exits_at_0915:
                print(f"\nExits at 09:15:")
                for pe in exits_at_0915:
                    print(f"  - TP{pe.tp_level} at {pe.price:.5f}")
                    
                # Verify the fix
                tp2_exit = any(pe.tp_level == 2 for pe in exits_at_0915)
                tp1_pb_exit = any(pe.tp_level == 0 and trade.exit_reason.value == 'tp1_pullback' for pe in exits_at_0915)
                
                if tp2_exit and not tp1_pb_exit:
                    print("\n✅ FIX SUCCESSFUL: TP2 was hit and no TP1 pullback in same candle")
                elif tp2_exit and tp1_pb_exit:
                    print("\n❌ FIX FAILED: Both TP2 and TP1 pullback happened in same candle")
                elif not tp2_exit and candle['High'] >= trade.take_profits[1]:
                    print("\n❌ FIX FAILED: TP2 should have been hit but wasn't")
        
        break
else:
    print("Trade not found for 2025-05-02 06:00:00")