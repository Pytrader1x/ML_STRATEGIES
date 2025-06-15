"""
Debug actual exit reason from trade
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, ExitReason
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

# Find the March 30 21:15 trade
for trade in results['trades']:
    if str(trade.entry_time) == '2025-03-30 21:15:00':
        print("="*60)
        print("Exit Reason Analysis")
        print("="*60)
        
        print(f"trade.exit_reason: {trade.exit_reason}")
        print(f"type: {type(trade.exit_reason)}")
        print(f"str(trade.exit_reason): {str(trade.exit_reason)}")
        print(f"repr(trade.exit_reason): {repr(trade.exit_reason)}")
        
        # Check equality
        print(f"\nEquality checks:")
        print(f"  == ExitReason.TP1_PULLBACK: {trade.exit_reason == ExitReason.TP1_PULLBACK}")
        print(f"  == 'tp1_pullback': {trade.exit_reason == 'tp1_pullback'}")
        print(f"  == 'ExitReason.TP1_PULLBACK': {trade.exit_reason == 'ExitReason.TP1_PULLBACK'}")
        
        # String checks
        exit_str = str(trade.exit_reason)
        print(f"\nString checks on '{exit_str}':")
        print(f"  'TP1_PULLBACK' in str: {'TP1_PULLBACK' in exit_str}")
        print(f"  'tp1_pullback' in str: {'tp1_pullback' in exit_str}")
        print(f"  'pullback' in str.lower(): {'pullback' in exit_str.lower()}")
        
        break