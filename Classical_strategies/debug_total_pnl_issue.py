"""
Debug why total P&L shows as 1.4k instead of 1.2k
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

# Find trades with similar P&L values
print("Looking for trades with P&L around $1,400...")
print("="*80)

for trade in results['trades']:
    if 1350 < trade.pnl < 1450:  # Look for trades around $1,400
        print(f"\nTrade: {trade.entry_time}")
        print(f"Total P&L: ${trade.pnl:.2f}")
        print(f"Exit Reason: {trade.exit_reason}")
        
        # Check partial exits
        if hasattr(trade, 'partial_exits') and trade.partial_exits:
            sum_partials = sum(pe.pnl for pe in trade.partial_exits)
            print(f"Sum of partials: ${sum_partials:.2f}")
            
            # Check if this might be displayed on March 30
            for pe in trade.partial_exits:
                if '2025-03-30' in str(pe.time) or '2025-03-31' in str(pe.time):
                    print(f"  - Has exit on March 30/31: {pe.time}")

# Also check the specific March 30 trade
print("\n" + "="*80)
print("MARCH 30 21:15 TRADE:")
print("="*80)

for trade in results['trades']:
    if str(trade.entry_time) == '2025-03-30 21:15:00':
        print(f"\nEntry: {trade.entry_time}")
        print(f"Total P&L: ${trade.pnl:.2f}")
        print(f"Total P&L / 1000: {trade.pnl/1000:.1f}k")
        print(f"Rounded: {round(trade.pnl/1000, 1)}k")
        
        # Test different rounding scenarios
        print(f"\nDifferent formatting tests:")
        print(f"  {trade.pnl:.0f} / 1000 = {trade.pnl/1000:.1f}")
        print(f"  Round to nearest 100: ${round(trade.pnl, -2)}")
        print(f"  Round to nearest 100 / 1000: {round(trade.pnl, -2)/1000:.1f}k")
        
        # Could there be a different trade being displayed?
        print(f"\nChecking for position overlap...")
        exit_time = trade.exit_time
        
        # Look for other trades that might be visible at this time
        for other_trade in results['trades']:
            if other_trade.entry_time <= exit_time <= other_trade.exit_time:
                if other_trade != trade:
                    print(f"  Overlapping trade: {other_trade.entry_time} -> {other_trade.exit_time}")
                    print(f"    P&L: ${other_trade.pnl:.2f}")