"""
Verify P&L display fix
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

# Find the March 30 trade
for trade in results['trades']:
    if str(trade.entry_time) == '2025-03-30 21:15:00':
        print("="*80)
        print("MARCH 30 TRADE P&L VERIFICATION:")
        print("="*80)
        
        print(f"\nTrade P&L: ${trade.pnl:.2f}")
        print(f"Expected display: $+{trade.pnl/1000:.1f}k")
        
        # Verify partial exits sum
        total = sum(pe.pnl for pe in trade.partial_exits)
        print(f"\nPartial exits sum: ${total:.2f}")
        print(f"Matches trade P&L: {'✅' if abs(total - trade.pnl) < 0.01 else '❌'}")
        
        # Show what will be displayed
        print(f"\nExit markers will show:")
        for i, pe in enumerate(trade.partial_exits):
            pnl_text = f"${pe.pnl:+.0f}" if pe.pnl < 1000 else f"${pe.pnl/1000:+.1f}k"
            print(f"  Exit {i+1}: {pnl_text}")
        
        total_text = f"$+{trade.pnl/1000:.1f}k" if trade.pnl >= 1000 else f"$+{trade.pnl:.0f}"
        print(f"  Total: {total_text}")
        
        break