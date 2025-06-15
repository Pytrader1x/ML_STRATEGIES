"""
Debug what P&L value is being passed to plotting
"""

import pandas as pd
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import ProductionPlotter
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
        print("MARCH 30 TRADE DATA:")
        print("="*80)
        
        # Create trade dict as it would be passed to plotting
        trade_dict = trade.to_dict()
        
        print(f"\nTrade dict keys: {list(trade_dict.keys())}")
        print(f"\nP&L value in dict: {trade_dict.get('pnl')}")
        print(f"Type: {type(trade_dict.get('pnl'))}")
        
        # Check if there's any confusion with other fields
        for key in trade_dict:
            if 'pnl' in key.lower() or 'profit' in key.lower():
                print(f"{key}: {trade_dict[key]}")
        
        # Test formatting
        pnl = trade_dict.get('pnl')
        if pnl:
            print(f"\nFormatting tests:")
            print(f"  Raw: ${pnl:.2f}")
            print(f"  /1000: {pnl/1000:.1f}k")
            print(f"  With sign: $+{pnl/1000:.1f}k")
            
        # Check cumulative P&L
        print(f"\nChecking if cumulative P&L might be involved...")
        all_trades = results['trades']
        cumulative = 0
        for t in all_trades:
            cumulative += t.pnl
            if t.entry_time == trade.entry_time:
                print(f"Cumulative P&L up to this trade: ${cumulative:.2f}")
                print(f"Cumulative / 1000: {cumulative/1000:.1f}k")
                break