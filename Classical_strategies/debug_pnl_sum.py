"""
Debug P&L sum discrepancy for March 30 21:15 trade
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

# Find the March 30 21:15 trade
target_trade = None
for trade in results['trades']:
    if str(trade.entry_time) == '2025-03-30 21:15:00':
        target_trade = trade
        break

if target_trade:
    print("="*80)
    print("MARCH 30 21:15 TRADE ANALYSIS")
    print("="*80)
    
    print(f'\nTrade Entry: {target_trade.entry_time}')
    print(f'Direction: {target_trade.direction.value}')
    print(f'Entry Price: {target_trade.entry_price:.5f}')
    print(f'Position: {target_trade.position_size/1e6:.2f}M')
    print(f'Exit Reason: {target_trade.exit_reason}')
    print(f'Total Trade P&L: ${target_trade.pnl:.2f}')
    
    print('\n' + '-'*60)
    print('PARTIAL EXITS:')
    print('-'*60)
    
    total_from_partials = 0
    for i, pe in enumerate(target_trade.partial_exits):
        pe_pips = 0
        if target_trade.direction.value == 'short':
            pe_pips = (target_trade.entry_price - pe.price) * 10000
        else:
            pe_pips = (pe.price - target_trade.entry_price) * 10000
            
        print(f'\nPartial Exit {i+1} (TP{pe.tp_level}):')
        print(f'  Time: {pe.time}')
        print(f'  Price: {pe.price:.5f}')
        print(f'  Size: {pe.size/1e6:.2f}M')
        print(f'  Pips: {pe_pips:.1f}')
        print(f'  P&L: ${pe.pnl:.2f}')
        total_from_partials += pe.pnl
    
    print('\n' + '='*60)
    print('P&L VERIFICATION:')
    print('='*60)
    print(f'Sum of partial P&Ls: ${total_from_partials:.2f}')
    print(f'Trade total P&L: ${target_trade.pnl:.2f}')
    print(f'Difference: ${target_trade.pnl - total_from_partials:.2f}')
    
    if abs(target_trade.pnl - total_from_partials) > 0.01:
        print('\n❌ DISCREPANCY FOUND!')
        
        # Manual calculation check
        print('\n' + '-'*60)
        print('MANUAL P&L CALCULATION:')
        print('-'*60)
        
        # For short: P&L = (Entry - Exit) × 10,000 × Size(M) × $100/pip
        manual_total = 0
        for i, pe in enumerate(target_trade.partial_exits):
            if target_trade.direction.value == 'short':
                pips = (target_trade.entry_price - pe.price) * 10000
            else:
                pips = (pe.price - target_trade.entry_price) * 10000
            
            size_m = pe.size / 1e6
            manual_pnl = pips * size_m * 100
            
            print(f'\nExit {i+1}:')
            print(f'  Pips: {pips:.1f}')
            print(f'  Size: {size_m:.2f}M')
            print(f'  Manual P&L: {pips:.1f} × {size_m:.2f} × $100 = ${manual_pnl:.2f}')
            manual_total += manual_pnl
        
        print(f'\nManual Total: ${manual_total:.2f}')
        print(f'Strategy Total: ${target_trade.pnl:.2f}')
        print(f'Partial Sum: ${total_from_partials:.2f}')
    else:
        print('\n✅ P&L values match correctly')