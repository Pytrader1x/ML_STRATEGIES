#!/usr/bin/env python3
"""
Analyze stop loss outcomes - distinguish between losing SL, breakeven SL, and profitable SL
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import os
import warnings
import time

warnings.filterwarnings('ignore')

def load_and_prepare_data(currency_pair, start_date, end_date):
    """Load and prepare data for a specific currency pair and date range"""
    
    possible_paths = ['data', '../data']
    data_path = None
    for path in possible_paths:
        file_path = os.path.join(path, f'{currency_pair}_MASTER_15M.csv')
        if os.path.exists(file_path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Cannot find data for {currency_pair}")
    
    file_path = os.path.join(data_path, f'{currency_pair}_MASTER_15M.csv')
    
    print(f"Loading {currency_pair} data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total data points: {len(df):,}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    return df

def analyze_sl_outcomes(trades):
    """Analyze stop loss outcomes - loss, breakeven, or profit"""
    
    sl_outcomes = {
        'sl_with_loss': [],
        'sl_breakeven': [],
        'sl_with_profit': [],
        'no_sl_exit': []
    }
    
    # Define breakeven threshold (in dollars)
    breakeven_threshold = 50  # Consider within $50 as breakeven
    
    for trade in trades:
        if trade.exit_reason and 'stop_loss' in trade.exit_reason.value:
            # This trade exited via stop loss
            if trade.pnl < -breakeven_threshold:
                sl_outcomes['sl_with_loss'].append(trade)
            elif -breakeven_threshold <= trade.pnl <= breakeven_threshold:
                sl_outcomes['sl_breakeven'].append(trade)
            else:  # trade.pnl > breakeven_threshold
                sl_outcomes['sl_with_profit'].append(trade)
        else:
            sl_outcomes['no_sl_exit'].append(trade)
    
    return sl_outcomes

def analyze_exit_patterns_comprehensive(trades):
    """Comprehensive exit pattern analysis"""
    
    patterns = {
        # Pure exits
        'pure_sl_loss': 0,
        'pure_sl_breakeven': 0,
        'pure_sl_profit': 0,
        'pure_tp_full': 0,  # All 3 TPs hit
        
        # Partial then SL
        'partial_then_sl_loss': 0,
        'partial_then_sl_breakeven': 0,
        'partial_then_sl_profit': 0,
        
        # TP then exit
        'tp1_only': 0,
        'tp2_partial': 0,
        'tp_then_sl': 0,
        
        # Other
        'signal_flip': 0,
        'end_of_data': 0,
        'other': 0
    }
    
    detailed_trades = []
    breakeven_threshold = 50
    
    for i, trade in enumerate(trades):
        trade_info = {
            'num': i + 1,
            'direction': trade.direction.value,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'position_size': trade.position_size,
            'pnl': trade.pnl,
            'pnl_pct': (trade.pnl / (trade.entry_price * trade.position_size)) * 100,
            'tp_hits': trade.tp_hits,
            'partial_exits': len(trade.partial_exits),
            'exit_reason': trade.exit_reason.value if trade.exit_reason else 'unknown',
            'pattern': '',
            'sl_outcome': ''
        }
        
        # Determine pattern
        if trade.exit_reason:
            if 'stop_loss' in trade.exit_reason.value:
                # Stop loss exit - determine if pure or after partials
                if len(trade.partial_exits) == 0 and trade.tp_hits == 0:
                    # Pure stop loss
                    if trade.pnl < -breakeven_threshold:
                        patterns['pure_sl_loss'] += 1
                        trade_info['pattern'] = 'Pure SL (Loss)'
                        trade_info['sl_outcome'] = 'Loss'
                    elif -breakeven_threshold <= trade.pnl <= breakeven_threshold:
                        patterns['pure_sl_breakeven'] += 1
                        trade_info['pattern'] = 'Pure SL (Breakeven)'
                        trade_info['sl_outcome'] = 'Breakeven'
                    else:
                        patterns['pure_sl_profit'] += 1
                        trade_info['pattern'] = 'Pure SL (Profit)'
                        trade_info['sl_outcome'] = 'Profit'
                else:
                    # Partial exits then stop loss
                    if trade.pnl < -breakeven_threshold:
                        patterns['partial_then_sl_loss'] += 1
                        trade_info['pattern'] = 'Partial Exit → SL (Loss)'
                        trade_info['sl_outcome'] = 'Loss'
                    elif -breakeven_threshold <= trade.pnl <= breakeven_threshold:
                        patterns['partial_then_sl_breakeven'] += 1
                        trade_info['pattern'] = 'Partial Exit → SL (Breakeven)'
                        trade_info['sl_outcome'] = 'Breakeven'
                    else:
                        patterns['partial_then_sl_profit'] += 1
                        trade_info['pattern'] = 'Partial Exit → SL (Profit)'
                        trade_info['sl_outcome'] = 'Profit'
                        
            elif 'take_profit' in trade.exit_reason.value:
                if trade.tp_hits >= 3:
                    patterns['pure_tp_full'] += 1
                    trade_info['pattern'] = 'Full TP (TP3)'
                elif trade.tp_hits == 1:
                    patterns['tp1_only'] += 1
                    trade_info['pattern'] = 'TP1 Exit'
                elif trade.tp_hits == 2:
                    patterns['tp2_partial'] += 1
                    trade_info['pattern'] = 'TP2 Exit'
                    
            elif 'signal_flip' in trade.exit_reason.value:
                patterns['signal_flip'] += 1
                trade_info['pattern'] = 'Signal Flip'
                
            elif 'end_of_data' in trade.exit_reason.value:
                patterns['end_of_data'] += 1
                trade_info['pattern'] = 'End of Data'
                
            else:
                patterns['other'] += 1
                trade_info['pattern'] = f'Other ({trade.exit_reason.value})'
        
        detailed_trades.append(trade_info)
    
    return patterns, detailed_trades

def print_comprehensive_analysis(patterns, sl_outcomes, total_trades, config_name):
    """Print comprehensive analysis including SL outcomes"""
    
    print(f"\n{'='*80}")
    print(f"{config_name} - COMPREHENSIVE EXIT ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\nTotal Trades: {total_trades}")
    
    # Stop Loss Analysis
    total_sl_exits = len(sl_outcomes['sl_with_loss']) + len(sl_outcomes['sl_breakeven']) + len(sl_outcomes['sl_with_profit'])
    
    print(f"\n━━━ Stop Loss Outcome Analysis ━━━")
    print(f"Total Stop Loss Exits: {total_sl_exits} ({total_sl_exits/total_trades*100:.1f}%)")
    
    if total_sl_exits > 0:
        sl_loss_pct = len(sl_outcomes['sl_with_loss']) / total_sl_exits * 100
        sl_be_pct = len(sl_outcomes['sl_breakeven']) / total_sl_exits * 100
        sl_profit_pct = len(sl_outcomes['sl_with_profit']) / total_sl_exits * 100
        
        print(f"\nOf trades that hit Stop Loss:")
        print(f"  SL with LOSS:      {len(sl_outcomes['sl_with_loss']):>4} ({sl_loss_pct:>5.1f}%)")
        print(f"  SL at BREAKEVEN:   {len(sl_outcomes['sl_breakeven']):>4} ({sl_be_pct:>5.1f}%)")
        print(f"  SL with PROFIT:    {len(sl_outcomes['sl_with_profit']):>4} ({sl_profit_pct:>5.1f}%)")
        
        # Calculate average P&L for each SL outcome
        if sl_outcomes['sl_with_loss']:
            avg_sl_loss = np.mean([t.pnl for t in sl_outcomes['sl_with_loss']])
            print(f"\n  Avg P&L for SL losses: ${avg_sl_loss:,.2f}")
        
        if sl_outcomes['sl_with_profit']:
            avg_sl_profit = np.mean([t.pnl for t in sl_outcomes['sl_with_profit']])
            print(f"  Avg P&L for SL profits: ${avg_sl_profit:,.2f}")
    
    # Detailed Pattern Breakdown
    print(f"\n━━━ Detailed Exit Pattern Breakdown ━━━")
    
    print(f"\nPURE STOP LOSS EXITS:")
    pure_sl_total = patterns['pure_sl_loss'] + patterns['pure_sl_breakeven'] + patterns['pure_sl_profit']
    print(f"  Pure SL with Loss:       {patterns['pure_sl_loss']:>4} ({patterns['pure_sl_loss']/total_trades*100:>5.1f}%)")
    print(f"  Pure SL at Breakeven:    {patterns['pure_sl_breakeven']:>4} ({patterns['pure_sl_breakeven']/total_trades*100:>5.1f}%)")
    print(f"  Pure SL with Profit:     {patterns['pure_sl_profit']:>4} ({patterns['pure_sl_profit']/total_trades*100:>5.1f}%)")
    print(f"  Total Pure SL:           {pure_sl_total:>4} ({pure_sl_total/total_trades*100:>5.1f}%)")
    
    print(f"\nPARTIAL EXIT THEN STOP LOSS:")
    partial_sl_total = patterns['partial_then_sl_loss'] + patterns['partial_then_sl_breakeven'] + patterns['partial_then_sl_profit']
    print(f"  Partial → SL (Loss):     {patterns['partial_then_sl_loss']:>4} ({patterns['partial_then_sl_loss']/total_trades*100:>5.1f}%)")
    print(f"  Partial → SL (BE):       {patterns['partial_then_sl_breakeven']:>4} ({patterns['partial_then_sl_breakeven']/total_trades*100:>5.1f}%)")
    print(f"  Partial → SL (Profit):   {patterns['partial_then_sl_profit']:>4} ({patterns['partial_then_sl_profit']/total_trades*100:>5.1f}%)")
    print(f"  Total Partial → SL:      {partial_sl_total:>4} ({partial_sl_total/total_trades*100:>5.1f}%)")
    
    print(f"\nTAKE PROFIT EXITS:")
    tp_total = patterns['tp1_only'] + patterns['tp2_partial'] + patterns['pure_tp_full']
    print(f"  TP1 Exit:                {patterns['tp1_only']:>4} ({patterns['tp1_only']/total_trades*100:>5.1f}%)")
    print(f"  TP2 Exit:                {patterns['tp2_partial']:>4} ({patterns['tp2_partial']/total_trades*100:>5.1f}%)")
    print(f"  TP3 (Full TP):           {patterns['pure_tp_full']:>4} ({patterns['pure_tp_full']/total_trades*100:>5.1f}%)")
    print(f"  Total TP Exits:          {tp_total:>4} ({tp_total/total_trades*100:>5.1f}%)")
    
    print(f"\nOTHER EXITS:")
    print(f"  Signal Flip:             {patterns['signal_flip']:>4} ({patterns['signal_flip']/total_trades*100:>5.1f}%)")
    print(f"  End of Data:             {patterns['end_of_data']:>4} ({patterns['end_of_data']/total_trades*100:>5.1f}%)")
    print(f"  Other:                   {patterns['other']:>4} ({patterns['other']/total_trades*100:>5.1f}%)")
    
    # Summary Statistics
    print(f"\n━━━ Key Insights ━━━")
    actual_loss_exits = patterns['pure_sl_loss'] + patterns['partial_then_sl_loss']
    print(f"Trades ending in actual loss:     {actual_loss_exits:>4} ({actual_loss_exits/total_trades*100:>5.1f}%)")
    
    profitable_sl_exits = patterns['pure_sl_profit'] + patterns['partial_then_sl_profit']
    print(f"SL exits that were profitable:    {profitable_sl_exits:>4} ({profitable_sl_exits/total_trades*100:>5.1f}%)")

def show_sample_profitable_sl_trades(detailed_trades):
    """Show examples of profitable stop loss exits"""
    
    profitable_sl = [t for t in detailed_trades if 'SL' in t['pattern'] and 'Profit' in t['pattern']]
    
    if profitable_sl:
        print(f"\n━━━ Sample Profitable Stop Loss Exits ━━━")
        for trade in profitable_sl[:5]:
            print(f"\nTrade #{trade['num']}:")
            print(f"  Pattern: {trade['pattern']}")
            print(f"  Direction: {trade['direction'].upper()} @ {trade['entry_price']:.5f}")
            print(f"  Exit Price: {trade['exit_price']:.5f}")
            print(f"  P&L: ${trade['pnl']:,.2f} ({trade['pnl_pct']:.2f}%)")
            print(f"  TP Hits: {trade['tp_hits']}, Partial Exits: {trade['partial_exits']}")

def main():
    """Run comprehensive stop loss outcome analysis"""
    
    print("="*80)
    print("STOP LOSS OUTCOME ANALYSIS")
    print("Analyzing whether SL exits resulted in losses, breakeven, or profits")
    print("="*80)
    
    # Load data
    currency = 'AUDUSD'
    start_date = '2025-02-01'
    end_date = '2025-03-31'
    
    df = load_and_prepare_data(currency, start_date, end_date)
    
    # Test both configurations
    configs = [
        ("Config 1: Ultra-Tight Risk Management", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            realistic_costs=True,
            use_daily_sharpe=True
        )),
        ("Config 2: Scalping Strategy", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            realistic_costs=True,
            use_daily_sharpe=True
        ))
    ]
    
    for config_name, config in configs:
        # Create and run strategy
        strategy = OptimizedProdStrategy(config)
        print(f"\nRunning {config_name}...")
        results = strategy.run_backtest(df)
        
        # Analyze outcomes
        if 'trades' in results and results['trades']:
            sl_outcomes = analyze_sl_outcomes(results['trades'])
            patterns, detailed_trades = analyze_exit_patterns_comprehensive(results['trades'])
            
            print_comprehensive_analysis(patterns, sl_outcomes, results['total_trades'], config_name)
            show_sample_profitable_sl_trades(detailed_trades)
            
            # Save detailed analysis
            df_analysis = pd.DataFrame(detailed_trades)
            filename = f'results/{currency}_{config_name.replace(":", "").replace(" ", "_").lower()}_sl_analysis.csv'
            df_analysis.to_csv(filename, index=False)
            print(f"\nDetailed analysis saved to {filename}")

if __name__ == "__main__":
    main()