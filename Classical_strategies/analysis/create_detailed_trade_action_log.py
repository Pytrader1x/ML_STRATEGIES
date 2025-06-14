#!/usr/bin/env python3
"""
Create a detailed trade action log that shows every entry and exit action
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC

def create_detailed_action_log():
    """Create a log showing every trade action (entry, partial exit, full exit)"""
    
    # Load data
    possible_paths = ['data', '../data']
    data_path = None
    for path in possible_paths:
        file_path = os.path.join(path, 'AUDUSD_MASTER_15M.csv')
        if os.path.exists(file_path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("Cannot find AUDUSD data")
    
    file_path = os.path.join(data_path, 'AUDUSD_MASTER_15M.csv')
    
    print("Loading AUDUSD data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Filter to Feb-March 2025
    df = df[(df.index >= '2025-02-01') & (df.index <= '2025-03-31')]
    
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Test both configurations
    configs = [
        ("Config_1_Ultra-Tight_Risk", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            realistic_costs=True
        )),
        ("Config_2_Scalping", OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.001,
            sl_max_pips=5.0,
            realistic_costs=True
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n{'='*80}")
        print(f"Running {config_name}")
        print(f"{'='*80}")
        
        # Enable trade logging
        strategy = OptimizedProdStrategy(config)
        strategy.enable_trade_logging = True
        
        # Run backtest
        results = strategy.run_backtest(df)
        
        # Get the detailed trade log
        trade_log = results.get('trade_log', [])
        
        if not trade_log:
            print("No trade log found!")
            continue
            
        # Convert to DataFrame
        log_df = pd.DataFrame(trade_log)
        
        # Create comprehensive action log
        action_records = []
        trade_summaries = {}  # Track cumulative info per trade
        
        for _, log_entry in log_df.iterrows():
            trade_id = log_entry['trade_id']
            
            # Initialize trade summary if new trade
            if trade_id not in trade_summaries:
                trade_summaries[trade_id] = {
                    'entry_price': None,
                    'entry_size': 0,
                    'total_exits': 0,
                    'cumulative_pnl': 0,
                    'tp_hits': 0
                }
            
            summary = trade_summaries[trade_id]
            
            if log_entry['action'] == 'ENTRY':
                # Trade entry
                summary['entry_price'] = log_entry['entry_price']
                summary['entry_size'] = log_entry['size']
                
                action_records.append({
                    'timestamp': log_entry['timestamp'],
                    'trade_id': trade_id,
                    'action': 'ENTRY',
                    'direction': log_entry['direction'],
                    'price': log_entry['entry_price'],
                    'size': log_entry['size'],
                    'size_millions': log_entry['size'] / 1e6,
                    'exit_type': '',
                    'pnl': 0,
                    'pips': 0,
                    'cumulative_pnl': 0,
                    'remaining_size': log_entry['size'],
                    'sl_level': log_entry['sl_level'],
                    'tp1_level': log_entry['tp1_level'],
                    'tp2_level': log_entry['tp2_level'],
                    'tp3_level': log_entry['tp3_level'],
                    'reason': log_entry['reason']
                })
                
            elif 'EXIT' in log_entry['action']:
                # Exit action
                summary['total_exits'] += log_entry['size']
                summary['cumulative_pnl'] += log_entry['pnl']
                
                # Determine exit type
                exit_type = ''
                if 'TP' in log_entry['action']:
                    tp_num = log_entry['action'].replace('TP', '').replace('_EXIT', '')
                    exit_type = f'TP{tp_num}'
                    summary['tp_hits'] += 1
                elif log_entry['action'] == 'PARTIAL_EXIT':
                    exit_type = 'PARTIAL'
                elif log_entry['action'] == 'FINAL_EXIT':
                    exit_type = log_entry['reason'].replace('Exit: ', '')
                
                action_records.append({
                    'timestamp': log_entry['timestamp'],
                    'trade_id': trade_id,
                    'action': 'EXIT',
                    'direction': log_entry['direction'],
                    'price': log_entry['current_price'],
                    'size': log_entry['size'],
                    'size_millions': log_entry['size'] / 1e6,
                    'exit_type': exit_type,
                    'pnl': log_entry['pnl'],
                    'pips': log_entry['pips'],
                    'cumulative_pnl': summary['cumulative_pnl'],
                    'remaining_size': log_entry['remaining_size'],
                    'sl_level': log_entry['sl_level'],
                    'tp1_level': log_entry['tp1_level'],
                    'tp2_level': log_entry['tp2_level'],
                    'tp3_level': log_entry['tp3_level'],
                    'reason': log_entry['reason']
                })
        
        # Convert to DataFrame and save
        action_df = pd.DataFrame(action_records)
        
        # Save to CSV
        output_file = f'results/AUDUSD_{config_name}_detailed_action_log.csv'
        action_df.to_csv(output_file, index=False)
        print(f"\nSaved detailed action log to: {output_file}")
        
        # Print summary statistics
        print(f"\nAction Summary:")
        print(f"Total Entries: {len(action_df[action_df['action'] == 'ENTRY'])}")
        print(f"Total Exits: {len(action_df[action_df['action'] == 'EXIT'])}")
        
        # Exit type breakdown
        exit_df = action_df[action_df['action'] == 'EXIT']
        if len(exit_df) > 0:
            print(f"\nExit Type Breakdown:")
            for exit_type, count in exit_df['exit_type'].value_counts().items():
                print(f"  {exit_type}: {count}")
        
        # Show sample entries
        print(f"\nSample Action Log:")
        print(action_df.head(20).to_string())
        
        # Verify position sizing
        print(f"\nPosition Size Verification:")
        for trade_id in action_df['trade_id'].unique()[:5]:
            trade_actions = action_df[action_df['trade_id'] == trade_id]
            entries = trade_actions[trade_actions['action'] == 'ENTRY']
            exits = trade_actions[trade_actions['action'] == 'EXIT']
            
            if len(entries) > 0 and len(exits) > 0:
                total_entry = entries['size'].sum()
                total_exit = exits['size'].sum()
                print(f"  Trade {trade_id}: Entry={total_entry/1e6:.1f}M, Exit={total_exit/1e6:.1f}M, "
                      f"Difference={abs(total_entry-total_exit)/1e6:.3f}M")

if __name__ == "__main__":
    create_detailed_action_log()