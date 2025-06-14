#!/usr/bin/env python3
"""
Analyze the detailed trade log to provide insights
"""

import pandas as pd
import numpy as np

def analyze_trade_log(filename):
    """Analyze the detailed trade log"""
    
    # Load trade log
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("="*80)
    print("TRADE LOG ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"Total actions: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Entry analysis
    entries = df[df['action'] == 'ENTRY']
    print(f"\nTotal entries: {len(entries)}")
    print(f"Long entries: {len(entries[entries['direction'] == 'long'])}")
    print(f"Short entries: {len(entries[entries['direction'] == 'short'])}")
    
    # Position sizing analysis
    print("\nPosition Sizing:")
    print(f"Standard position (1M): {len(entries[entries['size'] == 1000000])}")
    print(f"Increased position (3M): {len(entries[entries['size'] == 3000000])}")
    print(f"Other sizes: {len(entries[(entries['size'] != 1000000) & (entries['size'] != 3000000)])}")
    
    # Exit analysis
    exits = df[df['action'].str.contains('EXIT')]
    print(f"\nTotal exits: {len(exits)}")
    
    # Exit reasons breakdown
    print("\nExit Reasons:")
    final_exits = df[df['action'] == 'FINAL_EXIT']
    exit_reasons = final_exits['reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count}")
    
    # Partial exit analysis
    partial_exits = df[df['action'] == 'PARTIAL_EXIT']
    tp_exits = df[df['action'].str.startswith('TP')]
    print(f"\nPartial exits: {len(partial_exits)}")
    print(f"TP exits: {len(tp_exits)}")
    
    # P&L Analysis
    print("\nP&L Analysis:")
    
    # Get final exits with P&L
    trade_results = []
    for trade_id in entries['trade_id'].unique():
        trade_entries = df[df['trade_id'] == trade_id]
        entry = trade_entries[trade_entries['action'] == 'ENTRY'].iloc[0]
        trade_exits = trade_entries[trade_entries['action'].str.contains('EXIT')]
        
        if len(trade_exits) > 0:
            total_pnl = trade_exits['pnl'].sum()
            total_pips = trade_exits['pips'].sum()
            
            trade_results.append({
                'trade_id': trade_id,
                'direction': entry['direction'],
                'entry_price': entry['entry_price'],
                'size': entry['size'],
                'confidence': entry['confidence'],
                'total_pnl': total_pnl,
                'total_pips': total_pips,
                'num_exits': len(trade_exits)
            })
    
    results_df = pd.DataFrame(trade_results)
    
    if len(results_df) > 0:
        winning_trades = results_df[results_df['total_pnl'] > 0]
        losing_trades = results_df[results_df['total_pnl'] <= 0]
        
        print(f"Total completed trades: {len(results_df)}")
        print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(results_df)*100:.1f}%)")
        print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(results_df)*100:.1f}%)")
        
        print(f"\nAverage P&L per trade: ${results_df['total_pnl'].mean():,.2f}")
        print(f"Average pips per trade: {results_df['total_pips'].mean():.1f}")
        
        if len(winning_trades) > 0:
            print(f"\nWinning trades:")
            print(f"  Average win: ${winning_trades['total_pnl'].mean():,.2f}")
            print(f"  Average winning pips: {winning_trades['total_pips'].mean():.1f}")
            print(f"  Largest win: ${winning_trades['total_pnl'].max():,.2f}")
        
        if len(losing_trades) > 0:
            print(f"\nLosing trades:")
            print(f"  Average loss: ${losing_trades['total_pnl'].mean():,.2f}")
            print(f"  Average losing pips: {losing_trades['total_pips'].mean():.1f}")
            print(f"  Largest loss: ${losing_trades['total_pnl'].min():,.2f}")
        
        # Position size analysis
        print(f"\nP&L by position size:")
        for size in results_df['size'].unique():
            size_trades = results_df[results_df['size'] == size]
            size_wins = size_trades[size_trades['total_pnl'] > 0]
            print(f"  {size/1e6:.0f}M position: {len(size_trades)} trades, "
                  f"{len(size_wins)/len(size_trades)*100:.1f}% win rate, "
                  f"Avg P&L: ${size_trades['total_pnl'].mean():,.2f}")
    
    # Risk analysis
    print("\nRisk Management:")
    print(f"Average SL distance: {(entries['entry_price'] - entries['sl_level']).abs().mean() * 10000:.1f} pips")
    print(f"Average TP1 distance: {(entries['tp1_level'] - entries['entry_price']).abs().mean() * 10000:.1f} pips")
    
    # Sample trades
    print("\nSample Trade Details (First 5 completed trades):")
    print("-"*80)
    
    count = 0
    for trade_id in entries['trade_id'].unique()[:5]:
        trade_data = df[df['trade_id'] == trade_id].sort_values('timestamp')
        entry_data = trade_data[trade_data['action'] == 'ENTRY'].iloc[0]
        exit_data = trade_data[trade_data['action'].str.contains('EXIT')]
        
        if len(exit_data) > 0:
            count += 1
            print(f"\nTrade #{count}:")
            print(f"  Entry: {entry_data['timestamp']} - {entry_data['direction'].upper()} @ {entry_data['entry_price']:.5f}")
            print(f"  Size: {entry_data['size']/1e6:.1f}M units")
            print(f"  SL: {entry_data['sl_level']:.5f} | TP1: {entry_data['tp1_level']:.5f}")
            print(f"  Reason: {entry_data['reason']}")
            
            print("  Exits:")
            for _, exit in exit_data.iterrows():
                print(f"    {exit['timestamp']} - {exit['action']} @ {exit['current_price']:.5f}")
                print(f"      Size: {exit['size']/1e6:.1f}M | P&L: ${exit['pnl']:,.2f} ({exit['pips']:.1f} pips)")

if __name__ == "__main__":
    analyze_trade_log('results/AUDUSD_detailed_trade_log_feb_mar_2025.csv')