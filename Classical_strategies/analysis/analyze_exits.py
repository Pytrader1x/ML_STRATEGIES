"""
Analyze exit reasons and their profitability
Focus on identifying improvement opportunities
"""

import pandas as pd
import numpy as np
from Prod_strategy import create_strategy
from technical_indicators_custom import TIC
import time
from datetime import datetime, timedelta
from collections import defaultdict

def analyze_exit_performance(results):
    """Analyze performance by exit reason"""
    
    # Group trades by exit reason and P&L
    exit_analysis = defaultdict(lambda: {
        'total': 0,
        'profitable': 0,
        'losses': 0,
        'total_pnl': 0,
        'total_profit': 0,
        'total_loss': 0,
        'pips_won': [],
        'pips_lost': [],
        'durations': [],
        'position_sizes': []
    })
    
    for trade in results['trades']:
        if not trade.exit_reason:
            continue
            
        reason = trade.exit_reason.value
        stats = exit_analysis[reason]
        
        # Calculate pip movement
        if trade.direction.value == 'long':
            pips = (trade.exit_price - trade.entry_price) * 10000
        else:
            pips = (trade.entry_price - trade.exit_price) * 10000
        
        # Duration
        if trade.exit_time and trade.entry_time:
            duration_hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            stats['durations'].append(duration_hours)
        
        # Update statistics
        stats['total'] += 1
        stats['total_pnl'] += trade.pnl
        stats['position_sizes'].append(trade.position_size / 1_000_000)
        
        if trade.pnl > 0:
            stats['profitable'] += 1
            stats['total_profit'] += trade.pnl
            stats['pips_won'].append(pips)
        else:
            stats['losses'] += 1
            stats['total_loss'] += trade.pnl
            stats['pips_lost'].append(pips)
    
    return exit_analysis

def print_exit_analysis(exit_analysis):
    """Print detailed exit analysis"""
    
    print("\n" + "=" * 100)
    print("DETAILED EXIT REASON ANALYSIS")
    print("=" * 100)
    
    # Sort by total trades
    sorted_exits = sorted(exit_analysis.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for reason, stats in sorted_exits:
        win_rate = (stats['profitable'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_pnl = stats['total_pnl'] / stats['total'] if stats['total'] > 0 else 0
        
        print(f"\n{reason.upper()}")
        print("-" * 60)
        print(f"Total Trades: {stats['total']}")
        print(f"Profitable: {stats['profitable']} ({stats['profitable']/stats['total']*100:.1f}%)")
        print(f"Losses: {stats['losses']} ({stats['losses']/stats['total']*100:.1f}%)")
        print(f"Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"Average P&L per trade: ${avg_pnl:,.2f}")
        
        if stats['pips_won']:
            print(f"Average winning pips: {np.mean(stats['pips_won']):.1f}")
            print(f"Max winning pips: {max(stats['pips_won']):.1f}")
        
        if stats['pips_lost']:
            print(f"Average losing pips: {np.mean(stats['pips_lost']):.1f}")
            print(f"Max losing pips: {min(stats['pips_lost']):.1f}")
        
        if stats['durations']:
            print(f"Average duration: {np.mean(stats['durations']):.1f} hours")
        
        print(f"Average position size: {np.mean(stats['position_sizes']):.1f}M")

def analyze_signal_flip_patterns(results, df):
    """Analyze patterns in signal flip exits"""
    
    print("\n" + "=" * 100)
    print("SIGNAL FLIP PATTERN ANALYSIS")
    print("=" * 100)
    
    signal_flip_trades = [t for t in results['trades'] if t.exit_reason and t.exit_reason.value == 'signal_flip']
    
    # Analyze time to signal flip
    time_to_flip = []
    flip_after_profit = 0
    flip_after_loss = 0
    
    for trade in signal_flip_trades:
        if trade.exit_time and trade.entry_time:
            hours = (trade.exit_time - trade.entry_time).total_seconds() / 3600
            time_to_flip.append(hours)
            
            # Check if trade was in profit when flipped
            if trade.direction.value == 'long':
                max_favorable = df.loc[trade.entry_time:trade.exit_time]['High'].max()
                max_profit_pips = (max_favorable - trade.entry_price) * 10000
            else:
                min_favorable = df.loc[trade.entry_time:trade.exit_time]['Low'].min()
                max_profit_pips = (trade.entry_price - min_favorable) * 10000
            
            if max_profit_pips > 5:  # Was in profit by at least 5 pips
                flip_after_profit += 1
            else:
                flip_after_loss += 1
    
    print(f"Total signal flip exits: {len(signal_flip_trades)}")
    print(f"Average time to flip: {np.mean(time_to_flip):.1f} hours")
    print(f"Flips after being in profit: {flip_after_profit} ({flip_after_profit/len(signal_flip_trades)*100:.1f}%)")
    print(f"Flips without reaching profit: {flip_after_loss} ({flip_after_loss/len(signal_flip_trades)*100:.1f}%)")

def analyze_trailing_stop_patterns(results):
    """Analyze trailing stop patterns"""
    
    print("\n" + "=" * 100)
    print("TRAILING STOP PATTERN ANALYSIS")
    print("=" * 100)
    
    tsl_trades = [t for t in results['trades'] if t.exit_reason and t.exit_reason.value == 'trailing_stop']
    
    # Group by P&L
    profitable_tsl = [t for t in tsl_trades if t.pnl > 0]
    losing_tsl = [t for t in tsl_trades if t.pnl <= 0]
    
    print(f"Total trailing stop exits: {len(tsl_trades)}")
    print(f"Profitable TSL exits: {len(profitable_tsl)} ({len(profitable_tsl)/len(tsl_trades)*100:.1f}%)")
    print(f"Losing TSL exits: {len(losing_tsl)} ({len(losing_tsl)/len(tsl_trades)*100:.1f}%)")
    
    # Analyze profitable TSL
    if profitable_tsl:
        avg_profit = np.mean([t.pnl for t in profitable_tsl])
        print(f"\nProfitable TSL trades:")
        print(f"  Average profit: ${avg_profit:,.2f}")
        
    # Analyze losing TSL (should be rare)
    if losing_tsl:
        print(f"\nLosing TSL trades (investigating anomaly):")
        for trade in losing_tsl[:5]:  # Show first 5
            if trade.direction.value == 'long':
                pips = (trade.exit_price - trade.entry_price) * 10000
            else:
                pips = (trade.entry_price - trade.exit_price) * 10000
            print(f"  Entry: {trade.entry_price:.5f}, Exit: {trade.exit_price:.5f}, Pips: {pips:.1f}, P&L: ${trade.pnl:.2f}")

def suggest_improvements(exit_analysis):
    """Suggest improvements based on analysis"""
    
    print("\n" + "=" * 100)
    print("IMPROVEMENT SUGGESTIONS")
    print("=" * 100)
    
    # Signal flip analysis
    signal_flip_stats = exit_analysis.get('signal_flip', {})
    if signal_flip_stats['total'] > 0:
        sf_loss_rate = signal_flip_stats['losses'] / signal_flip_stats['total'] * 100
        print(f"\n1. SIGNAL FLIP OPTIMIZATION:")
        print(f"   - {sf_loss_rate:.1f}% of signal flips result in losses")
        print(f"   - Average loss on signal flip: ${signal_flip_stats['total_loss']/signal_flip_stats['losses']:.2f}" if signal_flip_stats['losses'] > 0 else "")
        print("   - Consider: Adding a minimum profit threshold before allowing signal flip exits")
        print("   - Consider: Time-based filter (e.g., ignore flips in first 2 hours)")
    
    # Stop loss analysis
    sl_stats = exit_analysis.get('stop_loss', {})
    if sl_stats['total'] > 0:
        print(f"\n2. STOP LOSS OPTIMIZATION:")
        print(f"   - {sl_stats['total']} stop losses hit ({sl_stats['total']/sum(s['total'] for s in exit_analysis.values())*100:.1f}% of all trades)")
        print(f"   - Average SL loss: ${sl_stats['total_loss']/sl_stats['total']:.2f}")
        print("   - Consider: Dynamic stop loss based on volatility")
        print("   - Consider: Partial exits before stop loss")
    
    # Take profit analysis
    tp_total = sum(exit_analysis.get(f'take_profit_{i}', {}).get('total', 0) for i in [1, 2, 3])
    if tp_total > 0:
        print(f"\n3. TAKE PROFIT OPTIMIZATION:")
        print(f"   - Only {tp_total} trades reached TP3 (full exit)")
        print("   - Consider: Tighter TP levels in ranging markets")
        print("   - Consider: Dynamic TP based on momentum")
    
    # Position sizing
    print(f"\n4. POSITION SIZING INSIGHTS:")
    total_trades = sum(s['total'] for s in exit_analysis.values())
    all_positions = []
    for s in exit_analysis.values():
        if s['position_sizes']:
            all_positions.extend(s['position_sizes'])
    if all_positions:
        avg_position = np.mean(all_positions)
        print(f"   - Average position size: {avg_position:.1f}M")
        print("   - Consider: More aggressive sizing on high-confidence setups")

def main():
    """Run detailed exit analysis"""
    
    print("Loading data and running backtest...")
    
    # Load data
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Get last 2 years
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=730)
    df_2years = df[df.index >= start_date].copy()
    
    # Calculate indicators
    df_2years = TIC.add_neuro_trend_intelligent(df_2years, base_fast=10, base_slow=50, confirm_bars=3)
    df_2years = TIC.add_market_bias(df_2years, ha_len=350, ha_len2=30)
    df_2years = TIC.add_intelligent_chop(df_2years)
    
    # Run strategy
    strategy = create_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        exit_on_signal_flip=True,
        intelligent_sizing=True,
        relaxed_mode=False,
        verbose=False
    )
    
    results = strategy.run_backtest(df_2years)
    
    # Analyze exits
    exit_analysis = analyze_exit_performance(results)
    print_exit_analysis(exit_analysis)
    
    # Pattern analysis
    analyze_signal_flip_patterns(results, df_2years)
    analyze_trailing_stop_patterns(results)
    
    # Suggestions
    suggest_improvements(exit_analysis)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")

if __name__ == "__main__":
    main()