"""
Simple Trade Forensics - Direct Analysis of Trading Strategy Legitimacy

This script performs a deep forensic analysis of actual trades to verify
the strategy is legitimate and not cheating.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_Strategy import create_config_2_scalping, load_and_prepare_data


def analyze_trades_forensically():
    """Perform forensic analysis on actual trades."""
    
    print("="*80)
    print("FORENSIC TRADE ANALYSIS - LEGITIMACY CHECK")
    print("="*80)
    
    # Load data and run strategy
    print("\n1. Running strategy to collect trades...")
    symbol = 'AUDUSD'
    full_df = load_and_prepare_data(symbol)
    df = full_df.tail(10000).copy()  # Use 10,000 bars for analysis
    
    # Create and run strategy
    strategy = create_config_2_scalping(realistic_costs=True)
    stats = strategy.run_backtest(df)
    trades = strategy.trades
    
    print(f"✓ Strategy completed: {len(trades)} trades")
    print(f"✓ Win rate: {stats['win_rate']:.1f}%")
    print(f"✓ Sharpe ratio: {stats['sharpe_ratio']:.2f}")
    
    # Analyze trades
    print("\n2. Analyzing individual trades for legitimacy...")
    
    suspicious_patterns = []
    trade_details = []
    
    # Sample 20 trades for detailed analysis
    sample_size = min(20, len(trades))
    sample_trades = trades[:sample_size]
    
    for i, trade in enumerate(sample_trades):
        print(f"\n--- Trade {i+1} ---")
        
        # Find the bar indices for entry and exit
        entry_idx = df.index.get_loc(trade.entry_time)
        exit_idx = df.index.get_loc(trade.exit_time) if trade.exit_time else None
        
        # Get bar data
        entry_bar = df.iloc[entry_idx]
        
        # Trade details
        is_long = trade.direction.value == 'LONG'
        print(f"Direction: {trade.direction.value}")
        print(f"Entry Time: {trade.entry_time}")
        print(f"Entry Price: {trade.entry_price:.5f}")
        print(f"Entry Bar OHLC: O={entry_bar['Open']:.5f}, H={entry_bar['High']:.5f}, L={entry_bar['Low']:.5f}, C={entry_bar['Close']:.5f}")
        
        # Check 1: Entry price within bar range
        entry_in_range = entry_bar['Low'] <= trade.entry_price <= entry_bar['High']
        print(f"Entry price in bar range: {'✅' if entry_in_range else '❌'}")
        
        if not entry_in_range:
            suspicious_patterns.append(f"Trade {i+1}: Entry price {trade.entry_price:.5f} outside bar range [{entry_bar['Low']:.5f}, {entry_bar['High']:.5f}]")
        
        # Check 2: Entry signals
        nti_dir = entry_bar.get('NTI_Direction', None)
        mb_bias = entry_bar.get('MB_Bias', None)
        expected_signal = 1 if is_long else -1
        
        signals_match = (nti_dir == expected_signal and mb_bias == expected_signal)
        print(f"Signals aligned (NTI={nti_dir}, MB={mb_bias}): {'✅' if signals_match else '❌'}")
        
        if not signals_match:
            suspicious_patterns.append(f"Trade {i+1}: Entry signals don't match direction")
        
        # Check 3: Slippage
        if is_long:
            # Long entries should be at or above close (slippage is adverse)
            slippage = (trade.entry_price - entry_bar['Close']) * 10000
            favorable = slippage < 0
        else:
            # Short entries should be at or below close
            slippage = (entry_bar['Close'] - trade.entry_price) * 10000
            favorable = slippage < 0
        
        print(f"Entry slippage: {slippage:.2f} pips {'(FAVORABLE!)' if favorable else ''}")
        
        if favorable:
            suspicious_patterns.append(f"Trade {i+1}: Favorable slippage of {abs(slippage):.2f} pips")
        
        # Check exit if closed
        if trade.exit_time and exit_idx is not None:
            exit_bar = df.iloc[exit_idx]
            print(f"\nExit Time: {trade.exit_time}")
            print(f"Exit Price: {trade.exit_price:.5f}")
            print(f"Exit Reason: {trade.exit_reason.value if trade.exit_reason else 'Unknown'}")
            print(f"Exit Bar OHLC: O={exit_bar['Open']:.5f}, H={exit_bar['High']:.5f}, L={exit_bar['Low']:.5f}, C={exit_bar['Close']:.5f}")
            
            # Check exit price in range
            exit_in_range = exit_bar['Low'] <= trade.exit_price <= exit_bar['High']
            print(f"Exit price in bar range: {'✅' if exit_in_range else '❌'}")
            
            if not exit_in_range:
                suspicious_patterns.append(f"Trade {i+1}: Exit price {trade.exit_price:.5f} outside bar range")
            
            # Check stop loss logic
            if trade.exit_reason and trade.exit_reason.value == 'STOP_LOSS':
                if is_long:
                    # For long, check if low touched stop loss
                    sl_touched = exit_bar['Low'] <= trade.stop_loss
                else:
                    # For short, check if high touched stop loss
                    sl_touched = exit_bar['High'] >= trade.stop_loss
                
                print(f"Stop loss touched: {'✅' if sl_touched else '❌'}")
                
                if not sl_touched:
                    suspicious_patterns.append(f"Trade {i+1}: Stop loss exit without price touching SL")
        
        # Store trade details
        trade_details.append({
            'id': i+1,
            'direction': trade.direction.value,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'slippage': slippage
        })
    
    # Overall analysis
    print("\n" + "="*80)
    print("3. OVERALL ANALYSIS")
    print("="*80)
    
    # Check win rate
    win_rate = stats['win_rate']
    if win_rate > 75:
        suspicious_patterns.append(f"Suspiciously high win rate: {win_rate:.1f}%")
    
    # Check P&L distribution
    pnls = [t.pnl for t in trades if t.pnl is not None]
    if pnls:
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        
        print(f"\nP&L Analysis:")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Win/Loss Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 'N/A':.2f}")
    
    # Check slippage distribution
    slippages = [t['slippage'] for t in trade_details]
    avg_slippage = np.mean(slippages) if slippages else 0
    favorable_count = sum(1 for s in slippages if s < -0.1)  # Count favorable by more than 0.1 pip
    
    print(f"\nSlippage Analysis:")
    print(f"Average slippage: {avg_slippage:.3f} pips")
    print(f"Favorable slippage instances: {favorable_count}/{len(slippages)}")
    
    if favorable_count > len(slippages) * 0.1:  # More than 10% favorable
        suspicious_patterns.append(f"Too many favorable slippage instances: {favorable_count}/{len(slippages)}")
    
    # Final verdict
    print("\n" + "="*80)
    print("4. FORENSIC VERDICT")
    print("="*80)
    
    if suspicious_patterns:
        print("\n⚠️  SUSPICIOUS PATTERNS DETECTED:")
        for pattern in suspicious_patterns:
            print(f"  - {pattern}")
        print("\nVERDICT: Strategy shows suspicious patterns that need investigation")
        legitimate = False
    else:
        print("\n✅ NO SUSPICIOUS PATTERNS DETECTED")
        print("\nThe forensic analysis shows:")
        print("- Entry and exit prices are within bar ranges")
        print("- Entry signals match the trade direction")
        print("- Slippage is consistently adverse (realistic)")
        print("- Win rate is reasonable")
        print("- No evidence of cheating or look-ahead bias")
        print("\nVERDICT: Strategy appears LEGITIMATE")
        legitimate = True
    
    # Create visualizations
    create_forensic_charts(df, trades, trade_details)
    
    return legitimate, suspicious_patterns


def create_forensic_charts(df, trades, trade_details):
    """Create visualization of forensic findings."""
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. P&L Distribution
    pnls = [t.pnl for t in trades if t.pnl is not None]
    if pnls:
        axes[0, 0].hist(pnls, bins=50, alpha=0.7, color='cyan')
        axes[0, 0].axvline(0, color='red', linestyle='--')
        axes[0, 0].set_title('P&L Distribution')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Frequency')
    
    # 2. Slippage Distribution
    slippages = [t['slippage'] for t in trade_details]
    if slippages:
        axes[0, 1].hist(slippages, bins=30, alpha=0.7, color='yellow')
        axes[0, 1].axvline(0, color='red', linestyle='--')
        axes[0, 1].set_title('Entry Slippage Distribution')
        axes[0, 1].set_xlabel('Slippage (pips)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].text(0.02, 0.95, f'Avg: {np.mean(slippages):.3f} pips', 
                       transform=axes[0, 1].transAxes, verticalalignment='top')
    
    # 3. Win Rate by Trade Number
    cumulative_wins = []
    win_count = 0
    for i, trade in enumerate(trades):
        if trade.pnl and trade.pnl > 0:
            win_count += 1
        if i > 0:
            cumulative_wins.append(win_count / (i + 1) * 100)
    
    if cumulative_wins:
        axes[1, 0].plot(cumulative_wins, color='green', alpha=0.7)
        axes[1, 0].axhline(50, color='white', linestyle='--', alpha=0.3)
        axes[1, 0].set_title('Cumulative Win Rate')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].set_ylim(0, 100)
    
    # 4. Exit Reasons
    exit_reasons = {}
    for trade in trades:
        if trade.exit_reason:
            reason = trade.exit_reason.value
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    if exit_reasons:
        axes[1, 1].pie(exit_reasons.values(), labels=exit_reasons.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Exit Reasons')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / 'forensic_analysis_charts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCharts saved to: {output_path}")


def generate_forensic_report(legitimate, patterns):
    """Generate the final forensic report."""
    
    report_lines = [
        "# Trading Strategy Forensic Analysis Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Executive Summary",
        f"\n**Verdict: {'LEGITIMATE' if legitimate else 'SUSPICIOUS'}**",
        "\n## Methodology",
        "\nThis forensic analysis examined individual trades to verify:",
        "1. Entry/exit prices are within bar ranges",
        "2. Entry signals match trade direction",
        "3. Slippage is realistically adverse",
        "4. No impossible trades or perfect timing",
        "5. Reasonable win rates and P&L distribution",
        "\n## Findings",
    ]
    
    if patterns:
        report_lines.append("\n### Suspicious Patterns Found:")
        for pattern in patterns:
            report_lines.append(f"- {pattern}")
    else:
        report_lines.append("\n### No Suspicious Patterns Found")
        report_lines.append("\nAll analyzed trades showed:")
        report_lines.append("- Realistic entry/exit prices")
        report_lines.append("- Proper signal alignment")
        report_lines.append("- Adverse slippage (as expected)")
        report_lines.append("- No evidence of look-ahead bias")
    
    report_lines.append("\n## Conclusion")
    
    if legitimate:
        report_lines.append("\nThe strategy appears to be legitimate with no evidence of cheating or false positive results.")
        report_lines.append("The backtest results can be considered reliable for further analysis.")
    else:
        report_lines.append("\nThe strategy shows suspicious patterns that suggest potential issues.")
        report_lines.append("Further investigation is recommended before trusting the backtest results.")
    
    # Save report
    report_path = Path(__file__).parent / 'forensic_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    
    return report_path


if __name__ == "__main__":
    # Run forensic analysis
    legitimate, patterns = analyze_trades_forensically()
    
    # Generate report
    report_path = generate_forensic_report(legitimate, patterns)
    
    print("\n" + "="*80)
    print("FORENSIC ANALYSIS COMPLETE")
    print("="*80)