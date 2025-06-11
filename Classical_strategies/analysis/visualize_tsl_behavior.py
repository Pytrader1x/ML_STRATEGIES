"""
Visualize TSL behavior to understand the fix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import create_optimized_strategy, FOREX_PIP_SIZE
from technical_indicators_custom import TIC
from datetime import datetime

def visualize_tsl_behavior():
    """Visualize how TSL behaves with the new buffer"""
    
    print("Visualizing TSL Behavior with Initial Buffer")
    print("=" * 60)
    
    # Create synthetic example
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Example 1: Long trade
    entry_price = 1.0000
    prices = []
    tsl_old = []
    tsl_new = []
    min_profit_line = []
    
    # Simulate price movement
    for i in range(100):
        if i < 30:
            # Move up to 15 pips
            price = entry_price + (i / 30) * 15 * FOREX_PIP_SIZE
        elif i < 40:
            # Small pullback
            price = entry_price + 15 * FOREX_PIP_SIZE - ((i - 30) / 10) * 5 * FOREX_PIP_SIZE
        else:
            # Continue up with volatility
            base = entry_price + 10 * FOREX_PIP_SIZE
            trend = ((i - 40) / 60) * 20 * FOREX_PIP_SIZE
            noise = np.sin(i / 5) * 2 * FOREX_PIP_SIZE
            price = base + trend + noise
        
        prices.append(price)
        profit_pips = (price - entry_price) / FOREX_PIP_SIZE
        
        # Calculate TSL positions
        atr = 10 * FOREX_PIP_SIZE  # Assume 10 pip ATR
        trailing_multiplier = 1.2
        
        if profit_pips >= 15:
            # Old TSL (no buffer)
            old_tsl = price - (atr * trailing_multiplier)
            # Ensure minimum profit
            old_tsl = max(old_tsl, entry_price + 5 * FOREX_PIP_SIZE)
            
            # New TSL (with 2x buffer on first activation)
            if i == 30:  # First activation
                new_tsl = price - (atr * trailing_multiplier * 2.0)
            else:
                new_tsl = price - (atr * trailing_multiplier)
            # Ensure minimum profit
            new_tsl = max(new_tsl, entry_price + 5 * FOREX_PIP_SIZE)
            
            tsl_old.append(old_tsl)
            tsl_new.append(new_tsl)
        else:
            tsl_old.append(None)
            tsl_new.append(None)
        
        min_profit_line.append(entry_price + 5 * FOREX_PIP_SIZE)
    
    # Convert to pips for plotting
    x = list(range(len(prices)))
    prices_pips = [(p - entry_price) / FOREX_PIP_SIZE for p in prices]
    tsl_old_pips = [((t - entry_price) / FOREX_PIP_SIZE) if t else None for t in tsl_old]
    tsl_new_pips = [((t - entry_price) / FOREX_PIP_SIZE) if t else None for t in tsl_new]
    
    # Plot long trade
    ax1.plot(x, prices_pips, 'b-', label='Price', linewidth=2)
    ax1.plot(x, tsl_old_pips, 'r--', label='Old TSL (No Buffer)', linewidth=1.5)
    ax1.plot(x, tsl_new_pips, 'g-', label='New TSL (With Buffer)', linewidth=1.5)
    ax1.axhline(y=15, color='orange', linestyle=':', label='TSL Activation (15 pips)')
    ax1.axhline(y=5, color='purple', linestyle=':', label='Min Profit (5 pips)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax1.set_title('Long Trade - TSL Behavior Comparison', fontsize=14)
    ax1.set_xlabel('Time (bars)')
    ax1.set_ylabel('Pips from Entry')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    activation_idx = 30
    ax1.annotate('TSL Activates', xy=(activation_idx, 15), xytext=(activation_idx + 10, 20),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))
    
    # Show the buffer difference
    if tsl_old[35] and tsl_new[35]:
        buffer_diff = (tsl_new[35] - tsl_old[35]) / FOREX_PIP_SIZE
        ax1.annotate(f'Buffer: {buffer_diff:.1f} pips', 
                     xy=(35, tsl_old_pips[35]), 
                     xytext=(45, tsl_old_pips[35] - 5),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1))
    
    # Example 2: Show exit scenarios
    scenarios = ['Immediate Pullback', 'Gradual Rise', 'Sharp Rally']
    colors = ['red', 'blue', 'green']
    
    for idx, (scenario, color) in enumerate(zip(scenarios, colors)):
        y_offset = idx * 30
        x_start = 150 + y_offset
        
        # Entry to 15 pips
        x_vals = list(range(x_start, x_start + 20))
        y_vals = [i * 0.75 for i in range(20)]
        
        if scenario == 'Immediate Pullback':
            # Price pulls back immediately after TSL activation
            x_vals.extend(range(x_start + 20, x_start + 30))
            for i in range(10):
                y_vals.append(15 - i * 0.8)
        elif scenario == 'Gradual Rise':
            # Price continues gradually
            x_vals.extend(range(x_start + 20, x_start + 40))
            for i in range(20):
                y_vals.append(15 + i * 0.3)
        else:  # Sharp Rally
            # Price rallies strongly
            x_vals.extend(range(x_start + 20, x_start + 30))
            for i in range(10):
                y_vals.append(15 + i * 2)
        
        ax2.plot(x_vals, y_vals, color=color, linewidth=2, label=scenario)
    
    ax2.axhline(y=15, color='orange', linestyle=':', label='TSL Activation')
    ax2.axhline(y=5, color='purple', linestyle=':', label='Min Profit')
    ax2.set_title('Different Price Scenarios After TSL Activation', fontsize=14)
    ax2.set_xlabel('Time (arbitrary units)')
    ax2.set_ylabel('Pips from Entry')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../charts/tsl_behavior_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nChart saved to: ../charts/tsl_behavior_comparison.png")
    
    # Create a summary diagram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Draw the concept
    ax.text(0.5, 0.9, 'TSL Initial Buffer Concept', ha='center', fontsize=16, weight='bold')
    
    ax.text(0.1, 0.7, 'Problem:', fontsize=12, weight='bold')
    ax.text(0.1, 0.65, '• TSL activates at 15 pips profit', fontsize=10)
    ax.text(0.1, 0.6, '• ATR-based stop placed too close (e.g., 12 pips away)', fontsize=10)
    ax.text(0.1, 0.55, '• Small pullback triggers stop immediately', fontsize=10)
    ax.text(0.1, 0.5, '• Result: Many trades exit at ~15 pips', fontsize=10, color='red')
    
    ax.text(0.1, 0.35, 'Solution:', fontsize=12, weight='bold')
    ax.text(0.1, 0.3, '• Add initial buffer when TSL first activates', fontsize=10)
    ax.text(0.1, 0.25, '• Use 2x ATR multiplier on first activation (24 pips away)', fontsize=10)
    ax.text(0.1, 0.2, '• Subsequent updates use normal 1.2x multiplier', fontsize=10)
    ax.text(0.1, 0.15, '• Result: Trades have room to breathe after activation', fontsize=10, color='green')
    
    ax.text(0.6, 0.5, 'Formula:', fontsize=12, weight='bold')
    ax.text(0.6, 0.45, 'First activation:', fontsize=10)
    ax.text(0.6, 0.4, 'TSL = Price - (ATR × 1.2 × 2.0)', fontsize=10, family='monospace', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.6, 0.3, 'Subsequent updates:', fontsize=10)
    ax.text(0.6, 0.25, 'TSL = Price - (ATR × 1.2)', fontsize=10, family='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.savefig('../charts/tsl_buffer_concept.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Concept diagram saved to: ../charts/tsl_buffer_concept.png")
    print("\nKey insights:")
    print("1. The initial buffer prevents the TSL from being placed too close")
    print("2. This gives trades room to handle normal market volatility")
    print("3. After the initial activation, TSL tightens normally")
    print("4. Minimum profit of 5 pips is always guaranteed")

if __name__ == "__main__":
    visualize_tsl_behavior()