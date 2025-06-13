"""
Create visual diagrams for the strategy report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as patches

# Set up the style
plt.style.use('dark_background')

# Create figure with subplots for different diagrams
fig = plt.figure(figsize=(20, 28))

# 1. Strategy Overview Flowchart
ax1 = plt.subplot(4, 2, (1, 2))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.text(5, 9.5, 'STRATEGY LOGIC FLOWCHART', fontsize=18, weight='bold', ha='center')

# Define boxes with positions and text
boxes = [
    (1, 8, 'Price Data', 'lightblue'),
    (3, 8, 'Calculate\nNTI', 'lightgreen'),
    (5, 8, 'Calculate\nMB', 'lightgreen'),
    (7, 8, 'Calculate\nIC', 'lightgreen'),
    (9, 8, 'Check\nAlignment', 'yellow'),
    (5, 6, 'All Agree?', 'orange'),
    (2, 4, 'WAIT', 'gray'),
    (8, 4, 'ENTER\nTRADE', 'lime'),
    (8, 2, 'Manage\nPosition', 'cyan'),
]

for x, y, text, color in boxes:
    box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='white',
                         alpha=0.7, linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, y, text, ha='center', va='center', fontsize=11, weight='bold')

# Add arrows
arrows = [
    (1.7, 8, 0.6, 0),
    (3.7, 8, 0.6, 0),
    (5.7, 8, 0.6, 0),
    (7.7, 8, 0.6, 0),
    (9, 7.6, -4, -1.2),
    (4.3, 6, -1.8, -1.6),
    (5.7, 6, 1.8, -1.6),
    (8, 3.6, 0, -1.2),
]

for x, y, dx, dy in arrows:
    ax1.arrow(x, y, dx, dy, head_width=0.2, head_length=0.15, 
              fc='white', ec='white', alpha=0.8)

# Add labels
ax1.text(3.5, 5, 'NO', fontsize=10, ha='center', color='red')
ax1.text(6.5, 5, 'YES', fontsize=10, ha='center', color='green')

# 2. Entry Conditions
ax2 = plt.subplot(4, 2, 3)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.text(5, 9.5, 'LONG ENTRY CONDITIONS', fontsize=14, weight='bold', ha='center', color='green')

conditions = [
    '✓ NTI = +1 (Bullish)',
    '✓ MB = +1 (Bullish)', 
    '✓ IC ≠ Choppy',
    '✓ No Open Position',
    '',
    'ALL must be TRUE'
]

for i, condition in enumerate(conditions):
    color = 'lime' if '✓' in condition else 'yellow'
    weight = 'bold' if i == len(conditions)-1 else 'normal'
    ax2.text(5, 8 - i*1.2, condition, ha='center', va='center', 
             fontsize=12, color=color, weight=weight)

# 3. Short Entry Conditions
ax3 = plt.subplot(4, 2, 4)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.text(5, 9.5, 'SHORT ENTRY CONDITIONS', fontsize=14, weight='bold', ha='center', color='red')

conditions = [
    '✓ NTI = -1 (Bearish)',
    '✓ MB = -1 (Bearish)', 
    '✓ IC ≠ Choppy',
    '✓ No Open Position',
    '',
    'ALL must be TRUE'
]

for i, condition in enumerate(conditions):
    color = 'salmon' if '✓' in condition else 'yellow'
    weight = 'bold' if i == len(conditions)-1 else 'normal'
    ax3.text(5, 8 - i*1.2, condition, ha='center', va='center', 
             fontsize=12, color=color, weight=weight)

# 4. Risk Management
ax4 = plt.subplot(4, 2, 5)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.text(5, 9.5, 'RISK MANAGEMENT', fontsize=14, weight='bold', ha='center')

# Position sizing box
pos_box = FancyBboxPatch((0.5, 6.5), 9, 2,
                         boxstyle="round,pad=0.1",
                         facecolor='darkblue', edgecolor='cyan',
                         alpha=0.7, linewidth=2)
ax4.add_patch(pos_box)
ax4.text(5, 7.5, 'Position Sizing: 0.1% Risk Per Trade', 
         ha='center', fontsize=12, weight='bold', color='white')
ax4.text(5, 6.8, 'If SL = 5 pips → Smaller position\nIf SL = 2 pips → Larger position', 
         ha='center', fontsize=10, color='lightblue')

# Stop loss box
sl_box = FancyBboxPatch((0.5, 3.5), 4, 2,
                        boxstyle="round,pad=0.1",
                        facecolor='darkred', edgecolor='red',
                        alpha=0.7, linewidth=2)
ax4.add_patch(sl_box)
ax4.text(2.5, 4.5, 'Stop Loss', ha='center', fontsize=12, weight='bold', color='white')
ax4.text(2.5, 3.8, '• Max: 5 pips\n• 0.5 × ATR\n• Slippage: 0-2 pips', 
         ha='center', fontsize=9, color='salmon')

# Take profit box
tp_box = FancyBboxPatch((5.5, 3.5), 4, 2,
                        boxstyle="round,pad=0.1",
                        facecolor='darkgreen', edgecolor='lime',
                        alpha=0.7, linewidth=2)
ax4.add_patch(tp_box)
ax4.text(7.5, 4.5, 'Take Profit', ha='center', fontsize=12, weight='bold', color='white')
ax4.text(7.5, 3.8, '• TP1: 1 pip\n• TP2: 2 pips\n• TP3: 3 pips', 
         ha='center', fontsize=9, color='lightgreen')

# 5. Sample Trade Visualization
ax5 = plt.subplot(4, 1, 3)
ax5.set_xlim(-5, 55)
ax5.set_ylim(1.0845, 1.0865)

# Generate sample price data
np.random.seed(42)
x = np.arange(50)
base_price = 1.0850
trend = np.linspace(0, 0.001, 50)
noise = np.random.normal(0, 0.00005, 50)
prices = base_price + trend + noise

# Plot candlesticks
for i in range(len(prices)):
    if i == 0:
        color = 'green'
    else:
        color = 'green' if prices[i] > prices[i-1] else 'red'
    
    # Candle body
    height = abs(prices[i] - prices[i-1]) if i > 0 else 0.00005
    bottom = min(prices[i], prices[i-1]) if i > 0 else prices[i]
    rect = Rectangle((i-0.3, bottom), 0.6, height, 
                     facecolor=color, alpha=0.8)
    ax5.add_patch(rect)
    
    # Wicks
    high = prices[i] + np.random.uniform(0, 0.00005)
    low = prices[i] - np.random.uniform(0, 0.00005)
    ax5.plot([i, i], [low, high], color=color, alpha=0.6, linewidth=1)

# Entry point
entry_bar = 10
entry_price = prices[entry_bar]
ax5.scatter(entry_bar, entry_price, color='cyan', s=300, marker='^', 
            zorder=5, edgecolor='white', linewidth=2)
ax5.annotate('ENTRY\nNTI=+1\nMB=+1', xy=(entry_bar, entry_price), 
             xytext=(entry_bar-5, entry_price+0.0003),
             arrowprops=dict(arrowstyle='->', color='cyan', linewidth=2),
             fontsize=10, color='cyan', weight='bold', ha='center')

# Stop loss and take profits
sl_price = entry_price - 0.0005
tp1_price = entry_price + 0.0001
tp2_price = entry_price + 0.0002
tp3_price = entry_price + 0.0003

ax5.axhline(sl_price, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Stop Loss (-5 pips)')
ax5.axhline(tp1_price, color='lime', linestyle='--', alpha=0.7, linewidth=2, label='TP1 (+1 pip)')
ax5.axhline(tp2_price, color='green', linestyle='--', alpha=0.6, linewidth=2, label='TP2 (+2 pips)')
ax5.axhline(tp3_price, color='darkgreen', linestyle='--', alpha=0.5, linewidth=2, label='TP3 (+3 pips)')

# Exit point
exit_bar = 25
ax5.scatter(exit_bar, tp1_price, color='yellow', s=300, marker='v', 
            zorder=5, edgecolor='white', linewidth=2)
ax5.annotate('EXIT\nTP1 Hit\n+1 pip', xy=(exit_bar, tp1_price), 
             xytext=(exit_bar+5, tp1_price+0.0003),
             arrowprops=dict(arrowstyle='->', color='yellow', linewidth=2),
             fontsize=10, color='yellow', weight='bold', ha='center')

# Shaded profit area
ax5.fill_between(range(entry_bar, exit_bar+1), entry_price, tp1_price, 
                 alpha=0.2, color='green')

ax5.set_title('SAMPLE WINNING TRADE - EURUSD', fontsize=14, weight='bold', pad=10)
ax5.set_xlabel('Time (15-minute bars)', fontsize=12)
ax5.set_ylabel('Price', fontsize=12)
ax5.legend(loc='upper left', fontsize=10)
ax5.grid(True, alpha=0.3)

# 6. Performance Summary
ax6 = plt.subplot(4, 1, 4)
ax6.set_xlim(-0.5, 4.5)
ax6.set_ylim(0, 10)

# Currency performance bars
currencies = ['AUDUSD', 'GBPUSD', 'EURUSD', 'USDCAD', 'NZDUSD']
sharpe_ratios = [7.72, 8.83, 8.52, 8.28, 6.49]
colors = ['gold', 'silver', 'lightblue', 'lightgreen', 'orange']

bars = ax6.bar(range(5), sharpe_ratios, color=colors, edgecolor='white', 
                linewidth=2, alpha=0.8)

# Add value labels
for i, (bar, sharpe) in enumerate(zip(bars, sharpe_ratios)):
    ax6.text(i, sharpe + 0.2, f'{sharpe:.2f}', ha='center', fontsize=12, 
             weight='bold', color='white')

# Reference lines
ax6.axhline(1, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax6.text(4.5, 1.1, 'Good', ha='right', fontsize=10, color='red')
ax6.axhline(2, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
ax6.text(4.5, 2.1, 'Very Good', ha='right', fontsize=10, color='yellow')
ax6.axhline(3, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax6.text(4.5, 3.1, 'Excellent', ha='right', fontsize=10, color='green')

ax6.set_xticks(range(5))
ax6.set_xticklabels(currencies)
ax6.set_ylabel('Sharpe Ratio', fontsize=12)
ax6.set_title('PERFORMANCE ACROSS CURRENCY PAIRS', fontsize=14, weight='bold', pad=10)
ax6.grid(True, axis='y', alpha=0.3)

# Add overall title
fig.suptitle('SCALPING STRATEGY VISUAL GUIDE', fontsize=24, weight='bold', y=0.98)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.96)
plt.savefig('strategy_visual_guide.png', dpi=150, bbox_inches='tight', 
            facecolor='#0a0a0a', edgecolor='none')
plt.close()

print("Visual guide created: strategy_visual_guide.png")