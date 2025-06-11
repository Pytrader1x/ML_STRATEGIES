"""
Generate charts for the validation report
Creates visual representations of performance metrics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_sharpe_comparison_chart():
    """Create horizontal bar chart comparing Sharpe ratios with slippage"""
    
    # Data from validation results
    data = {
        'GBPUSD Config 1': 1.537,
        'GBPUSD Config 2': 1.503,
        'USDCAD Config 2': 1.483,
        'USDCAD Config 1': 1.360,
        'EURUSD Config 1': 1.340,
        'EURUSD Config 2': 1.303,
        'NZDUSD Config 1': 1.109,
        'NZDUSD Config 2': 1.087
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    strategies = list(data.keys())
    sharpes = list(data.values())
    colors = ['#2ecc71' if s > 1.4 else '#3498db' if s > 1.2 else '#f39c12' for s in sharpes]
    
    bars = ax.barh(strategies, sharpes, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, sharpe in zip(bars, sharpes):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{sharpe:.3f}', va='center', fontweight='bold')
    
    # Add reference line
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
    ax.axvline(x=1.5, color='green', linestyle='--', alpha=0.5, label='Target Sharpe = 1.5')
    
    # Formatting
    ax.set_xlabel('Sharpe Ratio (with 2-pip slippage)', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Performance Comparison - Realistic Execution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.8)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('validation_charts/sharpe_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_robustness_heatmap():
    """Create heatmap showing robustness across currencies and configs"""
    
    # Robustness data (% maintaining Sharpe > 1.0)
    data = {
        'Config 1': [95, 95, 85, 65],  # GBPUSD, EURUSD, USDCAD, NZDUSD
        'Config 2': [90, 100, 100, 55]
    }
    currencies = ['GBPUSD', 'EURUSD', 'USDCAD', 'NZDUSD']
    
    df = pd.DataFrame(data, index=currencies)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='d', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': '% Tests with Sharpe > 1.0'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    # Formatting
    ax.set_title('Strategy Robustness Across Currency Pairs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Currency Pair', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('validation_charts/robustness_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_slippage_impact_chart():
    """Create chart showing slippage impact on performance"""
    
    # Data structure: [No Slippage Sharpe, With Slippage Sharpe]
    config1_data = {
        'GBPUSD': [1.659, 1.537],
        'EURUSD': [1.450, 1.340],
        'USDCAD': [1.426, 1.360],
        'NZDUSD': [1.192, 1.109]
    }
    
    config2_data = {
        'GBPUSD': [1.704, 1.503],
        'EURUSD': [1.496, 1.303],
        'USDCAD': [1.722, 1.483],
        'NZDUSD': [1.377, 1.087]
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Config 1 plot
    currencies = list(config1_data.keys())
    x = np.arange(len(currencies))
    width = 0.35
    
    no_slip_1 = [config1_data[c][0] for c in currencies]
    with_slip_1 = [config1_data[c][1] for c in currencies]
    
    bars1 = ax1.bar(x - width/2, no_slip_1, width, label='No Slippage', alpha=0.8)
    bars2 = ax1.bar(x + width/2, with_slip_1, width, label='With 2-pip Slippage', alpha=0.8)
    
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title('Config 1: Ultra-Tight Risk Management', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(currencies)
    ax1.legend()
    ax1.set_ylim(0, 2.0)
    
    # Add degradation percentages
    for i, (ns, ws) in enumerate(zip(no_slip_1, with_slip_1)):
        degradation = (ws - ns) / ns * 100
        ax1.text(i, ws + 0.05, f'{degradation:.1f}%', ha='center', fontsize=10, color='red')
    
    # Config 2 plot
    no_slip_2 = [config2_data[c][0] for c in currencies]
    with_slip_2 = [config2_data[c][1] for c in currencies]
    
    bars3 = ax2.bar(x - width/2, no_slip_2, width, label='No Slippage', alpha=0.8)
    bars4 = ax2.bar(x + width/2, with_slip_2, width, label='With 2-pip Slippage', alpha=0.8)
    
    ax2.set_ylabel('Sharpe Ratio', fontsize=12)
    ax2.set_title('Config 2: Scalping Strategy', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(currencies)
    ax2.legend()
    ax2.set_ylim(0, 2.0)
    
    # Add degradation percentages
    for i, (ns, ws) in enumerate(zip(no_slip_2, with_slip_2)):
        degradation = (ws - ns) / ns * 100
        ax2.text(i, ws + 0.05, f'{degradation:.1f}%', ha='center', fontsize=10, color='red')
    
    plt.suptitle('Slippage Impact Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('validation_charts/slippage_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_radar_chart():
    """Create radar chart comparing key metrics"""
    
    # Metrics for best performers
    categories = ['Sharpe Ratio', 'Win Rate', 'Robustness', 'Low Drawdown', 'P&L Stability']
    
    # Normalize metrics to 0-100 scale
    config1_gbpusd = [
        1.537 / 2.0 * 100,  # Sharpe (normalized to max 2.0)
        70.4,                # Win Rate
        95,                  # Robustness
        100 - 4.5,          # Low DD (100 - actual DD)
        100 - 7.4           # P&L Stability (100 - degradation%)
    ]
    
    config2_gbpusd = [
        1.503 / 2.0 * 100,
        63.3,
        90,
        100 - 2.6,
        100 - 11.8
    ]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Number of variables
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    config1_gbpusd += config1_gbpusd[:1]
    config2_gbpusd += config2_gbpusd[:1]
    angles += angles[:1]
    
    # Plot
    ax.plot(angles, config1_gbpusd, 'o-', linewidth=2, label='Config 1: Ultra-Tight Risk', color='#3498db')
    ax.fill(angles, config1_gbpusd, alpha=0.25, color='#3498db')
    
    ax.plot(angles, config2_gbpusd, 'o-', linewidth=2, label='Config 2: Scalping', color='#e74c3c')
    ax.fill(angles, config2_gbpusd, alpha=0.25, color='#e74c3c')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=10)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('GBPUSD Strategy Comparison (Best Performers)', size=14, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    plt.savefig('validation_charts/performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_anti_cheating_results():
    """Create visual summary of anti-cheating validation"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'Anti-Cheating Validation Results', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Checks
    checks = [
        ('Look-Ahead Bias Check', '✓ PASS', '#2ecc71'),
        ('Trade Validity Check', '✓ PASS', '#2ecc71'),
        ('Fill Realism Check', '✓ PASS', '#2ecc71'),
        ('Data Snooping Check', '✓ PASS', '#2ecc71'),
        ('Indicator Integrity Check', '✓ PASS', '#2ecc71')
    ]
    
    y_pos = 4.5
    for check, result, color in checks:
        # Create box
        rect = Rectangle((1, y_pos - 0.35), 8, 0.7, 
                        facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(2, y_pos, check, fontsize=12, va='center')
        ax.text(8, y_pos, result, fontsize=12, fontweight='bold', 
                ha='right', va='center', color=color)
        
        y_pos -= 0.9
    
    # Summary
    ax.text(5, 0.3, '✅ ALL CHECKS PASSED - NO CHEATING DETECTED', 
            fontsize=14, fontweight='bold', ha='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('validation_charts/anti_cheating_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all validation charts"""
    
    # Create output directory
    import os
    os.makedirs('validation_charts', exist_ok=True)
    
    print("Generating validation charts...")
    
    # Generate charts
    create_sharpe_comparison_chart()
    print("✓ Sharpe comparison chart created")
    
    create_robustness_heatmap()
    print("✓ Robustness heatmap created")
    
    create_slippage_impact_chart()
    print("✓ Slippage impact chart created")
    
    create_performance_radar_chart()
    print("✓ Performance radar chart created")
    
    create_anti_cheating_results()
    print("✓ Anti-cheating summary created")
    
    print("\nAll charts saved to validation_charts/ directory")

if __name__ == "__main__":
    main()