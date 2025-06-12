"""
Advanced Performance Analysis for Strategy Optimization
Goal: Identify patterns that drive Sharpe ratios > 2.0 consistently
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_results():
    """Load all Monte Carlo results and perform deep analysis"""
    
    # Load both strategy results
    config1_df = pd.read_csv('results/AUDUSD_config_1_ultra-tight_risk_management_monte_carlo.csv')
    config2_df = pd.read_csv('results/AUDUSD_config_2_scalping_strategy_monte_carlo.csv')
    
    # Add strategy identifier
    config1_df['strategy'] = 'config_1_ultra_tight'
    config2_df['strategy'] = 'config_2_scalping'
    
    # Combine datasets
    combined_df = pd.concat([config1_df, config2_df], ignore_index=True)
    
    return combined_df, config1_df, config2_df

def analyze_sharpe_patterns(df):
    """Analyze what drives high Sharpe ratios"""
    
    print("=== SHARPE RATIO ANALYSIS ===")
    print(f"Overall Sharpe Statistics:")
    print(f"Mean: {df['sharpe_ratio'].mean():.3f}")
    print(f"Median: {df['sharpe_ratio'].median():.3f}")
    print(f"Std: {df['sharpe_ratio'].std():.3f}")
    print(f"Max: {df['sharpe_ratio'].max():.3f}")
    print(f"Min: {df['sharpe_ratio'].min():.3f}")
    
    # High performance periods (Sharpe > 2.0)
    high_sharpe = df[df['sharpe_ratio'] > 2.0]
    print(f"\nHigh Sharpe (>2.0) Periods: {len(high_sharpe)} out of {len(df)} ({len(high_sharpe)/len(df)*100:.1f}%)")
    
    if len(high_sharpe) > 0:
        print(f"High Sharpe Mean: {high_sharpe['sharpe_ratio'].mean():.3f}")
        print(f"High Sharpe Win Rate: {high_sharpe['win_rate'].mean():.1f}%")
        print(f"High Sharpe Profit Factor: {high_sharpe['profit_factor'].mean():.3f}")
        print(f"High Sharpe Max Drawdown: {high_sharpe['max_drawdown'].mean():.2f}%")
        print(f"High Sharpe Total Trades: {high_sharpe['total_trades'].mean():.0f}")
        
        # Years when high Sharpe occurred
        high_sharpe_years = high_sharpe['primary_year'].value_counts()
        print(f"\nHigh Sharpe Years Distribution:")
        for year, count in high_sharpe_years.head(10).items():
            print(f"  {year}: {count} periods")
    
    return high_sharpe

def analyze_market_conditions(df):
    """Analyze market conditions that favor high performance"""
    
    print("\n=== MARKET CONDITIONS ANALYSIS ===")
    
    # Year-based analysis
    yearly_stats = df.groupby('primary_year').agg({
        'sharpe_ratio': ['mean', 'std', 'max'],
        'win_rate': 'mean',
        'total_trades': 'mean',
        'max_drawdown': 'mean'
    }).round(3)
    
    print("Yearly Performance Summary:")
    print(yearly_stats)
    
    # Best performing years
    best_years = df.groupby('primary_year')['sharpe_ratio'].mean().sort_values(ascending=False)
    print(f"\nTop 5 Years by Average Sharpe:")
    for year, sharpe in best_years.head(5).items():
        year_data = df[df['primary_year'] == year]
        print(f"  {year}: {sharpe:.3f} (n={len(year_data)})")
    
    return yearly_stats

def analyze_strategy_comparison(config1_df, config2_df):
    """Compare the two strategy configurations"""
    
    print("\n=== STRATEGY COMPARISON ===")
    
    strategies = {
        'Config 1 (Ultra-tight)': config1_df,
        'Config 2 (Scalping)': config2_df
    }
    
    for name, df in strategies.items():
        print(f"\n{name}:")
        print(f"  Mean Sharpe: {df['sharpe_ratio'].mean():.3f}")
        print(f"  Sharpe > 2.0: {len(df[df['sharpe_ratio'] > 2.0])}/{len(df)} ({len(df[df['sharpe_ratio'] > 2.0])/len(df)*100:.1f}%)")
        print(f"  Mean Win Rate: {df['win_rate'].mean():.1f}%")
        print(f"  Mean Profit Factor: {df['profit_factor'].mean():.3f}")
        print(f"  Mean Max Drawdown: {df['max_drawdown'].mean():.2f}%")
        print(f"  Mean Total Trades: {df['total_trades'].mean():.0f}")

def identify_optimization_opportunities(df):
    """Identify specific areas for optimization"""
    
    print("\n=== OPTIMIZATION OPPORTUNITIES ===")
    
    # Correlation analysis
    numeric_cols = ['sharpe_ratio', 'win_rate', 'total_trades', 'max_drawdown', 
                   'profit_factor', 'avg_win', 'avg_loss', 'max_consec_wins', 'max_consec_losses']
    
    corr_with_sharpe = df[numeric_cols].corr()['sharpe_ratio'].sort_values(ascending=False)
    
    print("Correlation with Sharpe Ratio:")
    for metric, corr in corr_with_sharpe.items():
        if metric != 'sharpe_ratio':
            print(f"  {metric}: {corr:.3f}")
    
    # Trade frequency analysis
    print(f"\nTrade Frequency Analysis:")
    high_sharpe = df[df['sharpe_ratio'] > 2.0]
    low_sharpe = df[df['sharpe_ratio'] < 1.0]
    
    if len(high_sharpe) > 0 and len(low_sharpe) > 0:
        print(f"High Sharpe avg trades: {high_sharpe['total_trades'].mean():.0f}")
        print(f"Low Sharpe avg trades: {low_sharpe['total_trades'].mean():.0f}")
        
        print(f"High Sharpe avg win: ${high_sharpe['avg_win'].mean():.0f}")
        print(f"Low Sharpe avg win: ${low_sharpe['avg_win'].mean():.0f}")
        
        print(f"High Sharpe avg loss: ${high_sharpe['avg_loss'].mean():.0f}")
        print(f"Low Sharpe avg loss: ${low_sharpe['avg_loss'].mean():.0f}")

def generate_improvement_recommendations(df):
    """Generate specific recommendations for strategy improvement"""
    
    print("\n=== IMPROVEMENT RECOMMENDATIONS ===")
    
    high_sharpe = df[df['sharpe_ratio'] > 2.0]
    
    if len(high_sharpe) > 0:
        print("Based on high-performing periods, focus on:")
        
        # Win rate patterns
        target_win_rate = high_sharpe['win_rate'].mean()
        print(f"1. Target win rate: {target_win_rate:.1f}% (current avg: {df['win_rate'].mean():.1f}%)")
        
        # Risk-reward patterns
        avg_rr_high = abs(high_sharpe['avg_win'].mean() / high_sharpe['avg_loss'].mean())
        avg_rr_all = abs(df['avg_win'].mean() / df['avg_loss'].mean())
        print(f"2. Target risk-reward ratio: {avg_rr_high:.2f} (current avg: {avg_rr_all:.2f})")
        
        # Drawdown control
        target_dd = high_sharpe['max_drawdown'].mean()
        print(f"3. Target max drawdown: {target_dd:.2f}% (current avg: {df['max_drawdown'].mean():.2f}%)")
        
        # Trade frequency
        target_trades = high_sharpe['total_trades'].mean()
        print(f"4. Target trade frequency: {target_trades:.0f} trades per period")
        
        # Consecutive wins/losses
        target_consec_wins = high_sharpe['max_consec_wins'].mean()
        target_consec_losses = high_sharpe['max_consec_losses'].mean()
        print(f"5. Target consecutive patterns: {target_consec_wins:.0f} wins, {target_consec_losses:.0f} losses max")
    
    print("\nKey Areas for Enhancement:")
    print("1. Multi-timeframe confluence - reduce false signals")
    print("2. Dynamic position sizing based on volatility")
    print("3. Market regime detection (trending vs ranging)")
    print("4. Advanced entry/exit timing with momentum confirmation")
    print("5. Adaptive stop-loss and take-profit levels")

def main():
    """Run comprehensive performance analysis"""
    
    print("Starting Deep Performance Analysis...")
    print("="*60)
    
    # Load data
    combined_df, config1_df, config2_df = load_and_analyze_results()
    
    # Run analyses
    high_sharpe_periods = analyze_sharpe_patterns(combined_df)
    yearly_stats = analyze_market_conditions(combined_df)
    analyze_strategy_comparison(config1_df, config2_df)
    identify_optimization_opportunities(combined_df)
    generate_improvement_recommendations(combined_df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    
    return combined_df, high_sharpe_periods

if __name__ == "__main__":
    df, high_sharpe = main()