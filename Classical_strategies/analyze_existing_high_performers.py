"""
Analyze the existing strategy's high-performing periods (Sharpe > 2.0)
Focus on understanding what made those specific periods successful
"""

import pandas as pd
import numpy as np

def analyze_high_performers():
    """Analyze periods where existing strategy achieved Sharpe > 2.0"""
    
    print("ANALYZING EXISTING HIGH-PERFORMING PERIODS")
    print("="*60)
    
    # Load the scalping strategy results (had the highest Sharpes)
    df = pd.read_csv('results/AUDUSD_config_2_scalping_strategy_monte_carlo.csv')
    
    # Find Sharpe > 2.0 periods
    high_performers = df[df['sharpe_ratio'] > 2.0]
    
    print(f"Periods with Sharpe > 2.0: {len(high_performers)} out of {len(df)}")
    print(f"Success rate: {len(high_performers)/len(df)*100:.1f}%\\n")
    
    if len(high_performers) == 0:
        print("No periods with Sharpe > 2.0 found in recent run.")
        return
    
    for idx, row in high_performers.iterrows():
        print(f"PERIOD {row['iteration']}:")
        print(f"  Date Range: {row['start_date']} to {row['end_date']}")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
        print(f"  Total Return: {row['total_return']:.1f}%")
        print(f"  Win Rate: {row['win_rate']:.1f}%")
        print(f"  Total Trades: {row['total_trades']}")
        print(f"  Max Drawdown: {row['max_drawdown']:.2f}%")
        print(f"  Profit Factor: {row['profit_factor']:.3f}")
        print(f"  Avg Win: ${row['avg_win']:.0f}")
        print(f"  Avg Loss: ${row['avg_loss']:.0f}")
        print(f"  Risk-Reward: {abs(row['avg_win']/row['avg_loss']):.2f}")
        print(f"  Max Consec Wins: {row['max_consec_wins']}")
        print(f"  Max Consec Losses: {row['max_consec_losses']}")
        print()
    
    # Analyze common characteristics
    print("COMMON CHARACTERISTICS OF HIGH PERFORMERS:")
    print("-" * 40)
    
    avg_metrics = {
        'sharpe_ratio': high_performers['sharpe_ratio'].mean(),
        'win_rate': high_performers['win_rate'].mean(),
        'total_trades': high_performers['total_trades'].mean(),
        'max_drawdown': high_performers['max_drawdown'].mean(),
        'profit_factor': high_performers['profit_factor'].mean(),
        'avg_win': high_performers['avg_win'].mean(),
        'avg_loss': high_performers['avg_loss'].mean(),
        'max_consec_wins': high_performers['max_consec_wins'].mean(),
        'max_consec_losses': high_performers['max_consec_losses'].mean()
    }
    
    print(f"Average Sharpe: {avg_metrics['sharpe_ratio']:.3f}")
    print(f"Average Win Rate: {avg_metrics['win_rate']:.1f}%")
    print(f"Average Trades: {avg_metrics['total_trades']:.0f}")
    print(f"Average Max DD: {avg_metrics['max_drawdown']:.2f}%")
    print(f"Average Profit Factor: {avg_metrics['profit_factor']:.3f}")
    print(f"Average Risk-Reward: {abs(avg_metrics['avg_win']/avg_metrics['avg_loss']):.2f}")
    print(f"Average Max Consec Wins: {avg_metrics['max_consec_wins']:.0f}")
    print(f"Average Max Consec Losses: {avg_metrics['max_consec_losses']:.0f}")
    
    # Compare to all periods
    print("\\nCOMPARISON TO ALL PERIODS:")
    print("-" * 40)
    
    all_avg = {
        'sharpe_ratio': df['sharpe_ratio'].mean(),
        'win_rate': df['win_rate'].mean(),
        'total_trades': df['total_trades'].mean(),
        'max_drawdown': df['max_drawdown'].mean(),
        'profit_factor': df['profit_factor'].mean(),
        'avg_win': df['avg_win'].mean(),
        'avg_loss': df['avg_loss'].mean(),
        'max_consec_wins': df['max_consec_wins'].mean(),
        'max_consec_losses': df['max_consec_losses'].mean()
    }
    
    print("Metric                  | High Performers | All Periods | Difference")
    print("-" * 65)
    print(f"Sharpe Ratio           | {avg_metrics['sharpe_ratio']:11.3f} | {all_avg['sharpe_ratio']:9.3f} | +{avg_metrics['sharpe_ratio']-all_avg['sharpe_ratio']:7.3f}")
    print(f"Win Rate %             | {avg_metrics['win_rate']:11.1f} | {all_avg['win_rate']:9.1f} | {avg_metrics['win_rate']-all_avg['win_rate']:+8.1f}")
    print(f"Total Trades           | {avg_metrics['total_trades']:11.0f} | {all_avg['total_trades']:9.0f} | {avg_metrics['total_trades']-all_avg['total_trades']:+8.0f}")
    print(f"Max Drawdown %         | {avg_metrics['max_drawdown']:11.2f} | {all_avg['max_drawdown']:9.2f} | {avg_metrics['max_drawdown']-all_avg['max_drawdown']:+8.2f}")
    print(f"Profit Factor          | {avg_metrics['profit_factor']:11.3f} | {all_avg['profit_factor']:9.3f} | +{avg_metrics['profit_factor']-all_avg['profit_factor']:7.3f}")
    
    risk_reward_high = abs(avg_metrics['avg_win']/avg_metrics['avg_loss'])
    risk_reward_all = abs(all_avg['avg_win']/all_avg['avg_loss'])
    print(f"Risk-Reward Ratio      | {risk_reward_high:11.2f} | {risk_reward_all:9.2f} | +{risk_reward_high-risk_reward_all:7.2f}")
    
    print(f"Max Consec Wins        | {avg_metrics['max_consec_wins']:11.0f} | {all_avg['max_consec_wins']:9.0f} | {avg_metrics['max_consec_wins']-all_avg['max_consec_wins']:+8.0f}")
    print(f"Max Consec Losses      | {avg_metrics['max_consec_losses']:11.0f} | {all_avg['max_consec_losses']:9.0f} | {avg_metrics['max_consec_losses']-all_avg['max_consec_losses']:+8.0f}")
    
    # Key insights
    print("\\nKEY INSIGHTS:")
    print("-" * 40)
    
    if avg_metrics['total_trades'] > all_avg['total_trades']:
        print(f"✅ Higher trade frequency helps (+{avg_metrics['total_trades']-all_avg['total_trades']:.0f} trades)")
    
    if avg_metrics['profit_factor'] > all_avg['profit_factor']:
        print(f"✅ Better profit factor (+{avg_metrics['profit_factor']-all_avg['profit_factor']:.3f})")
    
    if risk_reward_high > risk_reward_all:
        print(f"✅ Better risk-reward ratio (+{risk_reward_high-risk_reward_all:.2f})")
    
    if abs(avg_metrics['max_drawdown']) < abs(all_avg['max_drawdown']):
        print(f"✅ Lower drawdowns ({avg_metrics['max_drawdown']:.2f}% vs {all_avg['max_drawdown']:.2f}%)")
    
    # Year analysis
    print("\\nTIME PERIOD ANALYSIS:")
    print("-" * 40)
    
    years = []
    for _, row in high_performers.iterrows():
        start_year = pd.to_datetime(row['start_date']).year
        years.append(start_year)
    
    year_counts = pd.Series(years).value_counts()
    print("Years with high performance:")
    for year, count in year_counts.items():
        print(f"  {year}: {count} period(s)")
    
    return high_performers, avg_metrics

def suggest_optimizations(high_performers, avg_metrics):
    """Suggest specific optimizations based on analysis"""
    
    print("\\n" + "="*60)
    print("OPTIMIZATION SUGGESTIONS")
    print("="*60)
    
    print("Based on analysis of existing high-performing periods:")
    print()
    
    print("1. 🎯 TRADE FREQUENCY OPTIMIZATION")
    print(f"   - Target: {avg_metrics['total_trades']:.0f}+ trades per period")
    print("   - Current high performers average more trades")
    print("   - Consider slightly relaxing entry conditions")
    print()
    
    print("2. 📈 PROFIT FACTOR ENHANCEMENT") 
    print(f"   - Target: {avg_metrics['profit_factor']:.2f}+ profit factor")
    print("   - Focus on improving win size vs loss size")
    print("   - Better exit timing may be key")
    print()
    
    risk_reward = abs(avg_metrics['avg_win']/avg_metrics['avg_loss'])
    print("3. ⚖️ RISK-REWARD OPTIMIZATION")
    print(f"   - Target: {risk_reward:.2f}+ risk-reward ratio")
    print("   - High performers have better RR ratios")
    print("   - Optimize take profit and stop loss levels")
    print()
    
    print("4. 🛡️ DRAWDOWN CONTROL")
    print(f"   - Target: Keep max drawdown around {avg_metrics['max_drawdown']:.1f}%")
    print("   - High performers have controlled drawdowns")
    print("   - Maintain existing risk management")
    print()
    
    print("5. 📊 INCREMENTAL IMPROVEMENTS")
    print("   - The existing strategy is already strong!")
    print("   - Focus on small, targeted improvements")
    print("   - Don't over-engineer - preserve what works")
    print()
    
    print("RECOMMENDED APPROACH:")
    print("-" * 20)
    print("❌ DON'T: Build completely new complex systems")
    print("✅ DO: Make targeted adjustments to existing logic")
    print("✅ DO: Increase trade frequency moderately") 
    print("✅ DO: Fine-tune exit conditions")
    print("✅ DO: Test specific parameter adjustments")

if __name__ == "__main__":
    high_performers, avg_metrics = analyze_high_performers()
    suggest_optimizations(high_performers, avg_metrics)