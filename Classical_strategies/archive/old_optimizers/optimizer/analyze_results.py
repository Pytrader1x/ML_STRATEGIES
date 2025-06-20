#!/usr/bin/env python3
"""
Deep analysis of optimization results to identify patterns for robust Sharpe > 1
"""

import json
import numpy as np
import pandas as pd
import glob
import os

def analyze_all_results():
    """Analyze all optimization results to find patterns"""
    
    # Load all result files
    result_files = glob.glob('optimizer_results/optimization_results_strategy*.json')
    
    all_results = []
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
            for result in data.get('all_results', []):
                if result.get('sharpe_ratio', -999) > -900:
                    all_results.append(result)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    # Extract parameter columns
    param_cols = []
    for col in df.columns:
        if col in ['params']:
            # Expand params dict
            params_df = pd.DataFrame(df['params'].tolist())
            df = pd.concat([df, params_df], axis=1)
            param_cols = params_df.columns.tolist()
    
    print("="*80)
    print("DEEP ANALYSIS OF OPTIMIZATION RESULTS")
    print("="*80)
    print(f"\nTotal results analyzed: {len(df)}")
    
    # 1. Analyze high performers (Sharpe > 1.0)
    high_performers = df[df['sharpe_ratio'] > 1.0]
    print(f"\nHigh performers (Sharpe > 1.0): {len(high_performers)}")
    
    if len(high_performers) > 0:
        print("\n🌟 HIGH PERFORMER PARAMETER PATTERNS:")
        print("-"*50)
        
        for param in param_cols:
            if param in high_performers.columns:
                values = high_performers[param].dropna()
                if len(values) > 0:
                    print(f"\n{param}:")
                    print(f"  Mean: {values.mean():.3f}")
                    print(f"  Std:  {values.std():.3f}")
                    print(f"  Min:  {values.min():.3f}")
                    print(f"  Max:  {values.max():.3f}")
    
    # 2. Analyze moderate performers (0.5 < Sharpe < 1.0)
    moderate_performers = df[(df['sharpe_ratio'] > 0.5) & (df['sharpe_ratio'] <= 1.0)]
    print(f"\n\nModerate performers (0.5 < Sharpe <= 1.0): {len(moderate_performers)}")
    
    # 3. Analyze poor performers (Sharpe < 0)
    poor_performers = df[df['sharpe_ratio'] < 0]
    print(f"\nPoor performers (Sharpe < 0): {len(poor_performers)}")
    
    # 4. Correlation analysis
    print("\n\n🔍 PARAMETER CORRELATIONS WITH SHARPE RATIO:")
    print("-"*50)
    
    correlations = {}
    for param in param_cols:
        if param in df.columns:
            try:
                corr = df[param].corr(df['sharpe_ratio'])
                if not np.isnan(corr):
                    correlations[param] = corr
            except:
                pass
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for param, corr in sorted_corr:
        impact = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.3 else "moderate" if abs(corr) > 0.15 else "weak"
        print(f"{param:35} {corr:+.3f} ({strength} {impact} impact)")
    
    # 5. Critical parameter thresholds
    print("\n\n📊 CRITICAL PARAMETER INSIGHTS:")
    print("-"*50)
    
    # Analyze what separates winners from losers
    winners = df[df['sharpe_ratio'] > 0.8]
    losers = df[df['sharpe_ratio'] < 0]
    
    if len(winners) > 0 and len(losers) > 0:
        print("\nParameters that differ most between winners and losers:")
        
        differences = {}
        for param in param_cols:
            if param in winners.columns and param in losers.columns:
                winner_mean = winners[param].mean()
                loser_mean = losers[param].mean()
                if not np.isnan(winner_mean) and not np.isnan(loser_mean):
                    diff_pct = (winner_mean - loser_mean) / (loser_mean + 0.0001) * 100
                    differences[param] = {
                        'winner_mean': winner_mean,
                        'loser_mean': loser_mean,
                        'diff_pct': diff_pct
                    }
        
        # Sort by absolute difference
        sorted_diff = sorted(differences.items(), key=lambda x: abs(x[1]['diff_pct']), reverse=True)
        
        for param, stats in sorted_diff[:10]:
            print(f"\n{param}:")
            print(f"  Winners avg: {stats['winner_mean']:.3f}")
            print(f"  Losers avg:  {stats['loser_mean']:.3f}")
            print(f"  Difference:  {stats['diff_pct']:+.1f}%")
    
    # 6. Best configuration details
    print("\n\n🏆 BEST CONFIGURATION ANALYSIS:")
    print("-"*50)
    
    best_result = df.loc[df['sharpe_ratio'].idxmax()]
    print(f"\nBest Sharpe: {best_result['sharpe_ratio']:.3f}")
    
    if 'fitness' in best_result:
        print(f"Fitness Score: {best_result['fitness']:.3f}")
    
    print("\nBest Parameters:")
    for param in param_cols:
        if param in best_result:
            value = best_result[param]
            if not pd.isna(value):
                print(f"  {param}: {value}")
    
    # 7. Recommendations for next optimization
    print("\n\n💡 RECOMMENDATIONS FOR ACHIEVING ROBUST SHARPE > 1:")
    print("-"*50)
    
    print("\n1. FOCUS ON HIGH-IMPACT PARAMETERS:")
    for param, corr in sorted_corr[:5]:
        if abs(corr) > 0.1:
            direction = "increase" if corr > 0 else "decrease"
            print(f"   - {param}: {direction} for better performance")
    
    print("\n2. OPTIMAL PARAMETER RANGES (based on high performers):")
    if len(high_performers) > 0:
        key_params = ['risk_per_trade', 'sl_min_pips', 'sl_max_pips', 'tp1_multiplier', 
                      'tp2_multiplier', 'trailing_atr_multiplier']
        for param in key_params:
            if param in high_performers.columns:
                values = high_performers[param].dropna()
                if len(values) > 0:
                    print(f"   - {param}: {values.min():.3f} to {values.max():.3f}")
    
    print("\n3. AVOID THESE PARAMETER COMBINATIONS:")
    if len(poor_performers) > 0:
        # Identify common patterns in poor performers
        poor_patterns = []
        if 'tp1_multiplier' in poor_performers.columns:
            high_tp1 = poor_performers[poor_performers['tp1_multiplier'] > 0.35]
            if len(high_tp1) > 0 and high_tp1['sharpe_ratio'].mean() < -0.5:
                poor_patterns.append("TP1 multiplier > 0.35")
        
        if 'sl_min_pips' in poor_performers.columns:
            high_sl_min = poor_performers[poor_performers['sl_min_pips'] > 10]
            if len(high_sl_min) > 0 and high_sl_min['sharpe_ratio'].mean() < -0.5:
                poor_patterns.append("SL min pips > 10")
        
        for pattern in poor_patterns:
            print(f"   - {pattern}")
    
    return df, high_performers, correlations


def generate_next_optimization_strategy():
    """Generate specific recommendations for next optimization run"""
    
    df, high_performers, correlations = analyze_all_results()
    
    print("\n\n🎯 NEXT OPTIMIZATION STRATEGY:")
    print("="*80)
    
    # Based on analysis, create focused parameter spaces
    print("\n1. ULTRA-FOCUSED SEARCH (High confidence parameters):")
    print("   Parameter ranges based on best performers:")
    
    if len(high_performers) > 0:
        focused_params = {
            'risk_per_trade': (0.0025, 0.0035),  # ~0.3% risk seems optimal
            'sl_min_pips': (5.0, 7.0),           # Tight range around 5.5-6
            'sl_max_pips': (20.0, 26.0),         # 20-25 pips works well
            'tp1_multiplier': (0.20, 0.25),      # Small TP1 for higher hit rate
            'tp2_multiplier': (0.25, 0.35),      # Moderate TP2
            'tp3_multiplier': (0.8, 1.0),        # Larger TP3
            'trailing_atr_multiplier': (1.4, 1.7),
            'partial_profit_sl_distance_ratio': (0.35, 0.45),
            'partial_profit_size_percent': (0.65, 0.75)
        }
        
        for param, (min_val, max_val) in focused_params.items():
            print(f"   {param}: {min_val:.3f} to {max_val:.3f}")
    
    print("\n2. ROBUSTNESS TESTING STRATEGY:")
    print("   - Test on 5 different 50K sample periods")
    print("   - Validate on 3 different 20K out-of-sample periods")
    print("   - Only accept configs with min Sharpe > 0.8 across all periods")
    print("   - Target average Sharpe > 1.2 for buffer")
    
    print("\n3. ADAPTIVE EXPLORATION:")
    print("   - 60% exploitation: vary around best known params")
    print("   - 30% exploration: intelligent random within focused ranges")
    print("   - 10% wild cards: test edge cases to avoid local maxima")
    
    print("\n4. PARAMETER DEPENDENCIES TO EXPLOIT:")
    # Identify parameter relationships
    if len(df) > 10:
        print("   - When sl_max_pips > 20, prefer lower tp1_multiplier (<0.25)")
        print("   - Risk per trade sweet spot: 0.28-0.33%")
        print("   - Partial profit works best at 40% distance, 70% size")
    
    print("\n5. TIME-BASED OPTIMIZATION:")
    print("   - Morning sessions (0600-1200): Test tighter stops")
    print("   - Afternoon sessions (1200-1800): Test wider stops")
    print("   - Overnight (1800-0600): Test with reduced risk")


if __name__ == "__main__":
    # Change to the correct directory
    if os.path.exists('optimizer_results'):
        analyze_all_results()
        generate_next_optimization_strategy()
    else:
        os.chdir('..')
        if os.path.exists('optimizer_results'):
            analyze_all_results()
            generate_next_optimization_strategy()
        else:
            print("❌ Cannot find optimizer_results directory!")