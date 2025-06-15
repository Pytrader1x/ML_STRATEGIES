"""
Final Monte Carlo Test - 25 Random Samples
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("🎲 FINAL MONTE CARLO VALIDATION - 25 RANDOM SAMPLES")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# Load data
print("\n📊 Loading AUDUSD data...")
data_path = '../data/AUDUSD_MASTER_15M.csv'
df = pd.read_csv(data_path)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

print(f"Total data points: {len(df):,}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Calculate indicators once
print("\n🔧 Calculating indicators...")
start_time = time.time()
df = TIC.add_neuro_trend_intelligent(df)
df = TIC.add_market_bias(df, ha_len=350, ha_len2=30)
df = TIC.add_intelligent_chop(df)
print(f"Indicators calculated in {time.time() - start_time:.1f} seconds")

# Create strategy with verified settings
print("\n⚙️ Strategy Configuration:")
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.005,
    sl_min_pips=3.0,
    sl_max_pips=10.0,
    sl_atr_multiplier=0.8,
    tp_atr_multipliers=(0.15, 0.25, 0.4),
    max_tp_percent=0.005,
    tsl_activation_pips=8.0,
    tsl_min_profit_pips=1.0,
    trailing_atr_multiplier=0.8,
    tp_range_market_multiplier=0.4,
    tp_trend_market_multiplier=0.6,
    tp_chop_market_multiplier=0.3,
    exit_on_signal_flip=True,
    partial_profit_before_sl=True,
    partial_profit_sl_distance_ratio=0.3,
    partial_profit_size_percent=0.7,
    relaxed_mode=True,
    realistic_costs=True,  # CRITICAL
    verbose=False,
    debug_decisions=False,
    use_daily_sharpe=True
)

print(f"  Risk per trade: {config.risk_per_trade:.1%}")
print(f"  Realistic costs: {config.realistic_costs}")
print(f"  Stop loss range: {config.sl_min_pips}-{config.sl_max_pips} pips")
print(f"  Entry slippage: {config.entry_slippage_pips} pips")
print(f"  Stop slippage: {config.stop_loss_slippage_pips} pips")

# Monte Carlo parameters
n_simulations = 25
sample_size = 8000  # ~83 days of data

# Ensure we can take samples
max_start_idx = len(df) - sample_size - 1000
if max_start_idx <= 0:
    print("ERROR: Insufficient data for Monte Carlo")
    exit(1)

# Run Monte Carlo
print(f"\n🎯 Running {n_simulations} simulations with {sample_size:,} bars each...")
print("-"*70)

strategy = OptimizedProdStrategy(config)
results = []

for i in range(n_simulations):
    # Generate truly random starting point
    start_idx = np.random.randint(1000, max_start_idx)
    end_idx = start_idx + sample_size
    
    # Extract contiguous sample
    sample_df = df.iloc[start_idx:end_idx].copy()
    
    # Record sample info
    start_date = sample_df.index[0]
    end_date = sample_df.index[-1]
    
    print(f"\nSimulation {i+1:2d}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", end='')
    
    try:
        # Run backtest
        start_time = time.time()
        result = strategy.run_backtest(sample_df)
        run_time = time.time() - start_time
        
        # Store results
        results.append({
            'sim': i + 1,
            'start_date': start_date,
            'end_date': end_date,
            'days': (end_date - start_date).days,
            'sharpe': result.get('sharpe_ratio', 0),
            'return': result.get('total_return', 0),
            'win_rate': result.get('win_rate', 0),
            'trades': result.get('total_trades', 0),
            'max_dd': result.get('max_drawdown', 0),
            'profit_factor': result.get('profit_factor', 0),
            'run_time': run_time
        })
        
        print(f" | Sharpe: {result.get('sharpe_ratio', 0):>6.2f} | Return: {result.get('total_return', 0):>6.1f}% | Trades: {result.get('total_trades', 0):>4d}")
        
    except Exception as e:
        print(f" | ERROR: {str(e)}")
        results.append({
            'sim': i + 1,
            'start_date': start_date,
            'end_date': end_date,
            'sharpe': 0,
            'return': 0,
            'trades': 0,
            'error': str(e)
        })

# Analyze results
print("\n" + "="*70)
print("MONTE CARLO RESULTS ANALYSIS")
print("="*70)

results_df = pd.DataFrame(results)
valid_results = results_df[results_df['trades'] >= 50]  # Filter for meaningful samples

if len(valid_results) > 0:
    # Calculate statistics
    stats = {
        'Sharpe Ratio': {
            'mean': valid_results['sharpe'].mean(),
            'std': valid_results['sharpe'].std(),
            'min': valid_results['sharpe'].min(),
            'max': valid_results['sharpe'].max(),
            'median': valid_results['sharpe'].median()
        },
        'Total Return %': {
            'mean': valid_results['return'].mean(),
            'std': valid_results['return'].std(),
            'min': valid_results['return'].min(),
            'max': valid_results['return'].max(),
            'median': valid_results['return'].median()
        },
        'Win Rate %': {
            'mean': valid_results['win_rate'].mean(),
            'std': valid_results['win_rate'].std(),
            'min': valid_results['win_rate'].min(),
            'max': valid_results['win_rate'].max(),
            'median': valid_results['win_rate'].median()
        }
    }
    
    # Print statistics table
    print(f"\n📊 Performance Statistics ({len(valid_results)} valid samples):")
    print("-"*70)
    print(f"{'Metric':<15} {'Mean':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
    print("-"*70)
    
    for metric, values in stats.items():
        print(f"{metric:<15} {values['mean']:>10.2f} {values['std']:>10.2f} "
              f"{values['min']:>10.2f} {values['max']:>10.2f} {values['median']:>10.2f}")
    
    # Distribution analysis
    print(f"\n📈 Distribution Analysis:")
    print(f"  Samples with Sharpe > 0: {(valid_results['sharpe'] > 0).sum()} ({(valid_results['sharpe'] > 0).sum()/len(valid_results)*100:.1f}%)")
    print(f"  Samples with Sharpe > 0.7: {(valid_results['sharpe'] > 0.7).sum()} ({(valid_results['sharpe'] > 0.7).sum()/len(valid_results)*100:.1f}%)")
    print(f"  Samples with Sharpe > 1.0: {(valid_results['sharpe'] > 1.0).sum()} ({(valid_results['sharpe'] > 1.0).sum()/len(valid_results)*100:.1f}%)")
    print(f"  Samples with Sharpe > 2.0: {(valid_results['sharpe'] > 2.0).sum()} ({(valid_results['sharpe'] > 2.0).sum()/len(valid_results)*100:.1f}%)")
    
    # Consistency check
    cv = valid_results['sharpe'].std() / valid_results['sharpe'].mean() if valid_results['sharpe'].mean() > 0 else float('inf')
    print(f"\n📊 Consistency Metrics:")
    print(f"  Coefficient of Variation: {cv:.3f}")
    print(f"  Consistency Score: {(1 - cv)*100:.1f}%" if cv < 1 else "  Consistency Score: Low")
    
    # Time period analysis
    print(f"\n🗓️ Time Period Coverage:")
    print(f"  Earliest sample: {valid_results['start_date'].min().strftime('%Y-%m-%d')}")
    print(f"  Latest sample: {valid_results['end_date'].max().strftime('%Y-%m-%d')}")
    print(f"  Average sample duration: {valid_results['days'].mean():.0f} days")
    
    # Performance summary
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print("-"*70)
    
    avg_sharpe = valid_results['sharpe'].mean()
    if avg_sharpe >= 2.0:
        rating = "EXCEPTIONAL ⭐⭐⭐⭐⭐"
    elif avg_sharpe >= 1.5:
        rating = "EXCELLENT ⭐⭐⭐⭐"
    elif avg_sharpe >= 1.0:
        rating = "VERY GOOD ⭐⭐⭐"
    elif avg_sharpe >= 0.7:
        rating = "GOOD ⭐⭐"
    else:
        rating = "NEEDS IMPROVEMENT ⭐"
    
    print(f"  Strategy Rating: {rating}")
    print(f"  Average Sharpe: {avg_sharpe:.3f}")
    print(f"  Success Rate: {(valid_results['sharpe'] > 0.7).sum()/len(valid_results)*100:.1f}%")
    print(f"  Risk-Adjusted Return: {avg_sharpe * np.sqrt(252):.1f}% annualized")
    
else:
    print("\n❌ ERROR: No valid results obtained")

# Save results
print("\n💾 Saving results...")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_df.to_csv(f'results/monte_carlo_25_samples_{timestamp}.csv', index=False)
print(f"Results saved to: results/monte_carlo_25_samples_{timestamp}.csv")

print("\n" + "="*70)
print("✅ MONTE CARLO VALIDATION COMPLETE")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)