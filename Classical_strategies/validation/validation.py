"""
AUDUSD Strategy Validation Script
Checks for look-ahead bias, trade size consistency, and other potential issues
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os
import random
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def create_config_1_ultra_tight_risk():
    """Configuration 1: Ultra-Tight Risk Management"""
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,  # 0.2% risk per trade
        sl_max_pips=10.0,
        sl_atr_multiplier=1.0,
        tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003,
        tsl_activation_pips=3,
        tsl_min_profit_pips=1,
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=0.8,
        tp_range_market_multiplier=0.5,
        tp_trend_market_multiplier=0.7,
        tp_chop_market_multiplier=0.3,
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        signal_flip_min_profit_pips=5.0,
        signal_flip_min_time_hours=1.0,
        signal_flip_partial_exit_percent=1.0,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.5,
        partial_profit_size_percent=0.5,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )
    return OptimizedProdStrategy(config)


def load_and_prepare_data(currency_pair='AUDUSD'):
    """Load and prepare data for AUDUSD"""
    file_path = f'../../data/{currency_pair}_MASTER_15M.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading {currency_pair} data...")
    df = pd.read_csv(file_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Total data points: {len(df):,}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    return df


def check_look_ahead_bias(df, strategy, sample_size=5000):
    """Test for look-ahead bias by comparing with random entry baseline"""
    print("\n" + "="*80)
    print("LOOK-AHEAD BIAS CHECK")
    print("="*80)
    
    # Select a random sample
    start_idx = np.random.randint(0, len(df) - sample_size)
    sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
    
    # Run normal strategy
    print("\nRunning normal strategy...")
    normal_results = strategy.run_backtest(sample_df)
    
    # Create random entry strategy
    print("\nRunning random entry baseline...")
    random_strategy = create_random_entry_strategy()
    random_results = random_strategy.run_backtest(sample_df)
    
    print("\nComparison Results:")
    print(f"{'Metric':<20} {'Normal Strategy':>15} {'Random Baseline':>15}")
    print("-" * 50)
    print(f"{'Sharpe Ratio':<20} {normal_results['sharpe_ratio']:>15.3f} {random_results['sharpe_ratio']:>15.3f}")
    print(f"{'Win Rate %':<20} {normal_results['win_rate']:>15.1f} {random_results['win_rate']:>15.1f}")
    print(f"{'Total Return %':<20} {normal_results['total_return']:>15.1f} {random_results['total_return']:>15.1f}")
    print(f"{'Max Drawdown %':<20} {normal_results['max_drawdown']:>15.1f} {random_results['max_drawdown']:>15.1f}")
    
    # Check if random baseline is suspiciously good
    if random_results['sharpe_ratio'] > 0.5:
        print("\n⚠️  WARNING: Random entry baseline has Sharpe > 0.5!")
        print("This may indicate look-ahead bias or implementation issues.")
    
    return normal_results, random_results


def create_random_entry_strategy():
    """Create a strategy that enters randomly for baseline comparison"""
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        sl_atr_multiplier=1.0,
        tp_atr_multipliers=(0.2, 0.3, 0.5),
        max_tp_percent=0.003,
        verbose=False
    )
    
    # Override the strategy to use random entries
    class RandomEntryStrategy(OptimizedProdStrategy):
        def __init__(self, config):
            super().__init__(config)
            self.random_seed = 42
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        def generate_signal(self, row, prev_row=None):
            """Generate random buy/sell signals"""
            # Random signal with 5% probability
            rand = random.random()
            if rand < 0.025:  # 2.5% chance for buy
                return 1
            elif rand < 0.05:  # 2.5% chance for sell
                return -1
            else:
                return 0
    
    return RandomEntryStrategy(config)


def analyze_trade_sizes(df, strategy, sample_size=5000):
    """Analyze trade sizes to check for consistency"""
    print("\n" + "="*80)
    print("TRADE SIZE ANALYSIS")
    print("="*80)
    
    # Run strategy
    start_idx = np.random.randint(0, len(df) - sample_size)
    sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
    
    # Enable verbose to get trade details
    strategy.config.verbose = True
    results = strategy.run_backtest(sample_df)
    strategy.config.verbose = False
    
    if 'trades' in results and results['trades']:
        trades = results['trades']
        
        # Extract trade sizes
        trade_sizes = []
        trade_values = []
        for trade in trades:
            if hasattr(trade, 'size'):
                trade_sizes.append(trade.size)
                trade_values.append(trade.size * trade.entry_price)
            elif hasattr(trade, 'quantity'):
                trade_sizes.append(trade.quantity)
                trade_values.append(trade.quantity * trade.entry_price)
        
        if trade_sizes:
            print(f"\nAnalyzing {len(trade_sizes)} trades:")
            print(f"Min trade size: {min(trade_sizes):,.2f} units")
            print(f"Max trade size: {max(trade_sizes):,.2f} units")
            print(f"Avg trade size: {np.mean(trade_sizes):,.2f} units")
            print(f"Std trade size: {np.std(trade_sizes):,.2f} units")
            
            print(f"\nTrade values (in base currency):")
            print(f"Min trade value: ${min(trade_values):,.2f}")
            print(f"Max trade value: ${max(trade_values):,.2f}")
            print(f"Avg trade value: ${np.mean(trade_values):,.2f}")
            
            # Check if sizes are increasing over time
            first_half_avg = np.mean(trade_sizes[:len(trade_sizes)//2])
            second_half_avg = np.mean(trade_sizes[len(trade_sizes)//2:])
            
            print(f"\nTemporal analysis:")
            print(f"First half avg size: {first_half_avg:,.2f}")
            print(f"Second half avg size: {second_half_avg:,.2f}")
            
            if second_half_avg > first_half_avg * 1.5:
                print("\n⚠️  WARNING: Trade sizes increase significantly over time!")
                print("This could indicate compound position sizing or look-ahead bias.")
        else:
            print("\nNo trade size information available in trade objects.")
    else:
        print("\nNo trades found in results.")
    
    return results


def inspect_random_trades(df, strategy, n_trades=10, sample_size=5000):
    """Inspect random trades in detail"""
    print("\n" + "="*80)
    print(f"RANDOM TRADE INSPECTION ({n_trades} trades)")
    print("="*80)
    
    # Run strategy
    start_idx = np.random.randint(0, len(df) - sample_size)
    sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
    
    results = strategy.run_backtest(sample_df)
    
    if 'trades' in results and results['trades']:
        trades = results['trades']
        
        # Select random trades
        selected_trades = random.sample(trades, min(n_trades, len(trades)))
        
        for i, trade in enumerate(selected_trades, 1):
            print(f"\n--- Trade {i} ---")
            
            # Extract trade details
            if hasattr(trade, '__dict__'):
                for key, value in trade.__dict__.items():
                    if key not in ['_df']:  # Skip dataframe references
                        print(f"{key}: {value}")
            else:
                print(f"Trade object: {trade}")
            
            # Try to find the entry logic
            if hasattr(trade, 'entry_time'):
                entry_idx = sample_df.index.get_loc(trade.entry_time)
                if entry_idx > 0:
                    entry_row = sample_df.iloc[entry_idx]
                    prev_row = sample_df.iloc[entry_idx-1]
                    
                    print(f"\nEntry conditions:")
                    print(f"NTI_Direction: {entry_row.get('NTI_Direction', 'N/A')}")
                    print(f"MB_Bias: {entry_row.get('MB_Bias', 'N/A')}")
                    print(f"IC_Signal: {entry_row.get('IC_Signal', 'N/A')}")
                    print(f"Price: {entry_row['Close']}")
                    
                    # Check if entry makes sense
                    if hasattr(trade, 'direction'):
                        if trade.direction == 1 and entry_row.get('NTI_Direction', 0) <= 0:
                            print("⚠️  WARNING: Long trade with non-positive NTI_Direction!")
                        elif trade.direction == -1 and entry_row.get('NTI_Direction', 0) >= 0:
                            print("⚠️  WARNING: Short trade with non-negative NTI_Direction!")
    else:
        print("\nNo trades found in results.")
    
    return results


def check_future_data_usage(df, strategy, sample_size=1000):
    """Check if strategy uses future data by running on truncated data"""
    print("\n" + "="*80)
    print("FUTURE DATA USAGE CHECK")
    print("="*80)
    
    # Select a random starting point
    start_idx = np.random.randint(0, len(df) - sample_size - 100)
    
    # Run on full sample
    full_sample = df.iloc[start_idx:start_idx + sample_size].copy()
    full_results = strategy.run_backtest(full_sample)
    
    # Run on truncated sample (missing last 100 bars)
    truncated_sample = df.iloc[start_idx:start_idx + sample_size - 100].copy()
    truncated_results = strategy.run_backtest(truncated_sample)
    
    # Compare last trades
    if 'trades' in full_results and 'trades' in truncated_results:
        full_trades = full_results['trades']
        truncated_trades = truncated_results['trades']
        
        print(f"\nFull sample trades: {len(full_trades)}")
        print(f"Truncated sample trades: {len(truncated_trades)}")
        
        # Check if early trades are identical
        min_trades = min(10, len(truncated_trades))
        differences = 0
        
        for i in range(min_trades):
            if i < len(full_trades) and i < len(truncated_trades):
                full_trade = full_trades[i]
                trunc_trade = truncated_trades[i]
                
                # Compare entry times
                if hasattr(full_trade, 'entry_time') and hasattr(trunc_trade, 'entry_time'):
                    if full_trade.entry_time != trunc_trade.entry_time:
                        differences += 1
                        print(f"Trade {i}: Entry time differs!")
        
        if differences == 0:
            print("\n✅ PASS: Early trades are consistent between full and truncated data")
        else:
            print(f"\n⚠️  WARNING: {differences} trades differ between full and truncated data!")
            print("This may indicate future data usage.")
    
    return full_results, truncated_results


def calculate_realistic_metrics(results):
    """Calculate realistic performance metrics with proper annualization"""
    print("\n" + "="*80)
    print("REALISTIC PERFORMANCE METRICS")
    print("="*80)
    
    if 'trades' in results and results['trades']:
        trades = results['trades']
        
        # Calculate P&L for each trade
        pnls = []
        for trade in trades:
            if hasattr(trade, 'pnl'):
                pnls.append(trade.pnl)
            elif hasattr(trade, 'profit_loss'):
                pnls.append(trade.profit_loss)
        
        if pnls:
            # Calculate realistic metrics
            total_pnl = sum(pnls)
            avg_pnl = np.mean(pnls)
            
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            print(f"Total P&L: ${total_pnl:,.2f}")
            print(f"Average P&L per trade: ${avg_pnl:,.2f}")
            print(f"Win rate: {len(winning_trades)/len(pnls)*100:.1f}%")
            
            if winning_trades:
                print(f"Average win: ${np.mean(winning_trades):,.2f}")
            if losing_trades:
                print(f"Average loss: ${np.mean(losing_trades):,.2f}")
            
            # Check for unrealistic returns
            if results.get('total_return', 0) > 200:
                print("\n⚠️  WARNING: Returns exceed 200%!")
                print("This is extremely rare in forex trading and may indicate an issue.")
            
            # Check average win/loss ratio
            if winning_trades and losing_trades:
                avg_win = np.mean(winning_trades)
                avg_loss = abs(np.mean(losing_trades))
                win_loss_ratio = avg_win / avg_loss
                
                print(f"\nWin/Loss ratio: {win_loss_ratio:.2f}")
                
                if win_loss_ratio > 3:
                    print("⚠️  WARNING: Win/loss ratio > 3 is suspiciously high for forex!")
    
    return results


def main():
    """Run all validation checks"""
    print("AUDUSD Strategy Validation")
    print("="*80)
    print(f"Validation started: {datetime.now()}")
    
    # Load data
    df = load_and_prepare_data('AUDUSD')
    
    # Create strategy
    strategy = create_config_1_ultra_tight_risk()
    
    # Run validation checks
    print("\n1. LOOK-AHEAD BIAS CHECK")
    normal_results, random_results = check_look_ahead_bias(df, strategy)
    
    print("\n2. TRADE SIZE ANALYSIS")
    trade_results = analyze_trade_sizes(df, strategy)
    
    print("\n3. RANDOM TRADE INSPECTION")
    inspect_results = inspect_random_trades(df, strategy, n_trades=10)
    
    print("\n4. FUTURE DATA USAGE CHECK")
    full_results, truncated_results = check_future_data_usage(df, strategy)
    
    print("\n5. REALISTIC METRICS CHECK")
    calculate_realistic_metrics(normal_results)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Check for red flags
    red_flags = []
    
    if random_results['sharpe_ratio'] > 0.5:
        red_flags.append("Random baseline has high Sharpe ratio")
    
    if normal_results.get('total_return', 0) > 200:
        red_flags.append("Unrealistic returns (>200%)")
    
    if normal_results.get('win_rate', 0) > 80:
        red_flags.append("Suspiciously high win rate (>80%)")
    
    if red_flags:
        print("\n🚨 RED FLAGS DETECTED:")
        for flag in red_flags:
            print(f"  - {flag}")
    else:
        print("\n✅ No major red flags detected")
    
    print(f"\nValidation completed: {datetime.now()}")


if __name__ == "__main__":
    main()