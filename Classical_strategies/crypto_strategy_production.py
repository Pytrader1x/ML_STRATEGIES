"""
Production-Ready Crypto Trading Strategy
Adapted from FX strategy to work with cryptocurrency markets
Uses percentage-based calculations instead of pip-based
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')


class CryptoStrategyAdapter:
    """
    Adapter class to make FX-based strategy work with crypto markets
    Handles the conversion between percentage-based crypto movements and pip-based FX logic
    """
    
    def __init__(self, base_strategy_config):
        self.config = base_strategy_config
        self.pip_multiplier = None
        self.base_price = None
    
    def normalize_prices(self, df):
        """
        Convert crypto prices to pip-like format that the FX strategy can understand
        A 1% move in crypto = 100 "pips" in the normalized data
        """
        # Store original prices
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'{col}_Original'] = df[col].copy()
        
        # Calculate normalization factor
        # We normalize to 10000 so that 1% = 100 units (like pips)
        self.base_price = df['Close'].iloc[0]
        self.pip_multiplier = 10000 / self.base_price
        
        # Apply normalization
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col] * self.pip_multiplier
        
        return df
    
    def denormalize_results(self, results):
        """
        Convert results back from pip-space to real dollar values
        """
        if self.pip_multiplier is None:
            return results
        
        # Convert P&L back to dollars
        if 'total_pnl' in results:
            results['total_pnl_dollars'] = results['total_pnl'] / self.pip_multiplier
        
        # Add percentage return based on initial capital
        if 'total_pnl_dollars' in results:
            results['return_pct'] = (results['total_pnl_dollars'] / self.config.initial_capital) * 100
        
        return results


def create_crypto_config_conservative():
    """
    Conservative configuration for crypto trading
    Uses wider stops and targets appropriate for crypto volatility
    """
    return OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.001,  # 0.1% risk per trade (conservative for crypto)
        
        # Risk Management - expressed as percentages
        sl_max_pips=0.03,  # 3% max stop loss
        sl_atr_multiplier=2.0,  # Higher multiplier for crypto volatility
        
        # Take Profit - expressed as percentages
        tp_atr_multipliers=(0.5, 1.0, 1.5),  # Wider targets for crypto
        max_tp_percent=0.06,  # 6% max take profit
        
        # Trailing Stop - expressed as percentages
        tsl_activation_pips=0.01,  # 1% activation
        tsl_min_profit_pips=0.003,  # 0.3% minimum profit
        tsl_initial_buffer_multiplier=1.5,
        trailing_atr_multiplier=1.2,
        
        # Market-specific multipliers
        tp_range_market_multiplier=0.6,
        tp_trend_market_multiplier=0.8,
        tp_chop_market_multiplier=0.4,
        sl_range_market_multiplier=0.8,
        
        # Exit conditions
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=0.005,  # 0.5% minimum profit
        signal_flip_min_time_hours=2.0,
        signal_flip_partial_exit_percent=0.5,
        
        # Additional features
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.6,
        partial_profit_size_percent=0.4,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )


def create_crypto_config_aggressive():
    """
    Aggressive configuration for crypto scalping
    Tighter stops but more frequent trades
    """
    return OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.0015,  # 0.15% risk per trade
        
        # Risk Management - expressed as percentages
        sl_max_pips=0.015,  # 1.5% max stop loss
        sl_atr_multiplier=1.0,
        
        # Take Profit - expressed as percentages
        tp_atr_multipliers=(0.3, 0.5, 0.8),
        max_tp_percent=0.04,  # 4% max take profit
        
        # Trailing Stop - expressed as percentages
        tsl_activation_pips=0.006,  # 0.6% activation
        tsl_min_profit_pips=0.002,  # 0.2% minimum profit
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=0.8,
        
        # Market-specific multipliers
        tp_range_market_multiplier=0.4,
        tp_trend_market_multiplier=0.6,
        tp_chop_market_multiplier=0.3,
        sl_range_market_multiplier=0.6,
        
        # Exit conditions
        exit_on_signal_flip=True,
        signal_flip_min_profit_pips=0.0,  # Exit immediately on signal flip
        signal_flip_min_time_hours=0.0,
        signal_flip_partial_exit_percent=1.0,
        
        # Additional features
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=0.4,
        partial_profit_size_percent=0.6,
        intelligent_sizing=False,
        sl_volatility_adjustment=True,
        verbose=False
    )


def validate_crypto_strategy(df, config_name, config, n_tests=10):
    """
    Validate crypto strategy with multiple checks
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING {config_name}")
    print(f"{'='*60}")
    
    # Create adapter and strategy
    adapter = CryptoStrategyAdapter(config)
    strategy = OptimizedProdStrategy(config)
    
    # Prepare data
    df_normalized = adapter.normalize_prices(df.copy())
    
    # Add indicators
    df_normalized = TIC.add_neuro_trend_intelligent(df_normalized)
    df_normalized = TIC.add_market_bias(df_normalized)
    df_normalized = TIC.add_intelligent_chop(df_normalized)
    
    # Run multiple tests
    test_results = []
    sample_size = 8000
    
    for i in range(n_tests):
        # Random sample
        max_start = len(df_normalized) - sample_size
        if max_start <= 0:
            print(f"Insufficient data for testing (need {sample_size} rows)")
            return None
        
        start_idx = np.random.randint(0, max_start)
        sample_df = df_normalized.iloc[start_idx:start_idx + sample_size].copy()
        
        # Get date range
        start_date = sample_df.index[0]
        end_date = sample_df.index[-1]
        
        # Run backtest
        try:
            results = strategy.run_backtest(sample_df)
            
            # Denormalize results
            results = adapter.denormalize_results(results)
            
            # Store results
            test_results.append({
                'test': i + 1,
                'start_date': start_date,
                'end_date': end_date,
                'sharpe_ratio': results['sharpe_ratio'],
                'total_pnl_dollars': results.get('total_pnl_dollars', 0),
                'return_pct': results.get('return_pct', 0),
                'win_rate': results['win_rate'],
                'max_drawdown': results['max_drawdown'],
                'profit_factor': results['profit_factor'],
                'total_trades': results['total_trades']
            })
            
        except Exception as e:
            print(f"Error in test {i+1}: {e}")
            continue
    
    if not test_results:
        print("No successful tests completed")
        return None
    
    # Calculate statistics
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in test_results])
    avg_return = np.mean([r['return_pct'] for r in test_results])
    avg_win_rate = np.mean([r['win_rate'] for r in test_results])
    avg_drawdown = np.mean([r['max_drawdown'] for r in test_results])
    sharpe_above_1 = sum(1 for r in test_results if r['sharpe_ratio'] > 1.0) / len(test_results) * 100
    
    # Validation checks
    print("\n1. PERFORMANCE CHECK")
    print(f"   Average Sharpe: {avg_sharpe:.3f}")
    print(f"   Average Return: {avg_return:.1f}%")
    print(f"   Tests with Sharpe > 1.0: {sharpe_above_1:.0f}%")
    
    print("\n2. RISK CHECK")
    print(f"   Average Max Drawdown: {avg_drawdown:.1f}%")
    print(f"   Average Win Rate: {avg_win_rate:.1f}%")
    
    print("\n3. CONSISTENCY CHECK")
    sharpe_std = np.std([r['sharpe_ratio'] for r in test_results])
    print(f"   Sharpe Std Dev: {sharpe_std:.3f}")
    print(f"   Consistency: {'Good' if sharpe_std < 0.5 else 'Poor'}")
    
    print("\n4. SANITY CHECKS")
    # Check for unrealistic returns
    max_return = max(r['return_pct'] for r in test_results)
    min_return = min(r['return_pct'] for r in test_results)
    print(f"   Return range: {min_return:.1f}% to {max_return:.1f}%")
    
    if max_return > 1000:
        print("   ⚠️ WARNING: Unrealistic returns detected!")
    else:
        print("   ✓ Returns appear realistic")
    
    # Check for reasonable trade frequency
    avg_trades = np.mean([r['total_trades'] for r in test_results])
    trades_per_day = avg_trades / (sample_size / 96)  # 96 bars per day
    print(f"   Avg trades per day: {trades_per_day:.1f}")
    
    if trades_per_day > 50:
        print("   ⚠️ WARNING: Excessive trading frequency!")
    else:
        print("   ✓ Trade frequency appears reasonable")
    
    return {
        'config_name': config_name,
        'avg_sharpe': avg_sharpe,
        'avg_return': avg_return,
        'avg_win_rate': avg_win_rate,
        'avg_drawdown': avg_drawdown,
        'sharpe_above_1': sharpe_above_1,
        'consistency': sharpe_std,
        'test_results': test_results
    }


def run_crypto_monte_carlo(crypto_pair='ETH', n_iterations=30, sample_size=8000):
    """
    Run Monte Carlo simulation for crypto trading
    """
    print(f"\n{'='*80}")
    print(f"CRYPTO MONTE CARLO SIMULATION - {crypto_pair}")
    print(f"{'='*80}")
    
    # Load data
    data_path = f'../crypto_data/{crypto_pair}USD_MASTER_15M.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"\nData loaded: {len(df):,} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Test both configurations
    configs = [
        ("Conservative", create_crypto_config_conservative()),
        ("Aggressive", create_crypto_config_aggressive())
    ]
    
    all_results = {}
    
    for config_name, config in configs:
        print(f"\n\nTesting {config_name} Configuration...")
        
        # Create adapter and strategy
        adapter = CryptoStrategyAdapter(config)
        strategy = OptimizedProdStrategy(config)
        
        # Storage for iteration results
        iteration_results = []
        
        for i in range(n_iterations):
            # Random sample
            max_start = len(df) - sample_size
            if max_start <= 0:
                print(f"Insufficient data for {sample_size} sample size")
                break
            
            start_idx = np.random.randint(0, max_start)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
            
            # Normalize prices
            sample_df = adapter.normalize_prices(sample_df)
            
            # Add indicators
            sample_df = TIC.add_neuro_trend_intelligent(sample_df)
            sample_df = TIC.add_market_bias(sample_df)
            sample_df = TIC.add_intelligent_chop(sample_df)
            
            # Run backtest
            try:
                results = strategy.run_backtest(sample_df)
                results = adapter.denormalize_results(results)
                
                iteration_results.append({
                    'iteration': i + 1,
                    'sharpe_ratio': results['sharpe_ratio'],
                    'total_pnl_dollars': results.get('total_pnl_dollars', 0),
                    'return_pct': results.get('return_pct', 0),
                    'win_rate': results['win_rate'],
                    'max_drawdown': results['max_drawdown'],
                    'profit_factor': results['profit_factor'],
                    'total_trades': results['total_trades']
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{n_iterations} iterations completed")
                    
            except Exception as e:
                print(f"  Error in iteration {i+1}: {e}")
                continue
        
        # Calculate averages
        if iteration_results:
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in iteration_results])
            avg_return = np.mean([r['return_pct'] for r in iteration_results])
            avg_win_rate = np.mean([r['win_rate'] for r in iteration_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in iteration_results])
            sharpe_above_1 = sum(1 for r in iteration_results if r['sharpe_ratio'] > 1.0) / len(iteration_results) * 100
            
            all_results[config_name] = {
                'avg_sharpe': avg_sharpe,
                'avg_return': avg_return,
                'avg_win_rate': avg_win_rate,
                'avg_drawdown': avg_drawdown,
                'sharpe_above_1_pct': sharpe_above_1,
                'iterations': iteration_results
            }
            
            print(f"\n  Summary for {config_name}:")
            print(f"  Average Sharpe: {avg_sharpe:.3f}")
            print(f"  Average Return: {avg_return:.1f}%")
            print(f"  Win Rate: {avg_win_rate:.1f}%")
            print(f"  Max Drawdown: {avg_drawdown:.1f}%")
            print(f"  % Sharpe > 1.0: {sharpe_above_1:.1f}%")
    
    return all_results


def main():
    """
    Main function to run crypto strategy validation
    """
    print("="*80)
    print("CRYPTO STRATEGY PRODUCTION VALIDATION")
    print("="*80)
    
    # Load ETH data for validation
    data_path = '../crypto_data/ETHUSD_MASTER_15M.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Run validation on both configs
    configs = [
        ("Conservative Crypto", create_crypto_config_conservative()),
        ("Aggressive Crypto", create_crypto_config_aggressive())
    ]
    
    validation_results = []
    for config_name, config in configs:
        result = validate_crypto_strategy(df, config_name, config, n_tests=20)
        if result:
            validation_results.append(result)
    
    # Run full Monte Carlo
    print("\n\n" + "="*80)
    print("RUNNING FULL MONTE CARLO SIMULATION")
    print("="*80)
    
    monte_carlo_results = run_crypto_monte_carlo('ETH', n_iterations=30, sample_size=8000)
    
    # Save results
    if monte_carlo_results:
        import json
        with open('results/crypto_validation_results.json', 'w') as f:
            json.dump({
                'validation': validation_results,
                'monte_carlo': monte_carlo_results,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, default=str)
        
        print("\n✅ Results saved to results/crypto_validation_results.json")
    
    print(f"\nValidation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()