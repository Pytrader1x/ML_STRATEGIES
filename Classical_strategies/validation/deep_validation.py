"""
Deep Validation Analysis for AUDUSD Strategy
Investigates why random entry baseline performs suspiciously well
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random

warnings.filterwarnings('ignore')

class DeepValidationAnalysis:
    def __init__(self):
        self.results = {}
        
    def load_data(self, currency_pair='AUDUSD'):
        """Load and prepare data"""
        print(f"Loading {currency_pair} data...")
        df = pd.read_csv(f'../../data/{currency_pair}_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Calculate returns for analysis
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        print(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        return df
    
    def analyze_data_integrity(self, df):
        """Check data for anomalies that could cause high random performance"""
        print("\n" + "="*80)
        print("DATA INTEGRITY ANALYSIS")
        print("="*80)
        
        # Check for gaps
        time_diff = df.index.to_series().diff()
        gaps = time_diff[time_diff > pd.Timedelta(minutes=16)]
        print(f"\nData gaps (>16 min): {len(gaps)}")
        if len(gaps) > 0:
            print("Sample gaps:")
            print(gaps.head())
        
        # Check for extreme moves
        extreme_moves = df[abs(df['returns']) > 0.02]  # >2% moves in 15 min
        print(f"\nExtreme moves (>2% in 15min): {len(extreme_moves)}")
        
        # Check for zero or constant prices
        zero_changes = df[df['returns'] == 0]
        print(f"Bars with zero price change: {len(zero_changes)}")
        
        # Analyze autocorrelation
        returns = df['returns'].dropna()
        autocorr_1 = returns.autocorr(lag=1)
        autocorr_5 = returns.autocorr(lag=5)
        autocorr_20 = returns.autocorr(lag=20)
        
        print(f"\nReturn autocorrelations:")
        print(f"  Lag 1: {autocorr_1:.4f}")
        print(f"  Lag 5: {autocorr_5:.4f}")
        print(f"  Lag 20: {autocorr_20:.4f}")
        
        # Check for trending behavior
        up_bars = len(df[df['returns'] > 0])
        down_bars = len(df[df['returns'] < 0])
        total_bars = len(df['returns'].dropna())
        
        print(f"\nPrice movement distribution:")
        print(f"  Up bars: {up_bars} ({up_bars/total_bars*100:.1f}%)")
        print(f"  Down bars: {down_bars} ({down_bars/total_bars*100:.1f}%)")
        
        return {
            'gaps': len(gaps),
            'extreme_moves': len(extreme_moves),
            'zero_changes': len(zero_changes),
            'autocorr_1': autocorr_1,
            'up_bar_pct': up_bars/total_bars*100
        }
    
    def test_random_strategies(self, df, n_tests=50, sample_size=5000):
        """Test multiple random entry strategies"""
        print("\n" + "="*80)
        print("RANDOM STRATEGY MONTE CARLO")
        print("="*80)
        
        random_results = []
        
        for i in range(n_tests):
            # Select random sample
            start_idx = np.random.randint(0, len(df) - sample_size)
            sample_df = df.iloc[start_idx:start_idx + sample_size].copy()
            
            # Add indicators
            sample_df = TIC.add_neuro_trend_intelligent(sample_df)
            sample_df = TIC.add_market_bias(sample_df)
            sample_df = TIC.add_intelligent_chop(sample_df)
            
            # Run random strategy
            random_strategy = self.create_pure_random_strategy()
            results = random_strategy.run_backtest(sample_df)
            
            random_results.append({
                'sharpe': results['sharpe_ratio'],
                'return': results['total_return'],
                'win_rate': results['win_rate'],
                'trades': results['total_trades']
            })
            
            if i % 10 == 0:
                print(f"  Test {i+1}/{n_tests}: Sharpe={results['sharpe_ratio']:.3f}")
        
        # Analyze results
        sharpes = [r['sharpe'] for r in random_results]
        returns = [r['return'] for r in random_results]
        win_rates = [r['win_rate'] for r in random_results]
        
        print(f"\nRandom Strategy Statistics (n={n_tests}):")
        print(f"  Sharpe Ratio: {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
        print(f"  Returns: {np.mean(returns):.1f}% ± {np.std(returns):.1f}%")
        print(f"  Win Rate: {np.mean(win_rates):.1f}% ± {np.std(win_rates):.1f}%")
        print(f"  Sharpe > 0.5: {sum(s > 0.5 for s in sharpes)} ({sum(s > 0.5 for s in sharpes)/n_tests*100:.1f}%)")
        print(f"  Sharpe > 0.0: {sum(s > 0.0 for s in sharpes)} ({sum(s > 0.0 for s in sharpes)/n_tests*100:.1f}%)")
        
        return random_results
    
    def create_pure_random_strategy(self):
        """Create a truly random entry strategy"""
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            verbose=False
        )
        
        class PureRandomStrategy(OptimizedProdStrategy):
            def __init__(self, config):
                super().__init__(config)
                self.trade_probability = 0.02  # 2% chance per bar
                self.last_trade_bar = -100
                
            def generate_signal(self, row, prev_row=None):
                """Generate pure random signals"""
                # Ensure minimum spacing between trades
                if hasattr(self, 'current_bar_index'):
                    if self.current_bar_index - self.last_trade_bar < 20:
                        return 0
                
                # Random signal
                if random.random() < self.trade_probability:
                    signal = 1 if random.random() < 0.5 else -1
                    self.last_trade_bar = getattr(self, 'current_bar_index', 0)
                    return signal
                return 0
        
        return PureRandomStrategy(config)
    
    def analyze_indicator_lookahead(self, df):
        """Check if indicators contain future information"""
        print("\n" + "="*80)
        print("INDICATOR LOOK-AHEAD ANALYSIS")
        print("="*80)
        
        # Add indicators
        df_test = df.copy()
        df_test = TIC.add_neuro_trend_intelligent(df_test)
        df_test = TIC.add_market_bias(df_test)
        df_test = TIC.add_intelligent_chop(df_test)
        
        # Check correlation with future returns
        future_returns = df_test['returns'].shift(-1)  # Next bar return
        
        indicators = ['NTI_Direction', 'MB_Bias', 'IC_Signal']
        
        for indicator in indicators:
            if indicator in df_test.columns:
                # Correlation with future returns
                corr_future = df_test[indicator].corr(future_returns)
                
                # Correlation with current returns
                corr_current = df_test[indicator].corr(df_test['returns'])
                
                # Correlation with past returns
                corr_past = df_test[indicator].corr(df_test['returns'].shift(1))
                
                print(f"\n{indicator} correlations:")
                print(f"  With future returns: {corr_future:.4f}")
                print(f"  With current returns: {corr_current:.4f}")
                print(f"  With past returns: {corr_past:.4f}")
                
                # Check if indicator predicts future better than past
                if abs(corr_future) > abs(corr_past) * 1.5:
                    print(f"  ⚠️  WARNING: {indicator} shows stronger correlation with future!")
    
    def test_different_market_regimes(self, df):
        """Test strategy in different market conditions"""
        print("\n" + "="*80)
        print("MARKET REGIME ANALYSIS")
        print("="*80)
        
        # Define market regimes
        regimes = {
            '2020 COVID Volatility': ('2020-03-01', '2020-06-30'),
            '2021 Trending': ('2021-01-01', '2021-12-31'),
            '2022 Bear Market': ('2022-01-01', '2022-12-31'),
            '2023 Recovery': ('2023-01-01', '2023-12-31'),
            '2024 Recent': ('2024-01-01', '2024-12-31')
        }
        
        # Test both normal and random strategies in each regime
        for regime_name, (start, end) in regimes.items():
            regime_df = df[start:end].copy()
            
            if len(regime_df) < 1000:
                print(f"\n{regime_name}: Insufficient data")
                continue
                
            # Add indicators
            regime_df = TIC.add_neuro_trend_intelligent(regime_df)
            regime_df = TIC.add_market_bias(regime_df)
            regime_df = TIC.add_intelligent_chop(regime_df)
            
            # Test normal strategy
            normal_strategy = self.create_normal_strategy()
            normal_results = normal_strategy.run_backtest(regime_df)
            
            # Test random strategy
            random_strategy = self.create_pure_random_strategy()
            random_results = random_strategy.run_backtest(regime_df)
            
            print(f"\n{regime_name} ({len(regime_df)} bars):")
            print(f"  Normal Strategy: Sharpe={normal_results['sharpe_ratio']:.3f}, Return={normal_results['total_return']:.1f}%")
            print(f"  Random Strategy: Sharpe={random_results['sharpe_ratio']:.3f}, Return={random_results['total_return']:.1f}%")
            
            # Check if random outperforms in any regime
            if random_results['sharpe_ratio'] > normal_results['sharpe_ratio'] * 0.8:
                print(f"  ⚠️  WARNING: Random strategy performs too well in {regime_name}!")
    
    def create_normal_strategy(self):
        """Create the normal strategy for comparison"""
        config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.002,
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            tsl_activation_pips=3,
            tsl_min_profit_pips=1,
            verbose=False
        )
        return OptimizedProdStrategy(config)
    
    def analyze_spread_impact(self, df, spread_pips=0.5):
        """Test impact of realistic spreads"""
        print("\n" + "="*80)
        print("SPREAD IMPACT ANALYSIS")
        print("="*80)
        
        # Test with different spread levels
        spreads = [0, 0.5, 1.0, 2.0]  # pips
        
        sample_df = df.iloc[-10000:].copy()  # Last 10k bars
        sample_df = TIC.add_neuro_trend_intelligent(sample_df)
        sample_df = TIC.add_market_bias(sample_df)
        sample_df = TIC.add_intelligent_chop(sample_df)
        
        for spread in spreads:
            # Modify data to include spread
            if spread > 0:
                sample_df['Ask'] = sample_df['Close'] + (spread * 0.0001)
                sample_df['Bid'] = sample_df['Close'] - (spread * 0.0001)
            
            # Test random strategy
            random_strategy = self.create_pure_random_strategy()
            results = random_strategy.run_backtest(sample_df)
            
            print(f"\nSpread = {spread} pips:")
            print(f"  Sharpe: {results['sharpe_ratio']:.3f}")
            print(f"  Return: {results['total_return']:.1f}%")
            print(f"  Win Rate: {results['win_rate']:.1f}%")
    
    def visualize_results(self, random_results):
        """Create visualizations of validation results"""
        plt.figure(figsize=(15, 10))
        
        # Sharpe distribution
        plt.subplot(2, 2, 1)
        sharpes = [r['sharpe'] for r in random_results]
        plt.hist(sharpes, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='Zero')
        plt.axvline(x=0.644, color='green', linestyle='--', label='Observed (0.644)')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.title('Random Strategy Sharpe Distribution')
        plt.legend()
        
        # Return distribution
        plt.subplot(2, 2, 2)
        returns = [r['return'] for r in random_results]
        plt.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Total Return (%)')
        plt.ylabel('Frequency')
        plt.title('Random Strategy Return Distribution')
        
        # Win rate distribution
        plt.subplot(2, 2, 3)
        win_rates = [r['win_rate'] for r in random_results]
        plt.hist(win_rates, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=50, color='red', linestyle='--', label='50%')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Frequency')
        plt.title('Random Strategy Win Rate Distribution')
        plt.legend()
        
        # Sharpe vs Win Rate scatter
        plt.subplot(2, 2, 4)
        plt.scatter(win_rates, sharpes, alpha=0.5)
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe vs Win Rate')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=50, color='red', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deep_validation_results.png', dpi=150)
        print("\nVisualization saved to deep_validation_results.png")
        plt.close()
    
    def run_full_analysis(self):
        """Run complete deep validation analysis"""
        print("="*80)
        print("DEEP VALIDATION ANALYSIS")
        print("="*80)
        print(f"Started: {datetime.now()}")
        
        # Load data
        df = self.load_data()
        
        # 1. Data integrity check
        integrity_results = self.analyze_data_integrity(df)
        self.results['data_integrity'] = integrity_results
        
        # 2. Indicator look-ahead analysis
        self.analyze_indicator_lookahead(df)
        
        # 3. Random strategy Monte Carlo
        random_results = self.test_random_strategies(df, n_tests=50)
        self.results['random_monte_carlo'] = random_results
        
        # 4. Market regime analysis
        self.test_different_market_regimes(df)
        
        # 5. Spread impact analysis
        self.analyze_spread_impact(df)
        
        # 6. Visualize results
        self.visualize_results(random_results)
        
        # Summary
        print("\n" + "="*80)
        print("DEEP VALIDATION SUMMARY")
        print("="*80)
        
        avg_random_sharpe = np.mean([r['sharpe'] for r in random_results])
        if avg_random_sharpe > 0.3:
            print("\n🚨 CRITICAL FINDINGS:")
            print(f"  - Average random Sharpe of {avg_random_sharpe:.3f} is too high")
            print("  - This indicates systematic bias in the data or implementation")
            print("  - DO NOT TRADE LIVE until resolved")
        
        print(f"\nCompleted: {datetime.now()}")
        return self.results


def main():
    """Run deep validation analysis"""
    analyzer = DeepValidationAnalysis()
    results = analyzer.run_full_analysis()
    
    # Save results
    import json
    with open('deep_validation_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, default=convert, indent=2)
    
    print("\nResults saved to deep_validation_results.json")


if __name__ == "__main__":
    main()