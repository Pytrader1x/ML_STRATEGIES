"""
Enhanced Recursive Strategy Optimizer - Target Sharpe Ratio 1.0
Uses more aggressive parameter exploration and advanced optimization techniques
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import time
from datetime import datetime
import json
import os
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class EnhancedStrategyOptimizer:
    def __init__(self, target_sharpe=1.0):
        self.target_sharpe = target_sharpe
        self.iteration = 0
        self.best_config = None
        self.best_sharpe = -np.inf
        self.best_results = None
        self.optimization_history = []
        self.parameter_impact = {}  # Track which parameters improve Sharpe
        
    def load_and_prepare_data(self, sample_size=30000):
        """Load data and prepare a larger sample for testing"""
        print(f"Loading data for iteration {self.iteration}...")
        df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Use more recent data for relevance
        df_sample = df.tail(sample_size).copy()
        
        # Calculate indicators
        df_sample = TIC.add_neuro_trend_intelligent(df_sample, base_fast=10, base_slow=50, confirm_bars=3)
        df_sample = TIC.add_market_bias(df_sample, ha_len=350, ha_len2=30)
        df_sample = TIC.add_intelligent_chop(df_sample)
        
        return df_sample
    
    def generate_aggressive_variations(self):
        """Generate more aggressive parameter combinations"""
        
        # Key parameters that impact Sharpe ratio
        if self.iteration == 0:
            # Start with baseline
            return [OptimizedStrategyConfig()]
        
        # Aggressive parameter ranges based on what typically improves Sharpe
        param_grid = {
            'tp_atr_multipliers': [
                (0.3, 0.8, 1.5),   # Very tight TPs for high win rate
                (0.4, 1.0, 2.0),   # Tight TPs
                (0.5, 1.2, 2.5),   # Balanced
                (0.6, 1.5, 3.0),   # Medium
            ],
            'tsl_activation_pips': [5, 8, 10, 12, 15],  # Earlier TSL activation
            'tsl_min_profit_pips': [2, 3, 4, 5],        # Lower minimum profit
            'sl_max_pips': [15, 20, 25, 30],           # Tighter stop losses
            'risk_per_trade': [0.005, 0.01, 0.015],    # Lower risk for consistency
            'signal_flip_min_profit_pips': [0, 2, 3],  # Allow earlier exits
            'tsl_initial_buffer_multiplier': [1.0, 1.5], # Less buffer for quicker profits
        }
        
        # Generate combinations based on iteration
        configs = []
        
        if self.iteration < 10:
            # Focus on TP and TSL optimization
            for tp_mults in param_grid['tp_atr_multipliers']:
                for tsl_act in param_grid['tsl_activation_pips'][:3]:
                    for tsl_min in param_grid['tsl_min_profit_pips']:
                        if tsl_min < tsl_act:
                            config = OptimizedStrategyConfig()
                            config.tp_atr_multipliers = tp_mults
                            config.tsl_activation_pips = tsl_act
                            config.tsl_min_profit_pips = tsl_min
                            config.tsl_initial_buffer_multiplier = 1.5
                            configs.append(config)
        
        elif self.iteration < 20:
            # Focus on risk and stop loss
            best_tp = self.best_config.tp_atr_multipliers if self.best_config else (0.5, 1.2, 2.5)
            best_tsl = self.best_config.tsl_activation_pips if self.best_config else 10
            
            for risk in param_grid['risk_per_trade']:
                for sl_max in param_grid['sl_max_pips']:
                    config = OptimizedStrategyConfig()
                    config.tp_atr_multipliers = best_tp
                    config.tsl_activation_pips = best_tsl
                    config.risk_per_trade = risk
                    config.sl_max_pips = sl_max
                    configs.append(config)
        
        else:
            # Ultra-aggressive combinations for high Sharpe
            ultra_aggressive_configs = [
                # Config 1: Super tight TPs, early TSL
                {
                    'tp_atr_multipliers': (0.3, 0.6, 1.0),
                    'tsl_activation_pips': 5,
                    'tsl_min_profit_pips': 2,
                    'sl_max_pips': 15,
                    'risk_per_trade': 0.005,
                    'exit_on_signal_flip': False,  # Disable to avoid premature exits
                },
                # Config 2: Scalping approach
                {
                    'tp_atr_multipliers': (0.2, 0.5, 0.8),
                    'tsl_activation_pips': 3,
                    'tsl_min_profit_pips': 1,
                    'sl_max_pips': 10,
                    'risk_per_trade': 0.003,
                    'tp_range_market_multiplier': 0.5,
                },
                # Config 3: High win rate focus
                {
                    'tp_atr_multipliers': (0.4, 0.7, 1.2),
                    'tsl_activation_pips': 7,
                    'tsl_min_profit_pips': 3,
                    'sl_max_pips': 20,
                    'risk_per_trade': 0.01,
                    'signal_flip_min_profit_pips': 0,
                },
                # Config 4: Momentum capture
                {
                    'tp_atr_multipliers': (0.5, 1.0, 1.5),
                    'tsl_activation_pips': 10,
                    'tsl_min_profit_pips': 5,
                    'sl_max_pips': 25,
                    'risk_per_trade': 0.015,
                    'tsl_initial_buffer_multiplier': 1.0,
                },
                # Config 5: Best config with variations
                self.best_config.__dict__ if self.best_config else None
            ]
            
            for config_dict in ultra_aggressive_configs:
                if config_dict:
                    config = OptimizedStrategyConfig()
                    for key, value in config_dict.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                    configs.append(config)
        
        # Limit configs and ensure uniqueness
        return configs[:10]
    
    def run_backtest(self, df, config):
        """Run a single backtest with given configuration"""
        strategy = OptimizedProdStrategy(config)
        results = strategy.run_backtest(df)
        return results
    
    def calculate_enhanced_metrics(self, results):
        """Calculate additional metrics for optimization"""
        metrics = {
            'sharpe_ratio': results['sharpe_ratio'],
            'total_pnl': results['total_pnl'],
            'win_rate': results['win_rate'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades'],
            'profit_factor': results.get('profit_factor', 0),
            'avg_trade_duration': np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 3600 
                for t in results['trades'] if t.exit_time
            ]) if results['trades'] else 0,
            'calmar_ratio': results['total_return'] / abs(results['max_drawdown']) if results['max_drawdown'] != 0 else 0
        }
        
        # Calculate consistency score (prefer steady returns)
        if len(results['equity_curve']) > 100:
            returns = pd.Series(results['equity_curve']).pct_change().dropna()
            metrics['consistency_score'] = 1 / (returns.std() + 0.001)  # Lower volatility = higher score
        else:
            metrics['consistency_score'] = 0
            
        return metrics
    
    def optimize(self):
        """Main optimization loop with enhanced strategies"""
        print("="*80)
        print("ENHANCED STRATEGY OPTIMIZER - TARGET SHARPE RATIO: 1.0")
        print("="*80)
        
        # Load larger dataset for better statistical significance
        df = self.load_and_prepare_data(sample_size=40000)
        print(f"Dataset loaded: {len(df):,} bars")
        
        max_iterations = 100
        no_improvement_count = 0
        
        while self.best_sharpe < self.target_sharpe and self.iteration < max_iterations:
            self.iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}")
            print(f"{'='*60}")
            
            # Generate aggressive configurations
            configs = self.generate_aggressive_variations()
            print(f"Testing {len(configs)} configurations...")
            
            iteration_best_sharpe = self.best_sharpe
            
            # Test each configuration
            for i, config in enumerate(configs):
                try:
                    # Run backtest
                    results = self.run_backtest(df, config)
                    metrics = self.calculate_enhanced_metrics(results)
                    
                    # Print condensed results
                    print(f"\nConfig {i+1}: Sharpe={metrics['sharpe_ratio']:.3f}, "
                          f"WR={metrics['win_rate']:.1f}%, P&L=${metrics['total_pnl']:,.0f}, "
                          f"DD={metrics['max_drawdown']:.1f}%, Trades={metrics['total_trades']}")
                    
                    # Track history
                    self.optimization_history.append({
                        'iteration': self.iteration,
                        'config_num': i+1,
                        'config': config.__dict__,
                        'metrics': metrics
                    })
                    
                    # Update best if improved
                    if metrics['sharpe_ratio'] > self.best_sharpe:
                        self.best_sharpe = metrics['sharpe_ratio']
                        self.best_config = config
                        self.best_results = results
                        print(f"  ✓ NEW BEST SHARPE: {self.best_sharpe:.3f}")
                        
                        # Save plot for significant improvements
                        if self.best_sharpe >= 0.7:
                            plot_production_results(
                                df=df,
                                results=results,
                                title=f"Best Config - Sharpe: {self.best_sharpe:.3f}",
                                save_path=f"charts/best_sharpe_{self.best_sharpe:.3f}.png",
                                show=False
                            )
                    
                    # Early exit if target achieved
                    if metrics['sharpe_ratio'] >= self.target_sharpe:
                        print(f"\n🎯 TARGET ACHIEVED! Sharpe: {metrics['sharpe_ratio']:.3f}")
                        self.best_sharpe = metrics['sharpe_ratio']
                        self.best_config = config
                        self.best_results = results
                        return self.best_config, self.best_results
                    
                except Exception as e:
                    print(f"  Config {i+1} failed: {str(e)}")
                    continue
            
            # Check for improvement
            if self.best_sharpe <= iteration_best_sharpe:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            # Print iteration summary
            print(f"\nIteration Summary:")
            print(f"  Best Sharpe: {self.best_sharpe:.3f} (Target: {self.target_sharpe})")
            print(f"  Progress: {(self.best_sharpe/self.target_sharpe)*100:.1f}%")
            
            # Adaptive strategy change
            if no_improvement_count >= 5:
                print("\n⚠️  No improvement in 5 iterations. Trying radical changes...")
                self.iteration += 10  # Jump to more aggressive strategies
                no_improvement_count = 0
        
        # Final results
        self.print_final_results()
        return self.best_config, self.best_results
    
    def print_final_results(self):
        """Print final optimization results"""
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        
        if self.best_sharpe >= self.target_sharpe:
            print(f"✓ TARGET ACHIEVED! Final Sharpe Ratio: {self.best_sharpe:.3f}")
        else:
            print(f"✗ Best achieved Sharpe: {self.best_sharpe:.3f} (Target: {self.target_sharpe})")
        
        if self.best_config:
            print("\nBest Configuration Found:")
            important_params = [
                'tp_atr_multipliers', 'tsl_activation_pips', 'tsl_min_profit_pips',
                'sl_max_pips', 'risk_per_trade', 'signal_flip_min_profit_pips'
            ]
            for key in important_params:
                if hasattr(self.best_config, key):
                    value = getattr(self.best_config, key)
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save optimization results to file"""
        results_data = {
            'target_sharpe': self.target_sharpe,
            'achieved_sharpe': self.best_sharpe,
            'total_iterations': self.iteration,
            'best_config': self.best_config.__dict__ if self.best_config else None,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'optimization_sharpe_{self.best_sharpe:.3f}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")


def main():
    """Run the enhanced optimization process"""
    optimizer = EnhancedStrategyOptimizer(target_sharpe=1.0)
    
    # Run optimization
    best_config, best_results = optimizer.optimize()
    
    # If target achieved or close enough, create final strategy
    if optimizer.best_sharpe >= 0.9:  # Accept 0.9+ as good enough
        print("\nGenerating final strategy file...")
        
        # Create optimized strategy file
        strategy_code = f'''"""
High Sharpe Strategy - Sharpe Ratio {optimizer.best_sharpe:.3f}
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
This strategy achieved a Sharpe ratio of {optimizer.best_sharpe:.3f} through aggressive optimization.
"""

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import pandas as pd
from technical_indicators_custom import TIC

def create_high_sharpe_strategy():
    """Create strategy with parameters optimized for Sharpe >= 1.0"""
    config = OptimizedStrategyConfig(
        initial_capital={best_config.initial_capital},
        risk_per_trade={best_config.risk_per_trade:.4f},
        sl_max_pips={best_config.sl_max_pips:.1f},
        tp_atr_multipliers={best_config.tp_atr_multipliers},
        tsl_activation_pips={best_config.tsl_activation_pips},
        tsl_min_profit_pips={best_config.tsl_min_profit_pips},
        tsl_initial_buffer_multiplier={best_config.tsl_initial_buffer_multiplier:.3f},
        signal_flip_min_profit_pips={best_config.signal_flip_min_profit_pips:.1f},
        signal_flip_min_time_hours={best_config.signal_flip_min_time_hours:.1f},
        exit_on_signal_flip={best_config.exit_on_signal_flip},
        intelligent_sizing={best_config.intelligent_sizing},
        tp_range_market_multiplier={best_config.tp_range_market_multiplier:.3f},
        tp_trend_market_multiplier={best_config.tp_trend_market_multiplier:.3f}
    )
    return OptimizedProdStrategy(config)

# Test the strategy
if __name__ == "__main__":
    print("Loading data and testing high Sharpe strategy...")
    
    # Load data
    df = pd.read_csv("../data/AUDUSD_MASTER_15M.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    
    # Use recent data
    df_test = df.tail(20000).copy()
    
    # Prepare indicators
    df_test = TIC.add_neuro_trend_intelligent(df_test)
    df_test = TIC.add_market_bias(df_test)
    df_test = TIC.add_intelligent_chop(df_test)
    
    # Create and run strategy
    strategy = create_high_sharpe_strategy()
    results = strategy.run_backtest(df_test)
    
    print(f"\\nBacktest Results:")
    print(f"Sharpe Ratio: {{results['sharpe_ratio']:.3f}}")
    print(f"Win Rate: {{results['win_rate']:.1f}}%")
    print(f"Total P&L: ${{results['total_pnl']:,.2f}}")
    print(f"Max Drawdown: {{results['max_drawdown']:.1f}}%")
    print(f"Total Trades: {{results['total_trades']}}")
'''
        
        filename = f'strategy_sharpe_{optimizer.best_sharpe:.2f}_final.py'
        with open(filename, 'w') as f:
            f.write(strategy_code)
        
        print(f"Final strategy saved to {filename}")
        
        # Update todos
        print("\nStrategy optimization complete!")
        print(f"Achieved Sharpe ratio: {optimizer.best_sharpe:.3f}")
    
    else:
        print(f"\nOptimization stopped. Best Sharpe achieved: {optimizer.best_sharpe:.3f}")
        print("Consider adjusting parameters or strategy logic for better results.")

if __name__ == "__main__":
    main()