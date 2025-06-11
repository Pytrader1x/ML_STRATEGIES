"""
Recursive Strategy Optimizer - Target Sharpe Ratio 1.0
This script will continuously optimize parameters until achieving Sharpe >= 1.0
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

class StrategyOptimizer:
    def __init__(self, target_sharpe=1.0):
        self.target_sharpe = target_sharpe
        self.iteration = 0
        self.best_config = None
        self.best_sharpe = -np.inf
        self.best_results = None
        self.optimization_history = []
        
    def load_and_prepare_data(self, sample_size=10000):
        """Load data and prepare a sample for testing"""
        print(f"Loading data for iteration {self.iteration}...")
        df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Use recent data for more relevant results
        df_sample = df.tail(sample_size).copy()
        
        # Calculate indicators
        df_sample = TIC.add_neuro_trend_intelligent(df_sample, base_fast=10, base_slow=50, confirm_bars=3)
        df_sample = TIC.add_market_bias(df_sample, ha_len=350, ha_len2=30)
        df_sample = TIC.add_intelligent_chop(df_sample)
        
        return df_sample
    
    def generate_config_variations(self, base_config):
        """Generate parameter variations based on previous results"""
        variations = []
        
        # Base parameters to optimize
        param_ranges = {
            'tp_atr_multipliers': [
                (0.5, 1.5, 3.0),  # Tight TPs
                (0.8, 2.0, 4.0),  # Medium TPs
                (1.0, 2.5, 5.0),  # Wide TPs
                (0.6, 1.8, 3.5),  # Balanced
                (0.7, 1.5, 2.5),  # Very tight
            ],
            'tsl_activation_pips': [10, 15, 20, 25, 30],
            'tsl_min_profit_pips': [3, 5, 7, 10],
            'sl_max_pips': [25, 30, 35, 40, 45],
            'signal_flip_min_profit_pips': [0, 3, 5, 7, 10],
            'signal_flip_min_time_hours': [0.5, 1, 2, 3, 4],
            'tsl_initial_buffer_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
            'risk_per_trade': [0.01, 0.015, 0.02, 0.025, 0.03],
            'signal_flip_partial_exit_percent': [0.3, 0.5, 0.7],
            'tp_range_market_multiplier': [0.5, 0.7, 0.9],
            'tp_trend_market_multiplier': [0.8, 1.0, 1.2],
            'sl_atr_multiplier': [1.5, 2.0, 2.5],
        }
        
        # Generate variations based on iteration
        if self.iteration == 0:
            # Start with default config
            variations.append(base_config)
        elif self.iteration < 5:
            # Try different TP multipliers
            for tp_mults in param_ranges['tp_atr_multipliers']:
                config = OptimizedStrategyConfig()
                config.tp_atr_multipliers = tp_mults
                variations.append(config)
        elif self.iteration < 10:
            # Optimize TSL parameters with best TP settings
            best_tp = self.best_config.tp_atr_multipliers if self.best_config else (0.8, 1.5, 2.5)
            for tsl_act in param_ranges['tsl_activation_pips']:
                for tsl_min in param_ranges['tsl_min_profit_pips']:
                    if tsl_min < tsl_act:
                        config = OptimizedStrategyConfig()
                        config.tp_atr_multipliers = best_tp
                        config.tsl_activation_pips = tsl_act
                        config.tsl_min_profit_pips = tsl_min
                        variations.append(config)
        elif self.iteration < 15:
            # Optimize risk and SL with best settings so far
            if self.best_config:
                for risk in param_ranges['risk_per_trade']:
                    for sl_max in param_ranges['sl_max_pips']:
                        config = OptimizedStrategyConfig()
                        # Copy best settings
                        config.tp_atr_multipliers = self.best_config.tp_atr_multipliers
                        config.tsl_activation_pips = self.best_config.tsl_activation_pips
                        config.tsl_min_profit_pips = self.best_config.tsl_min_profit_pips
                        # New variations
                        config.risk_per_trade = risk
                        config.sl_max_pips = sl_max
                        variations.append(config)
        elif self.iteration < 25:
            # Optimize signal flip and market conditions
            if self.best_config:
                for flip_min in param_ranges['signal_flip_min_profit_pips']:
                    for flip_time in param_ranges['signal_flip_min_time_hours']:
                        config = OptimizedStrategyConfig()
                        # Copy best settings
                        for key, value in self.best_config.__dict__.items():
                            setattr(config, key, value)
                        # New variations
                        config.signal_flip_min_profit_pips = flip_min
                        config.signal_flip_min_time_hours = flip_time
                        variations.append(config)
        else:
            # Fine-tune best parameters with smaller variations
            if self.best_config:
                # Create variations around best config
                for i in range(10):
                    config = OptimizedStrategyConfig()
                    # Copy all settings
                    for key, value in self.best_config.__dict__.items():
                        setattr(config, key, value)
                    
                    # Randomly adjust parameters by small amounts
                    if np.random.random() > 0.5:
                        tp1, tp2, tp3 = config.tp_atr_multipliers
                        config.tp_atr_multipliers = (
                            tp1 * np.random.uniform(0.9, 1.1),
                            tp2 * np.random.uniform(0.9, 1.1),
                            tp3 * np.random.uniform(0.9, 1.1)
                        )
                    if np.random.random() > 0.5:
                        config.tsl_activation_pips = int(config.tsl_activation_pips * np.random.uniform(0.9, 1.1))
                    if np.random.random() > 0.5:
                        config.risk_per_trade *= np.random.uniform(0.95, 1.05)
                    if np.random.random() > 0.5:
                        config.sl_max_pips = int(config.sl_max_pips * np.random.uniform(0.9, 1.1))
                    variations.append(config)
        
        return variations[:5]  # Limit to 5 variations per iteration
    
    def run_backtest(self, df, config):
        """Run a single backtest with given configuration"""
        strategy = OptimizedProdStrategy(config)
        results = strategy.run_backtest(df)
        return results
    
    def evaluate_results(self, results):
        """Extract key metrics from results"""
        return {
            'sharpe_ratio': results['sharpe_ratio'],
            'total_pnl': results['total_pnl'],
            'win_rate': results['win_rate'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades'],
            'profit_factor': results.get('profit_factor', 0)
        }
    
    def optimize(self):
        """Main optimization loop"""
        print("="*80)
        print("STRATEGY OPTIMIZER - TARGET SHARPE RATIO: 1.0")
        print("="*80)
        
        # Load data once
        df = self.load_and_prepare_data(sample_size=20000)  # Use more data for better results
        
        # Start with default config
        base_config = OptimizedStrategyConfig(
            initial_capital=100_000,
            risk_per_trade=0.02,
            sl_max_pips=45.0,
            intelligent_sizing=True,
            tsl_initial_buffer_multiplier=2.0
        )
        
        while self.best_sharpe < self.target_sharpe and self.iteration < 50:
            self.iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}")
            print(f"{'='*60}")
            
            # Generate config variations
            configs = self.generate_config_variations(base_config if not self.best_config else self.best_config)
            
            # Test each configuration
            for i, config in enumerate(configs):
                print(f"\nTesting configuration {i+1}/{len(configs)}...")
                
                try:
                    # Run backtest
                    results = self.run_backtest(df, config)
                    metrics = self.evaluate_results(results)
                    
                    # Print results
                    print(f"  Sharpe: {metrics['sharpe_ratio']:.3f}")
                    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
                    print(f"  P&L: ${metrics['total_pnl']:,.0f}")
                    print(f"  Max DD: {metrics['max_drawdown']:.1f}%")
                    
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
                        
                        # Save plot for best result
                        if self.best_sharpe >= 0.8:  # Only plot when getting close
                            plot_production_results(
                                df=df,
                                results=results,
                                title=f"Optimization Progress - Sharpe: {self.best_sharpe:.3f}",
                                save_path=f"charts/optimization_sharpe_{self.best_sharpe:.3f}.png",
                                show=False  # Don't show, just save
                            )
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue
            
            # Print iteration summary
            print(f"\nIteration {self.iteration} Summary:")
            print(f"  Best Sharpe so far: {self.best_sharpe:.3f}")
            print(f"  Target: {self.target_sharpe}")
            print(f"  Progress: {(self.best_sharpe/self.target_sharpe)*100:.1f}%")
            
            # Adjust strategy based on results
            if self.iteration % 5 == 0:
                self.analyze_and_adjust()
        
        # Final results
        self.print_final_results()
        
        return self.best_config, self.best_results
    
    def analyze_and_adjust(self):
        """Analyze results and suggest adjustments"""
        print("\nAnalyzing performance patterns...")
        
        # Get recent results
        recent = self.optimization_history[-20:]
        
        # Find patterns
        high_sharpe = [h for h in recent if h['metrics']['sharpe_ratio'] > 0.5]
        if high_sharpe:
            # Analyze what works
            avg_tsl_act = np.mean([h['config']['tsl_activation_pips'] for h in high_sharpe])
            avg_risk = np.mean([h['config']['risk_per_trade'] for h in high_sharpe])
            print(f"  High performing configs tend to have:")
            print(f"    - TSL activation around: {avg_tsl_act:.0f} pips")
            print(f"    - Risk per trade around: {avg_risk:.3f}")
    
    def print_final_results(self):
        """Print final optimization results"""
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        
        if self.best_sharpe >= self.target_sharpe:
            print(f"✓ TARGET ACHIEVED! Sharpe Ratio: {self.best_sharpe:.3f}")
        else:
            print(f"✗ Target not achieved. Best Sharpe: {self.best_sharpe:.3f}")
        
        if self.best_config:
            print("\nBest Configuration:")
            for key, value in self.best_config.__dict__.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Save results to file
        self.save_results()
    
    def save_results(self):
        """Save optimization results to file"""
        results_data = {
            'target_sharpe': self.target_sharpe,
            'achieved_sharpe': self.best_sharpe,
            'total_iterations': self.iteration,
            'best_config': self.best_config.__dict__ if self.best_config else None,
            'optimization_history': self.optimization_history[-10:],  # Last 10 results
            'timestamp': datetime.now().isoformat()
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\nResults saved to optimization_results.json")


def main():
    """Run the optimization process"""
    optimizer = StrategyOptimizer(target_sharpe=1.0)
    
    # Run optimization
    best_config, best_results = optimizer.optimize()
    
    # If target achieved, create final version
    if optimizer.best_sharpe >= optimizer.target_sharpe:
        print("\nGenerating final strategy file...")
        
        # Create optimized strategy file
        strategy_code = f'''"""
Optimized Strategy - Sharpe Ratio {optimizer.best_sharpe:.3f}
Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import pandas as pd
from technical_indicators_custom import TIC

def create_sharpe_optimized_strategy():
    """Create strategy with optimized parameters achieving Sharpe >= 1.0"""
    config = OptimizedStrategyConfig(
        initial_capital={best_config.initial_capital},
        risk_per_trade={best_config.risk_per_trade:.3f},
        sl_max_pips={best_config.sl_max_pips:.1f},
        tp_atr_multipliers={best_config.tp_atr_multipliers},
        tsl_activation_pips={best_config.tsl_activation_pips},
        tsl_min_profit_pips={best_config.tsl_min_profit_pips},
        tsl_initial_buffer_multiplier={best_config.tsl_initial_buffer_multiplier:.3f},
        signal_flip_min_profit_pips={best_config.signal_flip_min_profit_pips:.1f},
        signal_flip_min_time_hours={best_config.signal_flip_min_time_hours:.1f},
        intelligent_sizing={best_config.intelligent_sizing}
    )
    return OptimizedProdStrategy(config)

# Usage example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/AUDUSD_MASTER_15M.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    
    # Prepare indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create and run strategy
    strategy = create_sharpe_optimized_strategy()
    results = strategy.run_backtest(df.tail(10000))
    
    print(f"Sharpe Ratio: {{results['sharpe_ratio']:.3f}}")
    print(f"Total P&L: ${{results['total_pnl']:,.2f}}")
'''
        
        with open('strategy_sharpe_1_0_final.py', 'w') as f:
            f.write(strategy_code)
        
        print("Final strategy saved to strategy_sharpe_1_0_final.py")
    
    print("\nOptimization process complete!")

if __name__ == "__main__":
    main()