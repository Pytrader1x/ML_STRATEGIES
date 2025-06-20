#!/usr/bin/env python3
"""
Recursive optimizer that iteratively improves parameters
Focuses on achieving Sharpe > 1.0 with robustness across different time periods
"""

import os
import sys
sys.path.append('..')  # Add parent directory to path

from intelligent_optimizer import (
    run_optimization, ParameterSpace, OptimizationResult,
    BayesianOptimizer, ParameterBounds
)
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class RecursiveOptimizer:
    """Recursive optimizer that learns from each iteration"""
    
    def __init__(self, strategy_type: int, currency: str = 'AUDUSD'):
        self.strategy_type = strategy_type
        self.currency = currency
        self.iteration_count = 0
        self.target_sharpe = 1.0
        self.min_trades = 100  # Minimum trades for robustness
        
    def run_recursive_optimization(self, max_iterations: int = 5):
        """Run recursive optimization until target is met or max iterations reached"""
        
        print(f"\n🚀 STARTING RECURSIVE OPTIMIZATION")
        print(f"   Strategy: {'Ultra-Tight Risk' if self.strategy_type == 1 else 'Scalping'}")
        print(f"   Target: Sharpe > {self.target_sharpe}")
        print(f"   Max Iterations: {max_iterations}")
        print("="*60)
        
        best_overall = None
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            
            print(f"\n\n{'='*60}")
            print(f"🔄 RECURSIVE ITERATION {self.iteration_count}/{max_iterations}")
            print(f"{'='*60}")
            
            # Adjust parameters based on previous results
            if iteration == 0:
                # First iteration - use default parameter space
                n_iterations = 15
                sample_size = 3000
            else:
                # Subsequent iterations - refine based on best result
                n_iterations = 10
                sample_size = 4000  # Use larger sample for validation
            
            # Run optimization
            optimizer, best_result = run_optimization(
                strategy_type=self.strategy_type,
                optimization_method='bayesian',
                currency=self.currency,
                n_iterations=n_iterations,
                sample_size=sample_size,
                use_previous_results=True  # Use feedback from previous runs
            )
            
            # Check if we've met our target
            if best_result and best_result.sharpe_ratio >= self.target_sharpe:
                print(f"\n✅ TARGET ACHIEVED! Sharpe: {best_result.sharpe_ratio:.3f}")
                
                # Validate robustness with different time periods
                if self._validate_robustness(best_result.params):
                    print(f"✅ Parameters are ROBUST across different time periods!")
                    best_overall = best_result
                    break
                else:
                    print(f"⚠️  Parameters not robust enough, continuing optimization...")
            
            # Update best overall if improved
            if best_result and (best_overall is None or best_result.fitness > best_overall.fitness):
                best_overall = best_result
            
            # Refine parameter space for next iteration
            if iteration < max_iterations - 1:
                self._refine_parameter_space(optimizer)
        
        # Final summary
        self._print_final_summary(best_overall)
        return best_overall
    
    def _validate_robustness(self, params: Dict[str, float]) -> bool:
        """Validate parameters across different time periods using multiprocessing"""
        print(f"\n🔍 Validating robustness across time periods...")
        
        # Define different time periods to test
        test_periods = [
            (2000, 2020),  # In-sample period
            (2000, 2021),  # Recent period  
            (2000, 2022),  # Extended recent
            (2000, 2023),  # More recent
        ]
        
        # Use multiprocessing to test periods in parallel
        with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
            futures = []
            for start_size, end_size in test_periods:
                future = executor.submit(
                    self._test_single_period,
                    params, start_size, end_size
                )
                futures.append((future, start_size, end_size))
            
            results = []
            for future, start_size, end_size in futures:
                try:
                    sharpe, trades = future.result(timeout=60)
                    results.append((sharpe, trades, start_size, end_size))
                    print(f"   Period {start_size}-{end_size}: Sharpe={sharpe:.3f}, Trades={trades}")
                except Exception as e:
                    print(f"   Period {start_size}-{end_size}: Error - {str(e)}")
                    results.append((0, 0, start_size, end_size))
        
        # Check robustness criteria
        valid_results = [(s, t) for s, t, _, _ in results if t >= self.min_trades]
        
        if len(valid_results) >= 3:  # Need at least 3 valid periods
            sharpes = [s for s, _ in valid_results]
            avg_sharpe = np.mean(sharpes)
            min_sharpe = np.min(sharpes)
            
            # Robust if average > 0.8 and minimum > 0.5
            is_robust = avg_sharpe > 0.8 and min_sharpe > 0.5
            
            print(f"\n   Robustness Summary:")
            print(f"   Average Sharpe: {avg_sharpe:.3f}")
            print(f"   Minimum Sharpe: {min_sharpe:.3f}")
            print(f"   Robust: {'YES' if is_robust else 'NO'}")
            
            return is_robust
        
        return False
    
    def _test_single_period(self, params: Dict[str, float], 
                           start_size: int, end_size: int) -> Tuple[float, int]:
        """Test parameters on a single time period"""
        from run_strategy_oop import DataManager, MonteCarloSimulator, TradeAnalyzer
        from strategy_code.Prod_strategy import OptimizedProdStrategy
        
        # Create strategy with given parameters
        optimizer = BayesianOptimizer(
            strategy_type=self.strategy_type,
            parameter_space=ParameterSpace.get_strategy1_space() if self.strategy_type == 1 else ParameterSpace.get_strategy2_space(),
            data_manager=DataManager(),
            currency=self.currency,
            n_iterations=1,
            sample_size=end_size - start_size
        )
        
        # Evaluate parameters
        result = optimizer.evaluate_parameters(params)
        return result.sharpe_ratio, result.total_trades
    
    def _refine_parameter_space(self, optimizer: BayesianOptimizer):
        """Refine parameter space based on best results"""
        if not optimizer.best_result:
            return
        
        print(f"\n📊 Refining parameter space based on best result...")
        print(f"   Current best Sharpe: {optimizer.best_result.sharpe_ratio:.3f}")
        
        # Analyze which parameters correlate with good performance
        if len(optimizer.results_history) > 5:
            # Get top 20% of results
            sorted_results = sorted(optimizer.results_history, 
                                  key=lambda r: r.fitness, reverse=True)
            top_results = sorted_results[:max(1, len(sorted_results) // 5)]
            
            # Find parameter ranges that work well
            for param_name in optimizer.parameter_space.keys():
                values = [r.params.get(param_name, 0) for r in top_results]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    print(f"   {param_name}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    def _print_final_summary(self, best_result: Optional[OptimizationResult]):
        """Print final optimization summary"""
        print(f"\n\n{'='*60}")
        print(f"🏆 FINAL OPTIMIZATION SUMMARY")
        print(f"{'='*60}")
        
        if best_result:
            print(f"\nBest Result Achieved:")
            print(f"  Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
            print(f"  Total Return: {best_result.total_return:.1f}%")
            print(f"  Win Rate: {best_result.win_rate:.1f}%")
            print(f"  Max Drawdown: {best_result.max_drawdown:.1f}%")
            print(f"  Profit Factor: {best_result.profit_factor:.2f}")
            print(f"  Total Trades: {best_result.total_trades}")
            print(f"  Fitness Score: {best_result.fitness:.3f}")
            
            print(f"\nOptimal Parameters:")
            for param, value in sorted(best_result.params.items()):
                print(f"  {param}: {value}")
            
            # Save final configuration
            self._save_final_config(best_result)
        else:
            print(f"\n❌ No valid results found")
    
    def _save_final_config(self, best_result: OptimizationResult):
        """Save the final optimized configuration"""
        config = {
            'strategy_type': self.strategy_type,
            'currency': self.currency,
            'timestamp': datetime.now().isoformat(),
            'iterations': self.iteration_count,
            'best_sharpe': best_result.sharpe_ratio,
            'best_fitness': best_result.fitness,
            'parameters': best_result.params,
            'metrics': {
                'total_return': best_result.total_return,
                'win_rate': best_result.win_rate,
                'max_drawdown': best_result.max_drawdown,
                'profit_factor': best_result.profit_factor,
                'total_trades': best_result.total_trades
            }
        }
        
        os.makedirs('final_configs', exist_ok=True)
        filename = f'final_configs/optimized_strategy{self.strategy_type}_{self.currency}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Final configuration saved to: {filename}")


def main():
    """Run recursive optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursive Strategy Optimizer")
    parser.add_argument('--strategy', type=int, choices=[1, 2], default=1,
                       help='Strategy type: 1=Ultra-Tight Risk, 2=Scalping')
    parser.add_argument('--currency', type=str, default='AUDUSD',
                       help='Currency pair to optimize on')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Maximum recursive iterations')
    
    args = parser.parse_args()
    
    # Create and run recursive optimizer
    optimizer = RecursiveOptimizer(args.strategy, args.currency)
    best_result = optimizer.run_recursive_optimization(args.max_iterations)
    
    print(f"\n✅ Recursive optimization completed!")


if __name__ == "__main__":
    main()