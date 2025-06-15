#!/usr/bin/env python3
"""
Fast iterative optimizer using multiprocessing for M3 Pro chip
Tests on random 50K training samples and validates on 20K samples
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Add parent directory to path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from optimizer.intelligent_optimizer import ParameterBounds, ParameterSpace, OptimizationResult
from run_strategy_oop import DataManager
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig


def evaluate_params_worker(args):
    """Worker function for parallel parameter evaluation"""
    params, df_dict, strategy_type, sample_type = args
    
    try:
        # Reconstruct dataframe from dict (for multiprocessing)
        import pandas as pd
        df = pd.DataFrame(df_dict['data'])
        df.index = pd.to_datetime(df_dict['index'])
        
        # Create strategy config
        config = create_strategy_config(params, strategy_type)
        strategy = OptimizedProdStrategy(config)
        
        # Run backtest
        results = strategy.run_backtest(df)
        
        # Return key metrics
        return {
            'params': params,
            'sharpe': results.get('sharpe_ratio', -999),
            'return': results.get('total_return', 0),
            'win_rate': results.get('win_rate', 0),
            'max_dd': results.get('max_drawdown', 100),
            'pf': results.get('profit_factor', 0),
            'trades': results.get('total_trades', 0),
            'sample_type': sample_type
        }
    except Exception as e:
        return {
            'params': params,
            'sharpe': -999,
            'error': str(e),
            'sample_type': sample_type
        }


def create_strategy_config(params: Dict[str, float], strategy_type: int) -> OptimizedStrategyConfig:
    """Create strategy configuration from parameters"""
    return OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=params.get('risk_per_trade', 0.002),
        sl_min_pips=params.get('sl_min_pips', 5.0),
        sl_max_pips=params.get('sl_max_pips', 10.0),
        sl_atr_multiplier=params.get('sl_atr_multiplier', 1.0),
        tp_atr_multipliers=(
            params.get('tp1_multiplier', 0.2),
            params.get('tp2_multiplier', 0.3),
            params.get('tp3_multiplier', 0.5)
        ),
        max_tp_percent=0.003,
        tsl_activation_pips=params.get('tsl_activation_pips', 15),
        tsl_min_profit_pips=params.get('tsl_min_profit_pips', 1),
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=params.get('trailing_atr_multiplier', 1.2),
        tp_range_market_multiplier=params.get('tp_range_market_multiplier', 0.5),
        tp_trend_market_multiplier=params.get('tp_trend_market_multiplier', 0.7),
        tp_chop_market_multiplier=params.get('tp_chop_market_multiplier', 0.3),
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=params.get('partial_profit_sl_distance_ratio', 0.5),
        partial_profit_size_percent=params.get('partial_profit_size_percent', 0.5),
        intelligent_sizing=bool(params.get('use_intelligent_sizing', 0)),
        sl_volatility_adjustment=True,
        relaxed_mode=False,
        realistic_costs=True,
        verbose=False,
        debug_decisions=False,
        use_daily_sharpe=True
    )


class FastIterativeOptimizer:
    """Fast optimizer using multiprocessing"""
    
    def __init__(self, strategy_type: int = 1, currency: str = 'AUDUSD'):
        self.strategy_type = strategy_type
        self.currency = currency
        self.train_size = 50000  # 50K for training
        self.val_size = 20000    # 20K for validation
        self.n_cores = mp.cpu_count()  # Use all cores
        self.procedure_file = 'OPTIMIZATION_PROCEDURE.md'
        
        # Load data once
        print(f"Loading {currency} data...")
        self.data_manager = DataManager()
        self.df = self.data_manager.load_currency_data(currency)
        print(f"Loaded {len(self.df):,} rows. Using {self.n_cores} CPU cores for optimization.")
        
        # Track results
        self.all_results = []
        self.best_params = None
        self.best_sharpe = -float('inf')
    
    def run_10_iterations(self):
        """Run 10 fast optimization iterations"""
        
        print(f"\n{'='*60}")
        print(f"🚀 FAST ITERATIVE OPTIMIZATION")
        print(f"   Strategy: {'Ultra-Tight Risk' if self.strategy_type == 1 else 'Scalping'}")
        print(f"   Train: {self.train_size:,} rows | Validate: {self.val_size:,} rows")
        print(f"   CPU Cores: {self.n_cores} | Target: Sharpe > 1.0")
        print(f"{'='*60}")
        
        for iteration in range(1, 11):
            print(f"\n\n{'='*60}")
            print(f"📊 ITERATION {iteration}/10")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Get focused parameters for this iteration
            param_space = self._get_iteration_parameters(iteration)
            
            # Run parallel optimization
            best_iteration_params = self._run_parallel_optimization(param_space, iteration)
            
            # Update procedure document
            self._update_procedure(iteration, best_iteration_params, time.time() - start_time)
            
            print(f"\n⏱️  Iteration {iteration} completed in {time.time() - start_time:.1f}s")
        
        self._print_final_summary()
    
    def _run_parallel_optimization(self, param_space: Dict[str, ParameterBounds], iteration: int) -> Dict:
        """Run optimization in parallel across multiple cores"""
        
        # Number of parameter sets to test
        n_param_sets = min(20, 120 // self.n_cores)  # Adjust based on cores
        
        print(f"\n🔄 Testing {n_param_sets} parameter combinations in parallel...")
        
        # Generate parameter sets
        param_sets = []
        for i in range(n_param_sets):
            params = self._generate_params(param_space, i, n_param_sets)
            param_sets.append(params)
        
        # Get training and validation samples
        train_sample = self._get_random_sample(self.train_size)
        val_sample = self._get_random_sample(self.val_size, exclude_start=train_sample['start_idx'])
        
        # Prepare data for multiprocessing (convert to dict for pickling)
        train_dict = {
            'data': train_sample['df'].reset_index().to_dict('list'),
            'index': train_sample['df'].index.tolist()
        }
        val_dict = {
            'data': val_sample['df'].reset_index().to_dict('list'),
            'index': val_sample['df'].index.tolist()
        }
        
        # Evaluate all parameter sets in parallel
        best_robust_score = -float('inf')
        best_params = None
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Submit training tasks
            train_tasks = [(p, train_dict, self.strategy_type, 'train') for p in param_sets]
            train_futures = [executor.submit(evaluate_params_worker, task) for task in train_tasks]
            
            # Collect training results
            train_results = {}
            for future in as_completed(train_futures):
                try:
                    result = future.result(timeout=30)
                    if result and result['sharpe'] > -900:
                        param_key = str(sorted(result['params'].items()))
                        train_results[param_key] = result
                except Exception as e:
                    print(f"   Error in training: {str(e)}")
            
            # Validate promising parameters (Sharpe > 0.5)
            promising_params = [r['params'] for r in train_results.values() if r['sharpe'] > 0.5]
            
            if promising_params:
                print(f"\n   Validating {len(promising_params)} promising parameters...")
                
                # Submit validation tasks
                val_tasks = [(p, val_dict, self.strategy_type, 'val') for p in promising_params]
                val_futures = [executor.submit(evaluate_params_worker, task) for task in val_tasks]
                
                # Collect validation results and find best
                for future in as_completed(val_futures):
                    try:
                        val_result = future.result(timeout=30)
                        if val_result and val_result['sharpe'] > -900:
                            param_key = str(sorted(val_result['params'].items()))
                            
                            if param_key in train_results:
                                train_sharpe = train_results[param_key]['sharpe']
                                val_sharpe = val_result['sharpe']
                                
                                # Calculate robust score
                                avg_sharpe = (train_sharpe + val_sharpe) / 2
                                sharpe_diff = abs(train_sharpe - val_sharpe)
                                robust_score = avg_sharpe - 0.3 * sharpe_diff
                                
                                print(f"      Train: {train_sharpe:.3f}, Val: {val_sharpe:.3f}, Robust: {robust_score:.3f}")
                                
                                # Update best if both > 0.8
                                if train_sharpe > 0.8 and val_sharpe > 0.8 and robust_score > best_robust_score:
                                    best_robust_score = robust_score
                                    best_params = val_result['params']
                                    
                                    # Update global best
                                    if avg_sharpe > self.best_sharpe:
                                        self.best_sharpe = avg_sharpe
                                        self.best_params = best_params
                                        print(f"      🌟 New best! Avg Sharpe: {avg_sharpe:.3f}")
                                
                                # Store result
                                self.all_results.append({
                                    'iteration': iteration,
                                    'params': val_result['params'],
                                    'train_sharpe': train_sharpe,
                                    'val_sharpe': val_sharpe,
                                    'avg_sharpe': avg_sharpe,
                                    'robust_score': robust_score,
                                    'trades': val_result['trades']
                                })
                    except Exception as e:
                        print(f"   Error in validation: {str(e)}")
        
        return best_params or (param_sets[0] if param_sets else {})
    
    def _get_random_sample(self, size: int, exclude_start: Optional[int] = None) -> Dict:
        """Get random sample from dataset"""
        max_start = len(self.df) - size
        
        if exclude_start is not None:
            # Avoid overlap with excluded region
            exclude_end = exclude_start + self.val_size + self.train_size
            
            # Get valid start positions
            valid_starts = list(range(0, exclude_start - size)) + list(range(exclude_end, max_start))
            
            if valid_starts:
                start_idx = np.random.choice(valid_starts)
            else:
                start_idx = np.random.randint(0, max_start)
        else:
            start_idx = np.random.randint(0, max_start)
        
        sample_df = self.df.iloc[start_idx:start_idx + size].copy()
        
        return {
            'df': sample_df,
            'start_idx': start_idx,
            'start_date': sample_df.index[0],
            'end_date': sample_df.index[-1]
        }
    
    def _generate_params(self, param_space: Dict[str, ParameterBounds], idx: int, total: int) -> Dict:
        """Generate parameter set with some intelligence"""
        params = {}
        
        # Use best params as reference 30% of the time after iteration 3
        use_best = self.best_params and np.random.random() < 0.3 and idx > total // 3
        
        for name, bounds in param_space.items():
            if use_best and name in self.best_params:
                # Vary around best
                center = self.best_params[name]
                variation = (bounds.max_value - bounds.min_value) * 0.2
                value = np.clip(
                    center + np.random.uniform(-variation, variation),
                    bounds.min_value,
                    bounds.max_value
                )
            else:
                # Random sampling with Latin Hypercube style distribution
                segment = (bounds.max_value - bounds.min_value) / total
                value = bounds.min_value + (idx + np.random.random()) * segment
                value = np.clip(value, bounds.min_value, bounds.max_value)
            
            params[name] = bounds.round_value(value)
        
        return params
    
    def _get_iteration_parameters(self, iteration: int) -> Dict[str, ParameterBounds]:
        """Get focused parameters for each iteration"""
        base = ParameterSpace.get_strategy1_space()
        
        if iteration == 1:
            # Core parameters only
            return {
                'risk_per_trade': base['risk_per_trade'],
                'sl_min_pips': base['sl_min_pips'],
                'sl_max_pips': base['sl_max_pips'],
                'tp1_multiplier': base['tp1_multiplier'],
            }
        elif iteration == 2:
            # Add more TP parameters
            return {
                'risk_per_trade': ParameterBounds('risk_per_trade', 0.002, 0.004, float, 0.0001),
                'tp1_multiplier': ParameterBounds('tp1_multiplier', 0.15, 0.35, float, 0.025),
                'tp2_multiplier': ParameterBounds('tp2_multiplier', 0.2, 0.5, float, 0.05),
                'tp3_multiplier': ParameterBounds('tp3_multiplier', 0.5, 1.0, float, 0.1),
            }
        elif iteration == 3:
            # Stop loss focus
            return {
                'sl_min_pips': ParameterBounds('sl_min_pips', 5, 10, float, 0.5),
                'sl_max_pips': ParameterBounds('sl_max_pips', 15, 30, float, 1),
                'sl_atr_multiplier': ParameterBounds('sl_atr_multiplier', 1.0, 2.5, float, 0.1),
            }
        elif iteration == 4:
            # Trailing stop
            return {
                'tsl_activation_pips': base['tsl_activation_pips'],
                'tsl_min_profit_pips': base['tsl_min_profit_pips'],
                'trailing_atr_multiplier': base['trailing_atr_multiplier'],
            }
        elif iteration == 5:
            # Partial profits
            return {
                'partial_profit_sl_distance_ratio': base['partial_profit_sl_distance_ratio'],
                'partial_profit_size_percent': base['partial_profit_size_percent'],
            }
        elif iteration <= 7:
            # Refine around best (if found)
            if self.best_params:
                return self._create_refined_space(0.3)
            else:
                return self._get_iteration_parameters(1)  # Fallback to iteration 1
        else:
            # Fine-tune
            if self.best_params:
                return self._create_refined_space(0.15)
            else:
                return self._get_iteration_parameters(2)  # Fallback
    
    def _create_refined_space(self, reduction: float) -> Dict[str, ParameterBounds]:
        """Create refined parameter space around best parameters"""
        if not self.best_params:
            return ParameterSpace.get_strategy1_space()
        
        base = ParameterSpace.get_strategy1_space()
        refined = {}
        
        key_params = ['risk_per_trade', 'sl_min_pips', 'sl_max_pips', 'tp1_multiplier', 'tp2_multiplier']
        
        for param in key_params:
            if param in base and param in self.best_params:
                bounds = base[param]
                center = self.best_params[param]
                range_width = (bounds.max_value - bounds.min_value) * reduction
                
                refined[param] = ParameterBounds(
                    param,
                    max(bounds.min_value, center - range_width/2),
                    min(bounds.max_value, center + range_width/2),
                    bounds.param_type,
                    bounds.step_size,
                    bounds.is_percentage
                )
        
        return refined
    
    def _update_procedure(self, iteration: int, best_params: Dict, elapsed_time: float):
        """Update procedure document"""
        if not best_params:
            return
        
        # Find best result for this iteration
        iteration_results = [r for r in self.all_results if r['iteration'] == iteration]
        if not iteration_results:
            return
        
        best = max(iteration_results, key=lambda r: r['robust_score'])
        
        print(f"\n📝 Updating procedure document...")
        
        # Read current content
        with open(self.procedure_file, 'r') as f:
            content = f.read()
        
        # Update the iteration section
        marker = f"### Run {iteration} -"
        if marker in content:
            start = content.find(marker)
            end = content.find(f"### Run {iteration + 1} -", start)
            if end == -1:
                end = content.find("## Best Configurations", start)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            update = f"""{marker} Completed ({timestamp})
**Time**: {elapsed_time:.1f}s
**Best Results**:
- Train Sharpe: {best['train_sharpe']:.3f}
- Val Sharpe: {best['val_sharpe']:.3f}
- Avg Sharpe: {best['avg_sharpe']:.3f}
- Trades: {best['trades']}

"""
            
            content = content[:start] + update + content[end:]
            
            with open(self.procedure_file, 'w') as f:
                f.write(content)
    
    def _print_final_summary(self):
        """Print final summary"""
        print(f"\n\n{'='*60}")
        print("🏆 OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        
        if self.best_params and self.best_sharpe > 0:
            print(f"\nBest Configuration:")
            print(f"  Average Sharpe: {self.best_sharpe:.3f}")
            print(f"\nBest Parameters:")
            for param, value in sorted(self.best_params.items()):
                print(f"  {param}: {value}")
            
            # Save best configuration
            self._save_best_config()
        else:
            print("\n❌ No configuration achieved target Sharpe > 1.0")
    
    def _save_best_config(self):
        """Save best configuration to file"""
        if not self.best_params:
            return
        
        config = {
            'strategy_type': self.strategy_type,
            'currency': self.currency,
            'best_sharpe': self.best_sharpe,
            'parameters': self.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('optimized_configs', exist_ok=True)
        filename = f'optimized_configs/best_config_strategy{self.strategy_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Best configuration saved to: {filename}")


def main():
    """Run fast iterative optimization"""
    optimizer = FastIterativeOptimizer(strategy_type=1, currency='AUDUSD')
    optimizer.run_10_iterations()


if __name__ == "__main__":
    main()