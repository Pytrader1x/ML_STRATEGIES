#!/usr/bin/env python3
"""
Fast Recursive Self-Improving Optimizer
Optimized for speed while maintaining robustness testing
"""

import os
import sys
import json
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from optimizer.intelligent_optimizer import ParameterBounds
from run_strategy_oop import DataManager
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig


class FastRecursiveOptimizer:
    """Fast self-improving optimizer with reduced evaluation time"""
    
    def __init__(self, strategy_type: int = 1, currency: str = 'AUDUSD'):
        self.strategy_type = strategy_type
        self.currency = currency
        self.n_cores = mp.cpu_count()
        
        # Load data
        print(f"Loading {currency} data...")
        self.data_manager = DataManager()
        self.df = self.data_manager.load_currency_data(currency)
        print(f"Loaded {len(self.df):,} rows. Using {self.n_cores} CPU cores.")
        
        # Fast evaluation settings
        self.sample_size = 30000  # Smaller samples for speed
        self.n_periods = 3  # Fewer periods but still robust
        self.n_configs_per_gen = 10  # Fewer configs per generation
        
        # Track learning
        self.history = []
        self.best_params = None
        self.best_sharpe = -float('inf')
        
    def run_fast_optimization(self, n_generations: int = 40):
        """Run fast recursive optimization"""
        
        print(f"\n{'='*80}")
        print(f"🚀 FAST RECURSIVE OPTIMIZATION")
        print(f"   Generations: {n_generations}")
        print(f"   Configs per gen: {self.n_configs_per_gen}")
        print(f"   Sample size: {self.sample_size:,} rows")
        print(f"   Robustness periods: {self.n_periods}")
        print(f"{'='*80}")
        
        for gen in range(1, n_generations + 1):
            print(f"\n\n{'='*60}")
            print(f"🧬 GENERATION {gen}/{n_generations}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Get adaptive parameter space
            param_space = self._get_adaptive_space(gen)
            
            # Generate and test configurations
            results = self._test_generation(param_space, gen)
            
            # Update learning
            if results:
                best_gen = max(results, key=lambda x: x['robust_score'])
                self.history.append(best_gen)
                
                if best_gen['avg_sharpe'] > self.best_sharpe:
                    self.best_sharpe = best_gen['avg_sharpe']
                    self.best_params = best_gen['params']
                    print(f"\n🌟 New best! Avg Sharpe: {self.best_sharpe:.3f}")
                
                # Print generation summary
                print(f"\n📊 Generation {gen} Summary:")
                print(f"   Best Sharpe: {best_gen['avg_sharpe']:.3f}")
                print(f"   Min Sharpe: {best_gen['min_sharpe']:.3f}")
                print(f"   Trades/period: {best_gen['avg_trades']:.0f}")
                print(f"   Time: {time.time() - start_time:.1f}s")
                
                # Learn and adapt
                self._learn_from_results(results)
        
        self._final_report()
    
    def _get_adaptive_space(self, generation: int) -> Dict[str, ParameterBounds]:
        """Get parameter space that adapts each generation"""
        
        # Base ranges from analysis
        base_ranges = {
            'risk_per_trade': (0.002, 0.0035),
            'sl_min_pips': (5.0, 8.0),
            'sl_max_pips': (22.0, 28.0),  # Key insight: wider stops work
            'tp1_multiplier': (0.18, 0.28),  # Low TP1 for hit rate
            'tp2_multiplier': (0.25, 0.40),
            'trailing_atr_multiplier': (1.4, 1.8),
            'partial_profit_sl_distance_ratio': (0.35, 0.45),
            'partial_profit_size_percent': (0.65, 0.80),  # High partial profit
        }
        
        # Add more parameters for early generations
        if generation <= 2:
            base_ranges.update({
                'sl_atr_multiplier': (1.2, 2.0),
                'tp3_multiplier': (0.8, 1.1),
                'tsl_activation_pips': (12, 16),
                'tsl_min_profit_pips': (1.0, 2.5),
            })
        
        param_space = {}
        
        for param, (min_val, max_val) in base_ranges.items():
            # Narrow range around best after gen 2
            if generation > 2 and self.best_params and param in self.best_params:
                center = self.best_params[param]
                reduction = 0.3 - (generation - 3) * 0.05
                reduction = max(reduction, 0.1)
                
                range_width = (max_val - min_val) * reduction
                new_min = max(min_val, center - range_width/2)
                new_max = min(max_val, center + range_width/2)
            else:
                new_min, new_max = min_val, max_val
            
            param_type = int if param in ['tsl_activation_pips'] else float
            param_space[param] = ParameterBounds(param, new_min, new_max, param_type)
        
        return param_space
    
    def _test_generation(self, param_space: Dict[str, ParameterBounds], generation: int) -> List[Dict]:
        """Test configurations for this generation"""
        
        # Generate parameter sets
        param_sets = []
        for i in range(self.n_configs_per_gen):
            if i < 3 and self.best_params and generation > 1:
                # Exploit best known
                params = self._vary_around_best(param_space)
            else:
                # Explore
                params = {name: bounds.sample() for name, bounds in param_space.items()}
            
            # Always set use_intelligent_sizing to 0
            params['use_intelligent_sizing'] = 0
            param_sets.append(params)
        
        # Test in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Create random sample periods
            sample_periods = self._get_sample_periods()
            
            # Submit tasks
            futures = []
            for i, params in enumerate(param_sets):
                future = executor.submit(
                    evaluate_config_fast,
                    params, self.df, sample_periods, i
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                        if result['avg_sharpe'] > 0.8:
                            print(f"   ✅ Config {result['id']}: Sharpe {result['avg_sharpe']:.3f}")
                except Exception as e:
                    print(f"   ❌ Evaluation error: {str(e)}")
        
        return results
    
    def _get_sample_periods(self) -> List[Tuple[int, int]]:
        """Get random sample periods for testing"""
        periods = []
        max_start = len(self.df) - self.sample_size
        
        # Generate non-overlapping periods
        used_ranges = []
        attempts = 0
        
        while len(periods) < self.n_periods and attempts < 50:
            start = np.random.randint(0, max_start)
            end = start + self.sample_size
            
            # Check overlap
            overlap = False
            for used_start, used_end in used_ranges:
                if not (end < used_start or start > used_end):
                    overlap = True
                    break
            
            if not overlap:
                periods.append((start, end))
                used_ranges.append((start, end))
            
            attempts += 1
        
        return periods
    
    def _vary_around_best(self, param_space: Dict[str, ParameterBounds]) -> Dict:
        """Generate parameters by varying around best known"""
        params = {}
        
        for name, bounds in param_space.items():
            if name in self.best_params:
                center = self.best_params[name]
                std = (bounds.max_value - bounds.min_value) * 0.1
                value = np.clip(
                    np.random.normal(center, std),
                    bounds.min_value,
                    bounds.max_value
                )
                params[name] = bounds.round_value(value)
            else:
                params[name] = bounds.sample()
        
        return params
    
    def _learn_from_results(self, results: List[Dict]):
        """Learn patterns from results"""
        if len(results) < 3:
            return
        
        # Sort by robust score
        sorted_results = sorted(results, key=lambda x: x['robust_score'], reverse=True)
        
        # Identify convergence
        top_params = sorted_results[0]['params']
        
        print("\n🧠 Learning insights:")
        
        # Check parameter convergence
        converged_params = []
        for param in top_params:
            values = [r['params'].get(param, 0) for r in sorted_results[:3]]
            if np.std(values) < 0.1 * np.mean(values):
                converged_params.append(param)
        
        if converged_params:
            print(f"   Parameters converging: {', '.join(converged_params[:5])}")
        
        # Performance insights
        avg_sharpe = np.mean([r['avg_sharpe'] for r in results])
        if avg_sharpe > 0.5:
            print(f"   Good parameter region found (avg Sharpe: {avg_sharpe:.3f})")
        
    def _final_report(self):
        """Generate final optimization report"""
        print(f"\n\n{'='*80}")
        print("🏆 OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        
        if self.best_params and self.best_sharpe > 0:
            print(f"\n✅ BEST CONFIGURATION:")
            print(f"   Sharpe Ratio: {self.best_sharpe:.3f}")
            
            # Find full result
            best_full = max(self.history, key=lambda x: x['avg_sharpe'])
            print(f"   Min Sharpe: {best_full['min_sharpe']:.3f}")
            print(f"   Avg Trades: {best_full['avg_trades']:.0f}")
            print(f"   Win Rate: {best_full['avg_win_rate']:.1f}%")
            
            print(f"\n📊 OPTIMAL PARAMETERS:")
            for param, value in sorted(self.best_params.items()):
                if param != 'use_intelligent_sizing':  # Skip this since it's always 0
                    print(f"   {param}: {value}")
            
            # Save configuration
            self._save_config(best_full)
            
            if self.best_sharpe > 1.0:
                print(f"\n🎉 SUCCESS! Achieved Sharpe > 1.0")
                print(f"   Ready for production deployment")
            else:
                print(f"\n⚡ Progress made. Consider:")
                print(f"   - Running more generations")
                print(f"   - Fine-tuning around these parameters")


    def _save_config(self, result: Dict):
        """Save best configuration"""
        config = {
            'strategy_type': self.strategy_type,
            'currency': self.currency,
            'optimization_complete': datetime.now().isoformat(),
            'best_result': result,
            'parameters': result['params']
        }
        
        os.makedirs('optimized_configs', exist_ok=True)
        filename = f'optimized_configs/fast_recursive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration saved to: {filename}")


def evaluate_config_fast(params: Dict, df, sample_periods: List[Tuple], config_id: int) -> Optional[Dict]:
    """Fast evaluation of a configuration"""
    try:
        sharpes = []
        trades_list = []
        returns = []
        win_rates = []
        
        for start, end in sample_periods:
            # Get sample
            sample_df = df.iloc[start:end].copy()
            
            # Create strategy
            config = create_fast_config(params)
            strategy = OptimizedProdStrategy(config)
            
            # Run backtest
            results = strategy.run_backtest(sample_df)
            
            sharpe = results.get('sharpe_ratio', -999)
            if sharpe > -900:
                sharpes.append(sharpe)
                trades_list.append(results.get('total_trades', 0))
                returns.append(results.get('total_return', 0))
                win_rates.append(results.get('win_rate', 0))
        
        if len(sharpes) >= 2:  # Need at least 2 valid results
            avg_sharpe = np.mean(sharpes)
            min_sharpe = np.min(sharpes)
            sharpe_std = np.std(sharpes) if len(sharpes) > 1 else 0
            
            # Robust score
            robust_score = avg_sharpe - 0.3 * sharpe_std
            
            return {
                'id': config_id,
                'params': params,
                'avg_sharpe': avg_sharpe,
                'min_sharpe': min_sharpe,
                'sharpe_std': sharpe_std,
                'robust_score': robust_score,
                'avg_trades': np.mean(trades_list),
                'avg_return': np.mean(returns),
                'avg_win_rate': np.mean(win_rates)
            }
        
        return None
        
    except Exception as e:
        return None


def create_fast_config(params: Dict) -> OptimizedStrategyConfig:
    """Create strategy config from parameters"""
    return OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=params.get('risk_per_trade', 0.002),
        sl_min_pips=params.get('sl_min_pips', 5.0),
        sl_max_pips=params.get('sl_max_pips', 25.0),
        sl_atr_multiplier=params.get('sl_atr_multiplier', 1.5),
        tp_atr_multipliers=(
            params.get('tp1_multiplier', 0.2),
            params.get('tp2_multiplier', 0.3),
            params.get('tp3_multiplier', 0.9)
        ),
        max_tp_percent=0.003,
        tsl_activation_pips=params.get('tsl_activation_pips', 14),
        tsl_min_profit_pips=params.get('tsl_min_profit_pips', 2),
        tsl_initial_buffer_multiplier=1.0,
        trailing_atr_multiplier=params.get('trailing_atr_multiplier', 1.5),
        tp_range_market_multiplier=0.5,  # Fixed based on analysis
        tp_trend_market_multiplier=1.0,  # Fixed
        tp_chop_market_multiplier=0.4,   # Fixed
        sl_range_market_multiplier=0.7,
        exit_on_signal_flip=False,
        partial_profit_before_sl=True,
        partial_profit_sl_distance_ratio=params.get('partial_profit_sl_distance_ratio', 0.4),
        partial_profit_size_percent=params.get('partial_profit_size_percent', 0.7),
        intelligent_sizing=False,  # Always False!
        sl_volatility_adjustment=True,
        relaxed_mode=False,
        realistic_costs=True,
        verbose=False,
        debug_decisions=False,
        use_daily_sharpe=True
    )


def main():
    """Run fast recursive optimization"""
    optimizer = FastRecursiveOptimizer(strategy_type=1, currency='AUDUSD')
    optimizer.run_fast_optimization(n_generations=5)


if __name__ == "__main__":
    main()