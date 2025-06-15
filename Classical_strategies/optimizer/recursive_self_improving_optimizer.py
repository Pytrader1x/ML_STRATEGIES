#!/usr/bin/env python3
"""
Recursive Self-Improving Optimizer
Learns from each optimization run to intelligently adjust search space
Target: Robust Sharpe > 1.0 across multiple time periods
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

from optimizer.intelligent_optimizer import ParameterBounds, OptimizationResult
from run_strategy_oop import DataManager
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig


class OptimizationMemory:
    """Stores and analyzes results from all optimization runs"""
    
    def __init__(self):
        self.all_results = []
        self.high_performers = []  # Sharpe > 1.0
        self.robust_performers = []  # Consistent across periods
        self.failed_patterns = []  # Parameter combinations to avoid
        self.insights = {}
        self.generation = 0
        
    def add_result(self, result: Dict):
        """Add a result and update insights"""
        self.all_results.append(result)
        
        # Track high performers
        if result.get('avg_sharpe', 0) > 1.0:
            self.high_performers.append(result)
            
        # Track robust performers (low variance across periods)
        if result.get('min_sharpe', 0) > 0.8 and result.get('sharpe_std', 1) < 0.3:
            self.robust_performers.append(result)
    
    def get_best_params(self) -> Optional[Dict]:
        """Get best parameters prioritizing robustness"""
        if self.robust_performers:
            return max(self.robust_performers, key=lambda x: x['avg_sharpe'])['params']
        elif self.high_performers:
            return max(self.high_performers, key=lambda x: x['avg_sharpe'])['params']
        elif self.all_results:
            return max(self.all_results, key=lambda x: x.get('avg_sharpe', -999))['params']
        return None
    
    def analyze_patterns(self) -> Dict:
        """Analyze patterns in successful vs failed configurations"""
        if len(self.all_results) < 5:
            return {}
        
        # Separate winners and losers
        winners = [r for r in self.all_results if r.get('avg_sharpe', 0) > 0.8]
        losers = [r for r in self.all_results if r.get('avg_sharpe', 0) < 0]
        
        insights = {}
        
        if winners and losers:
            # Find parameters that differ most
            all_params = set()
            for r in winners + losers:
                all_params.update(r['params'].keys())
            
            for param in all_params:
                winner_vals = [r['params'].get(param, 0) for r in winners]
                loser_vals = [r['params'].get(param, 0) for r in losers]
                
                if winner_vals and loser_vals:
                    winner_mean = np.mean(winner_vals)
                    loser_mean = np.mean(loser_vals)
                    
                    # Calculate impact
                    if loser_mean != 0:
                        diff_pct = (winner_mean - loser_mean) / abs(loser_mean) * 100
                    else:
                        diff_pct = 100 if winner_mean > 0 else -100
                    
                    insights[param] = {
                        'winner_mean': winner_mean,
                        'loser_mean': loser_mean,
                        'diff_pct': diff_pct,
                        'recommendation': 'increase' if diff_pct > 20 else 'decrease' if diff_pct < -20 else 'neutral'
                    }
        
        self.insights = insights
        return insights


class SelfImprovingOptimizer:
    """Self-improving optimizer that learns from each generation"""
    
    def __init__(self, strategy_type: int = 1, currency: str = 'AUDUSD'):
        self.strategy_type = strategy_type
        self.currency = currency
        self.memory = OptimizationMemory()
        self.n_cores = mp.cpu_count()
        self.generation = 0
        self.improvement_threshold = 0.1  # 10% improvement needed
        
        # Load data
        print(f"Loading {currency} data...")
        self.data_manager = DataManager()
        self.df = self.data_manager.load_currency_data(currency)
        print(f"Loaded {len(self.df):,} rows. Using {self.n_cores} CPU cores.")
        
        # Base parameter space (will be refined each generation)
        self.base_param_space = self._get_initial_param_space()
        
    def _get_initial_param_space(self) -> Dict[str, Tuple[float, float]]:
        """Initial parameter space based on analysis"""
        return {
            'risk_per_trade': (0.002, 0.0035),      # 0.2-0.35% based on analysis
            'sl_min_pips': (5.0, 8.0),              # Tight minimum
            'sl_max_pips': (22.0, 28.0),            # Wide maximum (key insight!)
            'sl_atr_multiplier': (1.2, 2.0),        
            'tp1_multiplier': (0.18, 0.28),         # Low TP1 for hit rate
            'tp2_multiplier': (0.25, 0.40),         
            'tp3_multiplier': (0.75, 1.1),          
            'tsl_activation_pips': (12, 16),        
            'tsl_min_profit_pips': (1.0, 2.5),      # Lower is better
            'trailing_atr_multiplier': (1.3, 1.8),   
            'tp_range_market_multiplier': (0.4, 0.6), # Lower is better
            'tp_trend_market_multiplier': (0.9, 1.2),
            'tp_chop_market_multiplier': (0.35, 0.5),
            'partial_profit_sl_distance_ratio': (0.35, 0.45),
            'partial_profit_size_percent': (0.65, 0.80),  # High is better
            'use_intelligent_sizing': (0, 0)  # Always OFF based on analysis!
        }
    
    def run_recursive_optimization(self, n_generations: int = 50):
        """Run recursive optimization that improves each generation"""
        
        print(f"\n{'='*80}")
        print(f"🚀 RECURSIVE SELF-IMPROVING OPTIMIZATION")
        print(f"   Generations: {n_generations}")
        print(f"   Target: Robust Sharpe > 1.0")
        print(f"{'='*80}")
        
        best_ever_sharpe = -float('inf')
        stagnation_counter = 0
        
        for gen in range(1, n_generations + 1):
            self.generation = gen
            print(f"\n\n{'='*60}")
            print(f"🧬 GENERATION {gen}/{n_generations}")
            print(f"{'='*60}")
            
            # Get adaptive parameter space for this generation
            param_space = self._get_adaptive_param_space(gen)
            
            # Print generation strategy
            self._print_generation_strategy(gen, param_space)
            
            # Run optimization
            start_time = time.time()
            results = self._run_generation_optimization(param_space, gen)
            elapsed = time.time() - start_time
            
            # Analyze results
            if results:
                best_gen_result = max(results, key=lambda x: x['robust_score'])
                self.memory.add_result(best_gen_result)
                
                print(f"\n📊 Generation {gen} Results:")
                print(f"   Best Avg Sharpe: {best_gen_result['avg_sharpe']:.3f}")
                print(f"   Min Sharpe: {best_gen_result['min_sharpe']:.3f}")
                print(f"   Sharpe Std Dev: {best_gen_result['sharpe_std']:.3f}")
                print(f"   Robust Score: {best_gen_result['robust_score']:.3f}")
                print(f"   Time: {elapsed:.1f}s")
                
                # Check for improvement
                if best_gen_result['avg_sharpe'] > best_ever_sharpe * (1 + self.improvement_threshold):
                    best_ever_sharpe = best_gen_result['avg_sharpe']
                    stagnation_counter = 0
                    print(f"   🌟 New best! {(best_gen_result['avg_sharpe']/best_ever_sharpe - 1)*100:.1f}% improvement")
                else:
                    stagnation_counter += 1
                    print(f"   ⚠️  No significant improvement (stagnation: {stagnation_counter})")
                
                # Learn from results
                self._learn_from_generation(results)
                
                # If stagnating, try more exploration
                if stagnation_counter >= 3:
                    print(f"\n🔄 Increasing exploration due to stagnation...")
                    self.improvement_threshold *= 0.5  # Lower bar for improvement
            
            # Save checkpoint
            self._save_checkpoint(gen)
        
        # Final analysis and recommendations
        self._final_analysis()
    
    def _get_adaptive_param_space(self, generation: int) -> Dict[str, ParameterBounds]:
        """Get parameter space that adapts based on learnings"""
        
        # Analyze patterns from memory
        insights = self.memory.analyze_patterns()
        best_params = self.memory.get_best_params()
        
        param_space = {}
        
        for param, (min_val, max_val) in self.base_param_space.items():
            # Start with base range
            new_min, new_max = min_val, max_val
            
            # Adapt based on insights
            if param in insights:
                insight = insights[param]
                
                # If strong positive impact, shift range up
                if insight['diff_pct'] > 30 and insight['winner_mean'] > insight['loser_mean']:
                    shift = (max_val - min_val) * 0.2
                    new_min = min(max_val - (max_val - min_val) * 0.8, min_val + shift)
                    new_max = min(max_val + shift * 0.5, max_val * 1.2)
                
                # If strong negative impact, shift range down
                elif insight['diff_pct'] < -30 and insight['winner_mean'] < insight['loser_mean']:
                    shift = (max_val - min_val) * 0.2
                    new_max = max(min_val + (max_val - min_val) * 0.8, max_val - shift)
                    new_min = max(min_val - shift * 0.5, min_val * 0.8)
            
            # After generation 3, focus around best known values
            if generation > 3 and best_params and param in best_params:
                center = best_params[param]
                
                # Reduce range each generation
                reduction = 0.3 - (generation - 3) * 0.05
                reduction = max(reduction, 0.1)  # Minimum 10% range
                
                range_width = (max_val - min_val) * reduction
                new_min = max(min_val, center - range_width/2)
                new_max = min(max_val, center + range_width/2)
            
            # Create parameter bounds
            param_type = int if param in ['tsl_activation_pips'] else float
            step = 1 if param_type == int else (new_max - new_min) / 20
            
            param_space[param] = ParameterBounds(
                param, new_min, new_max, param_type, step
            )
        
        return param_space
    
    def _run_generation_optimization(self, param_space: Dict[str, ParameterBounds], generation: int) -> List[Dict]:
        """Run optimization for one generation"""
        
        # Number of configurations to test
        n_configs = 15 if generation <= 3 else 10  # More exploration early
        
        print(f"\n🔍 Testing {n_configs} configurations...")
        
        # Generate parameter sets
        param_sets = self._generate_intelligent_params(param_space, n_configs, generation)
        
        # Test each configuration on multiple periods
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            futures = []
            
            for i, params in enumerate(param_sets):
                future = executor.submit(
                    self._evaluate_robust_config,
                    params, i, generation
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    if result:
                        results.append(result)
                        
                        # Print progress
                        if result['avg_sharpe'] > 1.0:
                            print(f"   ✅ Config {result['config_id']}: Avg Sharpe {result['avg_sharpe']:.3f} (Min: {result['min_sharpe']:.3f})")
                        elif result['avg_sharpe'] > 0.5:
                            print(f"   ⚡ Config {result['config_id']}: Avg Sharpe {result['avg_sharpe']:.3f}")
                except Exception as e:
                    print(f"   ❌ Error in evaluation: {str(e)}")
        
        return results
    
    def _evaluate_robust_config(self, params: Dict, config_id: int, generation: int) -> Optional[Dict]:
        """Evaluate configuration on multiple time periods for robustness"""
        
        try:
            # Test on 5 different periods (3 train, 2 validation)
            period_results = []
            
            # Define test periods (50K samples each)
            n_periods = 5
            period_size = 50000
            
            # Generate random non-overlapping periods
            max_start = len(self.df) - period_size
            starts = sorted(np.random.choice(
                range(0, max_start - period_size, period_size // 2), 
                size=n_periods, 
                replace=False
            ))
            
            for i, start_idx in enumerate(starts):
                # Get period data
                period_df = self.df.iloc[start_idx:start_idx + period_size].copy()
                
                # Create strategy
                config = self._create_strategy_config(params)
                strategy = OptimizedProdStrategy(config)
                
                # Run backtest
                results = strategy.run_backtest(period_df)
                
                period_results.append({
                    'period': i,
                    'sharpe': results.get('sharpe_ratio', -999),
                    'return': results.get('total_return', 0),
                    'win_rate': results.get('win_rate', 0),
                    'trades': results.get('total_trades', 0)
                })
            
            # Calculate robustness metrics
            sharpes = [r['sharpe'] for r in period_results if r['sharpe'] > -900]
            
            if len(sharpes) >= 3:
                avg_sharpe = np.mean(sharpes)
                min_sharpe = np.min(sharpes)
                max_sharpe = np.max(sharpes)
                sharpe_std = np.std(sharpes)
                
                # Robust score favors consistency
                robust_score = avg_sharpe - 0.5 * sharpe_std - 0.2 * (max_sharpe - min_sharpe)
                
                return {
                    'config_id': config_id,
                    'generation': generation,
                    'params': params,
                    'period_results': period_results,
                    'avg_sharpe': avg_sharpe,
                    'min_sharpe': min_sharpe,
                    'max_sharpe': max_sharpe,
                    'sharpe_std': sharpe_std,
                    'robust_score': robust_score,
                    'avg_trades': np.mean([r['trades'] for r in period_results]),
                    'avg_win_rate': np.mean([r['win_rate'] for r in period_results])
                }
            
            return None
            
        except Exception as e:
            print(f"   Error evaluating config {config_id}: {str(e)}")
            return None
    
    def _generate_intelligent_params(self, param_space: Dict[str, ParameterBounds], 
                                   n_configs: int, generation: int) -> List[Dict]:
        """Generate parameter sets intelligently"""
        
        param_sets = []
        best_params = self.memory.get_best_params()
        
        for i in range(n_configs):
            params = {}
            
            # Strategy depends on generation and index
            if generation <= 2:
                # Early generations: more exploration
                strategy = 'explore'
            elif i < n_configs * 0.4 and best_params:
                # 40% exploitation of best known
                strategy = 'exploit'
            elif i < n_configs * 0.7:
                # 30% intelligent exploration
                strategy = 'smart_explore'
            else:
                # 30% random exploration
                strategy = 'random'
            
            for param_name, bounds in param_space.items():
                if strategy == 'exploit' and best_params and param_name in best_params:
                    # Vary around best known value
                    center = best_params[param_name]
                    variation = (bounds.max_value - bounds.min_value) * 0.15
                    value = np.clip(
                        np.random.normal(center, variation/3),
                        bounds.min_value,
                        bounds.max_value
                    )
                elif strategy == 'smart_explore':
                    # Use insights to bias sampling
                    if param_name in self.memory.insights:
                        insight = self.memory.insights[param_name]
                        if insight['recommendation'] == 'increase':
                            # Bias towards higher values
                            value = bounds.min_value + (bounds.max_value - bounds.min_value) * np.random.beta(2, 1)
                        elif insight['recommendation'] == 'decrease':
                            # Bias towards lower values
                            value = bounds.min_value + (bounds.max_value - bounds.min_value) * np.random.beta(1, 2)
                        else:
                            value = bounds.sample()
                    else:
                        value = bounds.sample()
                else:
                    # Random exploration
                    value = bounds.sample()
                
                params[param_name] = bounds.round_value(value)
            
            param_sets.append(params)
        
        return param_sets
    
    def _learn_from_generation(self, results: List[Dict]):
        """Learn patterns from generation results"""
        
        if len(results) < 3:
            return
        
        # Sort by robust score
        sorted_results = sorted(results, key=lambda x: x['robust_score'], reverse=True)
        
        # Identify patterns in top performers
        top_n = min(3, len(sorted_results) // 2)
        top_performers = sorted_results[:top_n]
        
        print(f"\n🧠 Learning from top {top_n} performers:")
        
        # Find consistent parameter values in top performers
        param_patterns = {}
        for param in self.base_param_space.keys():
            values = [r['params'][param] for r in top_performers]
            param_patterns[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Print key insights
        for param, stats in sorted(param_patterns.items(), key=lambda x: x[1]['std']):
            if stats['std'] < 0.1 * (self.base_param_space[param][1] - self.base_param_space[param][0]):
                print(f"   ✓ {param}: converging around {stats['mean']:.3f} (low variance)")
    
    def _create_strategy_config(self, params: Dict[str, float]) -> OptimizedStrategyConfig:
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
    
    def _print_generation_strategy(self, generation: int, param_space: Dict[str, ParameterBounds]):
        """Print strategy for current generation"""
        
        insights = self.memory.insights
        
        print(f"\n📋 Generation {generation} Strategy:")
        
        if generation == 1:
            print("   - Initial exploration with base ranges")
        elif generation <= 3:
            print("   - Broad exploration with slight bias from learnings")
        else:
            print("   - Focused exploitation around best performers")
            print("   - Reduced parameter ranges for convergence")
        
        if insights:
            print("\n   Key parameter adjustments:")
            for param, insight in sorted(insights.items(), 
                                       key=lambda x: abs(x[1]['diff_pct']), 
                                       reverse=True)[:5]:
                if abs(insight['diff_pct']) > 20:
                    print(f"   - {param}: {insight['recommendation']} " + 
                          f"({insight['diff_pct']:+.1f}% impact)")
    
    def _save_checkpoint(self, generation: int):
        """Save checkpoint after each generation"""
        
        checkpoint = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'all_results': self.memory.all_results,
                'high_performers': self.memory.high_performers,
                'robust_performers': self.memory.robust_performers,
                'insights': self.memory.insights
            },
            'best_params': self.memory.get_best_params(),
            'best_result': self.memory.robust_performers[0] if self.memory.robust_performers else None
        }
        
        os.makedirs('optimizer_checkpoints', exist_ok=True)
        filename = f'optimizer_checkpoints/generation_{generation}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filename, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _final_analysis(self):
        """Final analysis and recommendations"""
        
        print(f"\n\n{'='*80}")
        print("🏆 FINAL OPTIMIZATION RESULTS")
        print(f"{'='*80}")
        
        if self.memory.robust_performers:
            best = self.memory.robust_performers[0]
            
            print(f"\n✅ BEST ROBUST CONFIGURATION:")
            print(f"   Average Sharpe: {best['avg_sharpe']:.3f}")
            print(f"   Minimum Sharpe: {best['min_sharpe']:.3f}")
            print(f"   Sharpe Std Dev: {best['sharpe_std']:.3f}")
            print(f"   Robust Score: {best['robust_score']:.3f}")
            print(f"   Average Trades: {best['avg_trades']:.0f}")
            print(f"   Average Win Rate: {best['avg_win_rate']:.1f}%")
            
            print(f"\n📊 OPTIMAL PARAMETERS:")
            for param, value in sorted(best['params'].items()):
                print(f"   {param}: {value}")
            
            # Save final configuration
            final_config = {
                'strategy_type': self.strategy_type,
                'currency': self.currency,
                'optimization_complete': datetime.now().isoformat(),
                'generations_run': self.generation,
                'best_robust_result': best,
                'all_high_performers': self.memory.high_performers,
                'final_insights': self.memory.insights
            }
            
            os.makedirs('optimized_configs', exist_ok=True)
            filename = f'optimized_configs/final_robust_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            with open(filename, 'w') as f:
                json.dump(final_config, f, indent=2)
            
            print(f"\n💾 Final configuration saved to: {filename}")
            
            if best['avg_sharpe'] > 1.0 and best['min_sharpe'] > 0.8:
                print(f"\n🎉 SUCCESS! Achieved robust Sharpe > 1.0")
                print(f"   This configuration maintains Sharpe > 0.8 across all test periods")
            else:
                print(f"\n⚠️  Target not fully achieved. Consider:")
                print(f"   - Running more generations")
                print(f"   - Adjusting risk parameters")
                print(f"   - Testing on different market conditions")
        else:
            print("\n❌ No robust configurations found. Consider adjusting target criteria.")


def main():
    """Run the self-improving optimizer"""
    optimizer = SelfImprovingOptimizer(strategy_type=1, currency='AUDUSD')
    optimizer.run_recursive_optimization(n_generations=10)


if __name__ == "__main__":
    main()