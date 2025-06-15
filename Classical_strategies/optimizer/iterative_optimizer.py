#!/usr/bin/env python3
"""
Iterative optimizer that runs 10 focused optimization rounds
Each round learns from previous results and focuses on promising areas
Limited to ~2 minutes per run using multiprocessing
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

sys.path.append('..')
from intelligent_optimizer import (
    ParameterBounds, ParameterSpace, OptimizationResult,
    BayesianOptimizer, run_optimization
)
from run_strategy_oop import DataManager, MonteCarloSimulator, TradeAnalyzer
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


class IterativeOptimizer:
    """Manages iterative optimization with learning between runs"""
    
    def __init__(self, strategy_type: int = 1, currency: str = 'AUDUSD'):
        self.strategy_type = strategy_type
        self.currency = currency
        self.procedure_file = 'OPTIMIZATION_PROCEDURE.md'
        self.current_iteration = 1
        self.all_results = []
        self.time_limit = 120  # 2 minutes per iteration
        self.data_manager = DataManager()
        print(f"Loading {currency} data...")
        self.df = self.data_manager.load_currency_data(currency)
        print(f"Data loaded: {len(self.df):,} rows from {self.df.index[0]} to {self.df.index[-1]}")
        
    def run_10_iterations(self):
        """Run 10 optimization iterations with learning"""
        
        print(f"\n{'='*60}")
        print(f"🚀 STARTING 10-ITERATION OPTIMIZATION SEQUENCE")
        print(f"   Strategy: {'Ultra-Tight Risk' if self.strategy_type == 1 else 'Scalping'}")
        print(f"   Time limit: {self.time_limit}s per iteration")
        print(f"   Goal: Sharpe > 1.0 across multiple time periods")
        print(f"{'='*60}")
        
        # Load previous iteration count if resuming
        self.current_iteration = self._get_current_iteration()
        
        while self.current_iteration <= 10:
            print(f"\n\n{'='*60}")
            print(f"📊 ITERATION {self.current_iteration}/10")
            print(f"{'='*60}")
            
            # Get focused parameter space for this iteration
            param_space = self._get_focused_parameters()
            
            # Run optimization with time limit
            start_time = time.time()
            
            try:
                # Calculate iterations based on time budget
                # Assume ~10 seconds per iteration with multiprocessing
                n_iterations = min(12, self.time_limit // 10)
                
                print(f"\n🎯 Focus for this iteration:")
                for param, bounds in param_space.items():
                    if hasattr(bounds, 'min_value'):
                        print(f"   {param}: {bounds.min_value:.3f} - {bounds.max_value:.3f}")
                
                # Run robust optimization with multiple time periods
                best_result = self._run_robust_optimization(
                    param_space=param_space,
                    n_iterations=n_iterations
                )
                
                elapsed = time.time() - start_time
                print(f"\n⏱️  Iteration completed in {elapsed:.1f}s")
                
                # Analyze and document results
                if best_result:
                    self._analyze_results(best_result, self.all_results[-n_iterations:] if len(self.all_results) >= n_iterations else self.all_results)
                
                # Update procedure document
                self._update_procedure_doc(best_result, param_space)
                
            except Exception as e:
                print(f"\n❌ Error in iteration {self.current_iteration}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            self.current_iteration += 1
            
            # Brief pause between iterations
            if self.current_iteration <= 10:
                print(f"\n⏸️  Pausing before next iteration...")
                time.sleep(2)
        
        print(f"\n\n{'='*60}")
        print(f"✅ COMPLETED ALL 10 ITERATIONS")
        print(f"{'='*60}")
        self._print_final_summary()
    
    def _get_focused_parameters(self) -> Dict[str, ParameterBounds]:
        """Get focused parameter bounds based on iteration and previous results"""
        
        base_space = ParameterSpace.get_strategy1_space()
        
        if self.current_iteration == 1:
            # First iteration - use base space but fewer parameters
            return {
                'risk_per_trade': base_space['risk_per_trade'],
                'sl_min_pips': base_space['sl_min_pips'],
                'sl_max_pips': base_space['sl_max_pips'],
                'tp1_multiplier': base_space['tp1_multiplier'],
                'tp2_multiplier': base_space['tp2_multiplier'],
                'trailing_atr_multiplier': base_space['trailing_atr_multiplier'],
            }
        
        elif self.current_iteration == 2:
            # Focus on stop loss optimization
            return {
                'sl_min_pips': ParameterBounds('sl_min_pips', 5.0, 10.0, float, 0.5),
                'sl_max_pips': ParameterBounds('sl_max_pips', 20.0, 30.0, float, 1.0),
                'sl_atr_multiplier': ParameterBounds('sl_atr_multiplier', 1.5, 2.5, float, 0.1),
                'risk_per_trade': ParameterBounds('risk_per_trade', 0.0025, 0.0035, float, 0.0001),
            }
        
        elif self.current_iteration == 3:
            # Focus on take profit optimization
            return {
                'tp1_multiplier': ParameterBounds('tp1_multiplier', 0.15, 0.35, float, 0.025),
                'tp2_multiplier': ParameterBounds('tp2_multiplier', 0.2, 0.4, float, 0.025),
                'tp3_multiplier': ParameterBounds('tp3_multiplier', 0.6, 1.2, float, 0.1),
                'tp_range_market_multiplier': base_space['tp_range_market_multiplier'],
            }
        
        elif self.current_iteration == 4:
            # Focus on trailing stop
            return {
                'tsl_activation_pips': ParameterBounds('tsl_activation_pips', 10, 20, int),
                'tsl_min_profit_pips': ParameterBounds('tsl_min_profit_pips', 1, 4, float, 0.5),
                'trailing_atr_multiplier': ParameterBounds('trailing_atr_multiplier', 1.2, 2.0, float, 0.1),
            }
        
        elif self.current_iteration == 5:
            # Focus on partial profits
            return {
                'partial_profit_sl_distance_ratio': ParameterBounds('partial_profit_sl_distance_ratio', 0.3, 0.6, float, 0.05),
                'partial_profit_size_percent': ParameterBounds('partial_profit_size_percent', 0.4, 0.8, float, 0.05),
                'risk_per_trade': ParameterBounds('risk_per_trade', 0.002, 0.004, float, 0.0001),
            }
        
        elif self.current_iteration <= 7:
            # Iterations 6-7: Refine best parameters found so far
            best_params = self._get_best_parameters_so_far()
            return self._create_refined_space(best_params, reduction_factor=0.3)
        
        else:
            # Iterations 8-10: Fine-tune around best configuration
            best_params = self._get_best_parameters_so_far()
            return self._create_refined_space(best_params, reduction_factor=0.15)
    
    def _create_refined_space(self, center_params: Dict, reduction_factor: float) -> Dict[str, ParameterBounds]:
        """Create refined parameter space around best values"""
        base_space = ParameterSpace.get_strategy1_space()
        refined = {}
        
        # Select most impactful parameters
        key_params = ['risk_per_trade', 'sl_min_pips', 'sl_max_pips', 
                     'tp1_multiplier', 'tp2_multiplier', 'trailing_atr_multiplier']
        
        for param in key_params:
            if param in base_space and param in center_params:
                bounds = base_space[param]
                center = center_params[param]
                range_width = (bounds.max_value - bounds.min_value) * reduction_factor
                
                new_min = max(bounds.min_value, center - range_width/2)
                new_max = min(bounds.max_value, center + range_width/2)
                
                refined[param] = ParameterBounds(
                    param, new_min, new_max, 
                    bounds.param_type, bounds.step_size, bounds.is_percentage
                )
        
        return refined
    
    def _get_best_parameters_so_far(self) -> Dict:
        """Get best parameters from all previous runs"""
        # Check saved results
        if os.path.exists('optimizer_results'):
            import glob
            files = glob.glob('optimizer_results/optimization_results_strategy*.json')
            
            best_sharpe = -float('inf')
            best_params = {}
            
            for file in files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if data.get('best_result', {}).get('sharpe_ratio', -999) > best_sharpe:
                            best_sharpe = data['best_result']['sharpe_ratio']
                            best_params = data['best_result']['params']
                except:
                    pass
            
            if best_params:
                return best_params
        
        # Default fallback
        return {
            'risk_per_trade': 0.003,
            'sl_min_pips': 6.0,
            'sl_max_pips': 20.0,
            'tp1_multiplier': 0.25,
            'tp2_multiplier': 0.35,
            'trailing_atr_multiplier': 1.5
        }
    
    def _analyze_results(self, best_result: OptimizationResult, all_results: List[OptimizationResult]):
        """Analyze results to inform next iteration"""
        if not best_result:
            return
        
        print(f"\n📊 RESULTS ANALYSIS:")
        print(f"   Best Sharpe: {best_result.sharpe_ratio:.3f}")
        print(f"   Best Return: {best_result.total_return:.1f}%")
        print(f"   Best Win Rate: {best_result.win_rate:.1f}%")
        
        # Analyze parameter impact
        if len(all_results) > 3:
            # Get top 30% of results
            sorted_results = sorted(all_results, key=lambda r: r.fitness, reverse=True)
            top_results = sorted_results[:max(1, len(sorted_results)//3)]
            
            print(f"\n   Top performers analysis:")
            # Find common parameter ranges in top performers
            for param in ['risk_per_trade', 'sl_min_pips', 'tp1_multiplier']:
                if param in top_results[0].params:
                    values = [r.params.get(param, 0) for r in top_results]
                    print(f"   {param}: avg={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    def _update_procedure_doc(self, best_result: OptimizationResult, param_space: Dict):
        """Update the procedure document with results"""
        
        # Read current document
        with open(self.procedure_file, 'r') as f:
            content = f.read()
        
        # Find the section for current run
        run_marker = f"### Run {self.current_iteration} -"
        
        if run_marker in content:
            # Update existing section
            start_idx = content.find(run_marker)
            end_idx = content.find(f"### Run {self.current_iteration + 1} -", start_idx)
            if end_idx == -1:
                end_idx = content.find("## Best Configurations", start_idx)
            
            # Create updated section
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            if best_result:
                update = f"""{run_marker} Completed ({timestamp})
**Focus**: {self._get_iteration_focus()}
**Parameters**: {', '.join(param_space.keys())}
**Results**: 
- Best Sharpe: {best_result.sharpe_ratio:.3f}
- Return: {best_result.total_return:.1f}%
- Win Rate: {best_result.win_rate:.1f}%
- Trades: {best_result.total_trades}
- Key params: risk={best_result.params.get('risk_per_trade', 0)*100:.2f}%, sl_min={best_result.params.get('sl_min_pips', 0):.1f}
**Insights**: {self._get_iteration_insights(best_result)}

"""
            else:
                update = f"""{run_marker} Failed ({timestamp})
**Focus**: {self._get_iteration_focus()}
**Results**: No valid results
**Insights**: Parameters may be too restrictive

"""
            
            # Replace section
            new_content = content[:start_idx] + update + content[end_idx:]
            
            # Write back
            with open(self.procedure_file, 'w') as f:
                f.write(new_content)
    
    def _get_iteration_focus(self) -> str:
        """Get focus description for current iteration"""
        focuses = {
            1: "Initial broad search",
            2: "Stop loss optimization",
            3: "Take profit optimization", 
            4: "Trailing stop optimization",
            5: "Partial profit optimization",
            6: "Refine best parameters (wide)",
            7: "Refine best parameters (medium)",
            8: "Fine-tune best configuration",
            9: "Final optimization",
            10: "Validation run"
        }
        return focuses.get(self.current_iteration, "General optimization")
    
    def _get_iteration_insights(self, result: OptimizationResult) -> str:
        """Generate insights from results"""
        if result.sharpe_ratio > 1.5:
            return "Excellent configuration found"
        elif result.sharpe_ratio > 1.0:
            return "Good configuration, worth refining"
        elif result.sharpe_ratio > 0.5:
            return "Moderate performance, needs adjustment"
        else:
            return "Poor performance, avoid these parameters"
    
    def _get_current_iteration(self) -> int:
        """Get current iteration from procedure doc"""
        if not os.path.exists(self.procedure_file):
            return 1
            
        with open(self.procedure_file, 'r') as f:
            content = f.read()
        
        # Find completed runs
        for i in range(10, 0, -1):
            if f"### Run {i} - Completed" in content:
                return i + 1
        
        return 1
    
    def _run_robust_optimization(self, param_space: Dict[str, ParameterBounds], n_iterations: int) -> Optional[OptimizationResult]:
        """Run optimization with robustness testing across different time periods"""
        
        # Define training and validation periods
        # Split data into multiple non-overlapping periods for robustness
        total_rows = len(self.df)
        period_size = 50000  # ~6 months of 15min data
        
        # Create time periods for training and validation
        periods = []
        for i in range(0, total_rows - period_size, period_size // 2):
            periods.append((i, min(i + period_size, total_rows)))
        
        # Use different periods for training
        train_periods = periods[::3]  # Every 3rd period for training
        val_periods = periods[1::3]   # Different periods for validation
        
        print(f"\n🔍 Robust optimization with {len(train_periods)} training periods")
        
        # Track best configuration across all periods
        best_robust_result = None
        best_robust_score = -float('inf')
        
        # Test parameters on different training periods
        tested_params = []
        
        for i in range(n_iterations):
            # Generate parameter set
            params = self._generate_parameter_set(param_space, tested_params)
            tested_params.append(params)
            
            # Test on multiple training periods
            period_results = []
            
            # Use multiprocessing for parallel evaluation
            with ProcessPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
                futures = []
                
                # Submit evaluation tasks for each period
                for period_idx, (start, end) in enumerate(train_periods[:3]):  # Use 3 periods
                    future = executor.submit(
                        self._evaluate_on_period,
                        params, start, end, period_idx
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)
                        if result:
                            period_results.append(result)
                    except Exception as e:
                        print(f"   Error in evaluation: {str(e)}")
            
            # Calculate robustness score
            if period_results:
                sharpes = [r.sharpe_ratio for r in period_results]
                avg_sharpe = np.mean(sharpes)
                min_sharpe = np.min(sharpes)
                std_sharpe = np.std(sharpes)
                
                # Robust score favors consistency
                robust_score = avg_sharpe - 0.5 * std_sharpe
                
                # Only consider if minimum Sharpe > 0.5
                if min_sharpe > 0.5 and robust_score > best_robust_score:
                    best_robust_score = robust_score
                    best_robust_result = period_results[0]  # Use first period's detailed result
                    best_robust_result.sharpe_ratio = avg_sharpe  # Update with average
                    
                    print(f"\n   🌟 New robust best! Avg Sharpe: {avg_sharpe:.3f}, Min: {min_sharpe:.3f}, Std: {std_sharpe:.3f}")
                    
                    # Validate on out-of-sample periods
                    if avg_sharpe > 0.8:
                        self._validate_on_oos(params, val_periods[:2])
                
                # Store result
                avg_result = OptimizationResult(
                    params=params,
                    sharpe_ratio=avg_sharpe,
                    total_return=np.mean([r.total_return for r in period_results]),
                    win_rate=np.mean([r.win_rate for r in period_results]),
                    max_drawdown=np.max([r.max_drawdown for r in period_results]),
                    profit_factor=np.mean([r.profit_factor for r in period_results]),
                    total_trades=int(np.mean([r.total_trades for r in period_results])),
                    iteration=len(self.all_results)
                )
                self.all_results.append(avg_result)
        
        return best_robust_result
    
    def _evaluate_on_period(self, params: Dict[str, float], start_idx: int, end_idx: int, period_idx: int) -> Optional[OptimizationResult]:
        """Evaluate parameters on a specific time period"""
        try:
            # Get data slice
            period_df = self.df.iloc[start_idx:end_idx].copy()
            
            # Create strategy with parameters
            config = self._create_strategy_config(params)
            strategy = OptimizedProdStrategy(config)
            
            # Run backtest
            results = strategy.run_backtest(period_df)
            
            # Extract metrics
            return OptimizationResult(
                params=params,
                sharpe_ratio=results.get('sharpe_ratio', -999),
                total_return=results.get('total_return', 0),
                win_rate=results.get('win_rate', 0),
                max_drawdown=results.get('max_drawdown', 100),
                profit_factor=results.get('profit_factor', 0),
                total_trades=results.get('total_trades', 0),
                iteration=period_idx
            )
        except Exception as e:
            print(f"   Error evaluating period {period_idx}: {str(e)}")
            return None
    
    def _validate_on_oos(self, params: Dict[str, float], val_periods: List[Tuple[int, int]]):
        """Validate parameters on out-of-sample periods"""
        print(f"\n   🎯 Validating on {len(val_periods)} out-of-sample periods...")
        
        val_results = []
        for start, end in val_periods:
            result = self._evaluate_on_period(params, start, end, -1)
            if result:
                val_results.append(result.sharpe_ratio)
                print(f"      Period {self.df.index[start].date()} to {self.df.index[end-1].date()}: Sharpe={result.sharpe_ratio:.3f}")
        
        if val_results:
            print(f"   📋 Validation Summary: Avg={np.mean(val_results):.3f}, Min={np.min(val_results):.3f}")
    
    def _generate_parameter_set(self, param_space: Dict[str, ParameterBounds], tested_params: List[Dict]) -> Dict[str, float]:
        """Generate new parameter set avoiding tested combinations"""
        # Simple random generation with some intelligence
        params = {}
        
        for name, bounds in param_space.items():
            if tested_params and np.random.random() < 0.3:  # 30% chance to use variation of good params
                # Use variation of best parameters so far
                best_params = max(self.all_results, key=lambda r: r.fitness).params if self.all_results else tested_params[0]
                if name in best_params:
                    center = best_params[name]
                    variation = (bounds.max_value - bounds.min_value) * 0.2
                    value = np.clip(center + np.random.uniform(-variation, variation), bounds.min_value, bounds.max_value)
                    params[name] = bounds.round_value(value)
                else:
                    params[name] = bounds.round_value(bounds.sample())
            else:
                params[name] = bounds.round_value(bounds.sample())
        
        return params
    
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
    
    def _print_final_summary(self):
        """Print summary of all 10 iterations"""
        print("\n📊 FINAL SUMMARY OF 10 ITERATIONS")
        print("="*60)
        
        # Find best overall result
        best_sharpe = -float('inf')
        best_config = None
        
        if os.path.exists('optimizer_results'):
            import glob
            files = glob.glob('optimizer_results/optimization_results_strategy*.json')
            
            for file in sorted(files):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if data.get('best_result', {}).get('sharpe_ratio', -999) > best_sharpe:
                            best_sharpe = data['best_result']['sharpe_ratio']
                            best_config = data['best_result']
                except:
                    pass
        
        if best_config:
            print(f"\n🏆 BEST CONFIGURATION FOUND:")
            print(f"   Sharpe Ratio: {best_config['sharpe_ratio']:.3f}")
            print(f"   Total Return: {best_config['total_return']:.1f}%")
            print(f"   Win Rate: {best_config['win_rate']:.1f}%")
            print(f"   Max Drawdown: {best_config['max_drawdown']:.1f}%")
            print(f"   Profit Factor: {best_config['profit_factor']:.2f}")
            
            print(f"\n   Key Parameters:")
            params = best_config['params']
            print(f"   - Risk per trade: {params.get('risk_per_trade', 0)*100:.2f}%")
            print(f"   - Stop loss: {params.get('sl_min_pips', 0):.1f}-{params.get('sl_max_pips', 0):.1f} pips")
            print(f"   - TP multipliers: ({params.get('tp1_multiplier', 0):.2f}, {params.get('tp2_multiplier', 0):.2f}, {params.get('tp3_multiplier', 0):.2f})")


def main():
    """Run the iterative optimization process"""
    optimizer = IterativeOptimizer(strategy_type=1, currency='AUDUSD')
    optimizer.run_10_iterations()


if __name__ == "__main__":
    main()