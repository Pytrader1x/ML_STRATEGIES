"""
Intelligent Strategy Optimizer with Adaptive Learning
Uses feedback from each optimization round to inform next parameters
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
import json
from datetime import datetime
import os
from collections import defaultdict
import warnings
from scipy.stats import norm
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Import strategy components
import sys
sys.path.append('..')  # Add parent directory to path
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from run_strategy_oop import DataManager, MonteCarloSimulator, TradeAnalyzer

warnings.filterwarnings('ignore')

@dataclass
class ParameterBounds:
    """Defines bounds and characteristics for a parameter"""
    name: str
    min_value: float
    max_value: float
    param_type: type = float
    step_size: Optional[float] = None
    is_percentage: bool = False
    
    def sample(self) -> float:
        """Sample a value within bounds"""
        if self.param_type == int:
            return np.random.randint(int(self.min_value), int(self.max_value) + 1)
        else:
            return np.random.uniform(self.min_value, self.max_value)
    
    def round_value(self, value: float) -> float:
        """Round value according to parameter type and step size"""
        if self.param_type == int:
            return int(round(value))
        elif self.step_size:
            return round(value / self.step_size) * self.step_size
        else:
            return round(value, 4)


@dataclass
class OptimizationResult:
    """Container for optimization result"""
    params: Dict[str, float]
    sharpe_ratio: float
    total_return: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    total_trades: int
    iteration: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def fitness(self) -> float:
        """Combined fitness score for optimization"""
        # Weighted combination of metrics
        if self.total_trades < 50:  # Penalize strategies with too few trades
            trade_penalty = 0.5
        else:
            trade_penalty = 1.0
            
        # Normalize metrics
        sharpe_score = min(self.sharpe_ratio / 2.0, 1.0)  # Sharpe of 2.0 = perfect score
        dd_score = max(1.0 - (self.max_drawdown / 30.0), 0)  # 30% DD = 0 score
        pf_score = min((self.profit_factor - 1.0) / 2.0, 1.0)  # PF of 3.0 = perfect score
        
        # Weighted combination
        fitness = (0.5 * sharpe_score + 
                  0.3 * dd_score + 
                  0.2 * pf_score) * trade_penalty
                  
        return fitness


class ParameterSpace:
    """Defines the parameter space for optimization"""
    
    @staticmethod
    def get_strategy1_space() -> Dict[str, ParameterBounds]:
        """Parameter space for Ultra-Tight Risk Management strategy
        
        Insights from baseline:
        - 66.6% of trades hit SL -> need wider stops or better entries
        - Only 11.6% reach TP3 -> TPs may be too far
        - TP1 pullback at 12.8% is working -> keep partial profit strategy
        - Average Sharpe 0.585 needs improvement
        """
        return {
            # Risk parameters - allow slightly higher risk for better R:R
            'risk_per_trade': ParameterBounds('risk_per_trade', 0.0015, 0.004, float, 0.0001, True),
            
            # Stop loss parameters - widen the range based on high SL hits
            'sl_min_pips': ParameterBounds('sl_min_pips', 5.0, 12.0, float, 0.5),
            'sl_max_pips': ParameterBounds('sl_max_pips', 10.0, 25.0, float, 1.0),
            'sl_atr_multiplier': ParameterBounds('sl_atr_multiplier', 0.8, 2.0, float, 0.1),
            
            # Take profit multipliers - bring TPs closer for higher hit rate
            'tp1_multiplier': ParameterBounds('tp1_multiplier', 0.15, 0.4, float, 0.025),
            'tp2_multiplier': ParameterBounds('tp2_multiplier', 0.25, 0.6, float, 0.05),
            'tp3_multiplier': ParameterBounds('tp3_multiplier', 0.4, 1.0, float, 0.05),
            
            # Trailing stop parameters - optimize for better exits
            'tsl_activation_pips': ParameterBounds('tsl_activation_pips', 8, 20, int),
            'tsl_min_profit_pips': ParameterBounds('tsl_min_profit_pips', 1, 5, float, 0.5),
            'trailing_atr_multiplier': ParameterBounds('trailing_atr_multiplier', 0.8, 1.8, float, 0.1),
            
            # Market condition multipliers
            'tp_range_market_multiplier': ParameterBounds('tp_range_market_multiplier', 0.4, 0.9, float, 0.05),
            'tp_trend_market_multiplier': ParameterBounds('tp_trend_market_multiplier', 0.7, 1.2, float, 0.05),
            'tp_chop_market_multiplier': ParameterBounds('tp_chop_market_multiplier', 0.2, 0.5, float, 0.05),
            
            # Partial profit parameters - these are working well
            'partial_profit_sl_distance_ratio': ParameterBounds('partial_profit_sl_distance_ratio', 0.3, 0.7, float, 0.05),
            'partial_profit_size_percent': ParameterBounds('partial_profit_size_percent', 0.3, 0.7, float, 0.05),
            
            # Add intelligent sizing to potentially improve performance
            'use_intelligent_sizing': ParameterBounds('use_intelligent_sizing', 0, 1, int),
        }
    
    @staticmethod
    def get_strategy2_space() -> Dict[str, ParameterBounds]:
        """Parameter space for Scalping strategy
        
        Insights from baseline:
        - TERRIBLE performance: -4.483 average Sharpe
        - 69.9% hit SL -> stops are way too tight
        - Need complete rethink: wider stops, better R:R
        - Signal flip exit only 1% -> maybe not useful
        """
        return {
            # Risk parameters - allow higher risk since stops are tight
            'risk_per_trade': ParameterBounds('risk_per_trade', 0.001, 0.003, float, 0.0001, True),
            
            # Stop loss parameters - MUST be wider to avoid constant stops
            'sl_min_pips': ParameterBounds('sl_min_pips', 4.0, 10.0, float, 0.5),
            'sl_max_pips': ParameterBounds('sl_max_pips', 6.0, 15.0, float, 0.5),
            'sl_atr_multiplier': ParameterBounds('sl_atr_multiplier', 0.6, 1.5, float, 0.1),
            
            # Take profit multipliers - need better R:R ratio
            'tp1_multiplier': ParameterBounds('tp1_multiplier', 0.15, 0.4, float, 0.025),
            'tp2_multiplier': ParameterBounds('tp2_multiplier', 0.3, 0.7, float, 0.05),
            'tp3_multiplier': ParameterBounds('tp3_multiplier', 0.5, 1.2, float, 0.1),
            
            # Trailing stop parameters - give trades room to breathe
            'tsl_activation_pips': ParameterBounds('tsl_activation_pips', 6, 15, int),
            'tsl_min_profit_pips': ParameterBounds('tsl_min_profit_pips', 0.5, 3, float, 0.5),
            'trailing_atr_multiplier': ParameterBounds('trailing_atr_multiplier', 0.7, 1.5, float, 0.1),
            
            # Exit on signal flip - maybe disable since it's rarely used
            'exit_on_signal_flip': ParameterBounds('exit_on_signal_flip', 0, 1, int),
            'signal_flip_min_profit_pips': ParameterBounds('signal_flip_min_profit_pips', -1.0, 3.0, float, 0.5),
            'signal_flip_partial_exit_percent': ParameterBounds('signal_flip_partial_exit_percent', 0.3, 1.0, float, 0.1),
            
            # Partial profit parameters - crucial for scalping
            'partial_profit_sl_distance_ratio': ParameterBounds('partial_profit_sl_distance_ratio', 0.25, 0.6, float, 0.05),
            'partial_profit_size_percent': ParameterBounds('partial_profit_size_percent', 0.4, 0.8, float, 0.05),
            
            # Add relaxed mode option for more trades
            'use_relaxed_mode': ParameterBounds('use_relaxed_mode', 0, 1, int),
        }


class IntelligentOptimizer(ABC):
    """Base class for intelligent optimization strategies"""
    
    def __init__(self, 
                 strategy_type: int,
                 parameter_space: Dict[str, ParameterBounds],
                 data_manager: DataManager,
                 currency: str = 'AUDUSD',
                 n_iterations: int = 20,
                 sample_size: int = 4000,
                 n_processes: int = None,
                 previous_results: Optional[List[OptimizationResult]] = None):
        
        self.strategy_type = strategy_type
        self.parameter_space = parameter_space
        self.data_manager = data_manager
        self.currency = currency
        self.n_iterations = n_iterations
        self.sample_size = sample_size
        self.n_processes = n_processes or mp.cpu_count() - 1
        
        # Results tracking - include previous results if provided
        self.results_history: List[OptimizationResult] = previous_results or []
        self.best_result: Optional[OptimizationResult] = None
        self.performance_landscape: Dict[str, List[float]] = defaultdict(list)
        
        # If we have previous results, find the best one
        if self.results_history:
            self.best_result = max(self.results_history, key=lambda r: r.fitness)
            print(f"\n📦 Loaded {len(self.results_history)} previous results")
            print(f"   Previous best fitness: {self.best_result.fitness:.3f}")
        
        # Load data once
        self.df = data_manager.load_currency_data(currency)
        
    @abstractmethod
    def optimize(self) -> OptimizationResult:
        """Run optimization and return best result"""
        pass
    
    def evaluate_parameters(self, params: Dict[str, float]) -> OptimizationResult:
        """Evaluate a single parameter set"""
        # Create strategy config
        config = self._create_strategy_config(params)
        strategy = OptimizedProdStrategy(config)
        
        # Run backtest with smaller Monte Carlo for speed
        simulator = MonteCarloSimulator(strategy, TradeAnalyzer())
        results_df, extra_data = simulator.run_simulation(
            self.df, 
            type('Config', (), {
                'n_iterations': 5,  # Fewer iterations for optimization
                'sample_size': self.sample_size,
                'realistic_costs': True,
                'use_daily_sharpe': True,
                'debug_mode': False,
                'show_plots': False,
                'save_plots': False,
                'export_trades': False,
                'calendar_analysis': False,
                'date_range': None
            })()
        )
        
        # Extract average metrics
        avg_sharpe = results_df['sharpe_ratio'].mean()
        avg_return = results_df['total_return'].mean()
        avg_win_rate = results_df['win_rate'].mean()
        avg_max_dd = results_df['max_drawdown'].mean()
        avg_pf = results_df['profit_factor'].mean()
        avg_trades = results_df['total_trades'].mean()
        
        return OptimizationResult(
            params=params,
            sharpe_ratio=avg_sharpe,
            total_return=avg_return,
            win_rate=avg_win_rate,
            max_drawdown=avg_max_dd,
            profit_factor=avg_pf,
            total_trades=int(avg_trades),
            iteration=len(self.results_history)
        )
    
    def _create_strategy_config(self, params: Dict[str, float]) -> OptimizedStrategyConfig:
        """Create strategy config from parameters"""
        if self.strategy_type == 1:
            # Ultra-Tight Risk Management base config
            config = OptimizedStrategyConfig(
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
        else:
            # Scalping base config
            config = OptimizedStrategyConfig(
                initial_capital=1_000_000,
                risk_per_trade=params.get('risk_per_trade', 0.001),
                sl_min_pips=params.get('sl_min_pips', 3.0),
                sl_max_pips=params.get('sl_max_pips', 5.0),
                sl_atr_multiplier=params.get('sl_atr_multiplier', 0.5),
                tp_atr_multipliers=(
                    params.get('tp1_multiplier', 0.1),
                    params.get('tp2_multiplier', 0.2),
                    params.get('tp3_multiplier', 0.3)
                ),
                max_tp_percent=0.002,
                tsl_activation_pips=params.get('tsl_activation_pips', 8),
                tsl_min_profit_pips=params.get('tsl_min_profit_pips', 0.5),
                tsl_initial_buffer_multiplier=0.5,
                trailing_atr_multiplier=params.get('trailing_atr_multiplier', 0.8),
                tp_range_market_multiplier=0.3,
                tp_trend_market_multiplier=0.5,
                tp_chop_market_multiplier=0.2,
                sl_range_market_multiplier=0.5,
                exit_on_signal_flip=bool(params.get('exit_on_signal_flip', 1)),
                signal_flip_min_profit_pips=params.get('signal_flip_min_profit_pips', 0.0),
                signal_flip_min_time_hours=0.0,
                signal_flip_partial_exit_percent=params.get('signal_flip_partial_exit_percent', 1.0),
                partial_profit_before_sl=True,
                partial_profit_sl_distance_ratio=params.get('partial_profit_sl_distance_ratio', 0.3),
                partial_profit_size_percent=params.get('partial_profit_size_percent', 0.7),
                intelligent_sizing=False,
                sl_volatility_adjustment=True,
                relaxed_mode=bool(params.get('use_relaxed_mode', 0)),
                realistic_costs=True,
                verbose=False,
                debug_decisions=False,
                use_daily_sharpe=True
            )
        
        return config
    
    def save_results(self, filename: Optional[str] = None):
        """Save optimization results to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.makedirs('optimizer_results', exist_ok=True)
            filename = f'optimizer_results/optimization_results_strategy{self.strategy_type}_{timestamp}.json'
        
        results_data = {
            'strategy_type': self.strategy_type,
            'currency': self.currency,
            'n_iterations': self.n_iterations,
            'sample_size': self.sample_size,
            'best_result': {
                'params': self.best_result.params,
                'sharpe_ratio': self.best_result.sharpe_ratio,
                'total_return': self.best_result.total_return,
                'win_rate': self.best_result.win_rate,
                'max_drawdown': self.best_result.max_drawdown,
                'profit_factor': self.best_result.profit_factor,
                'total_trades': self.best_result.total_trades,
                'fitness': self.best_result.fitness
            } if self.best_result else None,
            'all_results': [
                {
                    'iteration': r.iteration,
                    'params': r.params,
                    'sharpe_ratio': r.sharpe_ratio,
                    'fitness': r.fitness
                } for r in self.results_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def plot_optimization_progress(self, save_only=True):
        """Plot optimization progress"""
        if not self.results_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Optimization Progress - Strategy {self.strategy_type}', fontsize=16)
        
        iterations = [r.iteration for r in self.results_history]
        
        # Sharpe ratio progress
        sharpes = [r.sharpe_ratio for r in self.results_history]
        axes[0, 0].plot(iterations, sharpes, 'b-', alpha=0.6)
        axes[0, 0].scatter(iterations, sharpes, c='blue', alpha=0.6)
        if self.best_result:
            best_idx = [i for i, r in enumerate(self.results_history) if r == self.best_result][0]
            axes[0, 0].scatter(iterations[best_idx], sharpes[best_idx], c='red', s=100, marker='*')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Fitness score progress
        fitness = [r.fitness for r in self.results_history]
        axes[0, 1].plot(iterations, fitness, 'g-', alpha=0.6)
        axes[0, 1].scatter(iterations, fitness, c='green', alpha=0.6)
        if self.best_result:
            axes[0, 1].scatter(iterations[best_idx], fitness[best_idx], c='red', s=100, marker='*')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Fitness Score')
        axes[0, 1].set_title('Fitness Score Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win rate vs Max DD scatter
        win_rates = [r.win_rate for r in self.results_history]
        max_dds = [r.max_drawdown for r in self.results_history]
        scatter = axes[1, 0].scatter(max_dds, win_rates, c=sharpes, cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Max Drawdown %')
        axes[1, 0].set_ylabel('Win Rate %')
        axes[1, 0].set_title('Win Rate vs Max Drawdown')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Sharpe Ratio')
        
        # Parameter importance (if enough results)
        if len(self.results_history) > 10:
            # Calculate correlation between parameters and fitness
            param_names = list(self.parameter_space.keys())
            correlations = []
            for param in param_names:
                param_values = [r.params.get(param, 0) for r in self.results_history]
                correlation = np.corrcoef(param_values, fitness)[0, 1]
                correlations.append(correlation)
            
            # Plot top parameters by absolute correlation
            sorted_idx = np.argsort(np.abs(correlations))[-8:]  # Top 8
            y_pos = np.arange(len(sorted_idx))
            axes[1, 1].barh(y_pos, [correlations[i] for i in sorted_idx])
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels([param_names[i] for i in sorted_idx])
            axes[1, 1].set_xlabel('Correlation with Fitness')
            axes[1, 1].set_title('Parameter Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('optimizer_plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'optimizer_plots/optimization_progress_strategy{self.strategy_type}_{timestamp}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {plot_filename}")
        
        if not save_only:
            plt.show()
        else:
            plt.close(fig)


class BayesianOptimizer(IntelligentOptimizer):
    """Bayesian optimization with Gaussian Process surrogate model"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start with higher exploration if no previous results
        self.exploration_weight = 2.0 if not self.results_history else 1.0
        
    def optimize(self) -> OptimizationResult:
        """Run Bayesian optimization"""
        print(f"\n🧠 Starting Bayesian Optimization for Strategy {self.strategy_type}")
        print(f"   Currency: {self.currency}")
        print(f"   Iterations: {self.n_iterations}")
        print(f"   Processes: {self.n_processes}")
        print("="*60)
        
        # Initial random exploration
        n_initial = min(10, self.n_iterations // 3)
        print(f"\n🎲 Initial random exploration ({n_initial} iterations)...")
        
        for i in range(n_initial):
            # Random sample
            params = self._random_sample()
            result = self.evaluate_parameters(params)
            self._update_results(result)
            self._print_iteration_summary(i + 1, result)
        
        # Bayesian optimization iterations
        print(f"\n🎯 Starting Bayesian-guided optimization...")
        
        for i in range(n_initial, self.n_iterations):
            # Use acquisition function to select next parameters
            params = self._select_next_parameters()
            result = self.evaluate_parameters(params)
            self._update_results(result)
            self._print_iteration_summary(i + 1, result)
            
            # Adaptive exploration
            if i % 5 == 0:
                self._adapt_exploration()
        
        print(f"\n✅ Optimization complete!")
        self._print_final_summary()
        
        return self.best_result
    
    def _random_sample(self) -> Dict[str, float]:
        """Generate random parameter sample"""
        params = {}
        for name, bounds in self.parameter_space.items():
            params[name] = bounds.round_value(bounds.sample())
        return params
    
    def _select_next_parameters(self) -> Dict[str, float]:
        """Select next parameters using Upper Confidence Bound acquisition"""
        # Simple UCB implementation - could be enhanced with GP
        n_candidates = 100
        candidates = []
        
        for _ in range(n_candidates):
            params = self._random_sample()
            
            # Calculate expected improvement based on nearby results
            ucb_score = self._calculate_ucb(params)
            candidates.append((params, ucb_score))
        
        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_ucb(self, params: Dict[str, float]) -> float:
        """Calculate Upper Confidence Bound for parameters"""
        if not self.results_history:
            return 0.0
        
        # Find similar parameters in history
        distances = []
        for result in self.results_history:
            dist = self._parameter_distance(params, result.params)
            distances.append((dist, result.fitness))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Weighted average of nearby points
        if distances[0][0] < 0.01:  # Very close to existing point
            return -1.0  # Discourage
        
        # Use k-nearest neighbors
        k = min(5, len(distances))
        weights = [1.0 / (d[0] + 0.1) for d in distances[:k]]
        total_weight = sum(weights)
        
        mean_fitness = sum(w * d[1] for w, d in zip(weights, distances[:k])) / total_weight
        
        # Add exploration bonus based on distance to nearest point
        exploration_bonus = self.exploration_weight * distances[0][0]
        
        return mean_fitness + exploration_bonus
    
    def _parameter_distance(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        """Calculate normalized distance between parameter sets"""
        distance = 0.0
        for name, bounds in self.parameter_space.items():
            val1 = params1.get(name, 0)
            val2 = params2.get(name, 0)
            # Normalize by parameter range
            normalized_diff = (val1 - val2) / (bounds.max_value - bounds.min_value)
            distance += normalized_diff ** 2
        return np.sqrt(distance)
    
    def _adapt_exploration(self):
        """Adapt exploration based on progress"""
        # Reduce exploration as we converge
        recent_fitness = [r.fitness for r in self.results_history[-5:]]
        if len(recent_fitness) >= 5:
            fitness_std = np.std(recent_fitness)
            if fitness_std < 0.05:  # Converging
                self.exploration_weight *= 0.8
                print(f"   📉 Reduced exploration weight to {self.exploration_weight:.2f}")
    
    def _update_results(self, result: OptimizationResult):
        """Update results history and best result"""
        self.results_history.append(result)
        
        if self.best_result is None or result.fitness > self.best_result.fitness:
            self.best_result = result
            print(f"   🌟 New best! Fitness: {result.fitness:.3f}")
    
    def _print_iteration_summary(self, iteration: int, result: OptimizationResult):
        """Print summary of iteration"""
        print(f"\n📊 Iteration {iteration}/{self.n_iterations}:")
        print(f"   Sharpe: {result.sharpe_ratio:.3f}, Return: {result.total_return:.1f}%, "
              f"Win Rate: {result.win_rate:.1f}%, DD: {result.max_drawdown:.1f}%")
        print(f"   Fitness: {result.fitness:.3f}, Trades: {result.total_trades}")
    
    def _print_final_summary(self):
        """Print final optimization summary"""
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION SUMMARY")
        print("="*60)
        
        if self.best_result:
            print(f"\nBest Result:")
            print(f"  Fitness Score: {self.best_result.fitness:.3f}")
            print(f"  Sharpe Ratio: {self.best_result.sharpe_ratio:.3f}")
            print(f"  Total Return: {self.best_result.total_return:.1f}%")
            print(f"  Win Rate: {self.best_result.win_rate:.1f}%")
            print(f"  Max Drawdown: {self.best_result.max_drawdown:.1f}%")
            print(f"  Profit Factor: {self.best_result.profit_factor:.2f}")
            print(f"  Total Trades: {self.best_result.total_trades}")
            
            print(f"\nOptimal Parameters:")
            for param, value in sorted(self.best_result.params.items()):
                print(f"  {param}: {value}")


class GridSearchOptimizer(IntelligentOptimizer):
    """Intelligent grid search with adaptive refinement"""
    
    def __init__(self, *args, grid_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size
        self.refinement_factor = 0.5
        
    def optimize(self) -> OptimizationResult:
        """Run adaptive grid search optimization"""
        print(f"\n🔍 Starting Adaptive Grid Search for Strategy {self.strategy_type}")
        print(f"   Currency: {self.currency}")
        print(f"   Initial grid size: {self.grid_size}")
        print(f"   Processes: {self.n_processes}")
        print("="*60)
        
        # Initial coarse grid
        current_bounds = dict(self.parameter_space)
        refinement_level = 0
        
        while len(self.results_history) < self.n_iterations:
            print(f"\n🎯 Refinement level {refinement_level}")
            
            # Generate grid points
            grid_points = self._generate_grid(current_bounds)
            
            # Evaluate in parallel
            with Pool(self.n_processes) as pool:
                results = pool.map(self.evaluate_parameters, grid_points)
            
            # Update results
            for result in results:
                result.iteration = len(self.results_history)
                self._update_results(result)
                
            # Find best region and refine
            if len(self.results_history) < self.n_iterations:
                current_bounds = self._refine_bounds(current_bounds)
                refinement_level += 1
        
        print(f"\n✅ Optimization complete!")
        self._print_final_summary()
        
        return self.best_result
    
    def _generate_grid(self, bounds: Dict[str, ParameterBounds]) -> List[Dict[str, float]]:
        """Generate grid points for current bounds"""
        param_names = list(bounds.keys())
        param_values = []
        
        for name, bound in bounds.items():
            if bound.param_type == int:
                values = np.linspace(bound.min_value, bound.max_value, self.grid_size)
                values = [int(round(v)) for v in values]
            else:
                values = np.linspace(bound.min_value, bound.max_value, self.grid_size)
                values = [bound.round_value(v) for v in values]
            param_values.append(values)
        
        # Generate all combinations
        grid_points = []
        from itertools import product
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            grid_points.append(params)
        
        print(f"   Generated {len(grid_points)} grid points")
        return grid_points
    
    def _refine_bounds(self, current_bounds: Dict[str, ParameterBounds]) -> Dict[str, ParameterBounds]:
        """Refine bounds around best result"""
        if not self.best_result:
            return current_bounds
        
        new_bounds = {}
        for name, bound in current_bounds.items():
            best_value = self.best_result.params.get(name, (bound.min_value + bound.max_value) / 2)
            
            # Calculate new range
            current_range = bound.max_value - bound.min_value
            new_range = current_range * self.refinement_factor
            
            # Center around best value
            new_min = max(bound.min_value, best_value - new_range / 2)
            new_max = min(bound.max_value, best_value + new_range / 2)
            
            new_bounds[name] = ParameterBounds(
                name=name,
                min_value=new_min,
                max_value=new_max,
                param_type=bound.param_type,
                step_size=bound.step_size,
                is_percentage=bound.is_percentage
            )
        
        return new_bounds
    
    def _update_results(self, result: OptimizationResult):
        """Update results history and best result"""
        self.results_history.append(result)
        
        if self.best_result is None or result.fitness > self.best_result.fitness:
            self.best_result = result
    
    def _print_final_summary(self):
        """Print final optimization summary"""
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION SUMMARY")
        print("="*60)
        
        if self.best_result:
            print(f"\nBest Result:")
            print(f"  Fitness Score: {self.best_result.fitness:.3f}")
            print(f"  Sharpe Ratio: {self.best_result.sharpe_ratio:.3f}")
            print(f"  Total Return: {self.best_result.total_return:.1f}%")
            print(f"  Win Rate: {self.best_result.win_rate:.1f}%")
            print(f"  Max Drawdown: {self.best_result.max_drawdown:.1f}%")
            print(f"  Profit Factor: {self.best_result.profit_factor:.2f}")
            print(f"  Total Trades: {self.best_result.total_trades}")
            
            print(f"\nOptimal Parameters:")
            for param, value in sorted(self.best_result.params.items()):
                print(f"  {param}: {value}")


def load_previous_results(strategy_type: int, currency: str) -> List[OptimizationResult]:
    """Load previous optimization results if they exist"""
    previous_results = []
    
    # Check for existing result files
    if os.path.exists('optimizer_results'):
        pattern = f'optimization_results_strategy{strategy_type}_*.json'
        import glob
        result_files = glob.glob(os.path.join('optimizer_results', pattern))
        
        for file in sorted(result_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    if data.get('currency') == currency:
                        # Convert saved results to OptimizationResult objects
                        for result in data.get('all_results', []):
                            opt_result = OptimizationResult(
                                params=result['params'],
                                sharpe_ratio=result['sharpe_ratio'],
                                total_return=result.get('total_return', 0),
                                win_rate=result.get('win_rate', 0),
                                max_drawdown=result.get('max_drawdown', 0),
                                profit_factor=result.get('profit_factor', 0),
                                total_trades=result.get('total_trades', 0),
                                iteration=len(previous_results)
                            )
                            previous_results.append(opt_result)
                print(f"\n📁 Loaded {len(previous_results)} results from {file}")
            except Exception as e:
                print(f"\n⚠️  Could not load {file}: {str(e)}")
    
    return previous_results


def run_optimization(strategy_type: int, 
                    optimization_method: str = 'bayesian',
                    currency: str = 'AUDUSD',
                    n_iterations: int = 30,
                    sample_size: int = 4000,
                    use_previous_results: bool = True):
    """Run optimization for a specific strategy"""
    
    # Get parameter space
    if strategy_type == 1:
        param_space = ParameterSpace.get_strategy1_space()
        strategy_name = "Ultra-Tight Risk Management"
    else:
        param_space = ParameterSpace.get_strategy2_space()
        strategy_name = "Scalping Strategy"
    
    print(f"\n🚀 STARTING OPTIMIZATION")
    print(f"   Strategy: {strategy_name}")
    print(f"   Method: {optimization_method.upper()}")
    print(f"   Currency: {currency}")
    print(f"   Parameters to optimize: {len(param_space)}")
    
    # Create data manager
    data_manager = DataManager()
    
    # Load previous results if requested
    previous_results = []
    if use_previous_results:
        previous_results = load_previous_results(strategy_type, currency)
    
    # Create optimizer
    if optimization_method == 'bayesian':
        optimizer = BayesianOptimizer(
            strategy_type=strategy_type,
            parameter_space=param_space,
            data_manager=data_manager,
            currency=currency,
            n_iterations=n_iterations,
            sample_size=sample_size,
            previous_results=previous_results
        )
    elif optimization_method == 'grid':
        optimizer = GridSearchOptimizer(
            strategy_type=strategy_type,
            parameter_space=param_space,
            data_manager=data_manager,
            currency=currency,
            n_iterations=n_iterations,
            sample_size=sample_size,
            grid_size=3,
            previous_results=previous_results
        )
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    # Run optimization
    start_time = time.time()
    best_result = optimizer.optimize()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏱️  Optimization completed in {elapsed_time/60:.1f} minutes")
    
    # Save results
    optimizer.save_results()
    
    # Plot progress (save only, no display)
    optimizer.plot_optimization_progress(save_only=True)
    
    return optimizer, best_result


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent Strategy Optimizer")
    parser.add_argument('--strategy', type=int, choices=[1, 2], default=1,
                       help='Strategy type: 1=Ultra-Tight Risk, 2=Scalping')
    parser.add_argument('--method', type=str, choices=['bayesian', 'grid'], default='bayesian',
                       help='Optimization method')
    parser.add_argument('--currency', type=str, default='AUDUSD',
                       help='Currency pair to optimize on')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Number of optimization iterations')
    parser.add_argument('--sample-size', type=int, default=4000,
                       help='Sample size for backtesting')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer, best_result = run_optimization(
        strategy_type=args.strategy,
        optimization_method=args.method,
        currency=args.currency,
        n_iterations=args.iterations,
        sample_size=args.sample_size
    )