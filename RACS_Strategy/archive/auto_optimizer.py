"""
Autonomous RACS Strategy Optimizer

This module implements a self-improving trading strategy that:
1. Runs Monte Carlo backtests on random data samples
2. Analyzes performance metrics
3. Adjusts parameters based on results
4. Iterates until Sharpe ratio > 1.0
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import backtrader as bt
from dataclasses import dataclass, asdict
import random
import subprocess
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tic import TIC
from racs_strategy import RACSStrategy
from backtest_racs import PandasData_Custom


@dataclass
class OptimizationResult:
    """Store results from each optimization iteration"""
    iteration: int
    timestamp: str
    params: dict
    sharpe_ratio: float
    total_return: float
    win_rate: float
    max_drawdown: float
    trades: int
    sample_info: dict
    yearly_performance: dict
    
    def to_dict(self):
        return asdict(self)


class ParameterOptimizer:
    """Intelligent parameter optimizer based on performance metrics"""
    
    def __init__(self):
        self.param_ranges = {
            'base_risk_pct': (0.005, 0.02, 0.001),  # (min, max, step)
            'min_confidence': (50, 80, 5),
            'min_nti_confidence': (60, 85, 5),
            'min_slope_power': (15, 30, 5),
            'range_penetration': (0.01, 0.05, 0.01),
            'range_target_pct': (0.6, 0.9, 0.1),
            'atr_stop_multi_trend': (0.5, 1.5, 0.25),
            'atr_stop_multi_range': (0.3, 0.7, 0.1),
            'efficiency_threshold': (0.2, 0.4, 0.05),
        }
        
        self.performance_history = []
        
    def suggest_improvements(self, result: OptimizationResult) -> dict:
        """Suggest parameter improvements based on performance metrics"""
        new_params = result.params.copy()
        
        # Analyze failure modes
        if result.sharpe_ratio < 0:
            # Losing money - be more conservative
            new_params['min_confidence'] = min(80, result.params['min_confidence'] + 10)
            new_params['min_nti_confidence'] = min(85, result.params['min_nti_confidence'] + 5)
            new_params['base_risk_pct'] = max(0.005, result.params['base_risk_pct'] * 0.8)
            
        elif result.win_rate < 50:
            # Low win rate - tighten entry filters
            new_params['min_slope_power'] = min(30, result.params['min_slope_power'] + 5)
            new_params['efficiency_threshold'] = min(0.4, result.params['efficiency_threshold'] + 0.05)
            
        elif result.max_drawdown > 15:
            # High drawdown - improve risk management
            new_params['atr_stop_multi_trend'] = max(0.5, result.params['atr_stop_multi_trend'] - 0.25)
            new_params['base_risk_pct'] = max(0.005, result.params['base_risk_pct'] * 0.9)
            
        elif result.trades < 20:
            # Too few trades - loosen filters slightly
            new_params['min_confidence'] = max(50, result.params['min_confidence'] - 5)
            new_params['min_slope_power'] = max(15, result.params['min_slope_power'] - 5)
            
        else:
            # Good performance but not at target - fine tune
            if result.sharpe_ratio < 0.5:
                # Focus on quality
                new_params['min_nti_confidence'] = min(85, result.params['min_nti_confidence'] + 2)
            elif result.sharpe_ratio < 1.0:
                # Balance quality and quantity
                if result.trades < 50:
                    new_params['min_confidence'] = max(50, result.params['min_confidence'] - 2)
                else:
                    new_params['efficiency_threshold'] = min(0.4, result.params['efficiency_threshold'] + 0.02)
        
        # Apply some randomization to explore parameter space
        for param, value in new_params.items():
            if param in self.param_ranges and random.random() < 0.2:  # 20% chance
                min_val, max_val, step = self.param_ranges[param]
                noise = random.choice([-step, 0, step])
                new_params[param] = max(min_val, min(max_val, value + noise))
        
        return new_params


class MonteCarloBacktester:
    """Run efficient Monte Carlo backtests on random data samples"""
    
    def __init__(self, data_path: str, sample_size: int = 5000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.full_data = None
        self.prepared_data = None
        
    def load_and_prepare_data(self):
        """Load data once and prepare indicators"""
        print("Loading full dataset...")
        self.full_data = pd.read_csv(self.data_path, parse_dates=['DateTime'], index_col='DateTime')
        
        # Add indicators once to the full dataset
        print("Adding indicators to full dataset (this may take a moment)...")
        self.prepared_data = self.full_data.copy()
        
        # Add indicators with error handling
        try:
            self.prepared_data = TIC.add_intelligent_chop(self.prepared_data, inplace=True)
            self.prepared_data = TIC.add_market_bias(self.prepared_data, inplace=True)
            self.prepared_data = TIC.add_neuro_trend_intelligent(self.prepared_data, inplace=True)
            self.prepared_data = TIC.add_super_trend(self.prepared_data, inplace=True)
            self.prepared_data = TIC.add_fractal_sr(self.prepared_data, inplace=True)
        except Exception as e:
            print(f"Error adding indicators: {e}")
            raise
        
        # Remove NaN rows
        original_len = len(self.prepared_data)
        self.prepared_data = self.prepared_data.dropna()
        print(f"Dropped {original_len - len(self.prepared_data)} NaN rows")
        
        if len(self.prepared_data) < self.sample_size:
            print(f"Warning: Prepared data ({len(self.prepared_data)} rows) is smaller than sample size ({self.sample_size})")
            self.sample_size = min(self.sample_size, len(self.prepared_data) - 1000)
        
    def get_random_sample(self) -> Tuple[pd.DataFrame, dict]:
        """Get a random contiguous sample from the prepared data"""
        if self.prepared_data is None:
            self.load_and_prepare_data()
        
        max_start = len(self.prepared_data) - self.sample_size
        if max_start <= 0:
            # Use all data if sample size is too large
            sample = self.prepared_data.copy()
            start_idx = 0
        else:
            start_idx = random.randint(0, max_start)
            sample = self.prepared_data.iloc[start_idx:start_idx + self.sample_size].copy()
        
        # Get sample info
        sample_info = {
            'start_date': str(sample.index[0]),
            'end_date': str(sample.index[-1]),
            'num_rows': len(sample),
            'start_idx': start_idx,
            'years': sorted(sample.index.year.unique().tolist())
        }
        
        return sample, sample_info
    
    def run_single_backtest(self, sample: pd.DataFrame, params: dict) -> dict:
        """Run a single backtest with given parameters"""
        cerebro = bt.Cerebro()
        
        # Add strategy with parameters
        cerebro.addstrategy(RACSStrategy, **params)
        
        # Create data feed
        data = PandasData_Custom(dataname=sample)
        cerebro.adddata(data)
        
        # Set broker parameters
        initial_cash = params.get('account_size', 10000)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                           riskfreerate=0.01, annualize=True, timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Years)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run backtest
        results = cerebro.run()
        strat = results[0]
        
        # Extract metrics
        final_value = cerebro.broker.getvalue()
        total_return = (final_value / initial_cash - 1) * 100
        
        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', 0) if sharpe else 0
        
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.max.drawdown if hasattr(drawdown, 'max') else 0
        
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.total.total if hasattr(trades.total, 'total') else 0
        won_trades = trades.won.total if hasattr(trades.won, 'total') else 0
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Get yearly returns
        yearly_returns = strat.analyzers.returns.get_analysis()
        yearly_performance = {str(year): ret * 100 for year, ret in yearly_returns.items() 
                            if isinstance(year, int)}
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trades': total_trades,
            'final_value': final_value,
            'yearly_performance': yearly_performance
        }


class AutoOptimizer:
    """Main optimizer that orchestrates the self-improvement loop"""
    
    def __init__(self, data_path: str, target_sharpe: float = 1.0, max_iterations: int = 100):
        self.data_path = data_path
        self.target_sharpe = target_sharpe
        self.max_iterations = max_iterations
        
        self.backtester = MonteCarloBacktester(data_path)
        self.optimizer = ParameterOptimizer()
        
        self.results_history = []
        self.best_result = None
        self.iteration = 0
        
        # Initial parameters
        self.current_params = {
            'account_size': 10000,
            'base_risk_pct': 0.01,
            'max_positions': 3,
            'min_confidence': 60.0,
            'yellow_confidence': 70.0,
            'min_nti_confidence': 70.0,
            'min_slope_power': 20.0,
            'range_penetration': 0.02,
            'range_target_pct': 0.8,
            'yellow_size_factor': 0.5,
            'blue_size_factor': 0.5,
            'high_vol_reduction': 0.5,
            'low_vol_bonus': 1.2,
            'golden_setup_bonus': 1.5,
            'atr_stop_multi_trend': 1.0,
            'atr_stop_multi_range': 0.5,
            'time_stop_multi': 4,
            'max_atr_normalized': 3.0,
            'max_bandwidth': 5.0,
            'min_bandwidth_bonus': 2.0,
            'efficiency_threshold': 0.3,
            'range_lookback': 20,
            'min_range_bars': 5,
        }
    
    def run_optimization_loop(self):
        """Main optimization loop"""
        print(f"\nStarting optimization loop. Target Sharpe: {self.target_sharpe}")
        print("=" * 80)
        
        # Load data once
        self.backtester.load_and_prepare_data()
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n--- Iteration {self.iteration} ---")
            
            # Get random sample
            sample, sample_info = self.backtester.get_random_sample()
            print(f"Sample: {sample_info['start_date']} to {sample_info['end_date']} ({sample_info['num_rows']} rows)")
            
            # Run backtest
            try:
                metrics = self.backtester.run_single_backtest(sample, self.current_params)
            except Exception as e:
                print(f"Backtest failed: {e}")
                continue
            
            # Create result
            result = OptimizationResult(
                iteration=self.iteration,
                timestamp=datetime.now().isoformat(),
                params=self.current_params.copy(),
                sharpe_ratio=metrics['sharpe_ratio'],
                total_return=metrics['total_return'],
                win_rate=metrics['win_rate'],
                max_drawdown=metrics['max_drawdown'],
                trades=metrics['trades'],
                sample_info=sample_info,
                yearly_performance=metrics['yearly_performance']
            )
            
            self.results_history.append(result)
            
            # Print results
            print(f"Sharpe: {result.sharpe_ratio:.3f}, Return: {result.total_return:.1f}%, " +
                  f"Win Rate: {result.win_rate:.1f}%, Trades: {result.trades}")
            
            # Update best result
            if self.best_result is None or result.sharpe_ratio > self.best_result.sharpe_ratio:
                self.best_result = result
                print(f"NEW BEST! Sharpe: {result.sharpe_ratio:.3f}")
                self.save_progress()
            
            # Check if target reached
            if result.sharpe_ratio >= self.target_sharpe:
                print(f"\nTARGET REACHED! Sharpe: {result.sharpe_ratio:.3f}")
                self.save_final_results()
                break
            
            # Get improved parameters
            self.current_params = self.optimizer.suggest_improvements(result)
            
            # Save progress periodically
            if self.iteration % 5 == 0:
                self.save_progress()
                self.commit_to_git(f"Optimization iteration {self.iteration}")
        
        if self.iteration >= self.max_iterations:
            print(f"\nMax iterations reached. Best Sharpe: {self.best_result.sharpe_ratio:.3f}")
            self.save_final_results()
    
    def save_progress(self):
        """Save current progress to files"""
        # Save results history
        history_data = [r.to_dict() for r in self.results_history]
        with open('optimization_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save best parameters
        if self.best_result:
            with open('best_parameters.json', 'w') as f:
                json.dump({
                    'sharpe_ratio': self.best_result.sharpe_ratio,
                    'parameters': self.best_result.params,
                    'metrics': {
                        'total_return': self.best_result.total_return,
                        'win_rate': self.best_result.win_rate,
                        'max_drawdown': self.best_result.max_drawdown,
                        'trades': self.best_result.trades
                    }
                }, f, indent=2)
        
        # Update CLAUDE.md
        self.update_claude_md()
    
    def update_claude_md(self):
        """Update CLAUDE.md with current progress"""
        content = f"""# RACS Strategy Optimization Progress

## Current Status
- Iteration: {self.iteration}
- Best Sharpe: {self.best_result.sharpe_ratio:.3f if self.best_result else 'N/A'}
- Target Sharpe: {self.target_sharpe}

## Best Parameters Found
```json
{json.dumps(self.best_result.params if self.best_result else self.current_params, indent=2)}
```

## Best Performance Metrics
{f'''- Sharpe Ratio: {self.best_result.sharpe_ratio:.3f}
- Total Return: {self.best_result.total_return:.1f}%
- Win Rate: {self.best_result.win_rate:.1f}%
- Max Drawdown: {self.best_result.max_drawdown:.1f}%
- Total Trades: {self.best_result.trades}''' if self.best_result else 'No results yet'}

## Recent Progress
{self._format_recent_progress()}

## Next Steps
1. Continue optimization from iteration {self.iteration + 1}
2. Current focus: {self._get_optimization_focus()}
3. Run: `python auto_optimizer.py` to continue

## To Resume
```python
# Load previous results
with open('optimization_history.json', 'r') as f:
    history = json.load(f)
    
# Continue from last iteration
optimizer = AutoOptimizer('data/AUDUSD_MASTER_15M.csv')
optimizer.iteration = {self.iteration}
optimizer.current_params = {json.dumps(self.current_params)}
optimizer.run_optimization_loop()
```
"""
        
        with open('CLAUDE.md', 'w') as f:
            f.write(content)
    
    def _format_recent_progress(self) -> str:
        """Format recent optimization progress"""
        if not self.results_history:
            return "No results yet"
        
        recent = self.results_history[-5:]  # Last 5 iterations
        lines = []
        for r in recent:
            lines.append(f"- Iteration {r.iteration}: Sharpe {r.sharpe_ratio:.3f}, " +
                        f"Return {r.total_return:.1f}%, Trades {r.trades}")
        return '\n'.join(lines)
    
    def _get_optimization_focus(self) -> str:
        """Determine current optimization focus"""
        if not self.best_result:
            return "Initial exploration"
        
        if self.best_result.sharpe_ratio < 0:
            return "Achieving positive returns"
        elif self.best_result.win_rate < 50:
            return "Improving win rate"
        elif self.best_result.max_drawdown > 15:
            return "Reducing drawdown"
        elif self.best_result.trades < 30:
            return "Increasing trade frequency"
        else:
            return "Fine-tuning for higher Sharpe"
    
    def save_final_results(self):
        """Save final optimization results"""
        # Detailed report
        report = {
            'optimization_summary': {
                'total_iterations': self.iteration,
                'target_sharpe': self.target_sharpe,
                'achieved_sharpe': self.best_result.sharpe_ratio if self.best_result else None,
                'success': self.best_result.sharpe_ratio >= self.target_sharpe if self.best_result else False
            },
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'parameter_evolution': self._analyze_parameter_evolution(),
            'performance_by_year': self._analyze_yearly_performance()
        }
        
        with open('optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nOptimization complete. Report saved to optimization_report.json")
    
    def _analyze_parameter_evolution(self) -> dict:
        """Analyze how parameters evolved during optimization"""
        if not self.results_history:
            return {}
        
        param_evolution = {}
        for param in self.current_params.keys():
            if param == 'account_size':
                continue
            values = [r.params.get(param, 0) for r in self.results_history]
            param_evolution[param] = {
                'initial': values[0],
                'final': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values)
            }
        
        return param_evolution
    
    def _analyze_yearly_performance(self) -> dict:
        """Analyze performance by year across all tests"""
        yearly_data = {}
        
        for result in self.results_history:
            for year, perf in result.yearly_performance.items():
                if year not in yearly_data:
                    yearly_data[year] = []
                yearly_data[year].append(perf)
        
        yearly_summary = {}
        for year, perfs in yearly_data.items():
            yearly_summary[year] = {
                'mean_return': np.mean(perfs),
                'std_return': np.std(perfs),
                'min_return': min(perfs),
                'max_return': max(perfs),
                'samples': len(perfs)
            }
        
        return yearly_summary
    
    def commit_to_git(self, message: str):
        """Commit current results to git"""
        try:
            # Stage relevant files
            subprocess.run(['git', 'add', 'optimization_history.json', 
                          'best_parameters.json', 'CLAUDE.md'], 
                          capture_output=True)
            
            # Commit
            subprocess.run(['git', 'commit', '-m', f'RACS Auto-Optimization: {message}'], 
                          capture_output=True)
            
            print(f"Git commit: {message}")
        except Exception as e:
            print(f"Git commit failed: {e}")


def main():
    """Main entry point"""
    data_path = "../data/AUDUSD_MASTER_15M.csv"
    
    # Check if continuing from previous run
    if os.path.exists('optimization_history.json'):
        print("Found previous optimization history. Loading...")
        # Could implement resume functionality here
    
    # Create and run optimizer
    optimizer = AutoOptimizer(data_path, target_sharpe=1.0, max_iterations=100)
    optimizer.run_optimization_loop()


if __name__ == "__main__":
    main()