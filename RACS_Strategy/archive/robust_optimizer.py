"""
Robust RACS Strategy Optimizer

This module implements a self-improving trading strategy that:
1. Runs multiple Monte Carlo backtests to ensure robust performance
2. Targets consistent Sharpe > 1 across different market periods
3. Analyzes performance stability and adjusts for robustness
4. Accepts brief drawdowns but ensures overall profitability
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import backtrader as bt
from dataclasses import dataclass, field
import random
import subprocess
import sys
from collections import defaultdict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tic import TIC
from racs_strategy import RACSStrategy
# Import the fixed PandasData_Custom
from backtest_racs import PandasData_Custom


@dataclass
class RobustMetrics:
    """Metrics focused on robustness across periods"""
    mean_sharpe: float
    median_sharpe: float
    min_sharpe: float
    max_sharpe: float
    sharpe_std: float
    periods_above_1: int
    total_periods: int
    consistency_score: float  # % of periods with Sharpe > 1
    stability_score: float    # 1 - coefficient of variation
    profitable_years: int
    total_years: int
    worst_year_return: float
    best_year_return: float
    recovery_ability: float   # How quickly strategy recovers from drawdowns


@dataclass
class OptimizationRun:
    """Store results from a single parameter set tested across multiple samples"""
    run_id: int
    timestamp: str
    params: dict
    sample_results: List[dict] = field(default_factory=list)
    robust_metrics: Optional[RobustMetrics] = None
    
    def calculate_robust_metrics(self):
        """Calculate robustness metrics across all samples"""
        if not self.sample_results:
            return
        
        sharpes = [r['sharpe_ratio'] for r in self.sample_results]
        
        # Yearly performance analysis
        all_yearly = defaultdict(list)
        for result in self.sample_results:
            for year, ret in result.get('yearly_performance', {}).items():
                all_yearly[year].append(ret)
        
        profitable_years = sum(1 for year_rets in all_yearly.values() 
                              if np.mean(year_rets) > 0)
        
        worst_year = min(np.mean(rets) for rets in all_yearly.values()) if all_yearly else 0
        best_year = max(np.mean(rets) for rets in all_yearly.values()) if all_yearly else 0
        
        # Calculate recovery ability (average recovery time from drawdowns)
        recovery_scores = []
        for result in self.sample_results:
            if result['max_drawdown'] > 5:  # Significant drawdown
                # Simple heuristic: good recovery if profitable despite drawdown
                recovery_scores.append(1.0 if result['total_return'] > 0 else 0.0)
        recovery_ability = np.mean(recovery_scores) if recovery_scores else 1.0
        
        self.robust_metrics = RobustMetrics(
            mean_sharpe=np.mean(sharpes),
            median_sharpe=np.median(sharpes),
            min_sharpe=min(sharpes),
            max_sharpe=max(sharpes),
            sharpe_std=np.std(sharpes),
            periods_above_1=sum(1 for s in sharpes if s > 1.0),
            total_periods=len(sharpes),
            consistency_score=sum(1 for s in sharpes if s > 1.0) / len(sharpes) * 100,
            stability_score=1 - (np.std(sharpes) / np.mean(sharpes)) if np.mean(sharpes) > 0 else 0,
            profitable_years=profitable_years,
            total_years=len(all_yearly),
            worst_year_return=worst_year,
            best_year_return=best_year,
            recovery_ability=recovery_ability
        )


class RobustParameterOptimizer:
    """Optimizer focused on robust performance across market conditions"""
    
    def __init__(self):
        self.param_ranges = {
            'base_risk_pct': (0.005, 0.015, 0.0025),
            'min_confidence': (55, 75, 5),
            'min_nti_confidence': (65, 80, 5),
            'min_slope_power': (15, 25, 2.5),
            'range_penetration': (0.015, 0.035, 0.005),
            'range_target_pct': (0.7, 0.85, 0.05),
            'atr_stop_multi_trend': (0.75, 1.25, 0.125),
            'atr_stop_multi_range': (0.4, 0.6, 0.05),
            'efficiency_threshold': (0.25, 0.35, 0.025),
            'max_atr_normalized': (2.5, 3.5, 0.25),
            'yellow_confidence': (65, 75, 5),
        }
        
    def suggest_robust_improvements(self, run: OptimizationRun) -> dict:
        """Suggest improvements focused on robustness"""
        metrics = run.robust_metrics
        params = run.params.copy()
        
        if metrics is None:
            return params
        
        # Priority 1: Achieve positive mean Sharpe
        if metrics.mean_sharpe < 0.3:
            # Too aggressive, need more conservative approach
            params['base_risk_pct'] = max(0.005, params['base_risk_pct'] * 0.8)
            params['min_confidence'] = min(75, params['min_confidence'] + 5)
            params['min_nti_confidence'] = min(80, params['min_nti_confidence'] + 5)
            
        # Priority 2: Improve consistency (more periods above 1.0)
        elif metrics.consistency_score < 50:  # Less than 50% of periods above 1.0
            if metrics.min_sharpe < -0.5:
                # Has very bad periods, tighten risk management
                params['atr_stop_multi_trend'] = max(0.75, params['atr_stop_multi_trend'] - 0.125)
                params['max_atr_normalized'] = max(2.5, params['max_atr_normalized'] - 0.25)
            else:
                # Moderate adjustments for consistency
                params['efficiency_threshold'] = min(0.35, params['efficiency_threshold'] + 0.025)
                params['min_slope_power'] = min(25, params['min_slope_power'] + 2.5)
        
        # Priority 3: Reduce volatility while maintaining returns
        elif metrics.stability_score < 0.7:  # High volatility in results
            params['yellow_confidence'] = min(75, params['yellow_confidence'] + 5)
            params['range_penetration'] = max(0.015, params['range_penetration'] - 0.005)
        
        # Priority 4: Fine-tune for consistent Sharpe > 1
        else:
            # Good baseline, make small adjustments
            if metrics.mean_sharpe < 1.0:
                # Need slightly better entries
                params['min_nti_confidence'] = min(80, params['min_nti_confidence'] + 2)
            elif metrics.consistency_score < 80:
                # Good mean but not consistent enough
                params['min_confidence'] = min(75, params['min_confidence'] + 2)
            
            # If worst year is really bad, improve risk management
            if metrics.worst_year_return < -10:
                params['base_risk_pct'] = max(0.005, params['base_risk_pct'] * 0.9)
        
        # Add controlled randomization for exploration
        for param, value in params.items():
            if param in self.param_ranges and random.random() < 0.15:
                min_val, max_val, step = self.param_ranges[param]
                # Smaller random steps for fine-tuning
                noise = random.choice([-step/2, 0, step/2]) if metrics.mean_sharpe > 0.8 else random.choice([-step, 0, step])
                params[param] = max(min_val, min(max_val, value + noise))
        
        return params


class RobustMonteCarloTester:
    """Run comprehensive Monte Carlo tests for robustness"""
    
    def __init__(self, data_path: str, samples_per_test: int = 10, sample_size: int = 5000):
        self.data_path = data_path
        self.samples_per_test = samples_per_test
        self.sample_size = sample_size
        self.full_data = None
        self.prepared_data = None
        
    def load_and_prepare_data(self):
        """Load data once and prepare indicators"""
        print("Loading and preparing data...")
        self.full_data = pd.read_csv(self.data_path, parse_dates=['DateTime'], index_col='DateTime')
        
        # Create a smaller test set first to check
        test_size = min(1000, len(self.full_data))
        test_data = self.full_data.iloc[:test_size].copy()
        
        print(f"Testing indicator preparation on {test_size} rows...")
        try:
            # Test on small sample first
            test_data = TIC.add_intelligent_chop(test_data, inplace=True)
            if len(test_data) == 0:
                raise ValueError("Intelligent Chop indicator removed all data")
                
            test_data = TIC.add_market_bias(test_data, inplace=True)
            test_data = TIC.add_neuro_trend_intelligent(test_data, inplace=True)
            test_data = TIC.add_super_trend(test_data, inplace=True)
            test_data = TIC.add_fractal_sr(test_data, inplace=True)
            
            print(f"Test successful. {len(test_data)} rows remain after indicators.")
            
        except Exception as e:
            print(f"Error in test preparation: {e}")
            raise
        
        # Now prepare full dataset
        print("Preparing full dataset...")
        self.prepared_data = self.full_data.copy()
        
        # Add indicators to the full dataset at once
        print("Adding Intelligent Chop...")
        self.prepared_data = TIC.add_intelligent_chop(self.prepared_data, inplace=True)
        
        print("Adding Market Bias...")
        self.prepared_data = TIC.add_market_bias(self.prepared_data, inplace=True)
        
        print("Adding NeuroTrend Intelligent...")
        self.prepared_data = TIC.add_neuro_trend_intelligent(self.prepared_data, inplace=True)
        
        print("Adding SuperTrend...")
        self.prepared_data = TIC.add_super_trend(self.prepared_data, inplace=True)
        
        print("Adding Fractal Support/Resistance...")
        self.prepared_data = TIC.add_fractal_sr(self.prepared_data, inplace=True)
        
        # Remove NaN rows - but keep SR columns with NaN since they're sparse
        original_len = len(self.prepared_data)
        
        # Get list of columns that are not SR-related
        non_sr_columns = [col for col in self.prepared_data.columns 
                         if not col.startswith('SR_')]
        
        # Only drop rows where non-SR columns have NaN
        self.prepared_data = self.prepared_data.dropna(subset=non_sr_columns)
        
        print(f"Data prepared. {len(self.prepared_data)} rows available (dropped {original_len - len(self.prepared_data)} NaN rows)")
        print(f"Note: SR (Support/Resistance) columns may contain NaN values as they are sparse indicators.")
        
        if len(self.prepared_data) < self.sample_size * 2:
            self.sample_size = len(self.prepared_data) // 3
            print(f"Adjusted sample size to {self.sample_size}")
    
    def get_diverse_samples(self) -> List[Tuple[pd.DataFrame, dict]]:
        """Get diverse samples covering different market periods"""
        if self.prepared_data is None:
            self.load_and_prepare_data()
        
        samples = []
        data_len = len(self.prepared_data)
        
        # Strategy: Get samples from different parts of the data
        # 1. Early period sample
        # 2. Middle period samples  
        # 3. Recent period sample
        # 4. Random samples
        
        sections = min(4, self.samples_per_test)
        section_size = data_len // sections
        
        for i in range(self.samples_per_test):
            if i < sections:
                # Systematic sampling from different sections
                section_start = i * section_size
                max_start = min(section_start + section_size - self.sample_size, data_len - self.sample_size)
                start_idx = random.randint(section_start, max(section_start, max_start))
            else:
                # Random sampling
                start_idx = random.randint(0, data_len - self.sample_size)
            
            sample = self.prepared_data.iloc[start_idx:start_idx + self.sample_size].copy()
            
            sample_info = {
                'start_date': str(sample.index[0]),
                'end_date': str(sample.index[-1]),
                'num_rows': len(sample),
                'start_idx': start_idx,
                'years': sorted(sample.index.year.unique().tolist()),
                'section': i if i < sections else 'random'
            }
            
            samples.append((sample, sample_info))
        
        return samples
    
    def test_parameters(self, params: dict) -> List[dict]:
        """Test parameters on multiple diverse samples"""
        samples = self.get_diverse_samples()
        results = []
        
        print(f"Testing on {len(samples)} diverse samples...")
        
        for i, (sample, sample_info) in enumerate(samples):
            print(f"  Sample {i+1}: {sample_info['start_date'][:10]} to {sample_info['end_date'][:10]}", end='')
            
            try:
                result = self._run_single_backtest(sample, params)
                result['sample_info'] = sample_info
                results.append(result)
                print(f" - Sharpe: {result['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f" - FAILED: {str(e)}")
                # Add failed result
                results.append({
                    'sharpe_ratio': -999,
                    'total_return': -100,
                    'win_rate': 0,
                    'max_drawdown': 100,
                    'trades': 0,
                    'sample_info': sample_info,
                    'error': str(e)
                })
        
        return results
    
    def _run_single_backtest(self, sample: pd.DataFrame, params: dict) -> dict:
        """Run a single backtest"""
        cerebro = bt.Cerebro()
        
        # Add strategy
        cerebro.addstrategy(RACSStrategy, **params)
        
        # Create data feed
        data = PandasData_Custom(dataname=sample)
        cerebro.adddata(data)
        
        # Set broker
        initial_cash = params.get('account_size', 10000)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.001)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                           riskfreerate=0.01, annualize=True, timeframe=bt.TimeFrame.Days)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Years)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # Run
        results = cerebro.run()
        strat = results[0]
        
        # Extract metrics
        final_value = cerebro.broker.getvalue()
        total_return = (final_value / initial_cash - 1) * 100
        
        sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio', 0) if sharpe and 'sharperatio' in sharpe else 0
        
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.max.drawdown if hasattr(drawdown, 'max') else 0
        
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.total.total if hasattr(trades.total, 'total') else 0
        won_trades = trades.won.total if hasattr(trades.won, 'total') else 0
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Yearly returns
        yearly_returns = strat.analyzers.returns.get_analysis()
        yearly_performance = {str(year): ret * 100 for year, ret in yearly_returns.items() 
                            if isinstance(year, int)}
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trades': total_trades,
            'yearly_performance': yearly_performance
        }


class RobustAutoOptimizer:
    """Main optimizer focused on robust performance"""
    
    def __init__(self, data_path: str, target_consistency: float = 75.0, max_iterations: int = 50):
        self.data_path = data_path
        self.target_consistency = target_consistency  # % of periods with Sharpe > 1
        self.max_iterations = max_iterations
        
        self.tester = RobustMonteCarloTester(data_path, samples_per_test=10)
        self.optimizer = RobustParameterOptimizer()
        
        self.runs_history = []
        self.best_run = None
        self.iteration = 0
        
        # Starting parameters - conservative baseline
        self.current_params = {
            'account_size': 10000,
            'base_risk_pct': 0.008,
            'max_positions': 3,
            'min_confidence': 65.0,
            'yellow_confidence': 70.0,
            'min_nti_confidence': 72.0,
            'min_slope_power': 20.0,
            'range_penetration': 0.025,
            'range_target_pct': 0.75,
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
    
    def run_robust_optimization(self):
        """Main optimization loop focused on robustness"""
        print(f"\n{'='*80}")
        print(f"Starting ROBUST optimization. Target: {self.target_consistency}% consistency with Sharpe > 1")
        print(f"{'='*80}\n")
        
        # Load data once
        self.tester.load_and_prepare_data()
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}/{self.max_iterations}")
            print(f"{'='*60}")
            
            # Create run
            run = OptimizationRun(
                run_id=self.iteration,
                timestamp=datetime.now().isoformat(),
                params=self.current_params.copy()
            )
            
            # Test parameters on multiple samples
            run.sample_results = self.tester.test_parameters(self.current_params)
            
            # Calculate robust metrics
            run.calculate_robust_metrics()
            self.runs_history.append(run)
            
            # Display results
            self._display_run_results(run)
            
            # Check if best
            if self._is_better_run(run):
                self.best_run = run
                print(f"\n🎯 NEW BEST! Consistency: {run.robust_metrics.consistency_score:.1f}%")
                self.save_progress()
            
            # Check if target reached
            if run.robust_metrics.consistency_score >= self.target_consistency and run.robust_metrics.mean_sharpe > 1.0:
                print(f"\n✅ TARGET REACHED! Consistency: {run.robust_metrics.consistency_score:.1f}%")
                print(f"Mean Sharpe: {run.robust_metrics.mean_sharpe:.3f}")
                self.save_final_results()
                break
            
            # Get improved parameters
            self.current_params = self.optimizer.suggest_robust_improvements(run)
            
            # Save periodically
            if self.iteration % 3 == 0:
                self.save_progress()
                self.create_claude_md()
        
        if self.iteration >= self.max_iterations:
            print(f"\nMax iterations reached.")
            if self.best_run:
                print(f"Best consistency: {self.best_run.robust_metrics.consistency_score:.1f}%")
            self.save_final_results()
    
    def _display_run_results(self, run: OptimizationRun):
        """Display results from a run"""
        m = run.robust_metrics
        print(f"\nRobust Metrics:")
        print(f"  Mean Sharpe:    {m.mean_sharpe:.3f} (min: {m.min_sharpe:.3f}, max: {m.max_sharpe:.3f})")
        print(f"  Consistency:    {m.consistency_score:.1f}% ({m.periods_above_1}/{m.total_periods} periods > 1.0)")
        print(f"  Stability:      {m.stability_score:.3f}")
        print(f"  Profitable:     {m.profitable_years}/{m.total_years} years")
        print(f"  Year Range:     {m.worst_year_return:.1f}% to {m.best_year_return:.1f}%")
    
    def _is_better_run(self, run: OptimizationRun) -> bool:
        """Determine if this run is better than the current best"""
        if not self.best_run:
            return True
        
        current = run.robust_metrics
        best = self.best_run.robust_metrics
        
        # Priority 1: Positive mean Sharpe
        if current.mean_sharpe > 0 and best.mean_sharpe <= 0:
            return True
        if current.mean_sharpe <= 0 and best.mean_sharpe > 0:
            return False
        
        # Priority 2: Consistency score
        if abs(current.consistency_score - best.consistency_score) > 10:
            return current.consistency_score > best.consistency_score
        
        # Priority 3: Mean Sharpe (if consistency is similar)
        if abs(current.mean_sharpe - best.mean_sharpe) > 0.1:
            return current.mean_sharpe > best.mean_sharpe
        
        # Priority 4: Stability
        return current.stability_score > best.stability_score
    
    def save_progress(self):
        """Save current progress"""
        # Save runs history
        history_data = []
        for run in self.runs_history:
            run_data = {
                'run_id': run.run_id,
                'timestamp': run.timestamp,
                'params': run.params,
                'robust_metrics': {
                    'mean_sharpe': run.robust_metrics.mean_sharpe,
                    'consistency_score': run.robust_metrics.consistency_score,
                    'stability_score': run.robust_metrics.stability_score,
                    'periods_above_1': run.robust_metrics.periods_above_1,
                    'total_periods': run.robust_metrics.total_periods,
                    'worst_year': run.robust_metrics.worst_year_return,
                    'best_year': run.robust_metrics.best_year_return
                } if run.robust_metrics else None
            }
            history_data.append(run_data)
        
        with open('robust_optimization_history.json', 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save best parameters
        if self.best_run:
            with open('robust_best_parameters.json', 'w') as f:
                json.dump({
                    'params': self.best_run.params,
                    'metrics': {
                        'mean_sharpe': self.best_run.robust_metrics.mean_sharpe,
                        'consistency_score': self.best_run.robust_metrics.consistency_score,
                        'stability_score': self.best_run.robust_metrics.stability_score,
                        'min_sharpe': self.best_run.robust_metrics.min_sharpe,
                        'max_sharpe': self.best_run.robust_metrics.max_sharpe
                    }
                }, f, indent=2)
    
    def create_claude_md(self):
        """Create CLAUDE.md for persistence"""
        best_consistency = self.best_run.robust_metrics.consistency_score if self.best_run and self.best_run.robust_metrics else 0
        best_sharpe = self.best_run.robust_metrics.mean_sharpe if self.best_run and self.best_run.robust_metrics else 0
        
        content = f"""# RACS Robust Optimization Progress

## Objective
Achieve consistent Sharpe > 1 across diverse market periods with {self.target_consistency}% consistency.

## Current Status
- Iteration: {self.iteration}/{self.max_iterations}
- Best Consistency: {best_consistency:.1f}%
- Best Mean Sharpe: {best_sharpe:.3f}

## Best Parameters
```json
{json.dumps(self.best_run.params if self.best_run else self.current_params, indent=2)}
```

## Performance Summary
{self._format_performance_summary()}

## To Continue Optimization
```bash
python robust_optimizer.py
```

## Analysis Insights
{self._get_optimization_insights()}
"""
        
        with open('CLAUDE.md', 'w') as f:
            f.write(content)
    
    def _format_performance_summary(self) -> str:
        """Format performance summary"""
        if not self.best_run or not self.best_run.robust_metrics:
            return "No results yet"
        
        m = self.best_run.robust_metrics
        return f"""- Mean Sharpe: {m.mean_sharpe:.3f}
- Sharpe Range: {m.min_sharpe:.3f} to {m.max_sharpe:.3f}
- Consistency: {m.consistency_score:.1f}% periods with Sharpe > 1
- Stability Score: {m.stability_score:.3f}
- Profitable Years: {m.profitable_years}/{m.total_years}
- Worst Year: {m.worst_year_return:.1f}%
- Best Year: {m.best_year_return:.1f}%"""
    
    def _get_optimization_insights(self) -> str:
        """Get insights about optimization progress"""
        if not self.best_run:
            return "Starting optimization..."
        
        m = self.best_run.robust_metrics
        insights = []
        
        if m.mean_sharpe < 0.5:
            insights.append("- Strategy needs fundamental improvements to achieve positive returns")
        elif m.consistency_score < 50:
            insights.append("- Focus on improving consistency across different market periods")
        elif m.stability_score < 0.7:
            insights.append("- Reduce performance volatility while maintaining returns")
        elif m.worst_year_return < -10:
            insights.append("- Improve risk management during adverse market conditions")
        else:
            insights.append("- Fine-tuning parameters for optimal robustness")
        
        return '\n'.join(insights)
    
    def save_final_results(self):
        """Save comprehensive final results"""
        report = {
            'summary': {
                'total_iterations': self.iteration,
                'target_consistency': self.target_consistency,
                'achieved': self.best_run.robust_metrics.consistency_score >= self.target_consistency if self.best_run else False
            },
            'best_run': {
                'params': self.best_run.params,
                'metrics': {
                    'mean_sharpe': self.best_run.robust_metrics.mean_sharpe,
                    'median_sharpe': self.best_run.robust_metrics.median_sharpe,
                    'min_sharpe': self.best_run.robust_metrics.min_sharpe,
                    'max_sharpe': self.best_run.robust_metrics.max_sharpe,
                    'consistency_score': self.best_run.robust_metrics.consistency_score,
                    'stability_score': self.best_run.robust_metrics.stability_score,
                    'profitable_years': self.best_run.robust_metrics.profitable_years,
                    'total_years': self.best_run.robust_metrics.total_years
                }
            } if self.best_run else None,
            'optimization_path': self._analyze_optimization_path()
        }
        
        with open('robust_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFinal report saved to robust_optimization_report.json")
    
    def _analyze_optimization_path(self) -> dict:
        """Analyze how optimization progressed"""
        if not self.runs_history:
            return {}
        
        return {
            'sharpe_progression': [run.robust_metrics.mean_sharpe for run in self.runs_history if run.robust_metrics],
            'consistency_progression': [run.robust_metrics.consistency_score for run in self.runs_history if run.robust_metrics],
            'total_runs': len(self.runs_history)
        }


def main():
    """Main entry point"""
    data_path = "../data/AUDUSD_MASTER_15M.csv"
    
    # Create optimizer
    optimizer = RobustAutoOptimizer(data_path, target_consistency=75.0, max_iterations=50)
    
    # Run optimization
    optimizer.run_robust_optimization()
    
    # Commit results
    try:
        subprocess.run(['git', 'add', '.'], capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'RACS: Robust optimization complete'], capture_output=True)
    except:
        pass


if __name__ == "__main__":
    main()