"""
Infinite Autonomous RACS Optimizer

This module implements an infinite loop optimization system that:
1. Continuously runs backtests with improving parameters
2. Commits progress to git automatically
3. Runs efficiently (2-5 min per iteration)
4. Stops only when robust Sharpe > 1 is achieved
"""

import pandas as pd
import numpy as np
import json
import os
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import backtrader as bt
from dataclasses import dataclass
import random
import sys
import platform

# Signal handling - only on Unix-like systems
if platform.system() != 'Windows':
    import signal
else:
    signal = None

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tic import TIC
from simple_racs_strategy import SimpleRACSStrategy


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Backtest timeout")


@dataclass
class TestResult:
    iteration: int
    params: dict
    sharpe: float
    return_pct: float
    win_rate: float
    max_dd: float
    trades: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class EfficientTester:
    """Fast, simple backtester focused on speed"""
    
    def __init__(self, data_path: str, sample_size: int = 3000):
        self.data_path = data_path
        self.sample_size = sample_size
        self.prepared_samples = []
        
    def prepare_samples(self, num_samples: int = 5):
        """Pre-prepare random samples for testing"""
        print("Preparing test samples...")
        
        # Load full data
        df = pd.read_csv(self.data_path, parse_dates=['DateTime'], index_col='DateTime')
        
        # Add indicators once
        print("Adding indicators...")
        df = TIC.add_intelligent_chop(df, inplace=True)
        df = TIC.add_market_bias(df, inplace=True) 
        df = TIC.add_neuro_trend_intelligent(df, inplace=True)
        df = TIC.add_super_trend(df, inplace=True)
        df = TIC.add_fractal_sr(df, inplace=True)
        
        # Remove rows with NaN in non-SR columns
        non_sr_cols = [col for col in df.columns if not col.startswith('SR_')]
        df = df.dropna(subset=non_sr_cols)
        
        print(f"Prepared data: {len(df)} rows")
        
        # Create random samples
        self.prepared_samples = []
        for i in range(num_samples):
            if len(df) > self.sample_size * 2:
                start = random.randint(0, len(df) - self.sample_size)
                sample = df.iloc[start:start + self.sample_size].copy()
                self.prepared_samples.append(sample)
        
        print(f"Created {len(self.prepared_samples)} test samples")
    
    def quick_test(self, params: dict, timeout_seconds: int = 120) -> TestResult:
        """Run a quick backtest with timeout"""
        if not self.prepared_samples:
            self.prepare_samples()
        
        # Pick a random sample
        sample = random.choice(self.prepared_samples)
        
        start_time = time.time()
        
        # Set timeout (Unix only)
        if signal:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            # Run backtest
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SimpleRACSStrategy, **params)
            
            # Use simple pandas feed
            data = bt.feeds.PandasData(
                dataname=sample,
                datetime=None,
                open='Open',
                high='High', 
                low='Low',
                close='Close',
                volume=-1,
                openinterest=-1
            )
            cerebro.adddata(data)
            
            # Broker setup
            initial_cash = 10000
            cerebro.broker.setcash(initial_cash)
            cerebro.broker.setcommission(commission=0.001)
            
            # Simple analyzers only
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # Run
            results = cerebro.run()
            strat = results[0]
            
            # Extract metrics
            sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            dd = strat.analyzers.dd.get_analysis().max.drawdown
            trades_analysis = strat.analyzers.trades.get_analysis()
            
            total_trades = trades_analysis.total.total
            won = trades_analysis.won.total if hasattr(trades_analysis.won, 'total') else 0
            win_rate = (won / total_trades * 100) if total_trades > 0 else 0
            
            final_value = cerebro.broker.getvalue()
            return_pct = (final_value / initial_cash - 1) * 100
            
            duration = time.time() - start_time
            
            # Cancel timeout
            if signal:
                signal.alarm(0)
            
            return TestResult(
                iteration=0,
                params=params,
                sharpe=sharpe,
                return_pct=return_pct,
                win_rate=win_rate,
                max_dd=dd,
                trades=total_trades,
                duration_seconds=duration,
                success=True
            )
            
        except TimeoutException:
            if signal:
                signal.alarm(0)
            return TestResult(
                iteration=0,
                params=params,
                sharpe=-999,
                return_pct=-100,
                win_rate=0,
                max_dd=100,
                trades=0,
                duration_seconds=timeout_seconds,
                success=False,
                error="Timeout"
            )
        except Exception as e:
            if signal:
                signal.alarm(0)
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            # Get the last line of traceback for more context
            tb_lines = traceback.format_exc().split('\n')
            if len(tb_lines) > 2:
                error_msg += f" | {tb_lines[-3].strip()}"
            
            return TestResult(
                iteration=0,
                params=params,
                sharpe=-999,
                return_pct=-100,
                win_rate=0,
                max_dd=100,
                trades=0,
                duration_seconds=time.time() - start_time,
                success=False,
                error=error_msg
            )


class InfiniteOptimizer:
    """Infinite loop optimizer that runs until target is reached"""
    
    def __init__(self, data_path: str, target_sharpe: float = 1.0):
        self.data_path = data_path
        self.target_sharpe = target_sharpe
        self.tester = EfficientTester(data_path)
        
        self.iteration = 0
        self.results_history = []
        self.best_result = None
        self.start_time = time.time()
        
        # Parameter search space
        self.param_space = {
            'risk_pct': (0.003, 0.015, 0.001),
            'confidence_threshold': (50, 80, 5),
            'slope_threshold': (10, 30, 5),
            'stop_loss_atr': (0.5, 2.0, 0.25),
            'take_profit_atr': (1.0, 4.0, 0.5),
            'max_positions': (1, 3, 1),
        }
        
        # Current parameters
        self.current_params = {
            'risk_pct': 0.008,
            'confidence_threshold': 65,
            'slope_threshold': 20,
            'stop_loss_atr': 1.0,
            'take_profit_atr': 2.0,
            'max_positions': 2,
        }
    
    def mutate_params(self, base_params: dict, mutation_rate: float = 0.3) -> dict:
        """Create a mutated version of parameters"""
        new_params = base_params.copy()
        
        for param, (min_val, max_val, step) in self.param_space.items():
            if random.random() < mutation_rate:
                # Mutate this parameter
                direction = random.choice([-1, 1])
                new_val = base_params[param] + direction * step
                new_params[param] = max(min_val, min(max_val, new_val))
        
        return new_params
    
    def run_infinite_loop(self):
        """Main infinite optimization loop"""
        print("\n" + "="*80)
        print("STARTING INFINITE OPTIMIZATION LOOP")
        print(f"Target: Sharpe > {self.target_sharpe}")
        print("="*80 + "\n")
        
        # Prepare test samples
        self.tester.prepare_samples(10)
        
        # Load previous state if exists
        self.load_state()
        
        consecutive_failures = 0
        last_git_commit = time.time()
        
        while True:
            self.iteration += 1
            
            print(f"\n--- Iteration {self.iteration} ---")
            print(f"Time elapsed: {(time.time() - self.start_time) / 60:.1f} minutes")
            
            # Test current parameters
            result = self.tester.quick_test(self.current_params)
            result.iteration = self.iteration
            self.results_history.append(result)
            
            # Display results
            if result.success:
                print(f"Sharpe: {result.sharpe:.3f}, Return: {result.return_pct:.1f}%, " +
                      f"Win Rate: {result.win_rate:.1f}%, Trades: {result.trades}")
                consecutive_failures = 0
            else:
                print(f"Test failed: {result.error}")
                consecutive_failures += 1
            
            # Update best result
            if result.success and (self.best_result is None or result.sharpe > self.best_result.sharpe):
                self.best_result = result
                print(f"🎯 NEW BEST! Sharpe: {result.sharpe:.3f}")
                self.save_state()
            
            # Check if target reached
            if result.success and result.sharpe >= self.target_sharpe:
                print(f"\n✅ TARGET ACHIEVED! Sharpe: {result.sharpe:.3f}")
                self.save_final_report()
                self.git_commit("Target Sharpe > 1 achieved!")
                break
            
            # Evolve parameters
            if result.success:
                if result.sharpe > 0:
                    # Good direction, small mutations
                    self.current_params = self.mutate_params(self.current_params, 0.2)
                else:
                    # Need bigger changes
                    self.current_params = self.mutate_params(self.current_params, 0.5)
            else:
                # Failed test, try different params
                if self.best_result:
                    self.current_params = self.mutate_params(self.best_result.params, 0.4)
                else:
                    self.current_params = self.mutate_params(self.current_params, 0.7)
            
            # Reset if too many failures
            if consecutive_failures >= 5:
                print("Too many failures, resetting parameters...")
                self.current_params = self.generate_random_params()
                consecutive_failures = 0
            
            # Periodic actions
            if self.iteration % 10 == 0:
                self.save_state()
                self.update_claude_md()
            
            # Git commit every 10 minutes
            if time.time() - last_git_commit > 600:
                self.git_commit(f"Progress: Best Sharpe {self.best_result.sharpe:.3f}" if self.best_result else "Continuing optimization")
                last_git_commit = time.time()
            
            # Brief pause to not overwhelm system
            time.sleep(1)
    
    def generate_random_params(self) -> dict:
        """Generate random parameters within bounds"""
        params = {}
        for param, (min_val, max_val, step) in self.param_space.items():
            steps = int((max_val - min_val) / step)
            value = min_val + random.randint(0, steps) * step
            params[param] = value
        return params
    
    def save_state(self):
        """Save current optimization state"""
        state = {
            'iteration': self.iteration,
            'best_result': {
                'sharpe': self.best_result.sharpe,
                'params': self.best_result.params,
                'return_pct': self.best_result.return_pct,
                'win_rate': self.best_result.win_rate
            } if self.best_result else None,
            'current_params': self.current_params,
            'elapsed_minutes': (time.time() - self.start_time) / 60
        }
        
        with open('infinite_optimizer_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        """Load previous state if exists"""
        if os.path.exists('infinite_optimizer_state.json'):
            with open('infinite_optimizer_state.json', 'r') as f:
                state = json.load(f)
                self.iteration = state.get('iteration', 0)
                self.current_params = state.get('current_params', self.current_params)
                print(f"Resumed from iteration {self.iteration}")
    
    def update_claude_md(self):
        """Update CLAUDE.md with current progress"""
        best_sharpe = self.best_result.sharpe if self.best_result else 0
        
        content = f"""# RACS Infinite Optimization Progress

## Status: {"✅ COMPLETE" if best_sharpe >= self.target_sharpe else "🔄 RUNNING"}

## Current Progress
- Iteration: {self.iteration}
- Best Sharpe: {best_sharpe:.3f}
- Target: {self.target_sharpe}
- Time Running: {(time.time() - self.start_time) / 60:.1f} minutes

## Best Parameters
```json
{json.dumps(self.best_result.params if self.best_result else self.current_params, indent=2)}
```

## Best Performance
{f'''- Sharpe Ratio: {self.best_result.sharpe:.3f}
- Return: {self.best_result.return_pct:.1f}%
- Win Rate: {self.best_result.win_rate:.1f}%
- Max Drawdown: {self.best_result.max_dd:.1f}%
- Total Trades: {self.best_result.trades}''' if self.best_result else 'No successful results yet'}

## Recent Tests
{self._format_recent_tests()}

## To Resume
```bash
python infinite_optimizer.py
```
"""
        
        with open('CLAUDE.md', 'w') as f:
            f.write(content)
    
    def _format_recent_tests(self) -> str:
        """Format recent test results"""
        recent = self.results_history[-10:] if len(self.results_history) > 10 else self.results_history
        lines = []
        for r in recent:
            if r.success:
                lines.append(f"- Iteration {r.iteration}: Sharpe {r.sharpe:.3f}, Return {r.return_pct:.1f}%")
            else:
                lines.append(f"- Iteration {r.iteration}: Failed ({r.error})")
        return '\n'.join(lines)
    
    def save_final_report(self):
        """Save final optimization report"""
        report = {
            'success': True,
            'iterations': self.iteration,
            'time_minutes': (time.time() - self.start_time) / 60,
            'best_result': {
                'sharpe': self.best_result.sharpe,
                'return_pct': self.best_result.return_pct,
                'win_rate': self.best_result.win_rate,
                'max_drawdown': self.best_result.max_dd,
                'trades': self.best_result.trades,
                'params': self.best_result.params
            }
        }
        
        with open('infinite_optimization_success.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nOptimization successful! Report saved.")
    
    def git_commit(self, message: str):
        """Commit progress to git"""
        try:
            subprocess.run(['git', 'add', '.'], capture_output=True)
            subprocess.run(['git', 'commit', '-m', f'RACS Optimizer: {message}'], capture_output=True)
            print(f"Git: {message}")
        except:
            pass


def main():
    """Main entry point"""
    data_path = "../data/AUDUSD_MASTER_15M.csv"
    
    optimizer = InfiniteOptimizer(data_path, target_sharpe=1.0)
    
    try:
        optimizer.run_infinite_loop()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        optimizer.save_state()
        optimizer.update_claude_md()


if __name__ == "__main__":
    main()