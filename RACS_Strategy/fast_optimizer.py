"""
Fast RACS Optimizer - Simplified for speed and reliability

Runs continuously until Sharpe > 1 is achieved
"""

import pandas as pd
import numpy as np
import json
import os
import time
import subprocess
from datetime import datetime
import backtrader as bt
import random


class FastStrategy(bt.Strategy):
    """Ultra-simple trend following strategy"""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 50),
        ('risk_pct', 0.01),
        ('stop_multi', 2.0),
    )
    
    def __init__(self):
        self.sma_fast = bt.indicators.SMA(period=self.p.fast_period)
        self.sma_slow = bt.indicators.SMA(period=self.p.slow_period)
        self.atr = bt.indicators.ATR(period=14)
        self.order = None
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            # Entry logic
            if self.sma_fast[0] > self.sma_slow[0] and self.sma_fast[-1] <= self.sma_slow[-1]:
                # Bullish crossover
                size = (self.broker.getvalue() * self.p.risk_pct) / (self.p.stop_multi * self.atr[0])
                self.order = self.buy(size=size)
            elif self.sma_fast[0] < self.sma_slow[0] and self.sma_fast[-1] >= self.sma_slow[-1]:
                # Bearish crossover
                size = (self.broker.getvalue() * self.p.risk_pct) / (self.p.stop_multi * self.atr[0])
                self.order = self.sell(size=size)
        else:
            # Exit logic
            if self.position.size > 0:
                if self.sma_fast[0] < self.sma_slow[0]:
                    self.order = self.close()
            else:
                if self.sma_fast[0] > self.sma_slow[0]:
                    self.order = self.close()
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None


class FastOptimizer:
    """Fast optimization loop"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.iteration = 0
        self.best_sharpe = -999
        self.best_params = None
        self.start_time = time.time()
        
        # Load and prepare data once
        print("Loading data...")
        self.df = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
        
        # Use recent data for faster testing
        self.df = self.df[-100000:]  # Last ~2 years of 15min data
        print(f"Using {len(self.df)} rows of data")
        
    def test_params(self, params: dict) -> tuple:
        """Quick backtest"""
        try:
            cerebro = bt.Cerebro()
            cerebro.addstrategy(FastStrategy, **params)
            
            data = bt.feeds.PandasData(dataname=self.df)
            cerebro.adddata(data)
            
            cerebro.broker.setcash(10000)
            cerebro.broker.setcommission(commission=0.001)
            
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
            
            results = cerebro.run()
            strat = results[0]
            
            sharpe_analysis = strat.analyzers.sharpe.get_analysis()
            sharpe = sharpe_analysis.get('sharperatio', -999) if sharpe_analysis else -999
            
            dd_analysis = strat.analyzers.dd.get_analysis()
            dd = dd_analysis.max.drawdown if hasattr(dd_analysis, 'max') else 0
            
            final = cerebro.broker.getvalue()
            
            return sharpe, dd, final
            
        except Exception as e:
            return -999, 100, 0
    
    def run_infinite_loop(self):
        """Main optimization loop"""
        print("\n" + "="*60)
        print("FAST OPTIMIZER - Target: Sharpe > 1.0")
        print("="*60 + "\n")
        
        # Parameter ranges
        param_ranges = {
            'fast_period': (5, 20),
            'slow_period': (20, 100), 
            'risk_pct': (0.005, 0.02),
            'stop_multi': (1.0, 3.0),
        }
        
        # Current best
        current_params = {
            'fast_period': 10,
            'slow_period': 50,
            'risk_pct': 0.01,
            'stop_multi': 2.0,
        }
        
        last_git_commit = time.time()
        consecutive_improvements = 0
        
        while True:
            self.iteration += 1
            
            # Generate new params (mix of random and mutations)
            if self.iteration % 10 == 1:
                # Random exploration
                new_params = {
                    'fast_period': random.randint(*param_ranges['fast_period']),
                    'slow_period': random.randint(*param_ranges['slow_period']),
                    'risk_pct': random.uniform(*param_ranges['risk_pct']),
                    'stop_multi': random.uniform(*param_ranges['stop_multi']),
                }
            else:
                # Mutate best params
                base = self.best_params if self.best_params else current_params
                new_params = base.copy()
                
                # Mutate 1-2 parameters
                for _ in range(random.randint(1, 2)):
                    param = random.choice(list(param_ranges.keys()))
                    if param in ['fast_period', 'slow_period']:
                        new_params[param] = max(param_ranges[param][0], 
                                              min(param_ranges[param][1],
                                                  base[param] + random.randint(-5, 5)))
                    else:
                        delta = (param_ranges[param][1] - param_ranges[param][0]) * 0.1
                        new_params[param] = max(param_ranges[param][0],
                                              min(param_ranges[param][1],
                                                  base[param] + random.uniform(-delta, delta)))
            
            # Ensure slow > fast
            if new_params['slow_period'] <= new_params['fast_period']:
                new_params['slow_period'] = new_params['fast_period'] + 10
            
            # Test
            sharpe, dd, final = self.test_params(new_params)
            
            print(f"\nIteration {self.iteration} | Time: {(time.time()-self.start_time)/60:.1f}m")
            if sharpe != -999 and sharpe is not None:
                print(f"Sharpe: {sharpe:.3f} | DD: {dd:.1f}% | Final: ${final:.0f}")
            else:
                print(f"Test failed - Sharpe: {sharpe}")
            print(f"Params: Fast={new_params['fast_period']}, Slow={new_params['slow_period']}, " +
                  f"Risk={new_params['risk_pct']:.3f}, Stop={new_params['stop_multi']:.1f}")
            
            # Track best
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_params = new_params.copy()
                consecutive_improvements += 1
                print(f"🎯 NEW BEST! Sharpe: {sharpe:.3f}")
                
                # Save state
                self.save_state()
                
                # Check target
                if sharpe >= 1.0:
                    print(f"\n✅ SUCCESS! Sharpe > 1.0 achieved: {sharpe:.3f}")
                    self.save_final_report()
                    subprocess.run(['git', 'add', '.'], capture_output=True)
                    subprocess.run(['git', 'commit', '-m', f'RACS: SUCCESS! Sharpe {sharpe:.3f} > 1.0'], capture_output=True)
                    break
            else:
                consecutive_improvements = 0
            
            # Update current params if improving
            if sharpe > -1:
                current_params = new_params
            
            # Periodic actions
            if self.iteration % 20 == 0:
                self.update_claude_md()
            
            # Git commit every 10 mins
            if time.time() - last_git_commit > 600:
                self.git_commit()
                last_git_commit = time.time()
            
            # Reset if stuck
            if self.iteration > 50 and self.best_sharpe < 0:
                print("\nResetting search...")
                current_params = {
                    'fast_period': random.randint(5, 15),
                    'slow_period': random.randint(30, 80),
                    'risk_pct': random.uniform(0.005, 0.015),
                    'stop_multi': random.uniform(1.5, 2.5),
                }
    
    def save_state(self):
        """Save current state"""
        state = {
            'iteration': self.iteration,
            'best_sharpe': self.best_sharpe,
            'best_params': self.best_params,
            'elapsed_minutes': (time.time() - self.start_time) / 60
        }
        with open('fast_optimizer_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def update_claude_md(self):
        """Update progress file"""
        content = f"""# RACS Fast Optimization Progress

## Status: {"✅ SUCCESS" if self.best_sharpe >= 1.0 else "🔄 RUNNING"}

## Progress
- Iteration: {self.iteration}
- Best Sharpe: {self.best_sharpe:.3f}
- Time: {(time.time() - self.start_time) / 60:.1f} minutes

## Best Parameters
```json
{json.dumps(self.best_params if self.best_params else {}, indent=2)}
```

## To Continue
```bash
python fast_optimizer.py
```
"""
        with open('CLAUDE.md', 'w') as f:
            f.write(content)
    
    def save_final_report(self):
        """Save success report"""
        report = {
            'success': True,
            'iterations': self.iteration,
            'time_minutes': (time.time() - self.start_time) / 60,
            'best_sharpe': self.best_sharpe,
            'best_params': self.best_params
        }
        with open('optimization_success.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def git_commit(self):
        """Commit progress"""
        try:
            subprocess.run(['git', 'add', '.'], capture_output=True)
            subprocess.run(['git', 'commit', '-m', 
                          f'RACS Progress: Iteration {self.iteration}, Best Sharpe {self.best_sharpe:.3f}'], 
                          capture_output=True)
        except:
            pass


if __name__ == "__main__":
    optimizer = FastOptimizer("../data/AUDUSD_MASTER_15M.csv")
    optimizer.run_infinite_loop()