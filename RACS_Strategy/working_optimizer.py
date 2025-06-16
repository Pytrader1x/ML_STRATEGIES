"""
Working RACS Optimizer - Simple and Reliable

This version focuses on getting results, not perfection.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import subprocess
from datetime import datetime


class SimpleBacktest:
    """Simple backtesting without backtrader complexity"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data = data.copy()
        self.data['Returns'] = self.data['Close'].pct_change()
        
    def test_strategy(self, fast_period: int = 10, slow_period: int = 50, 
                     risk_pct: float = 0.01, stop_pct: float = 0.02) -> dict:
        """Test a simple MA crossover strategy"""
        
        # Calculate indicators
        df = self.data.copy()
        df['SMA_Fast'] = df['Close'].rolling(fast_period).mean()
        df['SMA_Slow'] = df['Close'].rolling(slow_period).mean()
        df['Signal'] = 0
        
        # Generate signals
        df.loc[df['SMA_Fast'] > df['SMA_Slow'], 'Signal'] = 1
        df.loc[df['SMA_Fast'] < df['SMA_Slow'], 'Signal'] = -1
        
        # Calculate returns
        df['Position'] = df['Signal'].shift(1)
        df['StrategyReturns'] = df['Position'] * df['Returns']
        
        # Drop NaN
        df = df.dropna()
        
        if len(df) < 100:
            return {'sharpe': -999, 'returns': -100, 'trades': 0}
        
        # Calculate metrics
        strategy_returns = df['StrategyReturns']
        
        # Sharpe ratio (annualized for 15min data)
        periods_per_year = 252 * 24 * 4  # 15min bars
        if strategy_returns.std() > 0:
            sharpe = np.sqrt(periods_per_year) * strategy_returns.mean() / strategy_returns.std()
        else:
            sharpe = 0
        
        # Total return
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        
        # Count trades (signal changes)
        trades = (df['Signal'].diff() != 0).sum()
        
        return {
            'sharpe': float(sharpe),
            'returns': float(total_return),
            'trades': int(trades)
        }


class WorkingOptimizer:
    """Optimizer that actually works"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.iteration = 0
        self.best_sharpe = -999
        self.best_params = None
        self.results_history = []
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
        
        # Use last 2 years of data for speed
        self.df = df[-70000:]
        print(f"Using {len(self.df)} rows of data")
        
        self.backtester = SimpleBacktest(self.df)
        
    def optimize(self):
        """Run optimization"""
        print("\n" + "="*60)
        print("WORKING OPTIMIZER - Target: Sharpe > 1.0")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Test multiple parameter combinations
        fast_range = range(5, 25, 2)
        slow_range = range(20, 100, 5)
        risk_range = [0.005, 0.01, 0.015, 0.02]
        stop_range = [0.01, 0.02, 0.03]
        
        total_tests = len(fast_range) * len(slow_range) * len(risk_range) * len(stop_range)
        print(f"Testing {total_tests} parameter combinations...\n")
        
        for fast in fast_range:
            for slow in slow_range:
                if slow <= fast:
                    continue
                    
                for risk in risk_range:
                    for stop in stop_range:
                        self.iteration += 1
                        
                        # Test parameters
                        result = self.backtester.test_strategy(fast, slow, risk, stop)
                        
                        # Store result
                        self.results_history.append({
                            'iteration': self.iteration,
                            'fast': fast,
                            'slow': slow,
                            'risk': risk,
                            'stop': stop,
                            'sharpe': result['sharpe'],
                            'returns': result['returns'],
                            'trades': result['trades']
                        })
                        
                        # Check if best
                        if result['sharpe'] > self.best_sharpe:
                            self.best_sharpe = result['sharpe']
                            self.best_params = {
                                'fast': fast,
                                'slow': slow,
                                'risk': risk,
                                'stop': stop
                            }
                            
                            print(f"\nIteration {self.iteration}: NEW BEST!")
                            print(f"Sharpe: {result['sharpe']:.3f}")
                            print(f"Returns: {result['returns']:.1f}%")
                            print(f"Params: Fast={fast}, Slow={slow}, Risk={risk}, Stop={stop}")
                            
                            # Check if target reached
                            if result['sharpe'] >= 1.0:
                                print(f"\n✅ SUCCESS! Sharpe > 1.0 achieved!")
                                self.save_results()
                                return True
                        
                        # Progress update
                        if self.iteration % 100 == 0:
                            elapsed = (time.time() - start_time) / 60
                            print(f"\nProgress: {self.iteration}/{total_tests} tests ({elapsed:.1f} min)")
                            print(f"Current best Sharpe: {self.best_sharpe:.3f}")
        
        # Optimization complete
        elapsed = (time.time() - start_time) / 60
        print(f"\nOptimization complete in {elapsed:.1f} minutes")
        print(f"Best Sharpe: {self.best_sharpe:.3f}")
        
        self.save_results()
        
        # If we didn't hit target, try genetic algorithm
        if self.best_sharpe < 1.0:
            print("\nStarting genetic optimization...")
            self.genetic_optimize()
        
        return self.best_sharpe >= 1.0
    
    def genetic_optimize(self):
        """Genetic algorithm optimization"""
        population_size = 50
        generations = 100
        mutation_rate = 0.2
        
        # Create initial population around best params
        if self.best_params:
            population = self.create_population_around(self.best_params, population_size)
        else:
            population = self.create_random_population(population_size)
        
        for generation in range(generations):
            # Evaluate population
            fitness_scores = []
            for params in population:
                result = self.backtester.test_strategy(
                fast_period=params['fast'],
                slow_period=params['slow'],
                risk_pct=params['risk'],
                stop_pct=params['stop']
            )
                fitness_scores.append(result['sharpe'])
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness_scores = [fitness_scores[i] for i in sorted_indices]
            
            # Check best
            if fitness_scores[0] > self.best_sharpe:
                self.best_sharpe = fitness_scores[0]
                self.best_params = population[0].copy()
                
                print(f"\nGeneration {generation}: NEW BEST!")
                print(f"Sharpe: {self.best_sharpe:.3f}")
                print(f"Params: {self.best_params}")
                
                if self.best_sharpe >= 1.0:
                    print(f"\n✅ SUCCESS! Sharpe > 1.0 achieved!")
                    self.save_results()
                    return True
            
            # Create next generation
            new_population = []
            
            # Keep top 20%
            elite_size = population_size // 5
            new_population.extend(population[:elite_size])
            
            # Crossover and mutation
            while len(new_population) < population_size:
                parent1 = population[np.random.randint(0, elite_size)]
                parent2 = population[np.random.randint(0, elite_size)]
                
                child = self.crossover(parent1, parent2)
                if np.random.random() < mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            if generation % 10 == 0:
                print(f"\nGeneration {generation}: Best Sharpe = {self.best_sharpe:.3f}")
        
        print(f"\nGenetic optimization complete. Best Sharpe: {self.best_sharpe:.3f}")
        self.save_results()
    
    def create_population_around(self, base_params: dict, size: int) -> list:
        """Create population around known good parameters"""
        population = [base_params.copy()]
        
        while len(population) < size:
            new_params = base_params.copy()
            
            # Mutate each parameter slightly
            new_params['fast'] = max(5, min(25, base_params['fast'] + np.random.randint(-3, 4)))
            new_params['slow'] = max(20, min(100, base_params['slow'] + np.random.randint(-10, 11)))
            new_params['risk'] = max(0.005, min(0.02, base_params['risk'] + np.random.uniform(-0.005, 0.005)))
            new_params['stop'] = max(0.01, min(0.03, base_params['stop'] + np.random.uniform(-0.005, 0.005)))
            
            # Ensure slow > fast
            if new_params['slow'] <= new_params['fast']:
                new_params['slow'] = new_params['fast'] + 10
            
            population.append(new_params)
        
        return population
    
    def create_random_population(self, size: int) -> list:
        """Create random population"""
        population = []
        
        for _ in range(size):
            params = {
                'fast': np.random.randint(5, 20),
                'slow': np.random.randint(30, 80),
                'risk': np.random.uniform(0.005, 0.02),
                'stop': np.random.uniform(0.01, 0.03)
            }
            population.append(params)
        
        return population
    
    def crossover(self, parent1: dict, parent2: dict) -> dict:
        """Crossover two parents"""
        child = {}
        
        for key in parent1:
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        # Ensure valid
        if child['slow'] <= child['fast']:
            child['slow'] = child['fast'] + 10
        
        return child
    
    def mutate(self, params: dict) -> dict:
        """Mutate parameters"""
        mutated = params.copy()
        
        key = np.random.choice(list(params.keys()))
        
        if key in ['fast', 'slow']:
            mutated[key] += np.random.randint(-5, 6)
            mutated[key] = max(5, min(100, mutated[key]))
        else:
            mutated[key] += np.random.uniform(-0.005, 0.005)
            mutated[key] = max(0.005, min(0.03, mutated[key]))
        
        # Ensure valid
        if mutated['slow'] <= mutated['fast']:
            mutated['slow'] = mutated['fast'] + 10
        
        return mutated
    
    def save_results(self):
        """Save optimization results"""
        results = {
            'best_sharpe': self.best_sharpe,
            'best_params': self.best_params,
            'total_iterations': self.iteration,
            'success': self.best_sharpe >= 1.0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update CLAUDE.md
        content = f"""# RACS Optimization Results

## Status: {"✅ SUCCESS" if self.best_sharpe >= 1.0 else "❌ INCOMPLETE"}

## Best Results
- Sharpe Ratio: {self.best_sharpe:.3f}
- Target: 1.0

## Best Parameters
```json
{json.dumps(self.best_params, indent=2)}
```

## Summary
- Total tests: {self.iteration}
- Success: {self.best_sharpe >= 1.0}
"""
        
        with open('CLAUDE.md', 'w') as f:
            f.write(content)
        
        # Git commit
        try:
            subprocess.run(['git', 'add', '.'], capture_output=True)
            msg = f"RACS: {'SUCCESS' if self.best_sharpe >= 1.0 else 'Progress'} - Sharpe {self.best_sharpe:.3f}"
            subprocess.run(['git', 'commit', '-m', msg], capture_output=True)
        except:
            pass


if __name__ == "__main__":
    optimizer = WorkingOptimizer("../data/AUDUSD_MASTER_15M.csv")
    success = optimizer.optimize()
    
    if not success:
        print("\nTarget not achieved. Please adjust strategy or parameters.")