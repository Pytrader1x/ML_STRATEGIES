"""
Walk-Forward Validation with Optuna Hyperparameter Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import optuna
from pathlib import Path
import json
from datetime import datetime
import logging
import sys
sys.path.append('..')

from quantlab import momentum, Backtest
from quantlab.costs import FXCosts
from sklearn.model_selection import TimeSeriesSplit
import scipy.stats as stats
from statsmodels.stats.stattools import jarque_bera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation with purged k-fold cross-validation"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 pair: str,
                 train_years: int = 3,
                 test_years: int = 1,
                 embargo_bars: int = 5):
        """
        Initialize validator
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        pair : str
            Currency pair
        train_years : int
            Training window in years
        test_years : int
            Test window in years
        embargo_bars : int
            Bars to skip between train/test (prevent leakage)
        """
        
        self.data = data
        self.pair = pair
        self.train_periods = train_years * 252 * 96  # 96 bars per day
        self.test_periods = test_years * 252 * 96
        self.embargo_bars = embargo_bars
        self.costs = FXCosts()
        
    def optimize_parameters(self, 
                          train_data: pd.DataFrame,
                          n_trials: int = 100) -> Dict:
        """
        Optimize parameters using Optuna on training data
        
        Returns best parameters found
        """
        
        def objective(trial):
            # Suggest parameters
            lookback = trial.suggest_int('lookback', 20, 60)
            entry_z = trial.suggest_float('entry_z', 1.0, 2.5)
            exit_z = trial.suggest_float('exit_z', 0.3, 1.0)
            
            # Ensure exit_z < entry_z
            if exit_z >= entry_z:
                return -10  # Bad parameter combination
            
            try:
                # Generate signals
                signals = momentum(
                    train_data['Close'],
                    lookback=lookback,
                    entry_z=entry_z,
                    exit_z=exit_z
                )
                
                # Run backtest
                backtest = Backtest(train_data)
                result = backtest.run(signals['signal'], self.pair)
                
                # Return negative Sharpe (Optuna minimizes)
                return -result.metrics['sharpe_ratio']
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return -10
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters
        best_params = study.best_params
        best_sharpe = -study.best_value
        
        logger.info(f"Best parameters: {best_params}, Sharpe: {best_sharpe:.3f}")
        
        return best_params
    
    def run_purged_kfold(self,
                        params: Dict,
                        n_splits: int = 5) -> List[Dict]:
        """
        Run purged k-fold cross-validation
        
        Parameters:
        -----------
        params : dict
            Strategy parameters
        n_splits : int
            Number of folds
            
        Returns:
        --------
        List of fold results
        """
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.data)):
            # Apply embargo
            train_idx = train_idx[:-self.embargo_bars]
            test_idx = test_idx[self.embargo_bars:]
            
            # Get data
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Generate signals on test data
            test_signals = momentum(test_data['Close'], **params)
            
            # Run backtest
            backtest = Backtest(test_data)
            result = backtest.run(test_signals['signal'], self.pair)
            
            fold_results.append({
                'fold': fold,
                'sharpe': result.metrics['sharpe_ratio'],
                'returns': result.metrics['total_return'],
                'max_dd': result.metrics['max_drawdown'],
                'num_trades': result.metrics['num_trades']
            })
            
        return fold_results
    
    def run_walk_forward(self,
                        optimize: bool = True,
                        n_trials: int = 50) -> Dict:
        """
        Run complete walk-forward analysis
        
        Parameters:
        -----------
        optimize : bool
            Whether to optimize parameters in each window
        n_trials : int
            Optuna trials per window
            
        Returns:
        --------
        Dict with all results and statistics
        """
        
        logger.info(f"Starting walk-forward analysis for {self.pair}")
        
        windows = []
        step_size = self.test_periods  # Non-overlapping windows
        
        # Calculate windows
        start_test = self.train_periods
        
        while start_test + self.test_periods <= len(self.data):
            # Define window
            train_start = max(0, start_test - self.train_periods)
            train_end = start_test - self.embargo_bars
            test_start = start_test
            test_end = min(start_test + self.test_periods, len(self.data))
            
            # Get data
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            logger.info(f"\nWindow {len(windows)+1}:")
            logger.info(f"Train: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Test: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Optimize parameters on training data
            if optimize:
                params = self.optimize_parameters(train_data, n_trials)
            else:
                # Use default parameters
                params = {'lookback': 40, 'entry_z': 1.5, 'exit_z': 0.5}
            
            # Generate signals and run backtest on test data
            test_signals = momentum(test_data['Close'], **params)
            backtest = Backtest(test_data)
            result = backtest.run(test_signals['signal'], self.pair)
            
            # Store window results
            window_result = {
                'window': len(windows) + 1,
                'train_start': str(train_data.index[0]),
                'train_end': str(train_data.index[-1]),
                'test_start': str(test_data.index[0]),
                'test_end': str(test_data.index[-1]),
                'parameters': params,
                'metrics': result.metrics,
                'equity_curve': result.equity_curve.to_list(),
                'num_trades': len(result.trades)
            }
            
            windows.append(window_result)
            
            # Move to next window
            start_test += step_size
        
        # Calculate aggregate statistics
        all_sharpes = [w['metrics']['sharpe_ratio'] for w in windows]
        all_returns = [w['metrics']['total_return'] for w in windows]
        all_dds = [w['metrics']['max_drawdown'] for w in windows]
        
        # Statistical tests
        sharpe_pvalue = self._calculate_sharpe_pvalue(all_sharpes)
        reality_check = self._whites_reality_check(all_returns)
        
        summary = {
            'pair': self.pair,
            'num_windows': len(windows),
            'aggregate_metrics': {
                'mean_sharpe': np.mean(all_sharpes),
                'median_sharpe': np.median(all_sharpes),
                'std_sharpe': np.std(all_sharpes),
                'min_sharpe': np.min(all_sharpes),
                'max_sharpe': np.max(all_sharpes),
                'positive_sharpe_pct': sum(1 for s in all_sharpes if s > 0) / len(all_sharpes) * 100,
                'mean_return': np.mean(all_returns),
                'mean_max_dd': np.mean(all_dds),
                'sharpe_pvalue': sharpe_pvalue,
                'reality_check_pvalue': reality_check
            },
            'windows': windows
        }
        
        return summary
    
    def _calculate_sharpe_pvalue(self, sharpes: List[float]) -> float:
        """
        Calculate p-value for Sharpe ratio using Ledoit-Wolf adjustment
        
        Tests H0: Sharpe = 0
        """
        
        # Convert to array
        sharpes = np.array(sharpes)
        
        # Basic t-test
        t_stat = np.sqrt(len(sharpes)) * np.mean(sharpes) / np.std(sharpes)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(sharpes) - 1))
        
        # Adjust for non-normality (simplified Ledoit-Wolf)
        jb_stat, jb_pvalue = jarque_bera(sharpes)
        if jb_pvalue < 0.05:  # Non-normal
            # Apply correction factor
            correction = 1 + (0.25 / len(sharpes))
            p_value *= correction
        
        return p_value
    
    def _whites_reality_check(self, returns: List[float]) -> float:
        """
        Simplified White's Reality Check
        Tests if strategy is better than random
        """
        
        # Bootstrap the returns
        n_bootstrap = 1000
        bootstrap_means = []
        
        returns = np.array(returns)
        
        for _ in range(n_bootstrap):
            # Random sample with replacement
            sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Calculate p-value
        actual_mean = np.mean(returns)
        p_value = sum(1 for m in bootstrap_means if m <= 0) / n_bootstrap
        
        return p_value


def create_tearsheet(results: Dict, output_path: Path):
    """Create performance tearsheet"""
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Sharpe ratios by window
    ax1 = axes[0, 0]
    sharpes = [w['metrics']['sharpe_ratio'] for w in results['windows']]
    ax1.bar(range(len(sharpes)), sharpes, color=['green' if s > 0 else 'red' for s in sharpes])
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(y=results['aggregate_metrics']['mean_sharpe'], color='blue', linestyle='--', 
                label=f"Mean: {results['aggregate_metrics']['mean_sharpe']:.2f}")
    ax1.set_xlabel('Window')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title(f"{results['pair']} - Sharpe Ratios by Window")
    ax1.legend()
    
    # 2. Returns distribution
    ax2 = axes[0, 1]
    returns = [w['metrics']['total_return'] for w in results['windows']]
    ax2.hist(returns, bins=20, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Returns Distribution')
    
    # 3. Parameter stability
    ax3 = axes[1, 0]
    lookbacks = [w['parameters']['lookback'] for w in results['windows']]
    entry_zs = [w['parameters']['entry_z'] for w in results['windows']]
    
    ax3_twin = ax3.twinx()
    ax3.plot(lookbacks, 'b-', label='Lookback', marker='o')
    ax3_twin.plot(entry_zs, 'r-', label='Entry Z', marker='s')
    ax3.set_xlabel('Window')
    ax3.set_ylabel('Lookback', color='b')
    ax3_twin.set_ylabel('Entry Z', color='r')
    ax3.set_title('Parameter Evolution')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Walk-Forward Analysis Summary
    
    Pair: {results['pair']}
    Windows: {results['num_windows']}
    
    Mean Sharpe: {results['aggregate_metrics']['mean_sharpe']:.3f}
    Median Sharpe: {results['aggregate_metrics']['median_sharpe']:.3f}
    Std Sharpe: {results['aggregate_metrics']['std_sharpe']:.3f}
    
    Positive Sharpe: {results['aggregate_metrics']['positive_sharpe_pct']:.1f}%
    Mean Return: {results['aggregate_metrics']['mean_return']:.1f}%
    Mean Max DD: {results['aggregate_metrics']['mean_max_dd']:.1f}%
    
    Statistical Significance:
    Sharpe p-value: {results['aggregate_metrics']['sharpe_pvalue']:.4f}
    Reality Check p-value: {results['aggregate_metrics']['reality_check_pvalue']:.4f}
    
    Verdict: {'SIGNIFICANT' if results['aggregate_metrics']['sharpe_pvalue'] < 0.05 else 'NOT SIGNIFICANT'}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Tearsheet saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')
    
    # Load data
    data_path = Path('../../data/AUDUSD_MASTER_15M.csv')
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    # Run walk-forward validation
    validator = WalkForwardValidator(data, 'AUDUSD')
    results = validator.run_walk_forward(optimize=True, n_trials=50)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'walk_forward_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create tearsheet
    create_tearsheet(results, output_dir / 'walk_forward_tearsheet.png')
    
    print(f"\nValidation complete!")
    print(f"Mean Sharpe: {results['aggregate_metrics']['mean_sharpe']:.3f}")
    print(f"p-value: {results['aggregate_metrics']['sharpe_pvalue']:.4f}")