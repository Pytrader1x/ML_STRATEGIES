"""
Robustness and Sensitivity Analysis
Tests strategy stability under various conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')

from quantlab import momentum, Backtest
from quantlab.costs import FXCosts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustnessAnalyzer:
    """Comprehensive robustness testing for trading strategies"""
    
    def __init__(self, data: pd.DataFrame, pair: str):
        self.data = data.copy()
        self.pair = pair
        self.base_params = {'lookback': 40, 'entry_z': 1.5, 'exit_z': 0.5}
        self.costs = FXCosts()
        
    def parameter_heatmap(self, 
                         param_ranges: Dict[str, List],
                         metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Create parameter sensitivity heatmap
        
        Parameters:
        -----------
        param_ranges : dict
            Parameter ranges to test, e.g.:
            {'lookback': [20, 30, 40, 50, 60],
             'entry_z': [1.0, 1.25, 1.5, 1.75, 2.0]}
        metric : str
            Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
        --------
        pd.DataFrame with results grid
        """
        
        logger.info(f"Creating parameter heatmap for {self.pair}")
        
        # Get parameter names and values
        param_names = list(param_ranges.keys())
        if len(param_names) != 2:
            raise ValueError("Heatmap requires exactly 2 parameters")
            
        param1_name, param2_name = param_names
        param1_values = param_ranges[param1_name]
        param2_values = param_ranges[param2_name]
        
        # Create results grid
        results = pd.DataFrame(index=param1_values, columns=param2_values)
        
        # Test each combination
        for p1_val in param1_values:
            for p2_val in param2_values:
                # Create parameter set
                params = self.base_params.copy()
                params[param1_name] = p1_val
                params[param2_name] = p2_val
                
                # Skip invalid combinations
                if 'exit_z' in params and 'entry_z' in params:
                    if params['exit_z'] >= params['entry_z']:
                        results.loc[p1_val, p2_val] = np.nan
                        continue
                
                try:
                    # Run backtest
                    result = self._run_single_backtest(params)
                    results.loc[p1_val, p2_val] = result['metrics'][metric]
                except:
                    results.loc[p1_val, p2_val] = np.nan
                    
        return results.astype(float)
    
    def parameter_stability(self, 
                          param_name: str,
                          base_value: float,
                          variation_pct: float = 50) -> pd.DataFrame:
        """
        Test parameter stability around base value
        
        Parameters:
        -----------
        param_name : str
            Parameter to test
        base_value : float
            Base parameter value
        variation_pct : float
            Percentage variation to test (±%)
            
        Returns:
        --------
        pd.DataFrame with stability results
        """
        
        # Create test range
        min_val = base_value * (1 - variation_pct/100)
        max_val = base_value * (1 + variation_pct/100)
        
        if param_name == 'lookback':
            test_values = np.arange(int(min_val), int(max_val)+1, 5)
        else:
            test_values = np.linspace(min_val, max_val, 21)
            
        results = []
        
        for test_val in test_values:
            params = self.base_params.copy()
            params[param_name] = test_val
            
            # Skip invalid combinations
            if 'exit_z' in params and 'entry_z' in params:
                if params['exit_z'] >= params['entry_z']:
                    continue
                    
            try:
                result = self._run_single_backtest(params)
                
                results.append({
                    param_name: test_val,
                    'variation_pct': (test_val - base_value) / base_value * 100,
                    'sharpe': result['metrics']['sharpe_ratio'],
                    'returns': result['metrics']['total_return'],
                    'max_dd': result['metrics']['max_drawdown'],
                    'num_trades': result['metrics']['num_trades']
                })
            except:
                continue
                
        return pd.DataFrame(results)
    
    def trade_delay_test(self, delays: List[int] = [0, 1, 2]) -> pd.DataFrame:
        """
        Test impact of trade execution delays
        
        Parameters:
        -----------
        delays : list
            List of delay periods (in bars)
            
        Returns:
        --------
        pd.DataFrame with delay impact results
        """
        
        logger.info("Testing trade execution delays")
        
        results = []
        
        for delay in delays:
            # Generate base signals
            signals_df = momentum(self.data['Close'], **self.base_params)
            base_signals = signals_df['signal']
            
            if delay > 0:
                # Shift entry signals by delay
                delayed_signals = base_signals.shift(delay).fillna(0)
            else:
                delayed_signals = base_signals
                
            # Run backtest
            backtest = Backtest(self.data)
            result = backtest.run(delayed_signals, self.pair)
            
            results.append({
                'delay_bars': delay,
                'sharpe': result.metrics['sharpe_ratio'],
                'returns': result.metrics['total_return'],
                'max_dd': result.metrics['max_drawdown'],
                'num_trades': result.metrics['num_trades'],
                'cost_impact': result.metrics['total_cost_impact_pct']
            })
            
        return pd.DataFrame(results)
    
    def price_noise_injection(self, 
                            noise_levels: List[float] = [0, 0.5, 1.0, 2.0]) -> pd.DataFrame:
        """
        Test strategy with price noise injection
        
        Parameters:
        -----------
        noise_levels : list
            Noise levels in pips
            
        Returns:
        --------
        pd.DataFrame with noise impact results
        """
        
        logger.info("Testing price noise injection")
        
        pip_size = self.costs.get_pip_size(self.pair)
        results = []
        
        for noise_pips in noise_levels:
            # Create noisy data
            noisy_data = self.data.copy()
            
            if noise_pips > 0:
                # Add random noise
                noise = np.random.normal(0, noise_pips * pip_size, len(self.data))
                
                # Add noise to all price columns
                for col in ['Open', 'High', 'Low', 'Close']:
                    noisy_data[col] = noisy_data[col] + noise
                    
                # Ensure OHLC consistency
                noisy_data['High'] = noisy_data[['Open', 'High', 'Low', 'Close']].max(axis=1)
                noisy_data['Low'] = noisy_data[['Open', 'High', 'Low', 'Close']].min(axis=1)
            
            # Run backtest on noisy data
            signals = momentum(noisy_data['Close'], **self.base_params)
            backtest = Backtest(noisy_data)
            result = backtest.run(signals['signal'], self.pair)
            
            results.append({
                'noise_pips': noise_pips,
                'sharpe': result.metrics['sharpe_ratio'],
                'returns': result.metrics['total_return'],
                'max_dd': result.metrics['max_drawdown'],
                'num_trades': result.metrics['num_trades']
            })
            
        return pd.DataFrame(results)
    
    def bootstrap_confidence_intervals(self, 
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Dict:
        """
        Calculate bootstrap confidence intervals for key metrics
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
        --------
        Dict with confidence intervals
        """
        
        logger.info(f"Calculating bootstrap CIs with {n_bootstrap} samples")
        
        # Generate base signals and returns
        signals = momentum(self.data['Close'], **self.base_params)
        backtest = Backtest(self.data)
        base_result = backtest.run(signals['signal'], self.pair)
        
        # Get daily returns
        daily_returns = base_result.returns.resample('D').sum()
        daily_returns = daily_returns[daily_returns != 0]  # Remove non-trading days
        
        # Bootstrap
        bootstrap_sharpes = []
        bootstrap_returns = []
        bootstrap_dds = []
        
        for _ in range(n_bootstrap):
            # Resample returns with replacement
            sample_returns = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
            
            # Calculate metrics
            if np.std(sample_returns) > 0:
                sharpe = np.sqrt(252) * np.mean(sample_returns) / np.std(sample_returns)
            else:
                sharpe = 0
                
            total_return = (np.prod(1 + sample_returns) - 1) * 100
            
            # Max drawdown
            cumulative = np.cumprod(1 + sample_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(np.min(drawdown)) * 100
            
            bootstrap_sharpes.append(sharpe)
            bootstrap_returns.append(total_return)
            bootstrap_dds.append(max_dd)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        results = {
            'sharpe_ratio': {
                'mean': np.mean(bootstrap_sharpes),
                'std': np.std(bootstrap_sharpes),
                'ci_lower': np.percentile(bootstrap_sharpes, lower_percentile),
                'ci_upper': np.percentile(bootstrap_sharpes, upper_percentile),
                'original': base_result.metrics['sharpe_ratio']
            },
            'total_return': {
                'mean': np.mean(bootstrap_returns),
                'std': np.std(bootstrap_returns),
                'ci_lower': np.percentile(bootstrap_returns, lower_percentile),
                'ci_upper': np.percentile(bootstrap_returns, upper_percentile),
                'original': base_result.metrics['total_return']
            },
            'max_drawdown': {
                'mean': np.mean(bootstrap_dds),
                'std': np.std(bootstrap_dds),
                'ci_lower': np.percentile(bootstrap_dds, lower_percentile),
                'ci_upper': np.percentile(bootstrap_dds, upper_percentile),
                'original': base_result.metrics['max_drawdown']
            }
        }
        
        return results
    
    def crisis_period_analysis(self) -> pd.DataFrame:
        """
        Analyze performance during crisis periods
        
        Returns:
        --------
        pd.DataFrame with crisis period performance
        """
        
        # Define crisis periods
        crisis_periods = [
            ('2008 Financial Crisis', '2008-01-01', '2009-03-31'),
            ('COVID-19 Crash', '2020-02-01', '2020-04-30'),
            ('2022 Gilt Crisis', '2022-09-01', '2022-10-31'),
            ('2015 China Devaluation', '2015-08-01', '2015-09-30'),
            ('2018 Volatility Spike', '2018-01-15', '2018-02-28')
        ]
        
        results = []
        
        for crisis_name, start_date, end_date in crisis_periods:
            # Check if we have data for this period
            if start_date < str(self.data.index[0]) or end_date > str(self.data.index[-1]):
                continue
                
            # Get crisis period data
            crisis_data = self.data[start_date:end_date]
            
            if len(crisis_data) < 100:  # Need minimum data
                continue
                
            # Run backtest
            try:
                signals = momentum(crisis_data['Close'], **self.base_params)
                backtest = Backtest(crisis_data)
                result = backtest.run(signals['signal'], self.pair)
                
                results.append({
                    'period': crisis_name,
                    'start': start_date,
                    'end': end_date,
                    'days': len(crisis_data) / 96,  # Approximate
                    'sharpe': result.metrics['sharpe_ratio'],
                    'returns': result.metrics['total_return'],
                    'max_dd': result.metrics['max_drawdown'],
                    'num_trades': result.metrics['num_trades']
                })
            except:
                continue
                
        return pd.DataFrame(results)
    
    def _run_single_backtest(self, params: Dict) -> Dict:
        """Run a single backtest with given parameters"""
        
        signals = momentum(self.data['Close'], **params)
        backtest = Backtest(self.data)
        result = backtest.run(signals['signal'], self.pair)
        
        return {
            'params': params,
            'metrics': result.metrics,
            'trades': len(result.trades)
        }
    
    def create_robustness_report(self, output_dir: Path):
        """Create comprehensive robustness report"""
        
        output_dir.mkdir(exist_ok=True)
        
        # Run all tests
        logger.info("Running comprehensive robustness analysis...")
        
        # 1. Parameter heatmap
        param_ranges = {
            'lookback': [20, 30, 40, 50, 60],
            'entry_z': [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        }
        heatmap_results = self.parameter_heatmap(param_ranges)
        
        # 2. Parameter stability
        lookback_stability = self.parameter_stability('lookback', 40, 50)
        entry_z_stability = self.parameter_stability('entry_z', 1.5, 50)
        
        # 3. Trade delays
        delay_results = self.trade_delay_test([0, 1, 2, 3])
        
        # 4. Price noise
        noise_results = self.price_noise_injection([0, 0.25, 0.5, 1.0])
        
        # 5. Bootstrap CIs
        bootstrap_results = self.bootstrap_confidence_intervals()
        
        # 6. Crisis periods
        crisis_results = self.crisis_period_analysis()
        
        # Create visualizations
        self._create_robustness_plots(
            heatmap_results, lookback_stability, entry_z_stability,
            delay_results, noise_results, bootstrap_results,
            output_dir
        )
        
        # Save numerical results
        results = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat(),
            'bootstrap_ci': bootstrap_results,
            'delay_impact': delay_results.to_dict('records'),
            'noise_impact': noise_results.to_dict('records'),
            'crisis_performance': crisis_results.to_dict('records') if len(crisis_results) > 0 else []
        }
        
        with open(output_dir / 'robustness_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Robustness report saved to {output_dir}")
        
        return results
    
    def _create_robustness_plots(self, heatmap_results, lookback_stab, entry_z_stab,
                                delay_results, noise_results, bootstrap_results,
                                output_dir):
        """Create robustness visualization plots"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Parameter heatmap
        plt.subplot(3, 3, 1)
        sns.heatmap(heatmap_results, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title(f'{self.pair} - Parameter Sensitivity Heatmap (Sharpe)')
        plt.xlabel('Entry Z-Score')
        plt.ylabel('Lookback Period')
        
        # 2. Lookback stability
        plt.subplot(3, 3, 2)
        plt.plot(lookback_stab['lookback'], lookback_stab['sharpe'], 'b-', marker='o')
        plt.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Base value')
        plt.xlabel('Lookback Period')
        plt.ylabel('Sharpe Ratio')
        plt.title('Lookback Parameter Stability')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 3. Entry Z stability
        plt.subplot(3, 3, 3)
        plt.plot(entry_z_stab['entry_z'], entry_z_stab['sharpe'], 'g-', marker='o')
        plt.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Base value')
        plt.xlabel('Entry Z-Score')
        plt.ylabel('Sharpe Ratio')
        plt.title('Entry Z-Score Stability')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 4. Trade delay impact
        plt.subplot(3, 3, 4)
        x = delay_results['delay_bars']
        plt.bar(x, delay_results['sharpe'], color='orange', alpha=0.7)
        plt.xlabel('Delay (bars)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Trade Execution Delay Impact')
        plt.grid(True, alpha=0.3)
        
        # 5. Price noise impact
        plt.subplot(3, 3, 5)
        plt.plot(noise_results['noise_pips'], noise_results['sharpe'], 'r-', marker='s')
        plt.xlabel('Noise Level (pips)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Price Noise Robustness')
        plt.grid(True, alpha=0.3)
        
        # 6. Bootstrap distributions
        plt.subplot(3, 3, 6)
        # Create sample distribution for visualization
        np.random.seed(42)
        sharpe_samples = np.random.normal(
            bootstrap_results['sharpe_ratio']['mean'],
            bootstrap_results['sharpe_ratio']['std'],
            1000
        )
        plt.hist(sharpe_samples, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(bootstrap_results['sharpe_ratio']['ci_lower'], color='red', 
                   linestyle='--', label=f"95% CI: [{bootstrap_results['sharpe_ratio']['ci_lower']:.2f}, "
                                       f"{bootstrap_results['sharpe_ratio']['ci_upper']:.2f}]")
        plt.axvline(bootstrap_results['sharpe_ratio']['ci_upper'], color='red', linestyle='--')
        plt.axvline(bootstrap_results['sharpe_ratio']['original'], color='green', 
                   linestyle='-', linewidth=2, label=f"Original: {bootstrap_results['sharpe_ratio']['original']:.2f}")
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Frequency')
        plt.title('Bootstrap Sharpe Distribution')
        plt.legend()
        
        # 7. Returns vs Risk scatter
        plt.subplot(3, 3, 7)
        param_results = []
        for lb in [20, 30, 40, 50, 60]:
            for ez in [1.0, 1.25, 1.5, 1.75, 2.0]:
                params = {'lookback': lb, 'entry_z': ez, 'exit_z': 0.5}
                try:
                    res = self._run_single_backtest(params)
                    param_results.append({
                        'returns': res['metrics']['total_return'],
                        'max_dd': res['metrics']['max_drawdown'],
                        'sharpe': res['metrics']['sharpe_ratio']
                    })
                except:
                    continue
                    
        if param_results:
            pr_df = pd.DataFrame(param_results)
            scatter = plt.scatter(pr_df['max_dd'], pr_df['returns'], 
                                c=pr_df['sharpe'], cmap='viridis', s=100, alpha=0.6)
            plt.colorbar(scatter, label='Sharpe Ratio')
            plt.xlabel('Max Drawdown (%)')
            plt.ylabel('Total Return (%)')
            plt.title('Risk-Return Profile')
            plt.grid(True, alpha=0.3)
        
        # 8. Summary text
        plt.subplot(3, 3, 9)
        plt.axis('off')
        
        summary_text = f"""
        Robustness Analysis Summary - {self.pair}
        
        Parameter Stability:
        - Sharpe > 0 in {(heatmap_results > 0).sum().sum()}/{heatmap_results.size} parameter combinations
        - Best Sharpe: {heatmap_results.max().max():.3f}
        
        Execution Robustness:
        - 1-bar delay impact: {(delay_results.loc[1, 'sharpe'] / delay_results.loc[0, 'sharpe'] - 1) * 100:.1f}%
        - 0.5 pip noise impact: {(noise_results.loc[noise_results['noise_pips'] == 0.5, 'sharpe'].values[0] / noise_results.loc[0, 'sharpe'] - 1) * 100:.1f}%
        
        Statistical Confidence:
        - Sharpe 95% CI: [{bootstrap_results['sharpe_ratio']['ci_lower']:.3f}, {bootstrap_results['sharpe_ratio']['ci_upper']:.3f}]
        - CI excludes zero: {'YES' if bootstrap_results['sharpe_ratio']['ci_lower'] > 0 else 'NO'}
        
        Overall Assessment: {'ROBUST' if bootstrap_results['sharpe_ratio']['ci_lower'] > 0.5 else 'MODERATE' if bootstrap_results['sharpe_ratio']['ci_lower'] > 0 else 'WEAK'}
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robustness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage
    data_path = Path('../../data/AUDUSD_MASTER_15M.csv')
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    # Use last 100k bars for speed
    data = data[-100000:]
    
    analyzer = RobustnessAnalyzer(data, 'AUDUSD')
    results = analyzer.create_robustness_report(Path('robustness_output'))
    
    print("\nRobustness Analysis Complete!")
    print(f"Sharpe 95% CI: [{results['bootstrap_ci']['sharpe_ratio']['ci_lower']:.3f}, "
          f"{results['bootstrap_ci']['sharpe_ratio']['ci_upper']:.3f}]")