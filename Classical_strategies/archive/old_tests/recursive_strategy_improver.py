"""
Recursive Strategy Improvement System
Uses feedback from each test to automatically improve parameters
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Store performance metrics for comparison"""
    sharpe_ratio: float
    total_return: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    avg_trade: float
    total_trades: int
    config_name: str
    
    def score(self) -> float:
        """Calculate overall score for ranking configurations"""
        # Weighted scoring system
        score = (
            self.sharpe_ratio * 40 +  # Most important
            self.win_rate * 0.3 +      # Important
            self.total_return * 0.2 +  # Important
            (100 - self.max_drawdown) * 0.1  # Risk control
        )
        return score


class RecursiveStrategyImprover:
    """Recursively improve strategy based on performance feedback"""
    
    def __init__(self, base_capital=1_000_000, test_periods=5):
        self.base_capital = base_capital
        self.test_periods = test_periods
        self.results_history = []
        self.best_config = None
        self.best_metrics = None
        self.iteration = 0
        self.improvement_log = []
        
    def load_test_data(self):
        """Load data for testing"""
        print("Loading test data...")
        data_path = '../data' if os.path.exists('../data') else 'data'
        
        # Load real data
        try:
            df = pd.read_csv(f'{data_path}/AUDUSD_MASTER_15M.csv')
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)
            
            # Use last 6 months for faster testing
            df = df.tail(17280)  # 180 days * 96 bars/day
            
            print("Calculating indicators...")
            df = TIC.add_neuro_trend_intelligent(df)
            df = TIC.add_market_bias(df)
            df = TIC.add_intelligent_chop(df)
            
            # Split into test periods
            period_size = len(df) // self.test_periods
            self.test_sets = []
            for i in range(self.test_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < self.test_periods - 1 else len(df)
                self.test_sets.append(df.iloc[start_idx:end_idx])
            
            print(f"Created {len(self.test_sets)} test periods")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating synthetic test data...")
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic data for testing"""
        self.test_sets = []
        for i in range(self.test_periods):
            dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
            np.random.seed(42 + i)
            
            # Create trending synthetic data
            trend = np.random.choice([-1, 0, 1])
            prices = [0.6500]
            for j in range(999):
                drift = trend * 0.00001 + np.random.randn() * 0.0002
                prices.append(prices[-1] + drift)
            
            df = pd.DataFrame({
                'Open': prices[:-1],
                'High': [p + abs(np.random.randn()) * 0.0001 for p in prices[:-1]],
                'Low': [p - abs(np.random.randn()) * 0.0001 for p in prices[:-1]],
                'Close': prices[1:],
                'NTI_Direction': np.random.choice([-1, 0, 1], 999, p=[0.3, 0.4, 0.3]),
                'NTI_Confidence': np.random.uniform(20, 80, 999),
                'MB_Bias': np.random.choice([-1, 0, 1], 999, p=[0.3, 0.4, 0.3]),
                'IC_Regime': np.random.choice([1, 2, 3, 4], 999, p=[0.2, 0.3, 0.3, 0.2]),
                'IC_RegimeName': np.random.choice(['Strong Trend', 'Weak Trend', 'Range', 'Chop'], 999),
                'IC_ATR_Normalized': np.random.uniform(0.0001, 0.0003, 999),
                'IC_ATR_MA': np.random.uniform(0.0001, 0.0003, 999),
                'MB_l2': [p - 0.002 for p in prices[:-1]],
                'MB_h2': [p + 0.002 for p in prices[:-1]]
            }, index=dates[:-1])
            
            self.test_sets.append(df)
    
    def create_base_config(self) -> OptimizedStrategyConfig:
        """Create the baseline configuration"""
        return OptimizedStrategyConfig(
            initial_capital=self.base_capital,
            risk_per_trade=0.005,
            base_position_size_millions=1.0,
            
            sl_min_pips=3.0,
            sl_max_pips=10.0,
            sl_atr_multiplier=0.8,
            
            tp_atr_multipliers=(0.15, 0.25, 0.4),
            max_tp_percent=0.005,
            
            tsl_activation_pips=8.0,
            tsl_min_profit_pips=1.0,
            
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.3,
            partial_profit_size_percent=0.7,
            
            relaxed_mode=True,
            relaxed_position_multiplier=0.5,
            
            intelligent_sizing=False,
            realistic_costs=True,
            
            verbose=False
        )
    
    def test_configuration(self, config: OptimizedStrategyConfig, config_name: str) -> PerformanceMetrics:
        """Test a configuration across all test periods"""
        results = []
        
        for i, test_df in enumerate(self.test_sets):
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(test_df)
            results.append(result)
        
        # Calculate average metrics
        avg_metrics = PerformanceMetrics(
            sharpe_ratio=np.mean([r['sharpe_ratio'] for r in results]),
            total_return=np.mean([r['total_return'] for r in results]),
            win_rate=np.mean([r['win_rate'] for r in results]),
            max_drawdown=np.mean([r['max_drawdown'] for r in results]),
            profit_factor=np.mean([r.get('profit_factor', 1.0) for r in results]),
            avg_trade=np.mean([r.get('avg_trade', 0) for r in results]),
            total_trades=int(np.mean([r['total_trades'] for r in results])),
            config_name=config_name
        )
        
        return avg_metrics
    
    def analyze_feedback(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """Analyze performance and determine improvements needed"""
        feedback = {}
        
        # Analyze Sharpe ratio
        if metrics.sharpe_ratio < 0:
            feedback['sharpe'] = "negative_sharpe"
        elif metrics.sharpe_ratio < 0.5:
            feedback['sharpe'] = "low_sharpe"
        elif metrics.sharpe_ratio < 1.0:
            feedback['sharpe'] = "moderate_sharpe"
        else:
            feedback['sharpe'] = "good_sharpe"
        
        # Analyze win rate
        if metrics.win_rate < 40:
            feedback['win_rate'] = "very_low_win_rate"
        elif metrics.win_rate < 50:
            feedback['win_rate'] = "low_win_rate"
        elif metrics.win_rate < 60:
            feedback['win_rate'] = "moderate_win_rate"
        else:
            feedback['win_rate'] = "good_win_rate"
        
        # Analyze drawdown
        if metrics.max_drawdown > 10:
            feedback['risk'] = "high_risk"
        elif metrics.max_drawdown > 5:
            feedback['risk'] = "moderate_risk"
        else:
            feedback['risk'] = "low_risk"
        
        # Analyze trade frequency
        if metrics.total_trades < 100:
            feedback['frequency'] = "low_frequency"
        elif metrics.total_trades > 500:
            feedback['frequency'] = "high_frequency"
        else:
            feedback['frequency'] = "moderate_frequency"
        
        return feedback
    
    def improve_config(self, base_config: OptimizedStrategyConfig, 
                      feedback: Dict[str, str], iteration: int) -> OptimizedStrategyConfig:
        """Create improved configuration based on feedback"""
        
        # Copy base config
        new_config = OptimizedStrategyConfig(
            initial_capital=base_config.initial_capital,
            risk_per_trade=base_config.risk_per_trade,
            base_position_size_millions=base_config.base_position_size_millions,
            sl_min_pips=base_config.sl_min_pips,
            sl_max_pips=base_config.sl_max_pips,
            sl_atr_multiplier=base_config.sl_atr_multiplier,
            tp_atr_multipliers=base_config.tp_atr_multipliers,
            max_tp_percent=base_config.max_tp_percent,
            tsl_activation_pips=base_config.tsl_activation_pips,
            tsl_min_profit_pips=base_config.tsl_min_profit_pips,
            partial_profit_before_sl=base_config.partial_profit_before_sl,
            partial_profit_sl_distance_ratio=base_config.partial_profit_sl_distance_ratio,
            partial_profit_size_percent=base_config.partial_profit_size_percent,
            relaxed_mode=base_config.relaxed_mode,
            relaxed_position_multiplier=base_config.relaxed_position_multiplier,
            intelligent_sizing=base_config.intelligent_sizing,
            realistic_costs=base_config.realistic_costs,
            verbose=False
        )
        
        improvements = []
        
        # Fix negative Sharpe
        if feedback['sharpe'] in ['negative_sharpe', 'low_sharpe']:
            if iteration == 1:
                # First try: Wider stops
                new_config.sl_min_pips = 5.0
                new_config.sl_max_pips = 15.0
                improvements.append("Widened stops to 5-15 pips")
            elif iteration == 2:
                # Second try: Better partial profits
                new_config.partial_profit_sl_distance_ratio = 0.5
                new_config.partial_profit_size_percent = 0.4
                improvements.append("Improved partial profit: 40% at 50% to TP")
            elif iteration == 3:
                # Third try: Wider targets
                new_config.tp_atr_multipliers = (0.3, 0.5, 0.8)
                new_config.max_tp_percent = 0.008
                improvements.append("Widened TP targets")
        
        # Fix low win rate
        if feedback['win_rate'] in ['very_low_win_rate', 'low_win_rate']:
            if iteration >= 2:
                # Tighter stops for higher win rate
                new_config.sl_atr_multiplier = 1.2
                new_config.tsl_activation_pips = 5.0
                improvements.append("Faster trailing stop activation")
        
        # Fix high risk
        if feedback['risk'] == 'high_risk':
            new_config.risk_per_trade = 0.002
            improvements.append("Reduced risk per trade to 0.2%")
        
        # Fix low frequency
        if feedback['frequency'] == 'low_frequency':
            new_config.relaxed_mode = True
            improvements.append("Enabled relaxed mode for more trades")
        
        # Progressive improvements
        if iteration >= 4:
            # Enable institutional sizing
            new_config.base_position_size_millions = 2.0
            new_config.relaxed_position_multiplier = 0.5
            improvements.append("Institutional sizing: 1M/2M")
        
        if iteration >= 5:
            # Enable intelligent sizing
            new_config.intelligent_sizing = True
            new_config.confidence_thresholds = (40.0, 60.0, 80.0)
            new_config.size_multipliers = (0.5, 0.75, 1.0, 1.5)
            improvements.append("Enabled intelligent sizing")
        
        self.improvement_log.append({
            'iteration': iteration,
            'improvements': improvements,
            'feedback': feedback
        })
        
        return new_config
    
    def run_recursive_improvement(self, max_iterations=10, target_sharpe=0.5):
        """Run recursive improvement process"""
        print("="*80)
        print("RECURSIVE STRATEGY IMPROVEMENT SYSTEM")
        print("="*80)
        
        # Load test data
        self.load_test_data()
        
        # Start with base configuration
        current_config = self.create_base_config()
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*60}")
            
            # Test current configuration
            config_name = f"Iteration_{iteration + 1}"
            metrics = self.test_configuration(current_config, config_name)
            
            # Store results
            self.results_history.append(metrics)
            
            # Print results
            print(f"\nResults:")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print(f"  Total Return: {metrics.total_return:.2f}%")
            print(f"  Win Rate: {metrics.win_rate:.1f}%")
            print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
            print(f"  Score: {metrics.score():.2f}")
            
            # Check if this is the best so far
            if self.best_metrics is None or metrics.score() > self.best_metrics.score():
                self.best_metrics = metrics
                self.best_config = current_config
                print("  🏆 New best configuration!")
            
            # Check if target reached
            if metrics.sharpe_ratio >= target_sharpe:
                print(f"\n✅ Target Sharpe ratio ({target_sharpe}) achieved!")
                break
            
            # Analyze feedback
            feedback = self.analyze_feedback(metrics)
            print(f"\nFeedback Analysis:")
            for key, value in feedback.items():
                print(f"  {key}: {value}")
            
            # Improve configuration
            current_config = self.improve_config(current_config, feedback, iteration + 1)
            
            if iteration < max_iterations - 1:
                print(f"\nImprovements for next iteration:")
                if self.improvement_log:
                    for imp in self.improvement_log[-1]['improvements']:
                        print(f"  • {imp}")
        
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print summary of all iterations"""
        print("\n" + "="*80)
        print("RECURSIVE IMPROVEMENT SUMMARY")
        print("="*80)
        
        # Create results dataframe
        results_data = []
        for i, metrics in enumerate(self.results_history):
            results_data.append({
                'Iteration': i + 1,
                'Sharpe': f"{metrics.sharpe_ratio:.3f}",
                'Return': f"{metrics.total_return:.1f}%",
                'Win Rate': f"{metrics.win_rate:.1f}%",
                'Drawdown': f"{metrics.max_drawdown:.1f}%",
                'Score': f"{metrics.score():.1f}"
            })
        
        df_results = pd.DataFrame(results_data)
        print("\n" + df_results.to_string(index=False))
        
        # Best configuration
        if self.best_metrics:
            print(f"\n🏆 BEST CONFIGURATION: {self.best_metrics.config_name}")
            print(f"   Sharpe: {self.best_metrics.sharpe_ratio:.3f}")
            print(f"   Return: {self.best_metrics.total_return:.2f}%")
            print(f"   Win Rate: {self.best_metrics.win_rate:.1f}%")
            print(f"   Score: {self.best_metrics.score():.2f}")
        
        # Improvement trajectory
        if len(self.results_history) > 1:
            first_score = self.results_history[0].score()
            last_score = self.results_history[-1].score()
            improvement = ((last_score - first_score) / abs(first_score)) * 100 if first_score != 0 else 0
            print(f"\n📈 Overall Improvement: {improvement:+.1f}%")
        
        # Save best configuration
        if self.best_config:
            self.save_best_config()
    
    def save_best_config(self):
        """Save the best configuration to file"""
        config_dict = {
            'initial_capital': self.best_config.initial_capital,
            'risk_per_trade': self.best_config.risk_per_trade,
            'base_position_size_millions': self.best_config.base_position_size_millions,
            'sl_min_pips': self.best_config.sl_min_pips,
            'sl_max_pips': self.best_config.sl_max_pips,
            'sl_atr_multiplier': self.best_config.sl_atr_multiplier,
            'tp_atr_multipliers': list(self.best_config.tp_atr_multipliers),
            'max_tp_percent': self.best_config.max_tp_percent,
            'tsl_activation_pips': self.best_config.tsl_activation_pips,
            'tsl_min_profit_pips': self.best_config.tsl_min_profit_pips,
            'partial_profit_before_sl': self.best_config.partial_profit_before_sl,
            'partial_profit_sl_distance_ratio': self.best_config.partial_profit_sl_distance_ratio,
            'partial_profit_size_percent': self.best_config.partial_profit_size_percent,
            'relaxed_mode': self.best_config.relaxed_mode,
            'relaxed_position_multiplier': self.best_config.relaxed_position_multiplier,
            'intelligent_sizing': self.best_config.intelligent_sizing,
            'performance': {
                'sharpe_ratio': self.best_metrics.sharpe_ratio,
                'total_return': self.best_metrics.total_return,
                'win_rate': self.best_metrics.win_rate,
                'max_drawdown': self.best_metrics.max_drawdown
            }
        }
        
        filename = f'optimized_configs/recursive_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('optimized_configs', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\n💾 Best configuration saved to: {filename}")
    
    def plot_improvement_trajectory(self):
        """Plot the improvement over iterations"""
        if len(self.results_history) < 2:
            return
        
        iterations = range(1, len(self.results_history) + 1)
        sharpes = [m.sharpe_ratio for m in self.results_history]
        returns = [m.total_return for m in self.results_history]
        win_rates = [m.win_rate for m in self.results_history]
        scores = [m.score() for m in self.results_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Sharpe ratio
        ax1.plot(iterations, sharpes, 'b-o')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Sharpe Ratio Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Returns
        ax2.plot(iterations, returns, 'g-o')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Total Return Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # Win rate
        ax3.plot(iterations, win_rates, 'm-o')
        ax3.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        ax3.set_title('Win Rate Evolution')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Win Rate (%)')
        ax3.grid(True, alpha=0.3)
        
        # Overall score
        ax4.plot(iterations, scores, 'r-o')
        ax4.set_title('Overall Score Evolution')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Score')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('charts/recursive_improvement_trajectory.png', dpi=150)
        print("\n📊 Improvement trajectory saved to: charts/recursive_improvement_trajectory.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Recursive Strategy Improvement System')
    parser.add_argument('--iterations', type=int, default=8, help='Maximum iterations')
    parser.add_argument('--target-sharpe', type=float, default=0.5, help='Target Sharpe ratio')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Initial capital')
    parser.add_argument('--test-periods', type=int, default=3, help='Number of test periods')
    parser.add_argument('--plot', action='store_true', help='Plot improvement trajectory')
    
    args = parser.parse_args()
    
    # Run recursive improvement
    improver = RecursiveStrategyImprover(
        base_capital=args.capital,
        test_periods=args.test_periods
    )
    
    improver.run_recursive_improvement(
        max_iterations=args.iterations,
        target_sharpe=args.target_sharpe
    )
    
    if args.plot:
        improver.plot_improvement_trajectory()


if __name__ == "__main__":
    main()