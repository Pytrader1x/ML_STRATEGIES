"""
Intelligent Self-Improving Strategy
This strategy uses recursive learning and feedback to optimize itself
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time
from collections import defaultdict
import random

warnings.filterwarnings('ignore')

@dataclass
class StrategyHypothesis:
    """Represents a hypothesis about what makes a good strategy"""
    name: str
    description: str
    parameters: Dict[str, float]
    confidence: float = 0.5
    test_count: int = 0
    success_count: int = 0
    
    def update_confidence(self, success: bool):
        """Update confidence based on test result"""
        self.test_count += 1
        if success:
            self.success_count += 1
        self.confidence = self.success_count / self.test_count if self.test_count > 0 else 0.5

@dataclass
class TestResult:
    """Container for test results"""
    hypothesis: StrategyHypothesis
    period: str
    sharpe: float
    return_pct: float
    win_rate: float
    max_drawdown: float
    trades: int
    success: bool
    feedback: str

class IntelligentStrategyOptimizer:
    """Self-improving strategy optimizer with feedback loops"""
    
    def __init__(self, currency_pair='AUDUSD', target_sharpe=0.7):
        self.currency_pair = currency_pair
        self.target_sharpe = target_sharpe
        self.hypotheses = []
        self.test_results = []
        self.learned_insights = defaultdict(list)
        self.parameter_ranges = self._initialize_parameter_ranges()
        self.df = None
        self.iteration = 0
        
    def _initialize_parameter_ranges(self):
        """Initialize parameter search ranges based on domain knowledge"""
        return {
            'risk_per_trade': (0.001, 0.01),  # 0.1% to 1%
            'sl_min_pips': (2.0, 15.0),
            'sl_max_pips': (10.0, 50.0),
            'sl_atr_multiplier': (0.5, 3.0),
            'tp1_multiplier': (0.1, 0.5),
            'tp2_multiplier': (0.2, 0.8),
            'tp3_multiplier': (0.5, 2.0),
            'tsl_activation_pips': (5.0, 25.0),
            'tsl_min_profit_pips': (0.5, 5.0),
            'trailing_atr_multiplier': (0.8, 2.5),
            'tp_range_market_multiplier': (0.3, 0.8),
            'tp_trend_market_multiplier': (0.8, 1.5),
            'tp_chop_market_multiplier': (0.2, 0.6),
            'partial_profit_sl_distance_ratio': (0.2, 0.8),
            'partial_profit_size_percent': (0.3, 0.8),
            'exit_on_signal_flip': [True, False],
            'partial_profit_before_sl': [True, False],
            'relaxed_mode': [True, False]
        }
    
    def load_data(self):
        """Load and prepare data"""
        data_path = 'data' if os.path.exists('data') else '../data'
        file_path = os.path.join(data_path, f'{self.currency_pair}_MASTER_15M.csv')
        
        print(f"Loading {self.currency_pair} data...")
        self.df = pd.read_csv(file_path)
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        # Calculate indicators
        print("Calculating indicators...")
        self.df = TIC.add_neuro_trend_intelligent(self.df)
        self.df = TIC.add_market_bias(self.df, ha_len=350, ha_len2=30)
        self.df = TIC.add_intelligent_chop(self.df)
        
        print(f"Data loaded: {len(self.df):,} rows from {self.df.index[0]} to {self.df.index[-1]}")
    
    def generate_initial_hypotheses(self):
        """Generate initial strategy hypotheses based on different trading philosophies"""
        
        # Hypothesis 1: Conservative Trend Following
        self.hypotheses.append(StrategyHypothesis(
            name="Conservative Trend Following",
            description="Large stops, small risk, ride trends with trailing stops",
            parameters={
                'risk_per_trade': 0.002,
                'sl_min_pips': 10.0,
                'sl_max_pips': 30.0,
                'sl_atr_multiplier': 2.0,
                'tp1_multiplier': 0.3,
                'tp2_multiplier': 0.6,
                'tp3_multiplier': 1.2,
                'tsl_activation_pips': 20.0,
                'tsl_min_profit_pips': 3.0,
                'trailing_atr_multiplier': 1.5,
                'tp_range_market_multiplier': 0.5,
                'tp_trend_market_multiplier': 1.2,
                'tp_chop_market_multiplier': 0.3,
                'partial_profit_sl_distance_ratio': 0.5,
                'partial_profit_size_percent': 0.5,
                'exit_on_signal_flip': False,
                'partial_profit_before_sl': True,
                'relaxed_mode': False
            }
        ))
        
        # Hypothesis 2: Aggressive Scalping
        self.hypotheses.append(StrategyHypothesis(
            name="Aggressive Scalping",
            description="Tight stops, higher risk, quick exits",
            parameters={
                'risk_per_trade': 0.005,
                'sl_min_pips': 3.0,
                'sl_max_pips': 10.0,
                'sl_atr_multiplier': 0.8,
                'tp1_multiplier': 0.15,
                'tp2_multiplier': 0.25,
                'tp3_multiplier': 0.4,
                'tsl_activation_pips': 8.0,
                'tsl_min_profit_pips': 1.0,
                'trailing_atr_multiplier': 0.8,
                'tp_range_market_multiplier': 0.4,
                'tp_trend_market_multiplier': 0.6,
                'tp_chop_market_multiplier': 0.3,
                'partial_profit_sl_distance_ratio': 0.3,
                'partial_profit_size_percent': 0.7,
                'exit_on_signal_flip': True,
                'partial_profit_before_sl': True,
                'relaxed_mode': True
            }
        ))
        
        # Hypothesis 3: Balanced Swing Trading
        self.hypotheses.append(StrategyHypothesis(
            name="Balanced Swing Trading",
            description="Medium stops, balanced risk/reward, flexible exits",
            parameters={
                'risk_per_trade': 0.003,
                'sl_min_pips': 6.0,
                'sl_max_pips': 20.0,
                'sl_atr_multiplier': 1.2,
                'tp1_multiplier': 0.25,
                'tp2_multiplier': 0.5,
                'tp3_multiplier': 0.8,
                'tsl_activation_pips': 15.0,
                'tsl_min_profit_pips': 2.0,
                'trailing_atr_multiplier': 1.2,
                'tp_range_market_multiplier': 0.6,
                'tp_trend_market_multiplier': 1.0,
                'tp_chop_market_multiplier': 0.4,
                'partial_profit_sl_distance_ratio': 0.4,
                'partial_profit_size_percent': 0.6,
                'exit_on_signal_flip': False,
                'partial_profit_before_sl': True,
                'relaxed_mode': False
            }
        ))
        
        # Hypothesis 4: ATR-Based Adaptive
        self.hypotheses.append(StrategyHypothesis(
            name="ATR-Based Adaptive",
            description="Fully volatility-based parameters",
            parameters={
                'risk_per_trade': 0.0025,
                'sl_min_pips': 5.0,
                'sl_max_pips': 25.0,
                'sl_atr_multiplier': 1.5,
                'tp1_multiplier': 0.2,
                'tp2_multiplier': 0.4,
                'tp3_multiplier': 1.0,
                'tsl_activation_pips': 12.0,
                'tsl_min_profit_pips': 1.5,
                'trailing_atr_multiplier': 1.3,
                'tp_range_market_multiplier': 0.5,
                'tp_trend_market_multiplier': 1.1,
                'tp_chop_market_multiplier': 0.35,
                'partial_profit_sl_distance_ratio': 0.45,
                'partial_profit_size_percent': 0.55,
                'exit_on_signal_flip': False,
                'partial_profit_before_sl': True,
                'relaxed_mode': False
            }
        ))
    
    def test_hypothesis(self, hypothesis: StrategyHypothesis, test_period: Tuple[str, str]) -> TestResult:
        """Test a hypothesis on a specific time period"""
        start_date, end_date = test_period
        period_name = f"{start_date} to {end_date}"
        
        # Filter data
        period_df = self.df.loc[start_date:end_date].copy()
        
        if len(period_df) < 100:
            return TestResult(
                hypothesis=hypothesis,
                period=period_name,
                sharpe=0,
                return_pct=0,
                win_rate=0,
                max_drawdown=0,
                trades=0,
                success=False,
                feedback="Insufficient data"
            )
        
        # Create strategy config
        config = self._create_config_from_hypothesis(hypothesis)
        strategy = OptimizedProdStrategy(config)
        
        # Run backtest
        try:
            result = strategy.run_backtest(period_df)
            
            sharpe = result.get('sharpe_ratio', 0)
            return_pct = result.get('total_return', 0)
            win_rate = result.get('win_rate', 0)
            max_dd = result.get('max_drawdown', 0)
            trades = result.get('total_trades', 0)
            
            # Determine success
            success = sharpe >= self.target_sharpe and trades >= 10
            
            # Generate feedback
            feedback = self._generate_feedback(result, hypothesis)
            
            return TestResult(
                hypothesis=hypothesis,
                period=period_name,
                sharpe=sharpe,
                return_pct=return_pct,
                win_rate=win_rate,
                max_drawdown=max_dd,
                trades=trades,
                success=success,
                feedback=feedback
            )
            
        except Exception as e:
            return TestResult(
                hypothesis=hypothesis,
                period=period_name,
                sharpe=0,
                return_pct=0,
                win_rate=0,
                max_drawdown=0,
                trades=0,
                success=False,
                feedback=f"Error: {str(e)}"
            )
    
    def _create_config_from_hypothesis(self, hypothesis: StrategyHypothesis) -> OptimizedStrategyConfig:
        """Create strategy config from hypothesis parameters"""
        params = hypothesis.parameters
        
        return OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=params['risk_per_trade'],
            sl_min_pips=params['sl_min_pips'],
            sl_max_pips=params['sl_max_pips'],
            sl_atr_multiplier=params['sl_atr_multiplier'],
            tp_atr_multipliers=(
                params['tp1_multiplier'],
                params['tp2_multiplier'],
                params['tp3_multiplier']
            ),
            max_tp_percent=0.005,
            tsl_activation_pips=params['tsl_activation_pips'],
            tsl_min_profit_pips=params['tsl_min_profit_pips'],
            tsl_initial_buffer_multiplier=1.0,
            trailing_atr_multiplier=params['trailing_atr_multiplier'],
            tp_range_market_multiplier=params['tp_range_market_multiplier'],
            tp_trend_market_multiplier=params['tp_trend_market_multiplier'],
            tp_chop_market_multiplier=params['tp_chop_market_multiplier'],
            sl_range_market_multiplier=0.7,
            exit_on_signal_flip=params['exit_on_signal_flip'],
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=params['partial_profit_before_sl'],
            partial_profit_sl_distance_ratio=params['partial_profit_sl_distance_ratio'],
            partial_profit_size_percent=params['partial_profit_size_percent'],
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            relaxed_position_multiplier=0.5,
            relaxed_mode=params['relaxed_mode'],
            realistic_costs=True,
            verbose=False,
            debug_decisions=False,
            use_daily_sharpe=True
        )
    
    def _generate_feedback(self, result: Dict, hypothesis: StrategyHypothesis) -> str:
        """Generate intelligent feedback based on results"""
        feedback_parts = []
        
        sharpe = result.get('sharpe_ratio', 0)
        win_rate = result.get('win_rate', 0)
        trades = result.get('total_trades', 0)
        avg_win = result.get('avg_win', 0)
        avg_loss = result.get('avg_loss', 0)
        max_dd = result.get('max_drawdown', 0)
        
        # Sharpe analysis
        if sharpe < 0.3:
            feedback_parts.append("Very poor risk-adjusted returns")
        elif sharpe < 0.7:
            feedback_parts.append("Below target Sharpe")
        else:
            feedback_parts.append("Good risk-adjusted returns")
        
        # Win rate analysis
        if win_rate < 40:
            feedback_parts.append("Low win rate - consider tighter stops or better entries")
        elif win_rate > 70:
            feedback_parts.append("High win rate but check if cutting winners too early")
        
        # Risk/reward analysis
        if avg_loss != 0:
            rr_ratio = abs(avg_win / avg_loss)
            if rr_ratio < 1:
                feedback_parts.append("Poor R/R ratio - winners smaller than losers")
            elif rr_ratio > 2:
                feedback_parts.append("Excellent R/R ratio")
        
        # Trade frequency
        if trades < 10:
            feedback_parts.append("Too few trades for reliable statistics")
        elif trades > 100:
            feedback_parts.append("Good sample size")
        
        # Drawdown
        if max_dd > 20:
            feedback_parts.append("High drawdown - reduce position size or tighten risk")
        
        return "; ".join(feedback_parts)
    
    def learn_from_results(self):
        """Extract insights from test results"""
        if not self.test_results:
            return
        
        # Group results by success
        successful_results = [r for r in self.test_results if r.success]
        failed_results = [r for r in self.test_results if not r.success]
        
        if successful_results:
            # Analyze successful parameters
            successful_params = defaultdict(list)
            for result in successful_results:
                for param, value in result.hypothesis.parameters.items():
                    if isinstance(value, (int, float)):
                        successful_params[param].append(value)
            
            # Store insights about successful parameter ranges
            for param, values in successful_params.items():
                if values:
                    self.learned_insights[f"{param}_successful_range"].append({
                        'min': min(values),
                        'max': max(values),
                        'mean': np.mean(values),
                        'std': np.std(values)
                    })
        
        # Analyze failure patterns
        if failed_results:
            failure_patterns = defaultdict(int)
            for result in failed_results:
                for part in result.feedback.split(";"):
                    failure_patterns[part.strip()] += 1
            
            # Store common failure reasons
            self.learned_insights['common_failures'] = [
                (reason, count) for reason, count in 
                sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)
            ]
    
    def generate_improved_hypothesis(self) -> StrategyHypothesis:
        """Generate a new hypothesis based on learned insights"""
        self.iteration += 1
        
        # Start with base parameters
        new_params = {}
        
        # Use successful parameter ranges if available
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, (list, bool)):
                # Boolean or categorical parameter
                new_params[param] = random.choice(min_val if isinstance(min_val, list) else [True, False])
            else:
                # Numeric parameter
                insight_key = f"{param}_successful_range"
                if insight_key in self.learned_insights and self.learned_insights[insight_key]:
                    # Use learned successful range
                    latest_insight = self.learned_insights[insight_key][-1]
                    mean = latest_insight['mean']
                    std = latest_insight['std']
                    
                    # Sample around successful mean with some exploration
                    value = np.random.normal(mean, std * 0.5)
                    value = np.clip(value, min_val, max_val)
                    new_params[param] = float(value)
                else:
                    # Random exploration
                    new_params[param] = random.uniform(min_val, max_val)
        
        # Apply specific learned adjustments
        if 'common_failures' in self.learned_insights:
            failures = self.learned_insights['common_failures']
            
            # Adjust based on common failures
            for failure, count in failures[:3]:  # Top 3 failures
                if "Low win rate" in failure:
                    new_params['sl_min_pips'] *= 0.8
                    new_params['tp1_multiplier'] *= 0.9
                elif "Poor R/R ratio" in failure:
                    new_params['tp2_multiplier'] *= 1.1
                    new_params['tp3_multiplier'] *= 1.2
                elif "High drawdown" in failure:
                    new_params['risk_per_trade'] *= 0.8
                elif "Too few trades" in failure:
                    new_params['relaxed_mode'] = True
                    new_params['sl_max_pips'] *= 1.2
        
        # Create hypothesis name
        name = f"Evolved_Gen{self.iteration}"
        if self.test_results:
            best_result = max([r for r in self.test_results if r.sharpe > 0], 
                            key=lambda x: x.sharpe, default=None)
            if best_result:
                name = f"Evolved_from_{best_result.hypothesis.name}_Gen{self.iteration}"
        
        return StrategyHypothesis(
            name=name,
            description=f"Evolved hypothesis based on {len(self.test_results)} previous tests",
            parameters=new_params
        )
    
    def run_optimization_cycle(self, max_iterations=20, test_periods=None):
        """Run the full optimization cycle"""
        if test_periods is None:
            test_periods = [
                ('2023-01-01', '2023-03-31'),
                ('2023-04-01', '2023-06-30'),
                ('2023-07-01', '2023-09-30'),
                ('2023-10-01', '2023-12-31'),
                ('2024-01-01', '2024-03-31'),
            ]
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        # Generate initial hypotheses
        self.generate_initial_hypotheses()
        
        print(f"\n🚀 Starting Intelligent Strategy Optimization")
        print(f"Target Sharpe: {self.target_sharpe}")
        print(f"Test Periods: {len(test_periods)}")
        print("="*60)
        
        best_hypothesis = None
        best_avg_sharpe = 0
        consecutive_failures = 0
        
        for iteration in range(max_iterations):
            print(f"\n📊 Iteration {iteration + 1}/{max_iterations}")
            
            # Select hypothesis to test
            if iteration < len(self.hypotheses):
                # Test initial hypotheses first
                hypothesis = self.hypotheses[iteration]
            else:
                # Generate new hypothesis based on learning
                hypothesis = self.generate_improved_hypothesis()
                self.hypotheses.append(hypothesis)
            
            print(f"Testing: {hypothesis.name}")
            print(f"Description: {hypothesis.description}")
            
            # Test across all periods
            period_results = []
            for period in test_periods:
                result = self.test_hypothesis(hypothesis, period)
                period_results.append(result)
                self.test_results.append(result)
                
                print(f"  {period[0]} to {period[1]}: Sharpe={result.sharpe:.3f}, "
                      f"Return={result.return_pct:.1f}%, Trades={result.trades}")
            
            # Calculate average performance
            valid_results = [r for r in period_results if r.trades >= 10]
            if valid_results:
                avg_sharpe = np.mean([r.sharpe for r in valid_results])
                success_rate = sum(1 for r in valid_results if r.success) / len(valid_results)
                
                print(f"\n  Average Sharpe: {avg_sharpe:.3f}")
                print(f"  Success Rate: {success_rate:.1%}")
                
                # Update hypothesis confidence
                hypothesis.update_confidence(avg_sharpe >= self.target_sharpe)
                
                # Track best hypothesis
                if avg_sharpe > best_avg_sharpe:
                    best_avg_sharpe = avg_sharpe
                    best_hypothesis = hypothesis
                    consecutive_failures = 0
                    print(f"  🎯 New best! Average Sharpe: {avg_sharpe:.3f}")
                else:
                    consecutive_failures += 1
                
                # Check if we've achieved target
                if avg_sharpe >= self.target_sharpe and success_rate >= 0.7:
                    print(f"\n✅ SUCCESS! Found robust strategy with Sharpe {avg_sharpe:.3f}")
                    break
            else:
                print("  ⚠️ Insufficient trades for evaluation")
                consecutive_failures += 1
            
            # Learn from results
            self.learn_from_results()
            
            # Print learning insights
            if iteration > 0 and iteration % 5 == 0:
                self._print_learning_summary()
            
            # Early stopping if stuck
            if consecutive_failures >= 5:
                print("\n⚠️ No improvement in 5 iterations, trying exploration...")
                # Force exploration by randomizing more
                self.parameter_ranges = self._widen_parameter_ranges()
                consecutive_failures = 0
        
        # Final summary
        self._print_final_summary(best_hypothesis, best_avg_sharpe)
        
        return best_hypothesis
    
    def _widen_parameter_ranges(self):
        """Widen parameter ranges for more exploration"""
        widened = {}
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, (int, float)):
                range_width = max_val - min_val
                widened[param] = (
                    max(0.001, min_val - range_width * 0.2),
                    max_val + range_width * 0.2
                )
            else:
                widened[param] = (min_val, max_val)
        return widened
    
    def _print_learning_summary(self):
        """Print what the optimizer has learned"""
        print("\n📚 Learning Summary:")
        
        # Print successful parameter ranges
        for param in ['risk_per_trade', 'sl_min_pips', 'tp1_multiplier', 'tsl_activation_pips']:
            key = f"{param}_successful_range"
            if key in self.learned_insights and self.learned_insights[key]:
                latest = self.learned_insights[key][-1]
                print(f"  {param}: {latest['mean']:.3f} ± {latest['std']:.3f}")
        
        # Print common failures
        if 'common_failures' in self.learned_insights:
            print("\n  Common Issues:")
            for reason, count in self.learned_insights['common_failures'][:3]:
                print(f"    - {reason} ({count} times)")
    
    def _print_final_summary(self, best_hypothesis: Optional[StrategyHypothesis], best_sharpe: float):
        """Print final optimization summary"""
        print("\n" + "="*60)
        print("🏁 OPTIMIZATION COMPLETE")
        print("="*60)
        
        if best_hypothesis and best_sharpe > 0:
            print(f"\n🏆 Best Strategy: {best_hypothesis.name}")
            print(f"Average Sharpe: {best_sharpe:.3f}")
            print(f"Confidence: {best_hypothesis.confidence:.1%}")
            print("\nOptimal Parameters:")
            for param, value in sorted(best_hypothesis.parameters.items()):
                if isinstance(value, float):
                    print(f"  {param}: {value:.3f}")
                else:
                    print(f"  {param}: {value}")
            
            # Save best configuration
            self._save_best_config(best_hypothesis)
        else:
            print("\n❌ No successful strategy found")
            print("Consider:")
            print("  - Lowering target Sharpe ratio")
            print("  - Expanding parameter ranges")
            print("  - Testing on different time periods")
    
    def _save_best_config(self, hypothesis: StrategyHypothesis):
        """Save best configuration to file"""
        config = {
            'name': hypothesis.name,
            'description': hypothesis.description,
            'parameters': hypothesis.parameters,
            'confidence': hypothesis.confidence,
            'test_count': hypothesis.test_count,
            'success_count': hypothesis.success_count,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f'optimized_configs/intelligent_strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs('optimized_configs', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration saved to: {filename}")

def main():
    """Main function to run the intelligent optimizer"""
    optimizer = IntelligentStrategyOptimizer(
        currency_pair='AUDUSD',
        target_sharpe=0.7  # Realistic target
    )
    
    # Run optimization
    best_strategy = optimizer.run_optimization_cycle(
        max_iterations=30,
        test_periods=[
            ('2023-01-01', '2023-03-31'),
            ('2023-04-01', '2023-06-30'),
            ('2023-07-01', '2023-09-30'),
            ('2023-10-01', '2023-12-31'),
            ('2024-01-01', '2024-03-31'),
            ('2024-04-01', '2024-06-30'),
        ]
    )
    
    return best_strategy

if __name__ == "__main__":
    main()