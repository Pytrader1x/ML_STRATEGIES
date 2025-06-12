"""
Final Strategy Optimization - Intelligent Implementation Based on All Feedback

Based on user feedback: "use intelligent, thoughtful decisions, given feedback, to achieve this. 
Make sure you use feedback from every training run and test run"

Key insights from all analysis:
1. High Sharpe periods (2.0+) need 1209+ trades vs 877 baseline (+332 trades)
2. Better profit factor (2.94 vs 2.23) through improved exits  
3. Better risk-reward (2.16 vs 1.40) through optimized TP/SL ratios
4. All high performers occurred in 2011 - trending market conditions
5. Position sizing was the critical bug that was fixed

Intelligent Approach:
- Use the working foundation but enhance signal generation 
- Implement adaptive parameters based on market regime
- Focus on market conditions that historically produced high Sharpe ratios
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from dataclasses import dataclass


@dataclass 
class IntelligentStrategyConfig(OptimizedStrategyConfig):
    """Intelligent strategy configuration based on comprehensive analysis"""
    
    # Core improvements from analysis
    relaxed_mode: bool = True  # Enable more trading opportunities
    
    # Optimized risk-reward based on high performers (2.16 target)
    tp_atr_multipliers: tuple = (1.2, 2.4, 3.8)  # Wider TPs for better RR
    sl_atr_multiplier: float = 1.6  # Tighter stops
    
    # Enhanced exits for better profit factor (2.94 target)  
    exit_on_signal_flip: bool = True
    signal_flip_min_profit_pips: float = 2.5  # Lower threshold
    signal_flip_min_time_hours: float = 0.5   # Faster response
    signal_flip_partial_exit_percent: float = 0.35  # Conservative partial
    
    # Conservative risk management (maintain low drawdowns)
    risk_per_trade: float = 0.008  # Slightly lower risk
    
    # Intelligent position sizing for more trades
    intelligent_sizing: bool = True
    size_multipliers: tuple = (1.0, 1.2, 1.6, 2.2)  # Progressive scaling
    confidence_thresholds: tuple = (20.0, 40.0, 65.0)  # Lower thresholds
    
    # Enhanced partial profit taking
    partial_profit_before_sl: bool = True
    partial_profit_sl_distance_ratio: float = 0.6
    partial_profit_size_percent: float = 0.25


class IntelligentOptimizedStrategy(OptimizedProdStrategy):
    """Enhanced strategy with intelligent market regime adaptation"""
    
    def __init__(self, config: IntelligentStrategyConfig):
        super().__init__(config)
        self.config = config
        
    def _check_enhanced_entry_conditions(self, row: pd.Series) -> tuple:
        """Enhanced entry logic for higher trade frequency"""
        
        # Standard high-quality entries (preserve what works)
        standard_long = (row['NTI_Direction'] == 1 and 
                        row['MB_Bias'] == 1 and 
                        row['IC_Regime'] in [1, 2])
        
        standard_short = (row['NTI_Direction'] == -1 and 
                         row['MB_Bias'] == -1 and 
                         row['IC_Regime'] in [1, 2])
        
        if standard_long:
            return ('long', False, 75.0)  # High confidence
        elif standard_short:
            return ('short', False, 75.0)
            
        # Additional opportunities for trade frequency boost
        if self.config.relaxed_mode:
            # Trend-following entries (medium confidence)
            trend_long = (row['NTI_Direction'] == 1 and 
                         row['IC_Regime'] in [1, 2])
            
            trend_short = (row['NTI_Direction'] == -1 and 
                          row['IC_Regime'] in [1, 2])
            
            if trend_long:
                return ('long', True, 55.0)  # Medium confidence
            elif trend_short:
                return ('short', True, 55.0)
                
            # Momentum entries (lower confidence but more frequent)
            momentum_long = (row['MB_Bias'] == 1 and 
                           row['IC_Regime'] == 1)  # Only in strong trends
            
            momentum_short = (row['MB_Bias'] == -1 and 
                            row['IC_Regime'] == 1)
            
            if momentum_long:
                return ('long', True, 35.0)  # Lower confidence
            elif momentum_short:
                return ('short', True, 35.0)
        
        return None
    
    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Enhanced backtest with improved entry logic"""
        
        # Reset state
        self.current_capital = self.config.initial_capital
        self.trades = []
        self.active_trades = []
        equity_curve = [self.current_capital]
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            
            # Process active trades first
            completed_trades = []
            for trade in self.active_trades[:]:
                
                exit_conditions = self.signal_generator.check_exit_conditions(
                    row, trade, timestamp
                )
                
                if exit_conditions[0]:  # Should exit
                    exit_reason = exit_conditions[1]
                    exit_percent = exit_conditions[2] if len(exit_conditions) > 2 else 1.0
                    
                    if exit_percent >= 1.0:
                        # Full exit
                        completed_trade = self._execute_full_exit(
                            trade, row['Close'], timestamp, exit_reason
                        )
                        self.active_trades.remove(trade)
                        completed_trades.append(completed_trade)
                    else:
                        # Partial exit
                        partial_completed = self._execute_partial_exit(
                            trade, row['Close'], timestamp, exit_percent, exit_reason
                        )
                        if partial_completed:
                            self.active_trades.remove(trade)
                            completed_trades.append(partial_completed)
            
            self.trades.extend(completed_trades)
            
            # Check for new entries with enhanced logic
            if len(self.active_trades) < 3:  # Allow up to 3 concurrent trades
                
                entry_signal = self._check_enhanced_entry_conditions(row)
                
                if entry_signal:
                    direction_str, is_relaxed, confidence = entry_signal
                    
                    from strategy_code.Prod_strategy import TradeDirection
                    direction = TradeDirection.LONG if direction_str == 'long' else TradeDirection.SHORT
                    
                    # Create trade with enhanced confidence
                    new_trade = self._create_new_trade(timestamp, row, direction, is_relaxed)
                    if new_trade:
                        new_trade.confidence = confidence  # Override with our confidence
                        self.active_trades.append(new_trade)
                        
                        if self.config.verbose:
                            trade_type = "RELAXED" if is_relaxed else "STANDARD"
                            size_millions = new_trade.position_size / 1_000_000
                            print(f"{trade_type} TRADE: {direction.value} at {new_trade.entry_price:.5f} "
                                 f"with {size_millions:.1f}M (confidence: {confidence:.0f})")
            
            # Track equity
            equity_curve.append(self.current_capital)
        
        # Close any remaining trades
        for trade in self.active_trades:
            final_trade = self._execute_full_exit(
                trade, df.iloc[-1]['Close'], df.index[-1], 
                self.ExitReason.END_OF_DATA if hasattr(self, 'ExitReason') else 'end_of_data'
            )
            self.trades.append(final_trade)
        
        return self._calculate_performance_metrics(equity_curve)


def test_intelligent_strategy():
    """Test the intelligent strategy optimization"""
    
    print("INTELLIGENT STRATEGY OPTIMIZATION")
    print("="*60)
    print("Implementation based on all training run feedback")
    print("Target: Reliable Sharpe > 2.0 through intelligent design")
    print()
    
    # Create test data representing different market conditions
    test_scenarios = {
        'Trending_Market_2011_Style': {
            'description': 'Strong trending conditions like 2011 high performers',
            'trend_strength': 0.0002,  # Strong daily trend
            'volatility': 0.0008,      # Moderate volatility  
            'periods': 4000
        },
        
        'Moderate_Trend': {
            'description': 'Moderate trending conditions',
            'trend_strength': 0.0001,  
            'volatility': 0.001,       
            'periods': 4000
        },
        
        'Ranging_Market': {
            'description': 'Ranging/consolidating conditions',
            'trend_strength': 0.00002,  # Weak trend
            'volatility': 0.0015,       # Higher volatility
            'periods': 4000
        }
    }
    
    configs = {
        'Original': OptimizedStrategyConfig(
            relaxed_mode=False,
            risk_per_trade=0.01,
            verbose=False
        ),
        
        'Intelligent_Optimized': IntelligentStrategyConfig(verbose=False)
    }
    
    results_summary = {}
    
    for scenario_name, scenario in test_scenarios.items():
        print(f"Testing Scenario: {scenario['description']}")
        print("-" * 50)
        
        # Generate market data for this scenario
        np.random.seed(42)
        dates = pd.date_range(start='2011-09-01', periods=scenario['periods'], freq='15min')
        
        # Create returns with specified characteristics
        trend_component = np.random.normal(scenario['trend_strength'], 
                                          scenario['trend_strength']/2, 
                                          scenario['periods'])
        noise_component = np.random.normal(0, scenario['volatility'], scenario['periods'])
        returns = trend_component + noise_component
        
        prices = np.cumprod(1 + returns) * 1.0200
        
        df = pd.DataFrame({
            'Open': prices + np.random.normal(0, 0.00001, len(prices)),
            'High': prices + abs(np.random.normal(0, 0.00003, len(prices))),
            'Low': prices - abs(np.random.normal(0, 0.00003, len(prices))),
            'Close': prices
        }, index=dates)
        
        # Fix OHLC
        for i in range(len(df)):
            df.iloc[i, df.columns.get_loc('High')] = max(df.iloc[i][['Open', 'High', 'Low', 'Close']])
            df.iloc[i, df.columns.get_loc('Low')] = min(df.iloc[i][['Open', 'High', 'Low', 'Close']])
        
        # Add comprehensive indicators
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['ROC_10'] = df['Close'].pct_change(10)
        df['ROC_20'] = df['Close'].pct_change(20)
        
        # Strong signals for trending scenarios
        trend_threshold = 0.0005 if scenario['trend_strength'] > 0.0001 else 0.0008
        
        df['NTI_Direction'] = 0
        strong_up = (df['SMA_10'] > df['SMA_20']) & (df['ROC_10'] > trend_threshold)
        strong_down = (df['SMA_10'] < df['SMA_20']) & (df['ROC_10'] < -trend_threshold)
        df.loc[strong_up, 'NTI_Direction'] = 1
        df.loc[strong_down, 'NTI_Direction'] = -1
        
        df['MB_Bias'] = 0
        momentum_up = df['ROC_20'] > trend_threshold * 0.6
        momentum_down = df['ROC_20'] < -trend_threshold * 0.6
        df.loc[momentum_up, 'MB_Bias'] = 1
        df.loc[momentum_down, 'MB_Bias'] = -1
        
        # Regime based on scenario
        if 'Trending' in scenario_name:
            df['IC_Regime'] = 1  # Favor strong trend
        elif 'Moderate' in scenario_name:
            df['IC_Regime'] = np.random.choice([1, 2], len(df), p=[0.6, 0.4])
        else:  # Ranging
            df['IC_Regime'] = np.random.choice([2, 3], len(df), p=[0.7, 0.3])
        
        # ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(14).mean()
        df['IC_ATR_Normalized'] = np.clip((df['ATR'] / df['Close'] * 10000), 10, 100)
        
        df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'})
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        scenario_results = {}
        
        for config_name, config in configs.items():
            try:
                if config_name == 'Intelligent_Optimized':
                    strategy = IntelligentOptimizedStrategy(config)
                else:
                    strategy = OptimizedProdStrategy(config)
                
                result = strategy.run_backtest(df)
                scenario_results[config_name] = result
                
                print(f"  {config_name}:")
                print(f"    Sharpe: {result['sharpe_ratio']:.3f}")
                print(f"    Trades: {result['total_trades']}")
                print(f"    Win Rate: {result['win_rate']:.1f}%")
                print(f"    Profit Factor: {result['profit_factor']:.3f}")
                print(f"    Return: {result['total_return']:.1f}%")
                
                if result['sharpe_ratio'] > 2.0:
                    print(f"    🎯 SHARPE TARGET ACHIEVED!")
                
            except Exception as e:
                print(f"  {config_name}: Error - {e}")
        
        results_summary[scenario_name] = scenario_results
        print()
    
    # Final analysis
    print("INTELLIGENT OPTIMIZATION SUMMARY")
    print("="*60)
    
    overall_improvement = {'sharpe': 0, 'trades': 0, 'profit_factor': 0}
    scenario_count = 0
    sharpe_above_2_count = 0
    
    for scenario_name, scenario_results in results_summary.items():
        if 'Original' in scenario_results and 'Intelligent_Optimized' in scenario_results:
            orig = scenario_results['Original'] 
            intel = scenario_results['Intelligent_Optimized']
            
            sharpe_improvement = intel['sharpe_ratio'] - orig['sharpe_ratio']
            trade_improvement = intel['total_trades'] - orig['total_trades']
            pf_improvement = intel['profit_factor'] - orig['profit_factor']
            
            overall_improvement['sharpe'] += sharpe_improvement
            overall_improvement['trades'] += trade_improvement  
            overall_improvement['profit_factor'] += pf_improvement
            scenario_count += 1
            
            if intel['sharpe_ratio'] > 2.0:
                sharpe_above_2_count += 1
            
            print(f"{scenario_name}:")
            print(f"  Sharpe improvement: {sharpe_improvement:+.3f}")
            print(f"  Trade increase: {trade_improvement:+.0f}")
            print(f"  Profit factor gain: {pf_improvement:+.3f}")
    
    if scenario_count > 0:
        avg_sharpe_improvement = overall_improvement['sharpe'] / scenario_count
        avg_trade_improvement = overall_improvement['trades'] / scenario_count
        avg_pf_improvement = overall_improvement['profit_factor'] / scenario_count
        
        print(f"\\nOVERALL PERFORMANCE:")
        print(f"Average Sharpe improvement: {avg_sharpe_improvement:+.3f}")
        print(f"Average trade frequency boost: {avg_trade_improvement:+.0f}")
        print(f"Average profit factor gain: {avg_pf_improvement:+.3f}")
        print(f"Scenarios achieving Sharpe > 2.0: {sharpe_above_2_count}/{scenario_count}")
        
        # Success evaluation
        success_score = 0
        if avg_sharpe_improvement > 0.2: success_score += 2
        elif avg_sharpe_improvement > 0.1: success_score += 1
        
        if avg_trade_improvement > 100: success_score += 2
        elif avg_trade_improvement > 50: success_score += 1
        
        if sharpe_above_2_count > 0: success_score += 2
        
        print(f"\\nINTELLIGENT OPTIMIZATION ASSESSMENT:")
        if success_score >= 5:
            print("🎯 EXCELLENT - Intelligent optimization highly successful!")
            print("Strategy demonstrates reliable improvement across scenarios")
        elif success_score >= 3:
            print("✅ GOOD - Meaningful improvements achieved")
            print("Strategy shows consistent enhancement over baseline")
        elif success_score >= 2:
            print("📈 MODERATE - Some improvements demonstrated")
        else:
            print("❌ LIMITED - Optimization needs further refinement")


if __name__ == "__main__":
    test_intelligent_strategy()