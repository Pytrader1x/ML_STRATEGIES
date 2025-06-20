"""
Fast Recursive Strategy Improver - Uses synthetic data for rapid iteration
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import warnings
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')

class FastRecursiveImprover:
    def __init__(self):
        self.iteration = 0
        self.results = []
        self.best_config = None
        self.best_score = -999
        
    def create_realistic_market_data(self, trend_bias=0, volatility=0.0002):
        """Create realistic synthetic market data"""
        np.random.seed(42 + self.iteration)
        
        # Create 5 days of 15-min data
        periods = 480  # 5 days * 96 bars/day
        dates = pd.date_range('2024-01-01', periods=periods, freq='15min')
        
        # Generate realistic price movement
        returns = np.random.normal(trend_bias * 0.00001, volatility, periods)
        prices = 0.6500 * np.exp(returns.cumsum())
        
        # Create OHLC data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = np.roll(prices, 1)
        df['Open'].iloc[0] = prices[0]
        
        # High/Low with realistic wicks
        df['High'] = df[['Open', 'Close']].max(axis=1) + np.abs(np.random.normal(0, 0.00005, periods))
        df['Low'] = df[['Open', 'Close']].min(axis=1) - np.abs(np.random.normal(0, 0.00005, periods))
        
        # Generate correlated indicators
        # NTI tends to follow price trend
        price_change = df['Close'].pct_change().fillna(0)
        df['NTI_Direction'] = np.where(price_change > 0.0001, 1, 
                                       np.where(price_change < -0.0001, -1, 0))
        df['NTI_Confidence'] = 50 + np.random.normal(0, 20, periods)
        df['NTI_Confidence'] = df['NTI_Confidence'].clip(0, 100)
        
        # Market Bias similar to NTI but with some lag
        df['MB_Bias'] = df['NTI_Direction'].shift(2).fillna(0)
        
        # IC Regime based on volatility
        rolling_std = df['Close'].rolling(20).std()
        df['IC_Regime'] = pd.cut(rolling_std, bins=4, labels=[1, 2, 3, 4]).fillna(2)
        df['IC_RegimeName'] = df['IC_Regime'].map({
            1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range', 4: 'Chop'
        })
        
        # ATR calculation
        df['IC_ATR_Normalized'] = rolling_std.fillna(0.0002)
        df['IC_ATR_MA'] = df['IC_ATR_Normalized'].rolling(10).mean().fillna(0.0002)
        
        # Market Bias bands
        df['MB_l2'] = df['Low'].rolling(5).min()
        df['MB_h2'] = df['High'].rolling(5).max()
        
        return df.dropna()
    
    def test_config(self, config, market_type='mixed'):
        """Test configuration on different market conditions"""
        results = []
        
        # Test on different market conditions
        market_configs = {
            'trending_up': (0.5, 0.0001),
            'trending_down': (-0.5, 0.0001),
            'volatile': (0, 0.0004),
            'calm': (0, 0.00005),
            'mixed': (0.1, 0.0002)
        }
        
        if market_type == 'all':
            test_markets = market_configs.items()
        else:
            test_markets = [(market_type, market_configs[market_type])]
        
        for market_name, (trend, vol) in test_markets:
            df = self.create_realistic_market_data(trend, vol)
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(df)
            results.append(result)
        
        # Average results
        avg_result = {
            'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
            'total_return': np.mean([r['total_return'] for r in results]),
            'win_rate': np.mean([r['win_rate'] for r in results]),
            'max_drawdown': np.mean([r['max_drawdown'] for r in results]),
            'total_trades': int(np.mean([r['total_trades'] for r in results]))
        }
        
        return avg_result
    
    def calculate_score(self, result):
        """Calculate optimization score"""
        # Penalize negative sharpe heavily
        sharpe_score = max(0, result['sharpe_ratio']) * 40
        
        # Reward win rate above 50%
        win_score = max(0, result['win_rate'] - 50) * 2
        
        # Reward positive returns
        return_score = max(-10, result['total_return']) * 5
        
        # Penalize high drawdown
        dd_score = max(0, 10 - result['max_drawdown']) * 3
        
        return sharpe_score + win_score + return_score + dd_score
    
    def improve_config(self, base_config, result, iteration):
        """Improve configuration based on results"""
        improvements = []
        
        # Create new config
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
        
        # Apply improvements based on iteration and results
        if iteration == 1:
            # Fix tight stops
            new_config.sl_min_pips = 5.0
            new_config.sl_max_pips = 15.0
            improvements.append("Widened stops: 5-15 pips")
            
        elif iteration == 2:
            # Fix partial profit
            new_config.partial_profit_sl_distance_ratio = 0.5
            new_config.partial_profit_size_percent = 0.5
            improvements.append("Balanced partial profit: 50% at 50%")
            
        elif iteration == 3:
            # Wider targets
            new_config.tp_atr_multipliers = (0.3, 0.5, 0.8)
            new_config.max_tp_percent = 0.008
            improvements.append("Wider TP targets: 0.3x/0.5x/0.8x ATR")
            
        elif iteration == 4:
            # Institutional sizing
            new_config.base_position_size_millions = 2.0
            new_config.relaxed_position_multiplier = 0.5
            improvements.append("Institutional sizing: 1M/2M")
            
        elif iteration == 5:
            # Enable intelligent sizing
            new_config.intelligent_sizing = True
            new_config.confidence_thresholds = (40.0, 60.0, 80.0)
            new_config.size_multipliers = (0.5, 0.75, 1.0, 1.25)
            improvements.append("Intelligent sizing enabled")
            
        elif iteration == 6:
            # Fine-tune based on previous results
            if result['win_rate'] < 50:
                new_config.tsl_activation_pips = 6.0
                improvements.append("Earlier TSL activation")
            if result['sharpe_ratio'] < 0:
                new_config.risk_per_trade = 0.003
                improvements.append("Reduced risk to 0.3%")
        
        return new_config, improvements
    
    def run_improvement_cycle(self, max_iterations=6):
        """Run the improvement cycle"""
        print("="*80)
        print("FAST RECURSIVE STRATEGY IMPROVEMENT")
        print("="*80)
        
        # Start with baseline
        current_config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
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
        
        for i in range(max_iterations):
            self.iteration = i
            
            print(f"\n{'='*60}")
            print(f"ITERATION {i + 1}")
            print(f"{'='*60}")
            
            # Test current config
            result = self.test_config(current_config, 'all')
            score = self.calculate_score(result)
            
            # Store results
            self.results.append({
                'iteration': i + 1,
                'config': current_config,
                'result': result,
                'score': score
            })
            
            # Print results
            print(f"Results:")
            print(f"  Sharpe: {result['sharpe_ratio']:.3f}")
            print(f"  Return: {result['total_return']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Max DD: {result['max_drawdown']:.2f}%")
            print(f"  Score: {score:.1f}")
            
            # Update best if improved
            if score > self.best_score:
                self.best_score = score
                self.best_config = current_config
                print("  🏆 New best!")
            
            # Stop if good enough
            if result['sharpe_ratio'] > 0.5 and result['win_rate'] > 55:
                print("\n✅ Target achieved!")
                break
            
            # Improve for next iteration
            if i < max_iterations - 1:
                current_config, improvements = self.improve_config(current_config, result, i + 1)
                print(f"\nImprovements for next iteration:")
                for imp in improvements:
                    print(f"  • {imp}")
        
        self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("IMPROVEMENT SUMMARY")
        print("="*80)
        
        # Results table
        print("\nIteration | Sharpe | Return | Win Rate | Score")
        print("-" * 50)
        for r in self.results:
            print(f"{r['iteration']:9d} | {r['result']['sharpe_ratio']:6.2f} | "
                  f"{r['result']['total_return']:6.1f}% | "
                  f"{r['result']['win_rate']:7.1f}% | {r['score']:6.1f}")
        
        # Improvement
        first_score = self.results[0]['score']
        last_score = self.results[-1]['score']
        improvement = ((last_score - first_score) / abs(first_score) * 100) if first_score != 0 else 0
        
        print(f"\n📈 Total Improvement: {improvement:+.1f}%")
        
        # Best config summary
        if self.best_config:
            print(f"\n🏆 BEST CONFIGURATION (Score: {self.best_score:.1f}):")
            best_result = next(r for r in self.results if r['score'] == self.best_score)['result']
            print(f"  Sharpe: {best_result['sharpe_ratio']:.3f}")
            print(f"  Return: {best_result['total_return']:.2f}%") 
            print(f"  Win Rate: {best_result['win_rate']:.1f}%")
            
            print(f"\n💡 Key Parameters:")
            print(f"  Stops: {self.best_config.sl_min_pips}-{self.best_config.sl_max_pips} pips")
            print(f"  Targets: {self.best_config.tp_atr_multipliers}")
            print(f"  Position: {self.best_config.base_position_size_millions}M base")
            print(f"  Intelligent Sizing: {self.best_config.intelligent_sizing}")


def main():
    improver = FastRecursiveImprover()
    improver.run_improvement_cycle(max_iterations=6)


if __name__ == "__main__":
    main()