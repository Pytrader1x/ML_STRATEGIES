"""
Targeted Optimization of Existing Strategy
Simple, focused improvements to increase Sharpe > 2.0 frequency

Based on analysis: Need +124 more trades and better profit factor
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig

@dataclass 
class TargetedOptimizationConfig(OptimizedStrategyConfig):
    """Targeted optimizations to existing proven strategy"""
    
    # TRADE FREQUENCY OPTIMIZATION
    # Target: +124 more trades (1024 vs 900)
    relaxed_mode: bool = True  # Enable relaxed entries for more trades
    aggressive_entry_timing: bool = True  # Slightly looser entry conditions
    
    # PROFIT FACTOR OPTIMIZATION  
    # Target: 2.57+ profit factor (currently 2.19)
    enhanced_exits: bool = True  # Better exit timing
    tp_atr_multipliers: tuple = (1.2, 2.2, 3.5)  # Wider TPs for better RR
    
    # RISK-REWARD OPTIMIZATION
    # Target: 1.71+ RR ratio (currently 1.38)  
    sl_atr_multiplier: float = 1.6  # Tighter stops
    smart_tp_scaling: bool = True  # Scale TPs based on momentum
    
    # MAINTAIN WHAT WORKS
    risk_per_trade: float = 0.015  # Keep conservative risk
    exit_on_signal_flip: bool = True  # Keep signal flip logic
    signal_flip_min_profit_pips: float = 3.0  # Keep existing threshold
    
    # FINE-TUNING FOR FREQUENCY
    minimum_signal_strength: float = 0.4  # Lower threshold for more trades
    entry_confirmation_buffer: float = 0.8  # Less strict confirmation

class TargetedOptimizedStrategy(OptimizedProdStrategy):
    """Targeted optimization of existing strategy"""
    
    def __init__(self, config: TargetedOptimizationConfig):
        super().__init__(config)
        self.config = config
        
        # Override signal generator for targeted improvements
        self.signal_generator = TargetedSignalGenerator(config)
    
    def _calculate_optimized_take_profits(self, entry_price: float, direction, atr: float, row) -> list:
        """Enhanced TP calculation for better risk-reward"""
        
        # Base distances from config
        tp_distances = [atr * mult for mult in self.config.tp_atr_multipliers]
        
        # Smart scaling based on momentum (if available)
        if self.config.smart_tp_scaling and 'NTI_Strength' in row:
            momentum = getattr(row, 'NTI_Strength', 0.5)
            if momentum > 0.7:  # Strong momentum - wider TPs
                tp_distances = [d * 1.2 for d in tp_distances]
            elif momentum < 0.3:  # Weak momentum - tighter TPs  
                tp_distances = [d * 0.8 for d in tp_distances]
        
        # Calculate actual price levels
        if direction.value == 'long':
            return [entry_price + (d * 0.0001) for d in tp_distances]  # 0.0001 = 1 pip for forex
        else:
            return [entry_price - (d * 0.0001) for d in tp_distances]

class TargetedSignalGenerator:
    """Enhanced signal generator for more trades while maintaining quality"""
    
    def __init__(self, config: TargetedOptimizationConfig):
        self.config = config
    
    def check_entry_conditions(self, row) -> tuple:
        """Enhanced entry conditions for higher trade frequency"""
        
        # Standard high-quality entries (maintain existing logic)
        standard_long = (row['NTI_Direction'] == 1 and 
                        row['MB_Bias'] == 1 and 
                        row['IC_Regime'] in [1, 2])
        
        standard_short = (row['NTI_Direction'] == -1 and 
                         row['MB_Bias'] == -1 and 
                         row['IC_Regime'] in [1, 2])
        
        if standard_long:
            return ('long', False)
        elif standard_short:
            return ('short', False)
        
        # OPTIMIZATION: More aggressive entry conditions for higher frequency
        if self.config.aggressive_entry_timing:
            # Allow entries with just NeuroTrend + favorable regime
            aggressive_long = (row['NTI_Direction'] == 1 and 
                              row['IC_Regime'] in [1, 2] and
                              row.get('NTI_Strength', 0.5) > self.config.minimum_signal_strength)
            
            aggressive_short = (row['NTI_Direction'] == -1 and 
                               row['IC_Regime'] in [1, 2] and  
                               row.get('NTI_Strength', 0.5) > self.config.minimum_signal_strength)
            
            if aggressive_long:
                return ('long', True)  # Mark as relaxed
            elif aggressive_short:
                return ('short', True)
        
        # Existing relaxed mode logic
        if self.config.relaxed_mode:
            if row['NTI_Direction'] == 1:
                return ('long', True)
            elif row['NTI_Direction'] == -1:
                return ('short', True)
        
        return None
    
    def check_exit_conditions(self, row, trade, current_time) -> tuple:
        """Enhanced exit conditions for better profit factor"""
        
        # Standard exit logic (maintain what works)
        current_price = row['Close']
        
        # Take profit checks
        for i, tp in enumerate(trade.take_profits):
            if ((trade.direction.value == 'long' and current_price >= tp) or
                (trade.direction.value == 'short' and current_price <= tp)):
                return (True, f"take_profit_{i+1}", 1.0 if i == 2 else 0.33)
        
        # Stop loss check  
        if ((trade.direction.value == 'long' and current_price <= trade.stop_loss) or
            (trade.direction.value == 'short' and current_price >= trade.stop_loss)):
            return (True, "stop_loss", 1.0)
        
        # Enhanced signal flip logic for better profit factor
        if self.config.exit_on_signal_flip:
            should_exit, exit_reason = self._check_enhanced_signal_flip(row, trade, current_time)
            if should_exit:
                return (True, "signal_flip", 0.6)  # Partial exit to lock profits
        
        return (False, None, 0.0)
    
    def _check_enhanced_signal_flip(self, row, trade, current_time) -> tuple:
        """Enhanced signal flip for better profit factor"""
        
        # Check minimum time requirement
        hours_in_trade = (current_time - trade.entry_time).total_seconds() / 3600
        if hours_in_trade < 1.0:  # Reduced from 2 hours
            return (False, None)
        
        # Check minimum profit
        current_price = row['Close']
        if trade.direction.value == 'long':
            profit_pips = (current_price - trade.entry_price) * 10000
        else:
            profit_pips = (trade.entry_price - current_price) * 10000
        
        if profit_pips < self.config.signal_flip_min_profit_pips:
            return (False, None)
        
        # Check for signal flip (same as existing)
        signal_flipped = False
        if trade.direction.value == 'long':
            if row['NTI_Direction'] == -1 or row['MB_Bias'] == -1:
                signal_flipped = True
        else:
            if row['NTI_Direction'] == 1 or row['MB_Bias'] == 1:
                signal_flipped = True
        
        return (signal_flipped, "signal_flip")
    
    def check_partial_profit_conditions(self, row, trade) -> bool:
        """Keep existing partial profit logic"""
        return False  # Disabled for simplicity

def test_targeted_optimization():
    """Test the targeted optimization approach"""
    
    print("TESTING TARGETED OPTIMIZATION")
    print("="*50)
    print("Goal: Increase Sharpe > 2.0 frequency from 16% to 25%+")
    print("Method: Focused improvements to existing proven strategy")
    print()
    
    # Test different optimization levels
    configs = {
        'Original': OptimizedStrategyConfig(verbose=False),
        'Light_Optimization': TargetedOptimizationConfig(
            relaxed_mode=True,
            aggressive_entry_timing=False,
            enhanced_exits=True,
            verbose=False
        ),
        'Moderate_Optimization': TargetedOptimizationConfig(
            relaxed_mode=True, 
            aggressive_entry_timing=True,
            enhanced_exits=True,
            minimum_signal_strength=0.4,
            verbose=False
        ),
        'Aggressive_Optimization': TargetedOptimizationConfig(
            relaxed_mode=True,
            aggressive_entry_timing=True, 
            enhanced_exits=True,
            minimum_signal_strength=0.3,
            tp_atr_multipliers=(1.0, 2.0, 3.0),
            sl_atr_multiplier=1.4,
            verbose=False
        )
    }
    
    # Quick test with synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', periods=1500, freq='1H')
    returns = np.random.normal(0, 0.001, 1500)
    prices = np.cumprod(1 + returns) * 0.75
    
    df = pd.DataFrame({
        'Open': prices + np.random.normal(0, 0.0001, 1500),
        'High': prices + abs(np.random.normal(0, 0.0002, 1500)),
        'Low': prices - abs(np.random.normal(0, 0.0002, 1500)),
        'Close': prices
    }, index=dates)
    
    # Add indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['NTI_Direction'] = np.where(df['SMA_20'] > df['SMA_50'] * 1.001, 1, 
                                  np.where(df['SMA_20'] < df['SMA_50'] * 0.999, -1, 0))
    df['NTI_Strength'] = np.clip(abs(df['SMA_20'] - df['SMA_50']) / df['Close'] * 100, 0.1, 1.0)
    df['ROC_10'] = df['Close'].pct_change(10)
    df['MB_Bias'] = np.where(df['ROC_10'] > 0.002, 1, 
                            np.where(df['ROC_10'] < -0.002, -1, 0))
    df['IC_Regime'] = np.random.choice([1, 2, 3], size=len(df), p=[0.3, 0.5, 0.2])
    df['IC_ATR_Normalized'] = np.random.uniform(20, 60, len(df))
    df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'})
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        if config_name == 'Original':
            strategy = OptimizedProdStrategy(config)
            result = strategy.run_backtest(df)
        else:
            strategy = TargetedOptimizedStrategy(config)  
            result = strategy.run_backtest(df)
        
        results[config_name] = result
        
        print(f"  Sharpe: {result['sharpe_ratio']:.3f}")
        print(f"  Trades: {result['total_trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.3f}")
        if result['total_trades'] > 0:
            print(f"  Return: {result['total_return']:.1f}%")
        print()
    
    # Compare results
    print("COMPARISON SUMMARY:")
    print("-" * 30)
    
    original_sharpe = results['Original']['sharpe_ratio'] 
    
    for config_name, result in results.items():
        if config_name == 'Original':
            continue
            
        sharpe_improvement = result['sharpe_ratio'] - original_sharpe
        trade_increase = result['total_trades'] - results['Original']['total_trades']
        
        print(f"{config_name}:")
        print(f"  Sharpe: {result['sharpe_ratio']:.3f} ({sharpe_improvement:+.3f})")
        print(f"  Trades: {result['total_trades']} ({trade_increase:+d})")
        
        if result['sharpe_ratio'] > 2.0:
            print(f"  🎯 TARGET ACHIEVED! Sharpe > 2.0")
        elif sharpe_improvement > 0:
            print(f"  ✅ Improvement over original")
        else:
            print(f"  ❌ No improvement")
        print()
    
    print("RECOMMENDATION:")
    print("-" * 15)
    
    best_config = max(results.keys(), key=lambda k: results[k]['sharpe_ratio'])
    best_sharpe = results[best_config]['sharpe_ratio']
    
    if best_sharpe > 2.0:
        print(f"🎯 SUCCESS! {best_config} achieved Sharpe > 2.0")
        print("Ready for live testing with this configuration.")
    elif best_sharpe > original_sharpe:
        print(f"📈 PROGRESS! {best_config} improved over original")
        print(f"Continue refinement to reach Sharpe > 2.0 target.")
    else:
        print("❌ No improvements found. Original strategy remains best.")
        print("Consider different optimization approaches.")

if __name__ == "__main__":
    test_targeted_optimization()