"""
Optimized Strategy for Consistent Sharpe > 2.0
Based on comprehensive analysis of trade patterns and performance drivers

Key Optimizations:
1. High-frequency approach with relaxed entry conditions
2. Smart signal flip exits for profit protection
3. Regime-aware position sizing
4. Enhanced risk-reward optimization
5. Proper position sizing controls

Author: Claude AI Strategy Optimizer
Date: 2025
Version: Final
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from strategy_code.Prod_strategy import (
    OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection, 
    ExitReason, Trade, PnLCalculator, FOREX_PIP_SIZE, MIN_LOT_SIZE, PIP_VALUE_PER_MILLION
)

logger = logging.getLogger(__name__)

@dataclass
class OptimizedSharpeConfig(OptimizedStrategyConfig):
    """Optimized configuration targeting consistent Sharpe > 2.0"""
    
    # High frequency trading approach
    relaxed_mode: bool = True
    aggressive_entry_mode: bool = True
    
    # Optimized risk management
    risk_per_trade: float = 0.015  # Slightly lower risk for consistency
    max_position_risk: float = 0.02  # Cap maximum risk
    
    # Enhanced take profit strategy
    tp_atr_multipliers: Tuple[float, float, float] = (1.2, 2.4, 3.6)  # Better risk-reward
    adaptive_tp_sizing: bool = True
    
    # Enhanced stop loss management
    sl_atr_multiplier: float = 1.8  # Tighter stops for better RR
    trailing_stop_aggressive: bool = True
    tsl_activation_pips: float = 12  # Earlier TSL activation
    
    # Signal flip optimization (key finding!)
    exit_on_signal_flip: bool = True
    signal_flip_min_profit_pips: float = 3.0  # Lower threshold for quicker exits
    signal_flip_min_time_hours: float = 1.0  # Faster flip detection
    signal_flip_partial_exit_percent: float = 0.7  # More aggressive partial exits
    
    # Position sizing controls (fix the unrealistic P&L issue)
    position_size_cap: float = 10.0  # Maximum position size in millions
    realistic_sizing: bool = True
    
    # Market regime optimization
    regime_position_adjustment: bool = True
    range_market_multiplier: float = 1.2  # Range markets performed well
    trend_market_multiplier: float = 1.0
    
    # Entry filter optimization
    minimum_signal_strength: float = 0.3  # Lower threshold for more trades
    entry_confirmation_required: bool = False  # Less filtering

class OptimizedSharpeStrategy(OptimizedProdStrategy):
    """Optimized strategy implementation"""
    
    def __init__(self, config: OptimizedSharpeConfig):
        super().__init__(config)
        self.config = config
        
        # Override signal generator with optimized version
        self.signal_generator = OptimizedSignalGenerator(config)
        
        # Enhanced tracking
        self.total_signal_flips = 0
        self.regime_trades = {1: 0, 2: 0, 3: 0}
    
    def _create_new_trade(self, entry_time: pd.Timestamp, row: pd.Series, 
                         direction: TradeDirection, is_relaxed: bool) -> Trade:
        """Create trade with optimized position sizing"""
        
        entry_price = row['Close']
        atr = row['IC_ATR_Normalized']
        regime = row.get('IC_Regime', 2)
        
        # Calculate position size with proper controls
        base_risk = self.config.risk_per_trade
        
        # Regime-based adjustment
        if self.config.regime_position_adjustment:
            if regime == 3:  # Range markets performed well
                regime_multiplier = self.config.range_market_multiplier
            else:
                regime_multiplier = self.config.trend_market_multiplier
        else:
            regime_multiplier = 1.0
        
        # Calculate risk amount
        risk_amount = self.current_capital * base_risk * regime_multiplier
        risk_amount = min(risk_amount, self.current_capital * self.config.max_position_risk)
        
        # Calculate position size based on stop loss distance
        sl_distance_pips = atr * self.config.sl_atr_multiplier
        pip_value = self.config.pip_value_per_million
        
        position_size = (risk_amount / (sl_distance_pips * pip_value)) * self.config.min_lot_size
        
        # Apply position size cap for realistic trading
        if self.config.realistic_sizing:
            max_size = self.config.position_size_cap * self.config.min_lot_size
            position_size = min(position_size, max_size)
        
        position_size = max(self.config.min_lot_size, position_size)
        
        # Calculate stop loss and take profits
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - (sl_distance_pips * FOREX_PIP_SIZE)
        else:
            stop_loss = entry_price + (sl_distance_pips * FOREX_PIP_SIZE)
        
        # Enhanced take profit calculation
        take_profits = self._calculate_optimized_take_profits(
            entry_price, direction, atr, row, regime
        )
        
        # Create trade
        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profits=take_profits,
            is_relaxed=is_relaxed
        )
        
        # Track regime trades
        self.regime_trades[regime] += 1
        
        return trade
    
    def _calculate_optimized_take_profits(self, entry_price: float, direction: TradeDirection,
                                        atr: float, row: pd.Series, regime: int) -> List[float]:
        """Calculate optimized take profit levels for better risk-reward"""
        
        # Base TP distances
        tp_distances = [atr * mult for mult in self.config.tp_atr_multipliers]
        
        # Regime-based adjustment
        if regime == 1:  # Strong trend - wider TPs
            tp_distances = [d * 1.2 for d in tp_distances]
        elif regime == 3:  # Range - tighter TPs
            tp_distances = [d * 0.8 for d in tp_distances]
        
        if direction == TradeDirection.LONG:
            take_profits = [entry_price + (d * FOREX_PIP_SIZE) for d in tp_distances]
        else:
            take_profits = [entry_price - (d * FOREX_PIP_SIZE) for d in tp_distances]
        
        return take_profits
    
    def run_optimized_backtest(self, df: pd.DataFrame) -> Dict:
        """Run optimized backtest with enhanced logic"""
        # Use the parent's backtest logic but with our optimized components
        results = self.run_backtest(df)
        
        # Add optimization-specific metrics
        results['total_signal_flips'] = self.total_signal_flips
        results['regime_distribution'] = self.regime_trades
        
        if self.trades:
            # Calculate actual risk-reward achieved
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]
            
            if winning_trades and losing_trades:
                avg_win = np.mean([t.pnl for t in winning_trades])
                avg_loss = abs(np.mean([t.pnl for t in losing_trades]))
                results['actual_risk_reward'] = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            # Calculate trade frequency (trades per 100 periods)
            results['trade_frequency'] = len(self.trades) / len(df) * 100
            
            # Calculate signal flip effectiveness
            signal_flip_trades = [t for t in self.trades if t.exit_reason == ExitReason.SIGNAL_FLIP]
            if signal_flip_trades:
                avg_flip_pnl = np.mean([t.pnl for t in signal_flip_trades])
                results['signal_flip_avg_pnl'] = avg_flip_pnl
                results['signal_flip_count'] = len(signal_flip_trades)
        
        return results

class OptimizedSignalGenerator:
    """Optimized signal generator for high-frequency approach"""
    
    def __init__(self, config: OptimizedSharpeConfig):
        self.config = config
    
    def check_entry_conditions(self, row: pd.Series) -> Optional[Tuple[TradeDirection, bool]]:
        """Enhanced entry conditions for higher frequency"""
        
        # Aggressive entry mode - either NeuroTrend OR MarketBias
        if self.config.aggressive_entry_mode:
            # Long signals
            aggressive_long = (
                (row['NTI_Direction'] == 1) or 
                (row['MB_Bias'] == 1 and row['IC_Regime'] in [1, 2])
            )
            
            # Short signals
            aggressive_short = (
                (row['NTI_Direction'] == -1) or 
                (row['MB_Bias'] == -1 and row['IC_Regime'] in [1, 2])
            )
            
            if aggressive_long:
                return (TradeDirection.LONG, False)
            elif aggressive_short:
                return (TradeDirection.SHORT, False)
        
        # Standard entry: all indicators must align (more conservative)
        standard_long = (row['NTI_Direction'] == 1 and 
                        row['MB_Bias'] == 1 and 
                        row['IC_Regime'] in [1, 2])
        
        standard_short = (row['NTI_Direction'] == -1 and 
                         row['MB_Bias'] == -1 and 
                         row['IC_Regime'] in [1, 2])
        
        if standard_long:
            return (TradeDirection.LONG, False)
        elif standard_short:
            return (TradeDirection.SHORT, False)
        
        # Relaxed mode: NeuroTrend alone with any regime
        if self.config.relaxed_mode:
            if row['NTI_Direction'] == 1:
                return (TradeDirection.LONG, True)
            elif row['NTI_Direction'] == -1:
                return (TradeDirection.SHORT, True)
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, trade: Trade, 
                            current_time: pd.Timestamp) -> Tuple[bool, ExitReason, float]:
        """Enhanced exit conditions with optimized signal flip logic"""
        
        # Check standard exit conditions first
        current_price = row['Close']
        
        # Take profit checks
        for i, tp in enumerate(trade.take_profits):
            if trade.direction == TradeDirection.LONG and current_price >= tp:
                return True, ExitReason(f"take_profit_{i+1}"), 1.0 if i == 2 else 0.33
            elif trade.direction == TradeDirection.SHORT and current_price <= tp:
                return True, ExitReason(f"take_profit_{i+1}"), 1.0 if i == 2 else 0.33
        
        # Stop loss check
        if ((trade.direction == TradeDirection.LONG and current_price <= trade.stop_loss) or
            (trade.direction == TradeDirection.SHORT and current_price >= trade.stop_loss)):
            return True, ExitReason.STOP_LOSS, 1.0
        
        # Trailing stop check
        if (trade.trailing_stop is not None and
            ((trade.direction == TradeDirection.LONG and current_price <= trade.trailing_stop) or
             (trade.direction == TradeDirection.SHORT and current_price >= trade.trailing_stop))):
            return True, ExitReason.TRAILING_STOP, 1.0
        
        # Enhanced signal flip check
        if self.config.exit_on_signal_flip:
            should_flip, exit_percent = self.check_optimized_signal_flip(row, trade, current_time)
            if should_flip:
                return True, ExitReason.SIGNAL_FLIP, exit_percent
        
        return False, None, 0.0
    
    def check_optimized_signal_flip(self, row: pd.Series, trade: Trade, 
                                  current_time: pd.Timestamp) -> Tuple[bool, float]:
        """Optimized signal flip detection"""
        
        # Check time filter (reduced from original)
        hours_in_trade = (current_time - trade.entry_time).total_seconds() / 3600
        if hours_in_trade < self.config.signal_flip_min_time_hours:
            return False, 0.0
        
        # Calculate current profit
        current_price = row['Close']
        if trade.direction == TradeDirection.LONG:
            profit_pips = (current_price - trade.entry_price) / FOREX_PIP_SIZE
        else:
            profit_pips = (trade.entry_price - current_price) / FOREX_PIP_SIZE
        
        # Check minimum profit requirement (reduced)
        if profit_pips < self.config.signal_flip_min_profit_pips:
            return False, 0.0
        
        # Check for signal flip (more sensitive)
        signal_flipped = False
        if trade.direction == TradeDirection.LONG:
            # Exit long if any bearish signal
            if row['NTI_Direction'] == -1 or row['MB_Bias'] == -1:
                signal_flipped = True
        else:
            # Exit short if any bullish signal
            if row['NTI_Direction'] == 1 or row['MB_Bias'] == 1:
                signal_flipped = True
        
        if not signal_flipped:
            return False, 0.0
        
        # More aggressive exit percentage
        return True, self.config.signal_flip_partial_exit_percent
    
    def check_partial_profit_conditions(self, row: pd.Series, trade: Trade) -> bool:
        """Check partial profit conditions (keep original logic)"""
        return False  # Disable for this optimization

def create_optimized_configs() -> Dict[str, OptimizedSharpeConfig]:
    """Create optimized configurations for different approaches"""
    
    configs = {
        'Balanced': OptimizedSharpeConfig(
            aggressive_entry_mode=True,
            relaxed_mode=True,
            risk_per_trade=0.015,
            signal_flip_min_profit_pips=3.0,
            realistic_sizing=True
        ),
        
        'High_Frequency': OptimizedSharpeConfig(
            aggressive_entry_mode=True,
            relaxed_mode=True,
            risk_per_trade=0.012,
            signal_flip_min_profit_pips=2.0,
            signal_flip_min_time_hours=0.5,
            realistic_sizing=True,
            tp_atr_multipliers=(1.0, 2.0, 3.0)
        ),
        
        'Conservative': OptimizedSharpeConfig(
            aggressive_entry_mode=False,
            relaxed_mode=False,
            risk_per_trade=0.01,
            signal_flip_min_profit_pips=5.0,
            realistic_sizing=True,
            tp_atr_multipliers=(1.5, 3.0, 4.5)
        )
    }
    
    return configs

def test_optimized_strategies():
    """Test the optimized strategies"""
    
    print("Testing Optimized Sharpe > 2.0 Strategies")
    print("="*50)
    
    # Create test data (reuse from previous)
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', periods=2000, freq='1H')
    returns = np.random.normal(0, 0.001, 2000)
    prices = np.cumprod(1 + returns) * 0.75
    
    df = pd.DataFrame({
        'Open': prices + np.random.normal(0, 0.0001, 2000),
        'High': prices + abs(np.random.normal(0, 0.0002, 2000)),
        'Low': prices - abs(np.random.normal(0, 0.0002, 2000)),
        'Close': prices
    }, index=dates)
    
    # Add indicators
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['NTI_Direction'] = np.where(df['SMA_20'] > df['SMA_50'] * 1.001, 1, 
                                  np.where(df['SMA_20'] < df['SMA_50'] * 0.999, -1, 0))
    df['ROC_10'] = df['Close'].pct_change(10)
    df['MB_Bias'] = np.where(df['ROC_10'] > 0.002, 1, 
                            np.where(df['ROC_10'] < -0.002, -1, 0))
    df['IC_Regime'] = np.random.choice([1, 2, 3], size=len(df), p=[0.3, 0.5, 0.2])
    df['IC_ATR_Normalized'] = np.random.uniform(20, 60, len(df))
    df['IC_RegimeName'] = df['IC_Regime'].map({1: 'Strong Trend', 2: 'Weak Trend', 3: 'Range'})
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Test configurations
    configs = create_optimized_configs()
    
    best_sharpe = 0
    best_config = None
    
    for config_name, config in configs.items():
        print(f"\\nTesting {config_name}...")
        
        strategy = OptimizedSharpeStrategy(config)
        results = strategy.run_optimized_backtest(df)
        
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Total Return: {results['total_return']:.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")
        
        if 'actual_risk_reward' in results:
            print(f"  Risk-Reward: {results['actual_risk_reward']:.2f}")
        if 'trade_frequency' in results:
            print(f"  Trade Frequency: {results['trade_frequency']:.1f} per 100 periods")
        
        if results['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['sharpe_ratio']
            best_config = config_name
    
    print(f"\\n{'='*50}")
    print(f"BEST CONFIGURATION: {best_config}")
    print(f"BEST SHARPE RATIO: {best_sharpe:.3f}")
    
    if best_sharpe >= 2.0:
        print("🎯 SUCCESS: Target Sharpe ≥ 2.0 achieved!")
    else:
        print(f"📈 Progress: {best_sharpe:.3f}/2.0 ({(best_sharpe/2.0*100):.1f}% of target)")
    
    return best_config, best_sharpe

if __name__ == "__main__":
    test_optimized_strategies()