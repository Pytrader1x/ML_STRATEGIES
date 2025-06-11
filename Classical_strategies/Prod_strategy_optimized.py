"""
Optimized Production Strategy Implementation
Incorporates improvements based on exit analysis insights

Key Optimizations:
1. Enhanced signal flip logic with profit threshold and time filter
2. Dynamic stop loss placement based on market conditions
3. Adaptive take profit levels based on market regime
4. Partial exit capabilities on signal flips

Author: Trading System
Date: 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

# Import base classes from original
from Prod_strategy import (
    TradeDirection, ExitReason, ConfidenceLevel, Trade, PartialExit,
    FOREX_PIP_SIZE, MIN_LOT_SIZE, PIP_VALUE_PER_MILLION,
    RiskManager, PnLCalculator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Configuration
# ============================================================================

@dataclass
class OptimizedStrategyConfig:
    """Enhanced strategy configuration with optimization parameters"""
    
    # Capital and risk management
    initial_capital: float = 100_000
    risk_per_trade: float = 0.02
    
    # Take profit configuration
    tp_atr_multipliers: Tuple[float, float, float] = (0.8, 1.5, 2.5)
    max_tp_percent: float = 0.01
    
    # Dynamic TP adjustments
    tp_range_market_multiplier: float = 0.7  # Tighter TPs in ranging markets
    tp_trend_market_multiplier: float = 1.0  # Normal TPs in trending markets
    tp_chop_market_multiplier: float = 0.5  # Very tight TPs in choppy markets
    
    # Stop loss configuration
    sl_atr_multiplier: float = 2.0
    sl_max_pips: float = 45.0  # Maximum stop loss in pips
    trailing_atr_multiplier: float = 1.2
    tsl_activation_pips: float = 15  # TSL activates after 15 pips in profit
    tsl_min_profit_pips: float = 5   # Guarantees minimum 5 pip profit once TSL activated
    
    # Dynamic SL adjustments
    sl_range_market_multiplier: float = 0.8  # Tighter stops in ranging markets
    sl_trend_market_multiplier: float = 1.0  # Normal stops in trending markets
    sl_volatility_adjustment: bool = True
    
    # Enhanced exit strategies
    exit_on_signal_flip: bool = True
    signal_flip_min_profit_pips: float = 5.0  # Minimum profit before allowing flip exit
    signal_flip_min_time_hours: float = 2.0  # Minimum time before allowing flip exit
    signal_flip_partial_exit_percent: float = 0.5  # Exit 50% on signal flip
    signal_flip_momentum_threshold: float = 0.7  # Require strong momentum for full exit
    
    # Partial profit taking
    partial_profit_before_sl: bool = True
    partial_profit_sl_distance_ratio: float = 0.7  # Take partial at 70% of SL distance
    partial_profit_size_percent: float = 0.33  # Take 33% as partial profit
    
    # Relaxed mode settings
    relaxed_mode: bool = False
    relaxed_tp_multiplier: float = 0.5
    relaxed_position_multiplier: float = 0.5
    relaxed_tsl_activation_pips: float = 8
    
    # Intelligent sizing
    intelligent_sizing: bool = True
    confidence_thresholds: Tuple[float, float, float] = (30.0, 50.0, 70.0)
    size_multipliers: Tuple[float, float, float, float] = (1.0, 1.0, 3.0, 5.0)
    tp_confidence_adjustment: bool = True
    
    # Trading constraints
    min_lot_size: float = MIN_LOT_SIZE
    pip_value_per_million: float = PIP_VALUE_PER_MILLION
    
    # Logging
    verbose: bool = False


# ============================================================================
# Enhanced Components
# ============================================================================

class OptimizedTakeProfitCalculator:
    """Enhanced TP calculator with market regime adaptation"""
    
    def __init__(self, config: OptimizedStrategyConfig):
        self.config = config
    
    def get_market_regime_multiplier(self, row: pd.Series) -> float:
        """Get TP multiplier based on market regime"""
        regime = row.get('IC_Regime', 0)
        regime_name = row.get('IC_RegimeName', '')
        
        # Adjust based on Intelligent Chop regime
        if regime in [3, 4] or 'Chop' in regime_name or 'Range' in regime_name:
            # Choppy/ranging market - use tighter TPs
            return self.config.tp_chop_market_multiplier
        elif regime == 2 or 'Weak' in regime_name:
            # Weak trend - use slightly tighter TPs
            return self.config.tp_range_market_multiplier
        else:
            # Strong trend - use normal TPs
            return self.config.tp_trend_market_multiplier
    
    def calculate_take_profits(self, entry_price: float, direction: TradeDirection,
                             atr: float, row: pd.Series, is_relaxed: bool = False,
                             confidence: float = 50.0) -> List[float]:
        """Calculate adaptive take profit levels"""
        
        # Get base multiplier
        tp_multiplier = self.config.relaxed_tp_multiplier if is_relaxed else 1.0
        
        # Apply market regime adjustment
        regime_multiplier = self.get_market_regime_multiplier(row)
        tp_multiplier *= regime_multiplier
        
        # Apply confidence-based adjustment if enabled
        if self.config.intelligent_sizing and self.config.tp_confidence_adjustment:
            _, confidence_tp_mult = RiskManager(self.config).get_position_size_multiplier(confidence)
            tp_multiplier *= confidence_tp_mult
        
        # Calculate TP distances with dynamic adjustments
        tp_distances = []
        max_distances = [0.003, 0.006, 0.01]  # Base max distances
        
        # Further tighten in choppy markets
        if regime_multiplier < 1.0:
            max_distances = [d * regime_multiplier for d in max_distances]
        
        for i, (atr_mult, max_dist) in enumerate(zip(self.config.tp_atr_multipliers, max_distances)):
            distance = atr * atr_mult * tp_multiplier
            distance = min(distance, entry_price * max_dist)
            tp_distances.append(distance)
        
        # Calculate actual TP levels
        if direction == TradeDirection.LONG:
            take_profits = [entry_price + dist for dist in tp_distances]
        else:
            take_profits = [entry_price - dist for dist in tp_distances]
        
        return take_profits


class OptimizedStopLossCalculator:
    """Enhanced SL calculator with dynamic adjustments"""
    
    def __init__(self, config: OptimizedStrategyConfig):
        self.config = config
    
    def get_volatility_adjustment(self, atr: float, recent_atr: float) -> float:
        """Adjust stop loss based on current vs recent volatility"""
        if not self.config.sl_volatility_adjustment:
            return 1.0
        
        # If current volatility is higher than recent average, widen stops
        if atr > recent_atr * 1.2:
            return 1.2
        # If current volatility is lower, tighten stops
        elif atr < recent_atr * 0.8:
            return 0.9
        return 1.0
    
    def calculate_initial_stop_loss(self, entry_price: float, direction: TradeDirection,
                                  atr: float, market_bias_data: Dict, row: pd.Series) -> float:
        """Calculate dynamic stop loss based on market conditions with maximum pip limit"""
        
        # Get base SL distance
        sl_distance = atr * self.config.sl_atr_multiplier
        
        # Apply market regime adjustment
        regime = row.get('IC_Regime', 0)
        if regime in [3, 4]:  # Ranging/choppy market
            sl_distance *= self.config.sl_range_market_multiplier
        else:  # Trending market
            sl_distance *= self.config.sl_trend_market_multiplier
        
        # Apply volatility adjustment if available
        if 'IC_ATR_MA' in row:
            vol_adj = self.get_volatility_adjustment(atr, row['IC_ATR_MA'])
            sl_distance *= vol_adj
        
        # Apply maximum pip limit
        max_sl_distance = self.config.sl_max_pips * FOREX_PIP_SIZE
        sl_distance = min(sl_distance, max_sl_distance)
        
        # Calculate stop loss levels
        if direction == TradeDirection.LONG:
            atr_stop = entry_price - sl_distance
            
            # Use Market Bias bar if available
            mb_low = market_bias_data.get('MB_l2')
            if mb_low and not pd.isna(mb_low):
                mb_stop = mb_low - 0.00005  # 0.5 pip buffer
                # Also apply max pip limit to MB stop
                mb_stop = max(mb_stop, entry_price - max_sl_distance)
                stop_loss = max(atr_stop, mb_stop)
            else:
                stop_loss = atr_stop
        else:
            atr_stop = entry_price + sl_distance
            
            # Use Market Bias bar if available
            mb_high = market_bias_data.get('MB_h2')
            if mb_high and not pd.isna(mb_high):
                mb_stop = mb_high + 0.00005  # 0.5 pip buffer
                # Also apply max pip limit to MB stop
                mb_stop = min(mb_stop, entry_price + max_sl_distance)
                stop_loss = min(atr_stop, mb_stop)
            else:
                stop_loss = atr_stop
        
        return stop_loss


class OptimizedSignalGenerator:
    """Enhanced signal generator with improved exit logic"""
    
    def __init__(self, config: OptimizedStrategyConfig):
        self.config = config
    
    def check_entry_conditions(self, row: pd.Series) -> Optional[Tuple[TradeDirection, bool]]:
        """Check if entry conditions are met (same as original)"""
        # Standard entry: all three indicators must align
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
        
        # Relaxed mode: NeuroTrend alone
        if self.config.relaxed_mode:
            if row['NTI_Direction'] == 1:
                return (TradeDirection.LONG, True)
            elif row['NTI_Direction'] == -1:
                return (TradeDirection.SHORT, True)
        
        return None
    
    def check_signal_flip_conditions(self, row: pd.Series, trade: Trade, 
                                   current_time: pd.Timestamp) -> Tuple[bool, float]:
        """Enhanced signal flip detection with filters"""
        
        # Check time filter
        hours_in_trade = (current_time - trade.entry_time).total_seconds() / 3600
        if hours_in_trade < self.config.signal_flip_min_time_hours:
            return False, 0.0
        
        # Calculate current profit in pips
        current_price = row['Close']
        if trade.direction == TradeDirection.LONG:
            profit_pips = (current_price - trade.entry_price) / FOREX_PIP_SIZE
        else:
            profit_pips = (trade.entry_price - current_price) / FOREX_PIP_SIZE
        
        # Check minimum profit requirement
        if profit_pips < self.config.signal_flip_min_profit_pips:
            return False, 0.0
        
        # Check for signal flip
        signal_flipped = False
        if trade.direction == TradeDirection.LONG:
            if row['NTI_Direction'] == -1 or row['MB_Bias'] == -1:
                signal_flipped = True
        else:
            if row['NTI_Direction'] == 1 or row['MB_Bias'] == 1:
                signal_flipped = True
        
        if not signal_flipped:
            return False, 0.0
        
        # Check momentum for full vs partial exit
        momentum_strength = abs(row.get('NTI_Strength', 0.5))
        if momentum_strength >= self.config.signal_flip_momentum_threshold:
            # Strong momentum - full exit
            exit_percent = 1.0
        else:
            # Weak momentum - partial exit
            exit_percent = self.config.signal_flip_partial_exit_percent
        
        return True, exit_percent
    
    def check_partial_profit_conditions(self, row: pd.Series, trade: Trade) -> bool:
        """Check if we should take partial profit before stop loss"""
        if not self.config.partial_profit_before_sl or trade.tp_hits > 0:
            return False
        
        current_price = row['Close']
        
        # Calculate distance to stop loss
        sl_distance = abs(trade.entry_price - trade.stop_loss)
        target_distance = sl_distance * self.config.partial_profit_sl_distance_ratio
        
        # Check if we've reached the partial profit level
        if trade.direction == TradeDirection.LONG:
            return current_price >= trade.entry_price + target_distance
        else:
            return current_price <= trade.entry_price - target_distance
    
    def check_exit_conditions(self, row: pd.Series, trade: Trade, 
                            current_time: pd.Timestamp) -> Tuple[bool, Optional[ExitReason], float]:
        """Enhanced exit condition checking"""
        current_price = row['Close']
        
        # Check take profit levels
        if trade.direction == TradeDirection.LONG:
            # Check for TP1 pullback
            if trade.tp_hits >= 2 and row['Low'] <= trade.take_profits[0]:
                return True, ExitReason.TP1_PULLBACK, 1.0
            
            # Check normal TPs
            for i, tp in enumerate(trade.take_profits):
                if i >= trade.tp_hits and row['High'] >= tp:
                    return True, ExitReason(f'take_profit_{i+1}'), 0.33
        else:
            # Check for TP1 pullback
            if trade.tp_hits >= 2 and row['High'] >= trade.take_profits[0]:
                return True, ExitReason.TP1_PULLBACK, 1.0
            
            # Check normal TPs
            for i, tp in enumerate(trade.take_profits):
                if i >= trade.tp_hits and row['Low'] <= tp:
                    return True, ExitReason(f'take_profit_{i+1}'), 0.33
        
        # Check stop loss
        current_stop = trade.trailing_stop if trade.trailing_stop is not None else trade.stop_loss
        
        if trade.direction == TradeDirection.LONG and current_price <= current_stop:
            exit_reason = ExitReason.TRAILING_STOP if trade.trailing_stop is not None else ExitReason.STOP_LOSS
            return True, exit_reason, 1.0
        elif trade.direction == TradeDirection.SHORT and current_price >= current_stop:
            exit_reason = ExitReason.TRAILING_STOP if trade.trailing_stop is not None else ExitReason.STOP_LOSS
            return True, exit_reason, 1.0
        
        # Check enhanced signal flip conditions
        if self.config.exit_on_signal_flip:
            flip_exit, exit_percent = self.check_signal_flip_conditions(row, trade, current_time)
            if flip_exit:
                return True, ExitReason.SIGNAL_FLIP, exit_percent
        
        return False, None, 0.0


# ============================================================================
# Optimized Strategy Class
# ============================================================================

class OptimizedProdStrategy:
    """Production strategy with optimizations based on performance analysis"""
    
    def __init__(self, config: Optional[OptimizedStrategyConfig] = None):
        """Initialize optimized strategy"""
        self.config = config or OptimizedStrategyConfig()
        
        # Initialize components
        self.risk_manager = RiskManager(self.config)
        self.tp_calculator = OptimizedTakeProfitCalculator(self.config)
        self.sl_calculator = OptimizedStopLossCalculator(self.config)
        self.signal_generator = OptimizedSignalGenerator(self.config)
        self.pnl_calculator = PnLCalculator(self.config)
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset strategy state"""
        self.current_capital = self.config.initial_capital
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.equity_curve = [self.config.initial_capital]
        self.partial_profit_taken = {}  # Track partial profits per trade
    
    def _execute_partial_exit(self, trade: Trade, exit_time: pd.Timestamp,
                             exit_price: float, exit_percent: float, 
                             exit_reason: Optional[ExitReason] = None) -> Optional[Trade]:
        """Execute a partial exit with custom percentage"""
        # Calculate exit size
        exit_size = trade.position_size * exit_percent
        trade.remaining_size -= exit_size
        
        # Calculate P&L
        partial_pnl, _ = self.pnl_calculator.calculate_pnl(
            trade.entry_price, exit_price, exit_size, trade.direction
        )
        
        # Update trade state
        trade.partial_pnl += partial_pnl
        
        # Record partial exit
        trade.partial_exits.append(PartialExit(
            time=exit_time,
            price=exit_price,
            size=exit_size,
            tp_level=0,  # Not a TP exit
            pnl=partial_pnl
        ))
        
        # Update capital
        self.current_capital += partial_pnl
        
        if self.config.verbose:
            logger.info(f"Partial exit ({exit_percent*100:.0f}%) at {exit_price:.5f}, P&L: ${partial_pnl:.2f}")
        
        # Check if trade is complete
        if trade.remaining_size <= 0:
            trade.remaining_size = 0
            trade.pnl = trade.partial_pnl
            trade.exit_time = exit_time
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason or ExitReason.SIGNAL_FLIP
            return trade
        
        return None  # Trade continues
    
    def _create_new_trade(self, entry_time: pd.Timestamp, row: pd.Series,
                         direction: TradeDirection, is_relaxed: bool) -> Trade:
        """Create a new trade with optimized parameters"""
        entry_price = row['Close']
        atr = row['IC_ATR_Normalized']
        confidence = row.get('NTI_Confidence', 50.0)
        
        # Calculate trade parameters
        market_bias_data = {
            'MB_l2': row.get('MB_l2'),
            'MB_h2': row.get('MB_h2')
        }
        
        # Dynamic stop loss
        stop_loss = self.sl_calculator.calculate_initial_stop_loss(
            entry_price, direction, atr, market_bias_data, row
        )
        
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss, self.current_capital, is_relaxed, confidence
        )
        
        # Dynamic take profits
        take_profits = self.tp_calculator.calculate_take_profits(
            entry_price, direction, atr, row, is_relaxed, confidence
        )
        
        return Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profits=take_profits,
            is_relaxed=is_relaxed,
            confidence=confidence
        )
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the optimized backtest"""
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'NTI_Direction',
                        'MB_Bias', 'IC_Regime', 'IC_ATR_Normalized', 'IC_RegimeName']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Reset state
        self.reset()
        
        # Main backtest loop
        for idx in range(1, len(df)):
            current_row = df.iloc[idx]
            current_time = df.index[idx]
            
            # Update equity curve
            self.equity_curve.append(self.current_capital)
            
            # Process open trade
            if self.current_trade is not None:
                # Check for partial profit taking
                if self.signal_generator.check_partial_profit_conditions(current_row, self.current_trade):
                    trade_id = id(self.current_trade)
                    if trade_id not in self.partial_profit_taken:
                        # Take partial profit
                        exit_price = current_row['Close']
                        completed = self._execute_partial_exit(
                            self.current_trade, current_time, exit_price, 
                            self.config.partial_profit_size_percent
                        )
                        self.partial_profit_taken[trade_id] = True
                        
                        if completed:
                            self.trades.append(self.current_trade)
                            self.current_trade = None
                            continue
                
                # Update trailing stop
                from Prod_strategy import StopLossCalculator
                base_sl_calc = StopLossCalculator(self.config)
                atr = current_row['IC_ATR_Normalized']
                new_trailing_stop = base_sl_calc.update_trailing_stop(
                    current_row['Close'], self.current_trade, atr
                )
                if new_trailing_stop is not None:
                    self.current_trade.trailing_stop = new_trailing_stop
                
                # Check exit conditions
                should_exit, exit_reason, exit_percent = self.signal_generator.check_exit_conditions(
                    current_row, self.current_trade, current_time
                )
                
                if should_exit:
                    # Determine exit price
                    exit_price = self._get_exit_price(current_row, self.current_trade, exit_reason)
                    
                    # Execute exit (partial or full)
                    if exit_percent < 1.0:
                        completed_trade = self._execute_partial_exit(
                            self.current_trade, current_time, exit_price, 
                            exit_percent, exit_reason
                        )
                    else:
                        # Full exit
                        completed_trade = self._execute_full_exit(
                            self.current_trade, current_time, exit_price, exit_reason
                        )
                    
                    if completed_trade is not None:
                        self.trades.append(self.current_trade)
                        self.current_trade = None
            
            # Check for new entry
            elif self.current_trade is None:
                entry_result = self.signal_generator.check_entry_conditions(current_row)
                
                if entry_result is not None:
                    direction, is_relaxed = entry_result
                    
                    # Create new trade
                    self.current_trade = self._create_new_trade(
                        current_time, current_row, direction, is_relaxed
                    )
                    
                    if self.config.verbose:
                        size_millions = self.current_trade.position_size / self.config.min_lot_size
                        trade_type = "RELAXED" if is_relaxed else "STANDARD"
                        logger.info(f"{trade_type} TRADE: {direction.value} at {self.current_trade.entry_price:.5f} "
                                   f"with {size_millions:.0f}M")
        
        # Close any remaining trade
        if self.current_trade is not None:
            last_row = df.iloc[-1]
            last_time = df.index[-1]
            
            completed_trade = self._execute_full_exit(
                self.current_trade, last_time, last_row['Close'], ExitReason.END_OF_DATA
            )
            if completed_trade:
                self.trades.append(self.current_trade)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics()
    
    def _get_exit_price(self, row: pd.Series, trade: Trade, exit_reason: ExitReason) -> float:
        """Determine exit price (same as original)"""
        if 'take_profit' in exit_reason.value:
            tp_index = int(exit_reason.value.split('_')[-1]) - 1
            return trade.take_profits[tp_index]
        elif exit_reason == ExitReason.TP1_PULLBACK:
            return trade.take_profits[0]
        elif exit_reason in [ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP]:
            if exit_reason == ExitReason.TRAILING_STOP and trade.trailing_stop is not None:
                return trade.trailing_stop
            else:
                return trade.stop_loss
        else:
            return row['Close']
    
    def _execute_full_exit(self, trade: Trade, exit_time: pd.Timestamp,
                          exit_price: float, exit_reason: ExitReason) -> Trade:
        """Execute a full trade exit (similar to original)"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Handle special exit types
        if 'take_profit' in exit_reason.value:
            tp_index = int(exit_reason.value.split('_')[-1]) - 1
            # Use original partial exit logic for TP exits
            exit_size = trade.position_size / 3
            trade.remaining_size -= exit_size
            
            partial_pnl, _ = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, exit_size, trade.direction
            )
            
            trade.partial_pnl += partial_pnl
            trade.tp_hits = tp_index + 1
            self.current_capital += partial_pnl
            
            if trade.tp_hits >= 3:
                trade.remaining_size = 0
                trade.pnl = trade.partial_pnl
                return trade
            
            return None  # Continue with partial position
        
        elif exit_reason == ExitReason.TP1_PULLBACK:
            exit_price = trade.take_profits[0]
        
        # Calculate final P&L
        if trade.remaining_size > 0:
            remaining_pnl, _ = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, trade.remaining_size, trade.direction
            )
            trade.pnl = trade.partial_pnl + remaining_pnl
            self.current_capital += remaining_pnl
        
        trade.remaining_size = 0
        
        # Calculate percentage return
        original_value = trade.entry_price * trade.position_size
        trade.pnl_percent = (trade.pnl / original_value) * 100
        
        return trade
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics (same as original)"""
        if not self.trades:
            return self._empty_metrics()
        
        # Basic metrics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.current_capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        # Win/Loss metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Maximum drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason.value if trade.exit_reason else 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'exit_reasons': exit_reasons,
            'final_capital': self.current_capital,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no trades"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'exit_reasons': {},
            'final_capital': self.current_capital,
            'trades': [],
            'equity_curve': self.equity_curve
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_optimized_strategy(**kwargs) -> OptimizedProdStrategy:
    """Create an optimized strategy instance"""
    config = OptimizedStrategyConfig(**kwargs)
    return OptimizedProdStrategy(config)


if __name__ == "__main__":
    logger.info("Optimized Production Strategy Module Loaded")