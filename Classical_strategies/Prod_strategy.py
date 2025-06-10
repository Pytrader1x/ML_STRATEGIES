"""
Production-ready implementation of Strategy 1
A quantitative trading strategy combining NeuroTrend, Market Bias, and Intelligent Chop indicators

Author: Trading System
Date: 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class TradeDirection(Enum):
    """Trade direction enumeration"""
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """Exit reason enumeration"""
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    TP1_PULLBACK = "tp1_pullback"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_FLIP = "signal_flip"
    END_OF_DATA = "end_of_data"


class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Trading constants
FOREX_PIP_SIZE = 0.0001
MIN_LOT_SIZE = 1_000_000  # 1M units
PIP_VALUE_PER_MILLION = 100  # $100 per pip per million


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PartialExit:
    """Represents a partial exit from a trade"""
    time: pd.Timestamp
    price: float
    size: float
    tp_level: int
    pnl: float


@dataclass
class Trade:
    """Comprehensive trade information"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: TradeDirection
    position_size: float
    stop_loss: float
    take_profits: List[float]
    is_relaxed: bool = False
    confidence: float = 50.0
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    trailing_stop: Optional[float] = None
    tp_hits: int = 0
    remaining_size: float = None
    partial_pnl: float = 0.0
    tp_exit_times: List[pd.Timestamp] = field(default_factory=list)
    tp_exit_prices: List[float] = field(default_factory=list)
    partial_exits: List[PartialExit] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize remaining size if not provided"""
        if self.remaining_size is None:
            self.remaining_size = self.position_size
        # Convert string direction to enum if needed
        if isinstance(self.direction, str):
            self.direction = TradeDirection(self.direction)


@dataclass
class StrategyConfig:
    """Strategy configuration parameters"""
    # Capital and risk management
    initial_capital: float = 100_000
    risk_per_trade: float = 0.02
    
    # Take profit configuration
    tp_atr_multipliers: Tuple[float, float, float] = (0.8, 1.5, 2.5)
    max_tp_percent: float = 0.01
    
    # Stop loss configuration
    sl_atr_multiplier: float = 2.0
    trailing_atr_multiplier: float = 1.2
    tsl_activation_pips: float = 15
    tsl_min_profit_pips: float = 5
    
    # Exit strategies
    exit_on_signal_flip: bool = True
    
    # Relaxed mode settings
    relaxed_mode: bool = False
    relaxed_tp_multiplier: float = 0.5
    relaxed_position_multiplier: float = 0.5
    relaxed_tsl_activation_pips: float = 8
    
    # Intelligent sizing
    intelligent_sizing: bool = False
    confidence_thresholds: Tuple[float, float, float] = (30.0, 50.0, 70.0)
    size_multipliers: Tuple[float, float, float, float] = (1.0, 1.0, 3.0, 5.0)
    tp_confidence_adjustment: bool = True
    
    # Trading constraints
    min_lot_size: float = MIN_LOT_SIZE
    pip_value_per_million: float = PIP_VALUE_PER_MILLION
    
    # Logging
    verbose: bool = False


# ============================================================================
# Strategy Components
# ============================================================================

class RiskManager:
    """Handles position sizing and risk calculations"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level from numeric confidence"""
        if confidence < self.config.confidence_thresholds[0]:
            return ConfidenceLevel.VERY_LOW
        elif confidence < self.config.confidence_thresholds[1]:
            return ConfidenceLevel.LOW
        elif confidence < self.config.confidence_thresholds[2]:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH
    
    def get_position_size_multiplier(self, confidence: float) -> Tuple[float, float]:
        """Get position size and TP multipliers based on confidence"""
        if not self.config.intelligent_sizing:
            return 1.0, 1.0
        
        level = self.get_confidence_level(confidence)
        
        multiplier_map = {
            ConfidenceLevel.VERY_LOW: (self.config.size_multipliers[0], 0.7),
            ConfidenceLevel.LOW: (self.config.size_multipliers[1], 0.85),
            ConfidenceLevel.MEDIUM: (self.config.size_multipliers[2], 1.0),
            ConfidenceLevel.HIGH: (self.config.size_multipliers[3], 1.0)
        }
        
        size_mult, tp_mult = multiplier_map[level]
        
        if not self.config.tp_confidence_adjustment:
            tp_mult = 1.0
            
        return size_mult, tp_mult
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              current_capital: float, is_relaxed: bool = False,
                              confidence: float = 50.0) -> float:
        """Calculate position size based on risk management rules"""
        # Default position size is 1M
        position_size_millions = 1.0
        
        # Apply intelligent sizing based on confidence
        if self.config.intelligent_sizing:
            size_millions, _ = self.get_position_size_multiplier(confidence)
            position_size_millions = size_millions
        
        # Apply relaxed mode position reduction
        if is_relaxed:
            position_size_millions *= self.config.relaxed_position_multiplier
            position_size_millions = max(1.0, round(position_size_millions))
        
        # Convert to units
        position_size = position_size_millions * self.config.min_lot_size
        
        # Check capital constraints
        required_margin = position_size * 0.01  # 1% margin requirement
        if required_margin > current_capital:
            affordable_millions = int(current_capital / (self.config.min_lot_size * 0.01))
            position_size = max(self.config.min_lot_size, affordable_millions * self.config.min_lot_size)
        
        return position_size


class TakeProfitCalculator:
    """Handles take profit level calculations"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def calculate_take_profits(self, entry_price: float, direction: TradeDirection,
                             atr: float, is_relaxed: bool = False,
                             confidence: float = 50.0) -> List[float]:
        """Calculate three take profit levels based on ATR"""
        # Apply mode-specific multipliers
        tp_multiplier = self.config.relaxed_tp_multiplier if is_relaxed else 1.0
        
        # Apply confidence-based TP adjustment
        if self.config.intelligent_sizing and self.config.tp_confidence_adjustment:
            _, confidence_tp_mult = RiskManager(self.config).get_position_size_multiplier(confidence)
            tp_multiplier *= confidence_tp_mult
        
        # Calculate TP distances
        tp_distances = []
        max_distances = [0.003, 0.006, 0.01]  # 0.3%, 0.6%, 1.0%
        
        for i, (atr_mult, max_dist) in enumerate(zip(self.config.tp_atr_multipliers, max_distances)):
            distance = atr * atr_mult * tp_multiplier
            distance = min(distance, entry_price * max_dist * tp_multiplier)
            tp_distances.append(distance)
        
        # Calculate actual TP levels
        if direction == TradeDirection.LONG:
            take_profits = [entry_price + dist for dist in tp_distances]
        else:
            take_profits = [entry_price - dist for dist in tp_distances]
        
        return take_profits


class StopLossCalculator:
    """Handles stop loss calculations"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def calculate_initial_stop_loss(self, entry_price: float, direction: TradeDirection,
                                  atr: float, market_bias_data: Dict) -> float:
        """Calculate initial stop loss based on ATR and Market Bias"""
        sl_distance = atr * self.config.sl_atr_multiplier
        
        if direction == TradeDirection.LONG:
            atr_stop = entry_price - sl_distance
            
            # Use Market Bias bar if available
            mb_low = market_bias_data.get('MB_l2')
            if mb_low and not pd.isna(mb_low):
                mb_stop = mb_low - 0.00005  # 0.5 pip buffer
                stop_loss = max(atr_stop, mb_stop)
            else:
                stop_loss = atr_stop
        else:
            atr_stop = entry_price + sl_distance
            
            # Use Market Bias bar if available
            mb_high = market_bias_data.get('MB_h2')
            if mb_high and not pd.isna(mb_high):
                mb_stop = mb_high + 0.00005  # 0.5 pip buffer
                stop_loss = min(atr_stop, mb_stop)
            else:
                stop_loss = atr_stop
        
        return stop_loss
    
    def update_trailing_stop(self, current_price: float, trade: Trade,
                           atr: float) -> Optional[float]:
        """Update trailing stop loss with pip-based activation"""
        activation_pips = (self.config.relaxed_tsl_activation_pips if trade.is_relaxed 
                          else self.config.tsl_activation_pips)
        min_profit_pips = self.config.tsl_min_profit_pips
        
        # Calculate profit in pips
        if trade.direction == TradeDirection.LONG:
            profit_pips = (current_price - trade.entry_price) / FOREX_PIP_SIZE
            
            if profit_pips >= activation_pips:
                # Ensure minimum profit
                min_profit_stop = trade.entry_price + (min_profit_pips * FOREX_PIP_SIZE)
                
                # ATR-based trailing stop
                atr_trailing_stop = current_price - (atr * self.config.trailing_atr_multiplier)
                
                # Use the higher (more conservative)
                new_trailing_stop = max(min_profit_stop, atr_trailing_stop)
                
                # Only update if higher than current
                if trade.trailing_stop is None or new_trailing_stop > trade.trailing_stop:
                    return new_trailing_stop
        else:
            profit_pips = (trade.entry_price - current_price) / FOREX_PIP_SIZE
            
            if profit_pips >= activation_pips:
                # Ensure minimum profit
                min_profit_stop = trade.entry_price - (min_profit_pips * FOREX_PIP_SIZE)
                
                # ATR-based trailing stop
                atr_trailing_stop = current_price + (atr * self.config.trailing_atr_multiplier)
                
                # Use the lower (more conservative)
                new_trailing_stop = min(min_profit_stop, atr_trailing_stop)
                
                # Only update if lower than current
                if trade.trailing_stop is None or new_trailing_stop < trade.trailing_stop:
                    return new_trailing_stop
        
        return trade.trailing_stop


class SignalGenerator:
    """Handles entry and exit signal generation"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def check_entry_conditions(self, row: pd.Series) -> Optional[Tuple[TradeDirection, bool]]:
        """Check if entry conditions are met"""
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
    
    def check_exit_conditions(self, row: pd.Series, trade: Trade) -> Tuple[bool, Optional[ExitReason]]:
        """Check if exit conditions are met"""
        current_price = row['Close']
        
        # Check take profit levels
        if trade.direction == TradeDirection.LONG:
            # Check for TP1 pullback
            if trade.tp_hits >= 2 and row['Low'] <= trade.take_profits[0]:
                return True, ExitReason.TP1_PULLBACK
            
            # Check normal TPs
            for i, tp in enumerate(trade.take_profits):
                if i >= trade.tp_hits and row['High'] >= tp:
                    return True, ExitReason(f'take_profit_{i+1}')
        else:
            # Check for TP1 pullback
            if trade.tp_hits >= 2 and row['High'] >= trade.take_profits[0]:
                return True, ExitReason.TP1_PULLBACK
            
            # Check normal TPs
            for i, tp in enumerate(trade.take_profits):
                if i >= trade.tp_hits and row['Low'] <= tp:
                    return True, ExitReason(f'take_profit_{i+1}')
        
        # Check stop loss
        current_stop = trade.trailing_stop if trade.trailing_stop is not None else trade.stop_loss
        
        if trade.direction == TradeDirection.LONG and current_price <= current_stop:
            exit_reason = ExitReason.TRAILING_STOP if trade.trailing_stop is not None else ExitReason.STOP_LOSS
            return True, exit_reason
        elif trade.direction == TradeDirection.SHORT and current_price >= current_stop:
            exit_reason = ExitReason.TRAILING_STOP if trade.trailing_stop is not None else ExitReason.STOP_LOSS
            return True, exit_reason
        
        # Check signal flip
        if self.config.exit_on_signal_flip:
            if trade.direction == TradeDirection.LONG:
                if row['NTI_Direction'] == -1 or row['MB_Bias'] == -1:
                    return True, ExitReason.SIGNAL_FLIP
            else:
                if row['NTI_Direction'] == 1 or row['MB_Bias'] == 1:
                    return True, ExitReason.SIGNAL_FLIP
        
        return False, None


class PnLCalculator:
    """Handles P&L calculations"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
    
    def calculate_pnl(self, entry_price: float, exit_price: float,
                     position_size: float, direction: TradeDirection) -> Tuple[float, float]:
        """Calculate P&L for a position"""
        if direction == TradeDirection.LONG:
            price_change_pips = (exit_price - entry_price) * 10000
        else:
            price_change_pips = (entry_price - exit_price) * 10000
        
        millions = position_size / self.config.min_lot_size
        pnl = millions * self.config.pip_value_per_million * price_change_pips
        
        return pnl, price_change_pips


# ============================================================================
# Main Strategy Class
# ============================================================================

class ProdStrategy:
    """Production-ready trading strategy implementation"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize strategy with configuration"""
        self.config = config or StrategyConfig()
        
        # Initialize components
        self.risk_manager = RiskManager(self.config)
        self.tp_calculator = TakeProfitCalculator(self.config)
        self.sl_calculator = StopLossCalculator(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.pnl_calculator = PnLCalculator(self.config)
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset strategy state"""
        self.current_capital = self.config.initial_capital
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.equity_curve = [self.config.initial_capital]
        self.drawdown_curve = []
    
    def _log_trade_entry(self, trade: Trade):
        """Log trade entry details"""
        if self.config.verbose:
            size_millions = trade.position_size / self.config.min_lot_size
            trade_type = "RELAXED" if trade.is_relaxed else "STANDARD"
            conf_level = self.risk_manager.get_confidence_level(trade.confidence).value
            
            logger.info(f"{trade_type} TRADE: {trade.direction.value} at {trade.entry_price:.5f} "
                       f"with {size_millions:.0f}M (conf: {trade.confidence:.1f}% - {conf_level})")
    
    def _log_trade_exit(self, trade: Trade, exit_type: str, exit_size: float, pnl: float):
        """Log trade exit details"""
        if self.config.verbose:
            millions_exited = exit_size / self.config.min_lot_size
            logger.info(f"{exit_type} at {trade.exit_time}: Exit {millions_exited:.1f}M "
                       f"at price {trade.exit_price:.5f}, P&L: ${pnl:.2f}")
    
    def _execute_partial_exit(self, trade: Trade, exit_time: pd.Timestamp,
                             exit_price: float, tp_level: int) -> Optional[Trade]:
        """Execute a partial take profit exit"""
        # Close 33% of original position
        exit_size = trade.position_size / 3
        trade.remaining_size -= exit_size
        
        # Calculate P&L
        partial_pnl, _ = self.pnl_calculator.calculate_pnl(
            trade.entry_price, exit_price, exit_size, trade.direction
        )
        
        # Update trade state
        trade.partial_pnl += partial_pnl
        trade.tp_hits = tp_level
        trade.tp_exit_times.append(exit_time)
        trade.tp_exit_prices.append(exit_price)
        
        # Record partial exit
        trade.partial_exits.append(PartialExit(
            time=exit_time,
            price=exit_price,
            size=exit_size,
            tp_level=tp_level,
            pnl=partial_pnl
        ))
        
        # Update capital
        self.current_capital += partial_pnl
        
        self._log_trade_exit(trade, f"TP{tp_level}", exit_size, partial_pnl)
        
        # Check if trade is complete
        if trade.tp_hits >= 3 or trade.remaining_size <= 0:
            trade.remaining_size = 0
            trade.pnl = trade.partial_pnl
            return trade
        
        return None  # Trade continues
    
    def _execute_full_exit(self, trade: Trade, exit_time: pd.Timestamp,
                          exit_price: float, exit_reason: ExitReason) -> Trade:
        """Execute a full trade exit"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Handle special exit types
        if 'take_profit' in exit_reason.value:
            tp_index = int(exit_reason.value.split('_')[-1]) - 1
            completed_trade = self._execute_partial_exit(trade, exit_time, exit_price, tp_index + 1)
            if completed_trade:
                return completed_trade
            return None  # Continue with partial position
        
        elif exit_reason == ExitReason.TP1_PULLBACK:
            exit_price = trade.take_profits[0]  # Exit at TP1 level
        
        # Calculate final P&L
        if trade.remaining_size > 0:
            remaining_pnl, _ = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, trade.remaining_size, trade.direction
            )
            trade.pnl = trade.partial_pnl + remaining_pnl
            self.current_capital += remaining_pnl
            
            self._log_trade_exit(trade, exit_reason.value.upper(), trade.remaining_size, remaining_pnl)
        
        trade.remaining_size = 0
        
        # Calculate percentage return
        original_value = trade.entry_price * trade.position_size
        trade.pnl_percent = (trade.pnl / original_value) * 100
        
        return trade
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the backtest on the provided DataFrame"""
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'NTI_Direction',
                        'MB_Bias', 'IC_Regime', 'IC_ATR_Normalized']
        
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
                # Update trailing stop
                atr = current_row['IC_ATR_Normalized']
                new_trailing_stop = self.sl_calculator.update_trailing_stop(
                    current_row['Close'], self.current_trade, atr
                )
                if new_trailing_stop is not None:
                    self.current_trade.trailing_stop = new_trailing_stop
                
                # Check exit conditions
                should_exit, exit_reason = self.signal_generator.check_exit_conditions(
                    current_row, self.current_trade
                )
                
                if should_exit:
                    # Determine exit price
                    exit_price = self._get_exit_price(current_row, self.current_trade, exit_reason)
                    
                    # Execute exit
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
                    
                    self._log_trade_entry(self.current_trade)
        
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
        """Determine the appropriate exit price based on exit reason"""
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
    
    def _create_new_trade(self, entry_time: pd.Timestamp, row: pd.Series,
                         direction: TradeDirection, is_relaxed: bool) -> Trade:
        """Create a new trade with all parameters"""
        entry_price = row['Close']
        atr = row['IC_ATR_Normalized']
        confidence = row.get('NTI_Confidence', 50.0)
        
        # Calculate trade parameters
        market_bias_data = {
            'MB_l2': row.get('MB_l2'),
            'MB_h2': row.get('MB_h2')
        }
        
        stop_loss = self.sl_calculator.calculate_initial_stop_loss(
            entry_price, direction, atr, market_bias_data
        )
        
        position_size = self.risk_manager.calculate_position_size(
            entry_price, stop_loss, self.current_capital, is_relaxed, confidence
        )
        
        take_profits = self.tp_calculator.calculate_take_profits(
            entry_price, direction, atr, is_relaxed, confidence
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
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
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

def create_strategy(**kwargs) -> ProdStrategy:
    """Create a strategy instance with custom configuration"""
    config = StrategyConfig(**kwargs)
    return ProdStrategy(config)


if __name__ == "__main__":
    # Example usage
    logger.info("Production Strategy Module Loaded")
    
    # Create strategy with custom configuration
    strategy = create_strategy(
        initial_capital=100_000,
        risk_per_trade=0.02,
        intelligent_sizing=True,
        exit_on_signal_flip=True,
        verbose=True
    )
    
    logger.info(f"Strategy initialized with capital: ${strategy.config.initial_capital:,.2f}")