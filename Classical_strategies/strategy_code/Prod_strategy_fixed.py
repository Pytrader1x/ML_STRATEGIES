"""
Fixed Production Strategy Implementation
Fixes the over-exiting bug and adds proper position tracking

Key Fixes:
1. Proper partial exit sizing based on remaining position
2. Comprehensive position tracking with all exits logged
3. Safety checks to prevent over-exiting
4. Clear trade progression tracking

Author: Trading System
Date: 2025
Version: 2.1 (Fixed)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def annualization_factor_from_df(df: pd.DataFrame) -> float:
    """
    Automatically detect data frequency from DataFrame index and return annualization factor.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        Square root of periods per year for Sharpe ratio calculation
    """
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        return np.sqrt(252)  # Default to daily
    
    # Calculate median time difference in seconds
    time_diffs = df.index.to_series().diff().dt.total_seconds().dropna()
    median_seconds = time_diffs.median()
    
    if median_seconds == 0 or pd.isna(median_seconds):
        return np.sqrt(252)  # Fallback to daily
    
    # Calculate periods per year based on median interval
    seconds_per_year = 365.25 * 24 * 3600
    periods_per_year = seconds_per_year / median_seconds
    
    # Common timeframes for validation
    if 840 <= median_seconds <= 960:  # 14-16 minutes
        return np.sqrt(252 * 96)  # 96 fifteen-minute bars per trading day
    elif 3300 <= median_seconds <= 3900:  # 55-65 minutes
        return np.sqrt(252 * 24)  # 24 hourly bars per trading day
    elif 14000 <= median_seconds <= 15000:  # ~4 hours
        return np.sqrt(252 * 6)  # 6 four-hour bars per trading day
    elif 82800 <= median_seconds <= 93600:  # ~23-26 hours
        return np.sqrt(252)  # Daily bars
    else:
        return np.sqrt(periods_per_year)


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
    exit_type: str  # "TP1", "TP2", "TP3", "PARTIAL", "STOP_LOSS", etc.


@dataclass
class Trade:
    """Comprehensive trade information with enhanced tracking"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: TradeDirection
    position_size: float  # Initial position size
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
    tp_hits: int = 0  # Number of TPs hit
    remaining_size: float = None  # Current remaining position
    partial_pnl: float = 0.0  # Cumulative P&L from partial exits
    tp_exit_times: List[pd.Timestamp] = field(default_factory=list)
    tp_exit_prices: List[float] = field(default_factory=list)
    partial_exits: List[PartialExit] = field(default_factory=list)
    
    # New fields for better tracking
    initial_position_size: float = None  # Store original size
    total_exited: float = 0.0  # Track total amount exited
    exit_history: List[Dict] = field(default_factory=list)  # Detailed exit history
    
    def __post_init__(self):
        """Initialize remaining size and initial position if not provided"""
        if self.remaining_size is None:
            self.remaining_size = self.position_size
        if self.initial_position_size is None:
            self.initial_position_size = self.position_size
        # Convert string direction to enum if needed
        if isinstance(self.direction, str):
            self.direction = TradeDirection(self.direction)
    
    def add_exit(self, exit_time: pd.Timestamp, exit_price: float, exit_size: float, 
                 exit_type: str, pnl: float, reason: str):
        """Add an exit to the trade history"""
        self.exit_history.append({
            'time': exit_time,
            'price': exit_price,
            'size': exit_size,
            'type': exit_type,
            'pnl': pnl,
            'reason': reason,
            'remaining_after': self.remaining_size - exit_size
        })
        self.total_exited += exit_size
        self.remaining_size -= exit_size
        
        # Safety check
        if self.remaining_size < 0:
            logger.error(f"WARNING: Remaining size negative! Trade {id(self)}: "
                        f"Initial={self.initial_position_size}, "
                        f"Exited={self.total_exited}, "
                        f"Remaining={self.remaining_size}")
            self.remaining_size = 0


# ============================================================================
# Base Strategy Components
# ============================================================================

class RiskManager:
    """Handles position sizing and risk calculations"""
    
    def __init__(self, config):
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


class PnLCalculator:
    """Handles P&L calculations"""
    
    def __init__(self, config):
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
# Enhanced Configuration
# ============================================================================

@dataclass
class OptimizedStrategyConfig:
    """Enhanced strategy configuration with optimization parameters"""
    
    # Capital and risk management
    initial_capital: float = 100_000
    risk_per_trade: float = 0.02
    
    # Realistic costs mode
    realistic_costs: bool = False
    entry_slippage_pips: float = 0.5  # Random slippage up to this amount
    stop_loss_slippage_pips: float = 2.0  # Up to 2 pips slippage on stop loss
    trailing_stop_slippage_pips: float = 1.0  # Up to 1 pip slippage on trailing stop
    take_profit_slippage_pips: float = 0.0  # No slippage on limit orders (TPs)
    
    # Take profit configuration
    tp_atr_multipliers: Tuple[float, float, float] = (0.8, 1.5, 2.5)
    max_tp_percent: float = 0.01
    
    # TP exit percentages (FIXED VALUES)
    tp1_exit_percent: float = 0.3333  # Exit 33.33% at TP1
    tp2_exit_percent: float = 0.5000  # Exit 50% of remaining at TP2 (33.33% of original)
    tp3_exit_percent: float = 1.0000  # Exit 100% of remaining at TP3 (33.33% of original)
    
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
    tsl_initial_buffer_multiplier: float = 2.0  # Initial TSL buffer to prevent immediate exits
    
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
    debug_decisions: bool = False  # Print detailed decision logic for analysis
    
    # Sharpe ratio calculation
    use_daily_sharpe: bool = True  # If True, resample to daily for Sharpe calculation; if False, use bar-level data


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
            
            # Check normal TPs - only check the next unprocessed TP level
            for i, tp in enumerate(trade.take_profits):
                if i == trade.tp_hits and row['High'] >= tp:  # Only check exact next level
                    # Return appropriate exit percentage for this TP level
                    if i == 0:  # TP1
                        return True, ExitReason(f'take_profit_{i+1}'), self.config.tp1_exit_percent
                    elif i == 1:  # TP2  
                        return True, ExitReason(f'take_profit_{i+1}'), self.config.tp2_exit_percent
                    else:  # TP3
                        return True, ExitReason(f'take_profit_{i+1}'), self.config.tp3_exit_percent
        else:
            # Check for TP1 pullback
            if trade.tp_hits >= 2 and row['High'] >= trade.take_profits[0]:
                return True, ExitReason.TP1_PULLBACK, 1.0
            
            # Check normal TPs - only check the next unprocessed TP level
            for i, tp in enumerate(trade.take_profits):
                if i == trade.tp_hits and row['Low'] <= tp:  # Only check exact next level
                    # Return appropriate exit percentage for this TP level
                    if i == 0:  # TP1
                        return True, ExitReason(f'take_profit_{i+1}'), self.config.tp1_exit_percent
                    elif i == 1:  # TP2
                        return True, ExitReason(f'take_profit_{i+1}'), self.config.tp2_exit_percent
                    else:  # TP3
                        return True, ExitReason(f'take_profit_{i+1}'), self.config.tp3_exit_percent
        
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
    """Production strategy with fixed exit sizing and enhanced tracking"""
    
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
        
        # Initialize random number generator for slippage
        np.random.seed(None)  # Use current time as seed for randomness
    
    def reset(self):
        """Reset strategy state"""
        self.current_capital = self.config.initial_capital
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.equity_curve = [self.config.initial_capital]
        self.partial_profit_taken = {}  # Track partial profits per trade
        
        # Initialize comprehensive trade logger
        self.trade_log_detailed = []
        self.enable_trade_logging = True
    
    def _log_trade_action(self, action_type: str, trade: Trade, price: float, size: float,
                         reason: str, indicators: dict, pnl: float = 0, cumulative_pnl: float = 0,
                         pips: float = 0, timestamp: pd.Timestamp = None):
        """Log detailed trade information"""
        if not self.enable_trade_logging:
            return
            
        log_entry = {
            'timestamp': timestamp or trade.entry_time,
            'action': action_type,
            'trade_id': id(trade),
            'direction': trade.direction.value,
            'entry_price': trade.entry_price,
            'current_price': price,
            'size': size,
            'reason': reason,
            'sl_level': trade.stop_loss,
            'tp1_level': trade.take_profits[0] if trade.take_profits else None,
            'tp2_level': trade.take_profits[1] if len(trade.take_profits) > 1 else None,
            'tp3_level': trade.take_profits[2] if len(trade.take_profits) > 2 else None,
            'pips': pips,
            'pnl': pnl,
            'cumulative_pnl': cumulative_pnl,
            'remaining_size': trade.remaining_size,
            'confidence': trade.confidence,
            'is_relaxed': trade.is_relaxed,
            'nti_direction': indicators.get('nti_direction', 0),
            'mb_bias': indicators.get('mb_bias', 0),
            'ic_regime': indicators.get('ic_regime', 0),
            'ic_regime_name': indicators.get('ic_regime_name', ''),
            'atr': indicators.get('atr', 0),
            'initial_size': trade.initial_position_size,
            'total_exited': trade.total_exited,
            'tp_hits': trade.tp_hits
        }
        
        self.trade_log_detailed.append(log_entry)
    
    def _apply_slippage(self, price: float, order_type: str, direction: TradeDirection) -> float:
        """Apply realistic slippage based on order type and direction"""
        if not self.config.realistic_costs:
            return price
        
        # Determine slippage amount based on order type
        if order_type == 'entry':
            # Random slippage for market entries
            slippage_pips = np.random.uniform(0, self.config.entry_slippage_pips)
        elif order_type == 'stop_loss':
            # Worse slippage for stop losses (always against us)
            slippage_pips = np.random.uniform(0, self.config.stop_loss_slippage_pips)
        elif order_type == 'trailing_stop':
            # Moderate slippage for trailing stops
            slippage_pips = np.random.uniform(0, self.config.trailing_stop_slippage_pips)
        elif order_type == 'take_profit':
            # No slippage for limit orders (take profits)
            slippage_pips = self.config.take_profit_slippage_pips
        else:
            slippage_pips = 0
        
        # Convert pips to price
        slippage_amount = slippage_pips * FOREX_PIP_SIZE
        
        # Apply slippage in the direction that's worse for us
        if order_type == 'entry':
            # For entries, we get worse price (buy higher, sell lower)
            if direction == TradeDirection.LONG:
                return price + slippage_amount
            else:
                return price - slippage_amount
        elif order_type in ['stop_loss', 'trailing_stop']:
            # For stops, we get worse fill (sell lower, buy higher)
            if direction == TradeDirection.LONG:
                return price - slippage_amount
            else:
                return price + slippage_amount
        else:
            # Take profits - no slippage (limit orders)
            return price
    
    def _update_trailing_stop(self, current_price: float, trade: Trade,
                              atr: float) -> Optional[float]:
        """Update trailing stop loss with pip-based activation"""
        activation_pips = (self.config.relaxed_tsl_activation_pips if trade.is_relaxed 
                          else self.config.tsl_activation_pips)
        min_profit_pips = self.config.tsl_min_profit_pips
        
        # Fix for abnormal ATR values - normalize if needed
        # If ATR seems unreasonably high (> 100 pips), assume it's not normalized
        if atr > 0.01:  # More than 100 pips
            # Assume ATR is in price terms, not normalized
            # For FOREX, typical ATR might be 10-30 pips
            atr = min(atr, 0.003)  # Cap at 30 pips max
        
        # Calculate profit in pips
        if trade.direction == TradeDirection.LONG:
            profit_pips = (current_price - trade.entry_price) / FOREX_PIP_SIZE
            
            if profit_pips >= activation_pips:
                # Ensure minimum profit
                min_profit_stop = trade.entry_price + (min_profit_pips * FOREX_PIP_SIZE)
                
                # ATR-based trailing stop with initial buffer
                # When TSL first activates, use a larger multiplier to prevent immediate exits
                if trade.trailing_stop is None:
                    # First activation - use larger buffer
                    atr_trailing_stop = current_price - (atr * self.config.trailing_atr_multiplier * self.config.tsl_initial_buffer_multiplier)
                else:
                    # Subsequent updates - use normal multiplier
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
                
                # ATR-based trailing stop with initial buffer
                # When TSL first activates, use a larger multiplier to prevent immediate exits
                if trade.trailing_stop is None:
                    # First activation - use larger buffer
                    atr_trailing_stop = current_price + (atr * self.config.trailing_atr_multiplier * self.config.tsl_initial_buffer_multiplier)
                else:
                    # Subsequent updates - use normal multiplier
                    atr_trailing_stop = current_price + (atr * self.config.trailing_atr_multiplier)
                
                # Use the lower (more conservative)
                new_trailing_stop = min(min_profit_stop, atr_trailing_stop)
                
                # Only update if lower than current
                if trade.trailing_stop is None or new_trailing_stop < trade.trailing_stop:
                    return new_trailing_stop
        
        return trade.trailing_stop
    
    def _execute_partial_exit(self, trade: Trade, exit_time: pd.Timestamp,
                             exit_price: float, exit_percent: float, 
                             exit_reason: Optional[ExitReason] = None) -> Optional[Trade]:
        """Execute a partial exit with custom percentage and safety checks"""
        
        # Safety check: ensure we have remaining position
        if trade.remaining_size <= 0:
            logger.warning(f"Cannot exit - no remaining position! Trade {id(trade)}")
            return None
        
        # Calculate exit size
        exit_size = trade.remaining_size * exit_percent
        
        # Safety check: ensure exit size doesn't exceed remaining
        exit_size = min(exit_size, trade.remaining_size)
        
        # Calculate P&L
        partial_pnl, pips = self.pnl_calculator.calculate_pnl(
            trade.entry_price, exit_price, exit_size, trade.direction
        )
        
        # Update trade state using the new add_exit method
        trade.add_exit(exit_time, exit_price, exit_size, 'PARTIAL', partial_pnl, 
                      exit_reason.value if exit_reason else 'Partial Exit')
        trade.partial_pnl += partial_pnl
        
        # Log partial exit
        if self.enable_trade_logging:
            indicators = {
                'nti_direction': 0,  # Will be updated by caller if available
                'mb_bias': 0,
                'ic_regime': 0,
                'ic_regime_name': '',
                'atr': 0
            }
            self._log_trade_action(
                'PARTIAL_EXIT',
                trade,
                exit_price,
                exit_size,
                f"Partial Exit ({exit_percent*100:.0f}%)",
                indicators,
                pnl=partial_pnl,
                cumulative_pnl=self.current_capital - self.config.initial_capital + partial_pnl,
                pips=pips,
                timestamp=exit_time
            )
        
        # Record partial exit
        trade.partial_exits.append(PartialExit(
            time=exit_time,
            price=exit_price,
            size=exit_size,
            tp_level=0,  # Not a TP exit
            pnl=partial_pnl,
            exit_type='PARTIAL'
        ))
        
        # Update capital
        self.current_capital += partial_pnl
        
        if self.config.verbose:
            logger.info(f"Partial exit ({exit_percent*100:.0f}%) at {exit_price:.5f}, "
                       f"Size: {exit_size/1e6:.2f}M, P&L: ${partial_pnl:.2f}, "
                       f"Remaining: {trade.remaining_size/1e6:.2f}M")
        
        # Check if trade is complete
        if trade.remaining_size <= 0:
            trade.remaining_size = 0
            trade.pnl = trade.partial_pnl
            trade.exit_time = exit_time
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason or ExitReason.SIGNAL_FLIP
            return trade
        
        return None  # Trade continues
    
    def _execute_full_exit(self, trade: Trade, exit_time: pd.Timestamp,
                          exit_price: float, exit_reason: ExitReason) -> Trade:
        """Execute a full trade exit with proper TP handling (FIXED)"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Handle special exit types
        if 'take_profit' in exit_reason.value:
            tp_index = int(exit_reason.value.split('_')[-1]) - 1
            
            # FIXED: Calculate exit size based on remaining position and TP level
            if tp_index == 0:  # TP1
                # Exit 33.33% of remaining (which is 33.33% of original on first hit)
                exit_size = trade.remaining_size * self.config.tp1_exit_percent
            elif tp_index == 1:  # TP2
                # Exit 50% of remaining (which gives us 33.33% of original)
                exit_size = trade.remaining_size * self.config.tp2_exit_percent
            else:  # TP3
                # Exit all remaining
                exit_size = trade.remaining_size * self.config.tp3_exit_percent
            
            # Safety check
            exit_size = min(exit_size, trade.remaining_size)
            
            # Calculate P&L
            partial_pnl, pips = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, exit_size, trade.direction
            )
            
            # Update trade state
            trade.add_exit(exit_time, exit_price, exit_size, f'TP{tp_index+1}', 
                          partial_pnl, f"Take Profit {tp_index+1} Hit")
            trade.partial_pnl += partial_pnl
            trade.tp_hits = tp_index + 1
            self.current_capital += partial_pnl
            
            # Record TP exit
            trade.partial_exits.append(PartialExit(
                time=exit_time,
                price=exit_price,
                size=exit_size,
                tp_level=tp_index + 1,
                pnl=partial_pnl,
                exit_type=f'TP{tp_index+1}'
            ))
            
            # Log TP exit
            if self.enable_trade_logging:
                indicators = {'nti_direction': 0, 'mb_bias': 0, 'ic_regime': 0, 'ic_regime_name': '', 'atr': 0}
                self._log_trade_action(
                    f'TP{tp_index+1}_EXIT',
                    trade,
                    exit_price,
                    exit_size,
                    f"Take Profit {tp_index+1} Hit",
                    indicators,
                    pnl=partial_pnl,
                    cumulative_pnl=self.current_capital - self.config.initial_capital,
                    pips=pips,
                    timestamp=exit_time
                )
            
            if self.config.verbose:
                logger.info(f"TP{tp_index+1} hit at {exit_price:.5f}, "
                           f"Exit size: {exit_size/1e6:.2f}M, P&L: ${partial_pnl:.2f}, "
                           f"Remaining: {trade.remaining_size/1e6:.2f}M")
            
            # Check if trade is complete
            if trade.tp_hits >= 3 or trade.remaining_size <= 0:
                trade.remaining_size = 0
                trade.pnl = trade.partial_pnl
                return trade
            
            # CRITICAL: After TP partial exit, we need to check if there are more TPs to hit
            # If all 3 TPs have been hit, close the trade completely
            if trade.tp_hits >= 3:
                trade.remaining_size = 0 
                trade.pnl = trade.partial_pnl
                return trade
            
            return None  # Continue with partial position for next TP level
        
        elif exit_reason == ExitReason.TP1_PULLBACK:
            exit_price = trade.take_profits[0]
        
        # Calculate final P&L for any remaining position
        if trade.remaining_size > 0:
            remaining_pnl, pips = self.pnl_calculator.calculate_pnl(
                trade.entry_price, exit_price, trade.remaining_size, trade.direction
            )
            trade.add_exit(exit_time, exit_price, trade.remaining_size, 
                          exit_reason.value, remaining_pnl, f"Exit: {exit_reason.value}")
            trade.pnl = trade.partial_pnl + remaining_pnl
            self.current_capital += remaining_pnl
            
            # Log final exit
            if self.enable_trade_logging:
                indicators = {'nti_direction': 0, 'mb_bias': 0, 'ic_regime': 0, 'ic_regime_name': '', 'atr': 0}
                self._log_trade_action(
                    'FINAL_EXIT',
                    trade,
                    exit_price,
                    trade.remaining_size,
                    f"Exit: {exit_reason.value}",
                    indicators,
                    pnl=remaining_pnl,
                    cumulative_pnl=self.current_capital - self.config.initial_capital,
                    pips=pips,
                    timestamp=exit_time
                )
        
        trade.remaining_size = 0
        
        # Calculate percentage return
        original_value = trade.entry_price * trade.position_size
        trade.pnl_percent = (trade.pnl / original_value) * 100
        
        # Final safety check
        if abs(trade.total_exited - trade.initial_position_size) > 1:  # Allow 1 unit rounding error
            logger.error(f"Position size mismatch! Trade {id(trade)}: "
                        f"Entered {trade.initial_position_size/1e6:.3f}M, "
                        f"Exited {trade.total_exited/1e6:.3f}M")
        
        return trade
    
    def _create_new_trade(self, entry_time: pd.Timestamp, row: pd.Series,
                         direction: TradeDirection, is_relaxed: bool) -> Trade:
        """Create a new trade with optimized parameters"""
        entry_price = row['Close']
        
        # Apply entry slippage if realistic costs mode is enabled
        entry_price = self._apply_slippage(entry_price, 'entry', direction)
        
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
        """Run the optimized backtest with optional debug logging"""
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'NTI_Direction',
                        'MB_Bias', 'IC_Regime', 'IC_ATR_Normalized', 'IC_RegimeName']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Reset state
        self.reset()
        
        # Store df reference for annualization factor calculation
        self.df = df
        
        if self.config.debug_decisions:
            print(f"\n=== STARTING BACKTEST ===")
            print(f"Data period: {df.index[0]} to {df.index[-1]}")
            print(f"Total rows: {len(df)}")
            print(f"Initial capital: ${self.config.initial_capital:,.0f}")
            print("=" * 50)
        
        # Main backtest loop
        for idx in range(1, len(df)):
            current_row = df.iloc[idx]
            current_time = df.index[idx]
            
            if self.config.debug_decisions:
                # Print current market state every 100 rows or when significant events happen
                if idx % 100 == 0 or self.current_trade is not None:
                    print(f"\nRow {idx:5d} | {current_time} | Price: {current_row['Close']:.5f}")
                    print(f"  NTI: {current_row['NTI_Direction']:2.0f} | MB: {current_row['MB_Bias']:2.0f} | IC: {current_row['IC_Regime']:2.0f} ({current_row['IC_RegimeName']})")
                    print(f"  ATR: {current_row['IC_ATR_Normalized']:.5f} | Capital: ${self.current_capital:,.0f}")
            
            # Calculate mark-to-market equity (including unrealized P&L)
            equity_value = self.current_capital
            if self.current_trade is not None:
                # Add unrealized P&L for open positions
                unrealized_pnl, _ = self.pnl_calculator.calculate_pnl(
                    self.current_trade.entry_price,
                    current_row['Close'],
                    self.current_trade.remaining_size,
                    self.current_trade.direction
                )
                equity_value += unrealized_pnl
            
            # Update equity curve with mark-to-market value
            self.equity_curve.append(equity_value)
            
            # Process open trade
            if self.current_trade is not None:
                if self.config.debug_decisions:
                    current_pnl = self.pnl_calculator.calculate_pnl(
                        self.current_trade.entry_price, current_row['Close'], 
                        self.current_trade.remaining_size, self.current_trade.direction
                    )[0]
                    print(f"  🔄 OPEN TRADE: {self.current_trade.direction.value} from {self.current_trade.entry_price:.5f}")
                    print(f"     Current P&L: ${current_pnl:,.0f} | SL: {self.current_trade.stop_loss:.5f} | Trailing: {self.current_trade.trailing_stop}")
                
                # Check for partial profit taking
                if self.signal_generator.check_partial_profit_conditions(current_row, self.current_trade):
                    trade_id = id(self.current_trade)
                    if trade_id not in self.partial_profit_taken:
                        if self.config.debug_decisions:
                            print(f"  ✅ PARTIAL PROFIT: Taking {self.config.partial_profit_size_percent*100:.0f}% profit at {current_row['Close']:.5f}")
                        
                        # Take partial profit
                        exit_price = current_row['Close']
                        completed = self._execute_partial_exit(
                            self.current_trade, current_time, exit_price, 
                            self.config.partial_profit_size_percent
                        )
                        self.partial_profit_taken[trade_id] = True
                        
                        if completed:
                            if self.config.debug_decisions:
                                print(f"  🏁 TRADE COMPLETED via partial profit")
                            self.trades.append(self.current_trade)
                            self.current_trade = None
                            continue
                
                # Update trailing stop
                atr = current_row['IC_ATR_Normalized']
                old_trailing_stop = self.current_trade.trailing_stop
                new_trailing_stop = self._update_trailing_stop(
                    current_row['Close'], self.current_trade, atr
                )
                if new_trailing_stop is not None:
                    self.current_trade.trailing_stop = new_trailing_stop
                    if self.config.debug_decisions and new_trailing_stop != old_trailing_stop:
                        print(f"  📈 TRAILING STOP updated: {old_trailing_stop} → {new_trailing_stop:.5f}")
                
                # Check exit conditions
                should_exit, exit_reason, exit_percent = self.signal_generator.check_exit_conditions(
                    current_row, self.current_trade, current_time
                )
                
                if should_exit:
                    if self.config.debug_decisions:
                        print(f"  🚪 EXIT SIGNAL: {exit_reason.value} ({exit_percent*100:.0f}% of position)")
                    
                    # Determine exit price
                    exit_price = self._get_exit_price(current_row, self.current_trade, exit_reason)
                    
                    if self.config.debug_decisions:
                        final_pnl = self.pnl_calculator.calculate_pnl(
                            self.current_trade.entry_price, exit_price, 
                            self.current_trade.remaining_size * exit_percent, self.current_trade.direction
                        )[0]
                        print(f"     Exit price: {exit_price:.5f} | Exit P&L: ${final_pnl:,.0f}")
                    
                    # Execute TP exit using the dedicated TP exit logic
                    if 'take_profit' in exit_reason.value:
                        completed_trade = self._execute_full_exit(
                            self.current_trade, current_time, exit_price, exit_reason
                        )
                    elif exit_percent < 1.0:
                        # Other partial exits (signal flip, etc.)
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
                        if self.config.debug_decisions:
                            print(f"  🏁 TRADE COMPLETED: Final P&L ${completed_trade.pnl:,.0f} | Total trades: {len(self.trades) + 1}")
                        self.trades.append(self.current_trade)
                        self.current_trade = None
            
            # Check for new entry
            elif self.current_trade is None:
                entry_result = self.signal_generator.check_entry_conditions(current_row)
                
                if entry_result is not None:
                    direction, is_relaxed = entry_result
                    
                    if self.config.debug_decisions:
                        entry_type = "RELAXED" if is_relaxed else "STANDARD"
                        print(f"  🎯 ENTRY SIGNAL: {entry_type} {direction.value}")
                        print(f"     Conditions: NTI={current_row['NTI_Direction']}, MB={current_row['MB_Bias']}, IC={current_row['IC_Regime']}")
                    
                    # Create new trade
                    self.current_trade = self._create_new_trade(
                        current_time, current_row, direction, is_relaxed
                    )
                    
                    # Log trade entry
                    if self.enable_trade_logging:
                        indicators = {
                            'nti_direction': current_row['NTI_Direction'],
                            'mb_bias': current_row['MB_Bias'],
                            'ic_regime': current_row['IC_Regime'],
                            'ic_regime_name': current_row.get('IC_RegimeName', ''),
                            'atr': current_row['IC_ATR_Normalized']
                        }
                        entry_reason = f"Signal: NTI={indicators['nti_direction']}, MB={indicators['mb_bias']}, IC={indicators['ic_regime_name']}"
                        if is_relaxed:
                            entry_reason += " (RELAXED)"
                        
                        self._log_trade_action(
                            'ENTRY', 
                            self.current_trade, 
                            self.current_trade.entry_price,
                            self.current_trade.position_size,
                            entry_reason,
                            indicators,
                            pnl=0,
                            cumulative_pnl=self.current_capital - self.config.initial_capital,
                            pips=0
                        )
                    
                    if self.config.debug_decisions:
                        size_millions = self.current_trade.position_size / self.config.min_lot_size
                        print(f"     Entry: {self.current_trade.entry_price:.5f} | Size: {size_millions:.1f}M")
                        print(f"     SL: {self.current_trade.stop_loss:.5f} | TPs: {[f'{tp:.5f}' for tp in self.current_trade.take_profits]}")
                    
                    if self.config.verbose:
                        size_millions = self.current_trade.position_size / self.config.min_lot_size
                        trade_type = "RELAXED" if is_relaxed else "STANDARD"
                        logger.info(f"{trade_type} TRADE: {direction.value} at {self.current_trade.entry_price:.5f} "
                                   f"with {size_millions:.0f}M")
                else:
                    # Debug why no entry was taken
                    if self.config.debug_decisions and idx % 500 == 0:  # Every 500 rows when no trade
                        print(f"  ⏸️  NO ENTRY: NTI={current_row['NTI_Direction']}, MB={current_row['MB_Bias']}, IC={current_row['IC_Regime']}")
        
        if self.config.debug_decisions:
            print(f"\n=== BACKTEST COMPLETED ===")
            print(f"Total trades executed: {len(self.trades)}")
            if self.current_trade is not None:
                print("Warning: Trade still open at end of data")
        
        # Close any remaining trade
        if self.current_trade is not None:
            last_row = df.iloc[-1]
            last_time = df.index[-1]
            
            if self.config.debug_decisions:
                print(f"🏁 CLOSING FINAL TRADE at end of data: {last_row['Close']:.5f}")
            
            completed_trade = self._execute_full_exit(
                self.current_trade, last_time, last_row['Close'], ExitReason.END_OF_DATA
            )
            if completed_trade:
                self.trades.append(self.current_trade)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics()
    
    def _get_exit_price(self, row: pd.Series, trade: Trade, exit_reason: ExitReason) -> float:
        """Determine exit price with slippage applied"""
        base_price = row['Close']
        
        if 'take_profit' in exit_reason.value:
            tp_index = int(exit_reason.value.split('_')[-1]) - 1
            base_price = trade.take_profits[tp_index]
            # Apply take profit slippage (should be 0 for limit orders)
            return self._apply_slippage(base_price, 'take_profit', trade.direction)
        elif exit_reason == ExitReason.TP1_PULLBACK:
            base_price = trade.take_profits[0]
            # TP1 pullback uses limit order, no slippage
            return self._apply_slippage(base_price, 'take_profit', trade.direction)
        elif exit_reason == ExitReason.STOP_LOSS:
            base_price = trade.stop_loss
            # Apply stop loss slippage
            return self._apply_slippage(base_price, 'stop_loss', trade.direction)
        elif exit_reason == ExitReason.TRAILING_STOP:
            base_price = trade.trailing_stop if trade.trailing_stop is not None else trade.stop_loss
            # Apply trailing stop slippage
            return self._apply_slippage(base_price, 'trailing_stop', trade.direction)
        else:
            # For signal flips and other market exits, use current price with entry slippage
            return self._apply_slippage(base_price, 'entry', trade.direction)
    
    def export_trades_to_csv(self, filepath: str = None):
        """Export all trades with detailed information to CSV"""
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        # Build comprehensive trade data
        trade_records = []
        
        for i, trade in enumerate(self.trades, 1):
            # Base trade information
            base_record = {
                'trade_id': i,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'direction': trade.direction.value if hasattr(trade.direction, 'value') else trade.direction,
                'initial_size': trade.initial_position_size,
                'initial_size_millions': trade.initial_position_size / 1e6,
                'confidence': trade.confidence,
                'is_relaxed': trade.is_relaxed,
                'entry_logic': 'Relaxed (NTI only)' if trade.is_relaxed else 'Standard (NTI+MB+IC)',
                'sl_price': trade.stop_loss,
                'sl_distance_pips': abs(trade.entry_price - trade.stop_loss) / FOREX_PIP_SIZE,
                'tp1_price': trade.take_profits[0] if len(trade.take_profits) > 0 else None,
                'tp2_price': trade.take_profits[1] if len(trade.take_profits) > 1 else None,
                'tp3_price': trade.take_profits[2] if len(trade.take_profits) > 2 else None,
                'tp1_distance_pips': (trade.take_profits[0] - trade.entry_price) / FOREX_PIP_SIZE * (1 if trade.direction.value == 'long' else -1) if len(trade.take_profits) > 0 else None,
                'tp2_distance_pips': (trade.take_profits[1] - trade.entry_price) / FOREX_PIP_SIZE * (1 if trade.direction.value == 'long' else -1) if len(trade.take_profits) > 1 else None,
                'tp3_distance_pips': (trade.take_profits[2] - trade.entry_price) / FOREX_PIP_SIZE * (1 if trade.direction.value == 'long' else -1) if len(trade.take_profits) > 2 else None,
            }
            
            # Add exit history details
            total_partial_pnl = 0
            partial_exits_summary = []
            
            if hasattr(trade, 'exit_history') and trade.exit_history:
                for j, exit_event in enumerate(trade.exit_history, 1):
                    exit_type = exit_event.get('type', '')
                    exit_size = exit_event.get('size', 0)
                    exit_pnl = exit_event.get('pnl', 0)
                    exit_price = exit_event.get('price', 0)
                    
                    # Calculate pips for this exit
                    if trade.direction.value == 'long':
                        exit_pips = (exit_price - trade.entry_price) / FOREX_PIP_SIZE
                    else:
                        exit_pips = (trade.entry_price - exit_price) / FOREX_PIP_SIZE
                    
                    partial_exits_summary.append({
                        'exit_num': j,
                        'exit_type': exit_type,
                        'exit_size_millions': exit_size / 1e6,
                        'exit_price': exit_price,
                        'exit_pips': exit_pips,
                        'exit_pnl': exit_pnl
                    })
                    
                    if 'TP' in exit_type or exit_type == 'PARTIAL':
                        total_partial_pnl += exit_pnl
            
            # Final exit information
            base_record.update({
                'exit_time': trade.exit_time,
                'final_exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason.value if hasattr(trade.exit_reason, 'value') else trade.exit_reason,
                'trade_duration_hours': (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else None,
                'tp_hits': trade.tp_hits,
                'total_exited': trade.total_exited,
                'total_exited_millions': trade.total_exited / 1e6,
                'position_integrity_ok': abs(trade.total_exited - trade.initial_position_size) <= 1,
            })
            
            # P&L breakdown
            base_record.update({
                'partial_pnl_total': total_partial_pnl,
                'final_pnl': trade.pnl,
                'pnl_per_million': trade.pnl / (trade.initial_position_size / 1e6) if trade.initial_position_size > 0 else 0,
            })
            
            # Add partial exits details
            for k in range(1, 4):  # Support up to 3 partial exits
                if k <= len(partial_exits_summary):
                    pe = partial_exits_summary[k-1]
                    base_record.update({
                        f'partial_exit_{k}_type': pe['exit_type'],
                        f'partial_exit_{k}_size_m': pe['exit_size_millions'],
                        f'partial_exit_{k}_price': pe['exit_price'],
                        f'partial_exit_{k}_pips': pe['exit_pips'],
                        f'partial_exit_{k}_pnl': pe['exit_pnl']
                    })
                else:
                    base_record.update({
                        f'partial_exit_{k}_type': None,
                        f'partial_exit_{k}_size_m': None,
                        f'partial_exit_{k}_price': None,
                        f'partial_exit_{k}_pips': None,
                        f'partial_exit_{k}_pnl': None
                    })
            
            trade_records.append(base_record)
        
        # Convert to DataFrame
        import pandas as pd
        trades_df = pd.DataFrame(trade_records)
        
        # Save to CSV
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'results/trades_export_{timestamp}.csv'
        
        trades_df.to_csv(filepath, index=False, float_format='%.6f')
        logger.info(f"Exported {len(trades_df)} trades to {filepath}")
        
        # Print summary
        print(f"\n📊 Trade Export Summary:")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Total P&L: ${trades_df['final_pnl'].sum():,.2f}")
        print(f"  Avg P&L per Trade: ${trades_df['final_pnl'].mean():,.2f}")
        print(f"  Win Rate: {(trades_df['final_pnl'] > 0).sum() / len(trades_df) * 100:.1f}%")
        print(f"  Position Integrity: {trades_df['position_integrity_ok'].sum()}/{len(trades_df)} OK")
        print(f"  Saved to: {filepath}")
        
        return filepath
    
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
        max_drawdown = abs(np.min(drawdown))  # Convert to positive magnitude
        
        # Sharpe ratio calculation
        if len(self.equity_curve) > 1 and hasattr(self, 'df') and self.df is not None:
            # Create equity DataFrame with timestamps
            equity_df = pd.DataFrame({
                'timestamp': self.df.index[:len(self.equity_curve)],
                'capital': self.equity_curve
            })
            equity_df.set_index('timestamp', inplace=True)
            
            # Check if we should use daily aggregation
            if self.config.use_daily_sharpe:
                # Aggregate to daily returns to remove intraday serial correlation
                # Use last value of each day (end-of-day equity)
                daily_equity = equity_df.resample('D').last().dropna()
                
                if len(daily_equity) > 1:
                    # Calculate daily returns
                    daily_returns = daily_equity['capital'].pct_change().dropna()
                    
                    # Calculate annualized Sharpe ratio with 252 trading days
                    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
                        sharpe_ratio = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
                    else:
                        sharpe_ratio = 0
                else:
                    # Not enough daily data, use bar-level calculation as fallback
                    returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
                    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
                        # Get appropriate annualization factor
                        ann_factor = annualization_factor_from_df(self.df)
                        sharpe_ratio = np.mean(returns) / np.std(returns, ddof=1) * ann_factor
                    else:
                        sharpe_ratio = 0
            else:
                # Use bar-level calculation without daily resampling
                returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
                if len(returns) > 1 and np.std(returns, ddof=1) > 0:
                    # Get appropriate annualization factor based on bar frequency
                    ann_factor = annualization_factor_from_df(self.df)
                    sharpe_ratio = np.mean(returns) / np.std(returns, ddof=1) * ann_factor
                else:
                    sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Exit reason breakdown and TP hit analysis
        exit_reasons = {}
        tp_hit_stats = {'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0, 'partial_exits': 0}
        sl_outcome_stats = {'sl_loss': 0, 'sl_breakeven': 0, 'sl_profit': 0, 'sl_total': 0}
        exit_pattern_stats = {
            'pure_sl': 0, 'partial_then_sl': 0, 'pure_tp': 0, 
            'tp_then_other': 0, 'other': 0
        }
        
        breakeven_threshold = 50  # $50 threshold for breakeven
        
        for trade in self.trades:
            # Count exit reasons
            reason = trade.exit_reason.value if trade.exit_reason else 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            # Count TP hits based on trade history
            if trade.tp_hits >= 1:
                tp_hit_stats['tp1_hits'] += 1
            if trade.tp_hits >= 2:
                tp_hit_stats['tp2_hits'] += 1
            if trade.tp_hits >= 3:
                tp_hit_stats['tp3_hits'] += 1
            
            # Count partial exits
            if len(trade.partial_exits) > 0:
                tp_hit_stats['partial_exits'] += 1
            
            # Analyze stop loss outcomes
            if trade.exit_reason and 'stop_loss' in trade.exit_reason.value:
                sl_outcome_stats['sl_total'] += 1
                if trade.pnl < -breakeven_threshold:
                    sl_outcome_stats['sl_loss'] += 1
                elif -breakeven_threshold <= trade.pnl <= breakeven_threshold:
                    sl_outcome_stats['sl_breakeven'] += 1
                else:
                    sl_outcome_stats['sl_profit'] += 1
                
                # Determine if pure SL or partial then SL
                if len(trade.partial_exits) == 0 and trade.tp_hits == 0:
                    exit_pattern_stats['pure_sl'] += 1
                else:
                    exit_pattern_stats['partial_then_sl'] += 1
            
            # Count other patterns
            elif trade.tp_hits >= 3:
                exit_pattern_stats['pure_tp'] += 1
            elif trade.tp_hits > 0:
                exit_pattern_stats['tp_then_other'] += 1
            else:
                exit_pattern_stats['other'] += 1
        
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
            'tp_hit_stats': tp_hit_stats,
            'sl_outcome_stats': sl_outcome_stats,
            'exit_pattern_stats': exit_pattern_stats,
            'final_capital': self.current_capital,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'trade_log': self.trade_log_detailed if self.enable_trade_logging else [],
            'strategy': self  # Include strategy instance for trade export
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
            'tp_hit_stats': {'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0, 'partial_exits': 0},
            'sl_outcome_stats': {'sl_loss': 0, 'sl_breakeven': 0, 'sl_profit': 0, 'sl_total': 0},
            'exit_pattern_stats': {'pure_sl': 0, 'partial_then_sl': 0, 'pure_tp': 0, 'tp_then_other': 0, 'other': 0},
            'final_capital': self.current_capital,
            'trades': [],
            'equity_curve': self.equity_curve,
            'trade_log': []
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_optimized_strategy(**kwargs) -> OptimizedProdStrategy:
    """Create an optimized strategy instance"""
    config = OptimizedStrategyConfig(**kwargs)
    return OptimizedProdStrategy(config)


if __name__ == "__main__":
    logger.info("Fixed Production Strategy Module Loaded")