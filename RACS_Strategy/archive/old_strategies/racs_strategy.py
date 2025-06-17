import backtrader as bt
import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional, Dict, List, Tuple


class MarketRegime(Enum):
    STRONG_TREND = "Strong Trend"
    WEAK_TREND = "Weak Trend"  
    QUIET_RANGE = "Quiet Range"
    VOLATILE_CHOP = "Volatile Chop"


class SignalType(Enum):
    TREND_LONG = "TREND_LONG"
    TREND_SHORT = "TREND_SHORT"
    RANGE_LONG = "RANGE_LONG"
    RANGE_SHORT = "RANGE_SHORT"
    EXIT = "EXIT"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    NONE = "NONE"


class RACSStrategy(bt.Strategy):
    """
    Regime-Adaptive Confluence Strategy (RACS)
    
    A sophisticated multi-regime trading strategy that adapts tactics based on
    market conditions identified by Intelligent Chop indicator.
    """
    
    params = (
        # Risk Management
        ('base_risk_pct', 0.01),  # 1% risk in strong trends
        ('max_positions', 3),
        ('account_size', 10000),
        
        # Regime Thresholds  
        ('min_confidence', 60.0),
        ('yellow_confidence', 70.0),
        
        # Entry Filters
        ('min_nti_confidence', 70.0),
        ('min_slope_power', 20.0),
        
        # Range Trading
        ('range_penetration', 0.02),  # 2% penetration allowed
        ('range_target_pct', 0.8),    # Target 80% of range
        
        # Position Sizing
        ('yellow_size_factor', 0.5),
        ('blue_size_factor', 0.5),
        ('high_vol_reduction', 0.5),
        ('low_vol_bonus', 1.2),
        ('golden_setup_bonus', 1.5),
        
        # Exit Management
        ('atr_stop_multi_trend', 1.0),
        ('atr_stop_multi_range', 0.5),
        ('time_stop_multi', 4),  # 4x average holding period
        
        # Advanced Filters
        ('max_atr_normalized', 3.0),  # 3%
        ('max_bandwidth', 5.0),       # 5%
        ('min_bandwidth_bonus', 2.0),  # 2% for low vol bonus
        ('efficiency_threshold', 0.3),
        
        # Range detection
        ('range_lookback', 20),
        ('min_range_bars', 5),  # Minimum bars in Blue regime for range trading
    )
    
    def __init__(self):
        # Track positions and performance
        self.active_positions = []
        self.closed_trades = []
        self.avg_holding_period = 20  # Initial estimate
        
        # Track regime history
        self.regime_history = []
        self.last_regime_change = 0
        
        # Track neutral bars for MB
        self.mb_neutral_bars = 0
        self.last_mb_bias = 0
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.trade_log = []
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.4f}, Size: {order.executed.size}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.4f}, Size: {order.executed.size}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
        
        # Track trade performance
        self.trade_count += 1
        if trade.pnl > 0:
            self.win_count += 1
            
        self.trade_log.append({
            'entry_date': bt.num2date(trade.dtopen),
            'exit_date': bt.num2date(trade.dtclose),
            'pnl': trade.pnl,
            'pnlcomm': trade.pnlcomm,
            'size': trade.size,
            'commission': trade.commission
        })
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')
    
    def get_market_regime(self) -> Tuple[MarketRegime, float]:
        """Extract market regime and confidence from current bar."""
        # Get regime as numeric value
        regime_num = int(self.data.IC_Regime[0])
        confidence = self.data.IC_Confidence[0]
        
        # Map numeric regime to enum
        # From indicators.py: 0=Chop, 1=Weak Trend, 2=Strong Trend, 3=Range
        regime_map = {
            0: MarketRegime.VOLATILE_CHOP,
            1: MarketRegime.WEAK_TREND,
            2: MarketRegime.STRONG_TREND,
            3: MarketRegime.QUIET_RANGE
        }
        
        return regime_map.get(regime_num, MarketRegime.VOLATILE_CHOP), confidence
    
    def check_directional_alignment(self, direction: int) -> Tuple[bool, List[str]]:
        """Check if all directional indicators align."""
        reasons = []
        
        # Check NTI
        nti_dir = self.data.NTI_Direction[0]
        if nti_dir != direction:
            return False, ["NTI direction mismatch"]
        
        # Check MB
        mb_bias = self.data.MB_Bias[0]
        if mb_bias != direction and self.mb_neutral_bars < 3:
            return False, ["Market Bias not aligned"]
        
        # Check price vs MB average
        price = self.data.close[0]
        mb_avg = self.data.MB_ha_avg[0]
        if direction > 0 and price <= mb_avg:
            return False, ["Price below MB average"]
        elif direction < 0 and price >= mb_avg:
            return False, ["Price above MB average"]
        
        reasons.append(f"NTI_Direction: {nti_dir}")
        reasons.append(f"MB_Bias: {mb_bias}")
        reasons.append(f"Price vs MB: {'Above' if price > mb_avg else 'Below'}")
        
        return True, reasons
    
    def check_supertrend_flip(self, direction: int) -> bool:
        """Check if SuperTrend just flipped in the desired direction."""
        if len(self.data) < 2:
            return False
        
        current_dir = self.data.SuperTrend_Direction[0]
        prev_dir = self.data.SuperTrend_Direction[-1]
        
        # Check for flip
        if direction > 0:  # Looking for bullish flip
            return current_dir == 1 and prev_dir == -1
        else:  # Looking for bearish flip
            return current_dir == -1 and prev_dir == 1
    
    def calculate_position_size(self, stop_distance: float, regime: MarketRegime) -> float:
        """Calculate position size based on regime and risk factors."""
        # Base risk by regime
        if regime == MarketRegime.STRONG_TREND:
            base_risk = self.p.base_risk_pct
            size_factor = 1.0
        elif regime == MarketRegime.WEAK_TREND:
            base_risk = self.p.base_risk_pct
            size_factor = self.p.yellow_size_factor
        elif regime == MarketRegime.QUIET_RANGE:
            base_risk = self.p.base_risk_pct
            size_factor = self.p.blue_size_factor
        else:  # VOLATILE_CHOP
            return 0.0
        
        # Volatility adjustments
        atr_norm = self.data.IC_ATR_Normalized[0]
        bandwidth = self.data.IC_BandWidth[0]
        
        if atr_norm > self.p.max_atr_normalized:
            size_factor *= self.p.high_vol_reduction
        elif bandwidth < self.p.min_bandwidth_bonus:
            size_factor *= self.p.low_vol_bonus
        
        # Golden setup bonus
        if self.is_golden_setup():
            size_factor *= self.p.golden_setup_bonus
        
        # Calculate position size
        account_value = self.broker.get_value()
        dollar_risk = account_value * base_risk * size_factor
        position_size = dollar_risk / stop_distance if stop_distance > 0 else 0
        
        return position_size
    
    def is_golden_setup(self) -> bool:
        """Check if this is a high-probability 'golden' setup."""
        ic_conf = self.data.IC_Confidence[0]
        nti_conf = self.data.NTI_Confidence[0]
        efficiency = self.data.IC_EfficiencyRatio[0]
        
        return (ic_conf > 80 and 
                nti_conf > 85 and 
                efficiency > self.p.efficiency_threshold)
    
    def find_range_boundaries(self) -> Tuple[float, float]:
        """Find range boundaries using fractals and Bollinger Bands."""
        lookback = min(self.p.range_lookback, len(self.data))
        
        # Method 1: Recent highs and lows
        recent_highs = [self.data.high[-i] for i in range(lookback)]
        recent_lows = [self.data.low[-i] for i in range(lookback)]
        
        # Method 2: Bollinger Bands
        bb_upper = self.data.IC_BB_Upper[0]
        bb_lower = self.data.IC_BB_Lower[0]
        
        # Method 3: Fractal levels (if available)
        fractal_highs = []
        fractal_lows = []
        
        for i in range(lookback):
            if hasattr(self.data, 'SR_FractalHighs'):
                try:
                    fh = self.data.SR_FractalHighs[-i]
                    if not np.isnan(fh) and fh > 0:
                        fractal_highs.append(fh)
                except (IndexError, TypeError):
                    pass
            if hasattr(self.data, 'SR_FractalLows'):
                try:
                    fl = self.data.SR_FractalLows[-i]
                    if not np.isnan(fl) and fl > 0:
                        fractal_lows.append(fl)
                except (IndexError, TypeError):
                    pass
        
        # Combine methods
        upper_levels = recent_highs + [bb_upper] + fractal_highs
        lower_levels = recent_lows + [bb_lower] + fractal_lows
        
        # Use percentiles to filter outliers
        upper = np.percentile(upper_levels, 75) if upper_levels else max(recent_highs)
        lower = np.percentile(lower_levels, 25) if lower_levels else min(recent_lows)
        
        return upper, lower
    
    def check_trend_entry(self, regime: MarketRegime, confidence: float) -> Optional[SignalType]:
        """Check for trend-following entry signals."""
        # Regime gate
        if regime == MarketRegime.VOLATILE_CHOP:
            return None
        
        if regime == MarketRegime.QUIET_RANGE:
            return None  # Use range strategy instead
        
        # Confidence requirements
        min_conf = self.p.yellow_confidence if regime == MarketRegime.WEAK_TREND else self.p.min_confidence
        if confidence < min_conf:
            return None
        
        # NTI requirements  
        nti_conf = self.data.NTI_Confidence[0]
        slope_power = self.data.NTI_SlopePower[0]
        nti_phase = self.data.NTI_TrendPhase[0]
        reversal_risk = self.data.NTI_ReversalRisk[0]
        
        if nti_conf < self.p.min_nti_confidence:
            return None
        if nti_phase != 'Impulse':
            return None
        if reversal_risk:
            return None
        
        # Check for long entry
        if slope_power > self.p.min_slope_power:
            aligned, reasons = self.check_directional_alignment(1)
            if aligned and self.check_supertrend_flip(1):
                return SignalType.TREND_LONG
        
        # Check for short entry
        elif slope_power < -self.p.min_slope_power:
            aligned, reasons = self.check_directional_alignment(-1)
            if aligned and self.check_supertrend_flip(-1):
                return SignalType.TREND_SHORT
        
        return None
    
    def check_range_entry(self, regime: MarketRegime, confidence: float) -> Optional[SignalType]:
        """Check for range-trading entry signals."""
        # Only trade ranges in blue regime
        if regime != MarketRegime.QUIET_RANGE:
            return None
        
        if confidence < self.p.min_confidence:
            return None
        
        # Check if we've been in range mode long enough
        if self.bars_in_current_regime() < self.p.min_range_bars:
            return None
        
        # Verify ranging behavior
        chop_index = self.data.IC_ChoppinessIndex[0]
        if chop_index < 50:  # Not choppy enough
            return None
        
        # Find range boundaries
        upper, lower = self.find_range_boundaries()
        range_size = upper - lower
        
        if range_size <= 0:
            return None
        
        price = self.data.close[0]
        mb_bias = self.data.MB_Bias[0]
        
        # Check for range long
        if price <= lower * (1 + self.p.range_penetration) and mb_bias >= 0:
            return SignalType.RANGE_LONG
        
        # Check for range short
        elif price >= upper * (1 - self.p.range_penetration) and mb_bias <= 0:
            return SignalType.RANGE_SHORT
        
        return None
    
    def check_exits(self) -> List[SignalType]:
        """Check if any exit conditions are triggered for open positions."""
        exit_signals = []
        regime, confidence = self.get_market_regime()
        
        # Red zone exit - immediate for all positions
        if regime == MarketRegime.VOLATILE_CHOP:
            return [SignalType.EXIT] * len(self.getpositions())
        
        for position in self.getpositions():
            if position.size == 0:
                continue
                
            # Check position-specific exits
            if position.size > 0:  # LONG
                # Trend exits
                if self.data.NTI_Direction[0] == -1:
                    exit_signals.append(SignalType.EXIT)
                    continue
                if self.data.SuperTrend_Direction[0] == -1:
                    exit_signals.append(SignalType.EXIT)
                    continue
                if self.data.NTI_ReversalRisk[0]:
                    exit_signals.append(SignalType.EXIT)
                    continue
                
                # Partial exit based on slope power reduction
                # If slope power reduced significantly, consider partial exit
                if abs(self.data.NTI_SlopePower[0]) < self.p.min_slope_power * 0.5 and not hasattr(position, 'partial_exit_done'):
                    exit_signals.append(SignalType.PARTIAL_EXIT)
                    position.partial_exit_done = True
                    continue
                    
            else:  # SHORT
                # Trend exits
                if self.data.NTI_Direction[0] == 1:
                    exit_signals.append(SignalType.EXIT)
                    continue
                if self.data.SuperTrend_Direction[0] == 1:
                    exit_signals.append(SignalType.EXIT)
                    continue
                if self.data.NTI_ReversalRisk[0]:
                    exit_signals.append(SignalType.EXIT)
                    continue
                
                # Partial exit based on slope power reduction
                # If slope power reduced significantly, consider partial exit
                if abs(self.data.NTI_SlopePower[0]) < self.p.min_slope_power * 0.5 and not hasattr(position, 'partial_exit_done'):
                    exit_signals.append(SignalType.PARTIAL_EXIT)
                    position.partial_exit_done = True
                    continue
        
        return exit_signals
    
    def bars_in_current_regime(self) -> int:
        """Calculate how many bars we've been in the current regime."""
        return len(self.data) - self.last_regime_change
    
    def next(self):
        # Update MB neutral bars tracking
        current_mb = self.data.MB_Bias[0]
        if current_mb == 0:
            self.mb_neutral_bars += 1
        else:
            self.mb_neutral_bars = 0
        self.last_mb_bias = current_mb
        
        # Track regime changes
        regime, confidence = self.get_market_regime()
        if not self.regime_history or regime != self.regime_history[-1]:
            self.regime_history.append(regime)
            self.last_regime_change = len(self.data)
        
        # Check exits first
        exit_signals = self.check_exits()
        
        for i, signal in enumerate(exit_signals):
            if signal == SignalType.EXIT:
                self.close()
                self.log(f'EXIT SIGNAL - Closing all positions')
            elif signal == SignalType.PARTIAL_EXIT:
                # Close 50% of position
                pos = self.getposition()
                if pos.size != 0:
                    self.sell(size=abs(pos.size) * 0.5) if pos.size > 0 else self.buy(size=abs(pos.size) * 0.5)
                    self.log(f'PARTIAL EXIT - Closing 50% of position')
        
        # Check for new entries if we have capacity
        if len(self.getpositions()) < self.p.max_positions:
            # Try trend entry first
            signal = self.check_trend_entry(regime, confidence)
            
            # If no trend signal, try range entry
            if not signal:
                signal = self.check_range_entry(regime, confidence)
            
            # Process signal
            if signal:
                # Calculate position sizing
                if signal in [SignalType.TREND_LONG, SignalType.RANGE_LONG]:
                    entry_price = self.data.high[0] + 0.0001  # Stop order above high
                    stop_price = self.data.SuperTrend_Line[0]
                    stop_distance = entry_price - stop_price
                    
                    if stop_distance > 0:
                        size = self.calculate_position_size(stop_distance, regime)
                        if size > 0:
                            self.buy(size=size)
                            self.log(f'{signal.value} - Size: {size:.2f}, Entry: {entry_price:.4f}, Stop: {stop_price:.4f}')
                
                elif signal in [SignalType.TREND_SHORT, SignalType.RANGE_SHORT]:
                    entry_price = self.data.low[0] - 0.0001  # Stop order below low
                    stop_price = self.data.SuperTrend_Line[0]
                    stop_distance = stop_price - entry_price
                    
                    if stop_distance > 0:
                        size = self.calculate_position_size(stop_distance, regime)
                        if size > 0:
                            self.sell(size=size)
                            self.log(f'{signal.value} - Size: {size:.2f}, Entry: {entry_price:.4f}, Stop: {stop_price:.4f}')
    
    def stop(self):
        """Called when backtesting is complete"""
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        
        self.log('=' * 50)
        self.log('STRATEGY PERFORMANCE SUMMARY')
        self.log('=' * 50)
        self.log(f'Starting Value: ${self.p.account_size:,.2f}')
        self.log(f'Ending Value: ${self.broker.getvalue():,.2f}')
        self.log(f'Total Return: {((self.broker.getvalue() / self.p.account_size - 1) * 100):.2f}%')
        self.log(f'Total Trades: {self.trade_count}')
        self.log(f'Winning Trades: {self.win_count}')
        self.log(f'Win Rate: {win_rate:.1f}%')
        self.log('=' * 50)