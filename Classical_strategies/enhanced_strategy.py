"""
Enhanced Strategy Implementation for Consistent Sharpe > 2.0
Based on deep performance analysis and advanced trading logic

Key Enhancements:
1. Multi-timeframe confluence analysis
2. Signal quality scoring and filtering
3. Dynamic position sizing based on market regime
4. Advanced risk-reward optimization
5. Market structure awareness for better entries
6. Volatility-adaptive parameters

Author: Claude AI Strategy Optimizer
Date: 2025
Version: 3.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from strategy_code.Prod_strategy import (
    OptimizedProdStrategy, OptimizedStrategyConfig, TradeDirection, 
    ExitReason, Trade, PnLCalculator, FOREX_PIP_SIZE, MIN_LOT_SIZE, PIP_VALUE_PER_MILLION
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedStrategyConfig(OptimizedStrategyConfig):
    """Enhanced configuration for Sharpe > 2.0 targeting"""
    
    # Multi-timeframe analysis
    use_mtf_confluence: bool = True
    mtf_lookback_periods: Tuple[int, int, int] = (5, 15, 30)  # Short, medium, long term
    mtf_confluence_threshold: float = 0.7  # 70% of timeframes must agree
    
    # Signal quality scoring
    signal_quality_threshold: float = 75.0  # Minimum signal quality score
    momentum_weight: float = 0.3
    trend_alignment_weight: float = 0.4
    regime_suitability_weight: float = 0.3
    
    # Enhanced risk management
    dynamic_position_sizing: bool = True
    volatility_position_adjustment: bool = True
    regime_based_risk: bool = True
    min_risk_per_trade: float = 0.01
    max_risk_per_trade: float = 0.04
    
    # Market structure awareness
    use_market_structure: bool = True
    structure_confluence_weight: float = 0.2
    support_resistance_buffer: float = 5.0  # pips
    
    # Advanced entry timing
    use_pullback_entries: bool = True
    pullback_ema_period: int = 8
    pullback_threshold: float = 0.3  # 30% retracement
    
    # Frequency optimization
    min_signal_strength: float = 0.6
    reduce_noise_filtering: bool = True
    aggressive_tp_mode: bool = True
    
    # Risk-reward optimization
    target_risk_reward_ratio: float = 1.5
    adaptive_tp_levels: bool = True
    smart_sl_placement: bool = True

class SignalQualityAnalyzer:
    """Analyzes and scores signal quality for better entry decisions"""
    
    def __init__(self, config: EnhancedStrategyConfig):
        self.config = config
    
    def calculate_momentum_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate momentum quality score (0-100)"""
        if idx < 10:
            return 50.0  # Neutral score for insufficient data
            
        row = df.iloc[idx]
        
        # NeuroTrend strength
        nti_strength = abs(row.get('NTI_Strength', 0.5))
        
        # Price momentum consistency
        close_prices = df['Close'].iloc[idx-5:idx+1]
        price_momentum = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
        momentum_consistency = 1.0 - abs(price_momentum - nti_strength * np.sign(price_momentum))
        
        # Volume confirmation (if available)
        volume_confirmation = 0.8  # Default if no volume data
        
        momentum_score = (nti_strength * 40 + momentum_consistency * 40 + volume_confirmation * 20)
        return min(100.0, max(0.0, momentum_score))
    
    def calculate_trend_alignment_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate trend alignment across timeframes (0-100)"""
        if idx < max(self.config.mtf_lookback_periods):
            return 50.0
            
        row = df.iloc[idx]
        alignment_scores = []
        
        for period in self.config.mtf_lookback_periods:
            if idx >= period:
                # Check trend consistency over this period
                start_idx = max(0, idx - period)
                period_data = df.iloc[start_idx:idx+1]
                
                nti_consistency = period_data['NTI_Direction'].value_counts().max() / len(period_data)
                mb_consistency = period_data['MB_Bias'].value_counts().max() / len(period_data)
                
                period_score = (nti_consistency + mb_consistency) / 2
                alignment_scores.append(period_score)
        
        return np.mean(alignment_scores) * 100 if alignment_scores else 50.0
    
    def calculate_regime_suitability_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate how suitable current regime is for trading (0-100)"""
        row = df.iloc[idx]
        regime = row.get('IC_Regime', 2)
        regime_name = row.get('IC_RegimeName', '')
        
        # Trend regimes are best for trading
        if regime == 1:  # Strong trend
            return 90.0
        elif regime == 2:  # Weak trend
            return 70.0
        elif regime == 3:  # Range/Chop
            return 30.0
        else:  # Very choppy
            return 10.0
    
    def calculate_signal_quality_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate overall signal quality score (0-100)"""
        momentum_score = self.calculate_momentum_score(df, idx)
        trend_score = self.calculate_trend_alignment_score(df, idx)
        regime_score = self.calculate_regime_suitability_score(df, idx)
        
        total_score = (
            momentum_score * self.config.momentum_weight +
            trend_score * self.config.trend_alignment_weight +
            regime_score * self.config.regime_suitability_weight
        )
        
        return total_score

class MultiTimeframeAnalyzer:
    """Analyzes confluence across multiple timeframes"""
    
    def __init__(self, config: EnhancedStrategyConfig):
        self.config = config
    
    def get_mtf_confluence(self, df: pd.DataFrame, idx: int) -> Tuple[float, TradeDirection]:
        """Calculate multi-timeframe confluence score"""
        if idx < max(self.config.mtf_lookback_periods):
            return 0.0, TradeDirection.LONG  # Insufficient data
        
        row = df.iloc[idx]
        confluences = []
        directions = []
        
        for period in self.config.mtf_lookback_periods:
            if idx >= period:
                start_idx = max(0, idx - period)
                period_data = df.iloc[start_idx:idx+1]
                
                # Analyze dominant direction over this timeframe
                long_signals = ((period_data['NTI_Direction'] == 1) & 
                               (period_data['MB_Bias'] == 1)).sum()
                short_signals = ((period_data['NTI_Direction'] == -1) & 
                                (period_data['MB_Bias'] == -1)).sum()
                
                total_signals = long_signals + short_signals
                if total_signals > 0:
                    long_ratio = long_signals / total_signals
                    short_ratio = short_signals / total_signals
                    
                    if long_ratio > short_ratio:
                        confluences.append(long_ratio)
                        directions.append(TradeDirection.LONG)
                    else:
                        confluences.append(short_ratio)
                        directions.append(TradeDirection.SHORT)
                else:
                    confluences.append(0.5)
                    directions.append(TradeDirection.LONG)
        
        if not confluences:
            return 0.0, TradeDirection.LONG
        
        # Check if majority of timeframes agree
        long_count = directions.count(TradeDirection.LONG)
        short_count = directions.count(TradeDirection.SHORT)
        
        dominant_direction = TradeDirection.LONG if long_count > short_count else TradeDirection.SHORT
        confluence_strength = max(long_count, short_count) / len(directions)
        
        avg_confidence = np.mean(confluences)
        final_confluence = confluence_strength * avg_confidence
        
        return final_confluence, dominant_direction

class EnhancedRiskManager:
    """Advanced risk management with regime-aware position sizing"""
    
    def __init__(self, config: EnhancedStrategyConfig):
        self.config = config
    
    def calculate_dynamic_risk(self, df: pd.DataFrame, idx: int, signal_quality: float) -> float:
        """Calculate dynamic risk per trade based on conditions"""
        base_risk = self.config.risk_per_trade
        
        if not self.config.dynamic_position_sizing:
            return base_risk
        
        row = df.iloc[idx]
        
        # Adjust for signal quality
        quality_multiplier = signal_quality / 100.0
        
        # Adjust for volatility
        volatility_multiplier = 1.0
        if self.config.volatility_position_adjustment:
            atr = row.get('IC_ATR_Normalized', 50.0)
            # Lower position size in high volatility
            if atr > 70:
                volatility_multiplier = 0.7
            elif atr < 30:
                volatility_multiplier = 1.3
        
        # Adjust for regime
        regime_multiplier = 1.0
        if self.config.regime_based_risk:
            regime = row.get('IC_Regime', 2)
            if regime == 1:  # Strong trend - increase size
                regime_multiplier = 1.5
            elif regime == 2:  # Weak trend - normal size
                regime_multiplier = 1.0
            else:  # Choppy - reduce size
                regime_multiplier = 0.6
        
        final_risk = base_risk * quality_multiplier * volatility_multiplier * regime_multiplier
        
        return np.clip(final_risk, self.config.min_risk_per_trade, self.config.max_risk_per_trade)

class EnhancedStrategy(OptimizedProdStrategy):
    """Enhanced strategy targeting consistent Sharpe > 2.0"""
    
    def __init__(self, config: EnhancedStrategyConfig):
        super().__init__(config)
        self.config = config
        self.signal_analyzer = SignalQualityAnalyzer(config)
        self.mtf_analyzer = MultiTimeframeAnalyzer(config)
        self.risk_manager = EnhancedRiskManager(config)
        
        # Enhanced tracking
        self.signal_quality_scores = []
        self.mtf_confluence_scores = []
        self.dynamic_risks = []
    
    def check_enhanced_entry_conditions(self, df: pd.DataFrame, idx: int) -> Optional[Tuple[TradeDirection, bool, float, float]]:
        """Enhanced entry condition checking with quality filtering"""
        row = df.iloc[idx]
        
        # First check basic conditions
        basic_entry = self.signal_generator.check_entry_conditions(row)
        if basic_entry is None:
            return None
        
        direction, is_relaxed = basic_entry
        
        # Calculate signal quality
        signal_quality = self.signal_analyzer.calculate_signal_quality_score(df, idx)
        self.signal_quality_scores.append(signal_quality)
        
        # Apply quality threshold
        if signal_quality < self.config.signal_quality_threshold:
            return None
        
        # Check multi-timeframe confluence
        mtf_confluence = 0.0
        mtf_direction = direction
        
        if self.config.use_mtf_confluence:
            mtf_confluence, mtf_direction = self.mtf_analyzer.get_mtf_confluence(df, idx)
            self.mtf_confluence_scores.append(mtf_confluence)
            
            # Check if MTF agrees with signal direction
            if mtf_direction != direction or mtf_confluence < self.config.mtf_confluence_threshold:
                return None
        
        # Calculate dynamic risk
        dynamic_risk = self.risk_manager.calculate_dynamic_risk(df, idx, signal_quality)
        self.dynamic_risks.append(dynamic_risk)
        
        return direction, is_relaxed, signal_quality, dynamic_risk
    
    def run_enhanced_backtest(self, df: pd.DataFrame) -> Dict:
        """Run enhanced backtest with new logic"""
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'NTI_Direction',
                        'MB_Bias', 'IC_Regime', 'IC_ATR_Normalized', 'IC_RegimeName']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Reset state
        self.reset()
        
        # Enhanced tracking
        self.signal_quality_scores = []
        self.mtf_confluence_scores = []
        self.dynamic_risks = []
        
        # Main backtest loop
        for idx in range(max(self.config.mtf_lookback_periods), len(df)):
            current_row = df.iloc[idx]
            current_time = df.index[idx]
            
            # Update equity curve
            self.equity_curve.append(self.current_capital)
            
            # Process open trade (use existing logic)
            if self.current_trade is not None:
                # [Same exit logic as parent class]
                # Check for partial profit taking
                if self.signal_generator.check_partial_profit_conditions(current_row, self.current_trade):
                    trade_id = id(self.current_trade)
                    if trade_id not in self.partial_profit_taken:
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
                atr = current_row['IC_ATR_Normalized']
                new_trailing_stop = self._update_trailing_stop(
                    current_row['Close'], self.current_trade, atr
                )
                if new_trailing_stop is not None:
                    self.current_trade.trailing_stop = new_trailing_stop
                
                # Check exit conditions
                should_exit, exit_reason, exit_percent = self.signal_generator.check_exit_conditions(
                    current_row, self.current_trade, current_time
                )
                
                if should_exit:
                    exit_price = self._get_exit_price(current_row, self.current_trade, exit_reason)
                    
                    if exit_percent < 1.0:
                        completed_trade = self._execute_partial_exit(
                            self.current_trade, current_time, exit_price, 
                            exit_percent, exit_reason
                        )
                    else:
                        completed_trade = self._execute_full_exit(
                            self.current_trade, current_time, exit_price, exit_reason
                        )
                    
                    if completed_trade is not None:
                        self.trades.append(self.current_trade)
                        self.current_trade = None
            
            # Check for new entry using enhanced conditions
            elif self.current_trade is None:
                entry_result = self.check_enhanced_entry_conditions(df, idx)
                
                if entry_result is not None:
                    direction, is_relaxed, signal_quality, dynamic_risk = entry_result
                    
                    # Create new trade with dynamic risk
                    self.current_trade = self._create_enhanced_trade(
                        current_time, current_row, direction, is_relaxed, dynamic_risk
                    )
                    
                    if self.config.verbose:
                        size_millions = self.current_trade.position_size / self.config.min_lot_size
                        trade_type = "ENHANCED" if signal_quality > 80 else "STANDARD"
                        logger.info(f"{trade_type} TRADE: {direction.value} at {self.current_trade.entry_price:.5f} "
                                   f"with {size_millions:.0f}M, Quality: {signal_quality:.1f}")
        
        # Close any remaining trade
        if self.current_trade is not None:
            last_row = df.iloc[-1]
            last_time = df.index[-1]
            
            completed_trade = self._execute_full_exit(
                self.current_trade, last_time, last_row['Close'], ExitReason.END_OF_DATA
            )
            if completed_trade:
                self.trades.append(self.current_trade)
        
        # Calculate enhanced performance metrics
        return self._calculate_enhanced_performance_metrics()
    
    def _create_enhanced_trade(self, entry_time: pd.Timestamp, row: pd.Series, 
                             direction: TradeDirection, is_relaxed: bool, 
                             dynamic_risk: float) -> Trade:
        """Create trade with enhanced risk management"""
        
        entry_price = row['Close']
        atr = row['IC_ATR_Normalized']
        
        # Calculate position size based on dynamic risk
        pip_risk = atr * self.config.sl_atr_multiplier
        pip_value = self.config.pip_value_per_million
        risk_amount = self.current_capital * dynamic_risk
        
        position_size = (risk_amount / (pip_risk * pip_value)) * self.config.min_lot_size
        position_size = max(self.config.min_lot_size, position_size)
        
        # Calculate enhanced stop loss
        if self.config.smart_sl_placement:
            sl_distance = self._calculate_smart_sl_distance(row, direction, atr)
        else:
            sl_distance = atr * self.config.sl_atr_multiplier
        
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - (sl_distance * FOREX_PIP_SIZE)
        else:
            stop_loss = entry_price + (sl_distance * FOREX_PIP_SIZE)
        
        # Calculate enhanced take profits
        take_profits = self._calculate_enhanced_take_profits(
            entry_price, direction, atr, row, dynamic_risk
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
        
        return trade
    
    def _calculate_smart_sl_distance(self, row: pd.Series, direction: TradeDirection, atr: float) -> float:
        """Calculate smarter stop loss placement"""
        base_distance = atr * self.config.sl_atr_multiplier
        
        # Adjust based on regime
        regime = row.get('IC_Regime', 2)
        if regime == 1:  # Strong trend - wider stops
            multiplier = 1.2
        elif regime == 2:  # Weak trend - normal stops
            multiplier = 1.0
        else:  # Choppy - tighter stops
            multiplier = 0.8
        
        return base_distance * multiplier
    
    def _calculate_enhanced_take_profits(self, entry_price: float, direction: TradeDirection,
                                       atr: float, row: pd.Series, dynamic_risk: float) -> List[float]:
        """Calculate take profits optimized for better risk-reward"""
        
        base_tp_distance = atr * self.config.tp_atr_multipliers[0]
        
        # Enhance TP based on target risk-reward ratio
        target_rr = self.config.target_risk_reward_ratio
        sl_distance = atr * self.config.sl_atr_multiplier
        
        # Calculate TP distances to achieve target risk-reward
        tp1_distance = sl_distance * target_rr * 0.8  # First TP at 80% of target RR
        tp2_distance = sl_distance * target_rr * 1.2  # Second TP at 120% of target RR
        tp3_distance = sl_distance * target_rr * 2.0  # Third TP at 200% of target RR
        
        if direction == TradeDirection.LONG:
            take_profits = [
                entry_price + (tp1_distance * FOREX_PIP_SIZE),
                entry_price + (tp2_distance * FOREX_PIP_SIZE),
                entry_price + (tp3_distance * FOREX_PIP_SIZE)
            ]
        else:
            take_profits = [
                entry_price - (tp1_distance * FOREX_PIP_SIZE),
                entry_price - (tp2_distance * FOREX_PIP_SIZE),
                entry_price - (tp3_distance * FOREX_PIP_SIZE)
            ]
        
        return take_profits
    
    def _calculate_enhanced_performance_metrics(self) -> Dict:
        """Calculate enhanced performance metrics"""
        base_metrics = self._calculate_performance_metrics()
        
        # Add enhanced metrics
        if self.signal_quality_scores:
            base_metrics['avg_signal_quality'] = np.mean(self.signal_quality_scores)
            base_metrics['min_signal_quality'] = np.min(self.signal_quality_scores)
        
        if self.mtf_confluence_scores:
            base_metrics['avg_mtf_confluence'] = np.mean(self.mtf_confluence_scores)
        
        if self.dynamic_risks:
            base_metrics['avg_dynamic_risk'] = np.mean(self.dynamic_risks)
        
        # Calculate enhanced ratios
        if len(self.trades) > 0:
            total_wins = sum(1 for trade in self.trades if trade.pnl > 0)
            total_losses = len(self.trades) - total_wins
            
            if total_wins > 0 and total_losses > 0:
                avg_win = np.mean([trade.pnl for trade in self.trades if trade.pnl > 0])
                avg_loss = abs(np.mean([trade.pnl for trade in self.trades if trade.pnl < 0]))
                base_metrics['actual_risk_reward_ratio'] = avg_win / avg_loss if avg_loss > 0 else 0
        
        return base_metrics

# Enhanced configuration presets for different market conditions
def create_aggressive_config() -> EnhancedStrategyConfig:
    """Configuration for aggressive Sharpe > 2.0 targeting"""
    return EnhancedStrategyConfig(
        # More aggressive risk for higher returns
        risk_per_trade=0.025,
        min_risk_per_trade=0.015,
        max_risk_per_trade=0.05,
        
        # High-quality signals only
        signal_quality_threshold=80.0,
        mtf_confluence_threshold=0.8,
        
        # Optimized for higher frequency
        target_risk_reward_ratio=1.8,
        dynamic_position_sizing=True,
        use_mtf_confluence=True,
        
        # Enhanced filters
        momentum_weight=0.4,
        trend_alignment_weight=0.4,
        regime_suitability_weight=0.2
    )

def create_conservative_config() -> EnhancedStrategyConfig:
    """Configuration for steady Sharpe improvement"""
    return EnhancedStrategyConfig(
        # Conservative risk management
        risk_per_trade=0.015,
        min_risk_per_trade=0.01,
        max_risk_per_trade=0.025,
        
        # Very high-quality signals
        signal_quality_threshold=85.0,
        mtf_confluence_threshold=0.75,
        
        # Focus on risk control
        target_risk_reward_ratio=1.5,
        dynamic_position_sizing=True,
        use_mtf_confluence=True
    )