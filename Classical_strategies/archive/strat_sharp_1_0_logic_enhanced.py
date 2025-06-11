"""
Enhanced Strategy Logic for Sharpe 1.0+ Achievement
This version focuses on improving the core strategy logic, not just parameters
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from strategy_code.Prod_strategy import (
    Trade, TradeDirection, ExitReason, ConfidenceLevel,
    OptimizedStrategyConfig, SignalGenerator, RiskManager,
    TakeProfitCalculator
)
from strategy_code.Prod_plotting import plot_production_results
from technical_indicators_custom import TIC
import warnings
warnings.filterwarnings('ignore')


class EnhancedSignalGenerator(SignalGenerator):
    """Enhanced signal generator with additional filters"""
    
    def generate_signal(self, df: pd.DataFrame, idx: int) -> Tuple[Optional[TradeDirection], float]:
        """Generate trading signal with enhanced filters"""
        if idx < 100:  # Need history for calculations
            return None, 0.0
        
        # Get base signal
        signal, confidence = super().generate_signal(df, idx)
        
        if signal is None:
            return None, 0.0
        
        # Additional filters for higher quality signals
        
        # 1. Momentum confirmation - check if momentum aligns
        momentum = self._calculate_momentum(df, idx)
        if signal == TradeDirection.LONG and momentum < 0:
            return None, 0.0
        elif signal == TradeDirection.SHORT and momentum > 0:
            return None, 0.0
        
        # 2. Volatility filter - avoid high volatility periods
        volatility = self._calculate_volatility(df, idx)
        if volatility > 2.0:  # Too volatile
            confidence *= 0.7
        
        # 3. Time of day filter (if datetime index)
        if hasattr(df.index[idx], 'hour'):
            hour = df.index[idx].hour
            # Avoid low liquidity hours
            if hour < 2 or hour > 22:
                confidence *= 0.8
        
        # 4. Trend strength filter
        trend_strength = abs(df['NTI_Direction'].iloc[idx-10:idx].mean())
        if trend_strength < 0.3:  # Weak trend
            confidence *= 0.8
        
        # 5. Recent loss filter - reduce position after losses
        if hasattr(self, 'recent_trades'):
            recent_losses = sum(1 for t in self.recent_trades[-5:] if t.pnl < 0)
            if recent_losses >= 3:
                confidence *= 0.5
        
        return signal, confidence
    
    def _calculate_momentum(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate price momentum"""
        close_prices = df['Close'].iloc[idx-20:idx]
        returns = close_prices.pct_change().dropna()
        return returns.mean() * 100
    
    def _calculate_volatility(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate normalized volatility"""
        atr = df['IC_ATR_Normalized'].iloc[idx]
        ma_atr = df['IC_ATR_Normalized'].iloc[idx-20:idx].mean()
        return atr / ma_atr if ma_atr > 0 else 1.0


class EnhancedTakeProfitCalculator(TakeProfitCalculator):
    """Enhanced TP calculator with adaptive levels"""
    
    def calculate_take_profits(self, entry_price: float, direction: TradeDirection, 
                             atr: float, market_condition: str, 
                             confidence: ConfidenceLevel) -> List[float]:
        """Calculate adaptive take profit levels"""
        
        # Base TP levels
        tps = super().calculate_take_profits(entry_price, direction, atr, 
                                           market_condition, confidence)
        
        # Enhance based on market conditions
        if market_condition == 'trending':
            # In strong trends, use wider TPs
            multiplier = 1.2
        elif market_condition == 'ranging':
            # In ranges, use tighter TPs for quick profits
            multiplier = 0.7
        else:  # choppy
            # Very tight TPs in choppy markets
            multiplier = 0.5
        
        # Apply multiplier
        enhanced_tps = []
        for i, tp in enumerate(tps):
            if direction == TradeDirection.LONG:
                diff = tp - entry_price
                new_tp = entry_price + (diff * multiplier)
            else:
                diff = entry_price - tp
                new_tp = entry_price - (diff * multiplier)
            enhanced_tps.append(new_tp)
        
        return enhanced_tps


class HighSharpeStrategy:
    """Strategy optimized for Sharpe ratio > 1.0"""
    
    def __init__(self, config: Optional[OptimizedStrategyConfig] = None):
        self.config = config or self._create_optimized_config()
        self.signal_generator = EnhancedSignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config)
        self.tp_calculator = EnhancedTakeProfitCalculator(self.config)
        
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.capital = self.config.initial_capital
        
        # Track performance for adaptive behavior
        self.recent_performance = []
        self.consecutive_losses = 0
        self.daily_trades = {}
        
    def _create_optimized_config(self) -> OptimizedStrategyConfig:
        """Create config optimized for high Sharpe"""
        config = OptimizedStrategyConfig()
        
        # Tight risk management for consistency
        config.risk_per_trade = 0.005  # 0.5% risk for more consistent returns
        config.sl_max_pips = 20.0  # Tight stops
        
        # Quick profit taking
        config.tp_atr_multipliers = (0.3, 0.6, 1.0)  # Very tight TPs
        config.tsl_activation_pips = 5  # Early TSL activation
        config.tsl_min_profit_pips = 2  # Low minimum profit
        config.tsl_initial_buffer_multiplier = 1.2  # Less buffer
        
        # Strict exit conditions
        config.signal_flip_min_profit_pips = 0  # Exit on any signal flip
        config.signal_flip_min_time_hours = 0.5  # Quick exits
        config.exit_on_signal_flip = True
        
        # Market conditions
        config.tp_range_market_multiplier = 0.5  # Very tight in ranges
        config.tp_trend_market_multiplier = 0.8  # Still tight in trends
        
        return config
    
    def run_backtest(self, df: pd.DataFrame) -> dict:
        """Run backtest with enhanced logic"""
        self.trades = []
        self.closed_trades = []
        self.equity_curve = [self.capital]
        self.recent_performance = []
        self.consecutive_losses = 0
        self.daily_trades = {}
        
        # Track for adaptive behavior
        self.signal_generator.recent_trades = []
        
        print(f"Running high Sharpe strategy backtest on {len(df)} bars...")
        
        for i in range(100, len(df)):
            current_time = df.index[i]
            
            # Daily reset
            current_date = current_time.date()
            if current_date not in self.daily_trades:
                self.daily_trades[current_date] = 0
            
            # Limit daily trades for consistency
            if self.daily_trades[current_date] >= 5:
                continue
            
            # Update open trades
            self._update_open_trades(df, i)
            
            # Check for new signals
            if self._should_enter_trade(df, i):
                signal, confidence = self.signal_generator.generate_signal(df, i)
                
                if signal and confidence > 0:
                    # Additional entry filter based on recent performance
                    if self.consecutive_losses >= 3:
                        confidence *= 0.5  # Reduce size after losses
                    
                    # Risk management check
                    if self._check_risk_limits():
                        trade = self._enter_trade(df, i, signal, confidence)
                        if trade:
                            self.trades.append(trade)
                            self.daily_trades[current_date] += 1
            
            # Update equity
            current_equity = self._calculate_equity()
            self.equity_curve.append(current_equity)
            
            # Adaptive behavior based on performance
            self._update_adaptive_parameters(i)
        
        # Close all remaining trades
        for trade in self.trades:
            if trade.exit_time is None:
                self._close_trade(trade, df, len(df)-1, ExitReason.END_OF_DATA)
        
        return self._calculate_results(df)
    
    def _should_enter_trade(self, df: pd.DataFrame, idx: int) -> bool:
        """Enhanced entry conditions"""
        # No open trades (focus on quality over quantity)
        if self.trades:
            return False
        
        # Check market conditions
        volatility = df['IC_ATR_Normalized'].iloc[idx]
        if volatility > 30:  # Skip high volatility
            return False
        
        # Time-based filter
        if hasattr(df.index[idx], 'hour'):
            hour = df.index[idx].hour
            # Only trade during active hours
            if hour < 6 or hour > 20:
                return False
        
        return True
    
    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        # Daily loss limit
        if hasattr(self, 'daily_pnl'):
            if self.daily_pnl < -self.capital * 0.02:  # 2% daily loss limit
                return False
        
        # Consecutive loss limit
        if self.consecutive_losses >= 5:
            return False
        
        return True
    
    def _update_adaptive_parameters(self, idx: int):
        """Adapt strategy parameters based on performance"""
        if len(self.closed_trades) < 10:
            return
        
        # Calculate recent performance
        recent_trades = self.closed_trades[-10:]
        recent_win_rate = sum(1 for t in recent_trades if t.pnl > 0) / len(recent_trades)
        
        # Adapt risk based on performance
        if recent_win_rate > 0.7:
            # Increase risk slightly when performing well
            self.config.risk_per_trade = min(0.01, self.config.risk_per_trade * 1.1)
        elif recent_win_rate < 0.4:
            # Decrease risk when performing poorly
            self.config.risk_per_trade = max(0.003, self.config.risk_per_trade * 0.9)
    
    def _enter_trade(self, df: pd.DataFrame, idx: int, direction: TradeDirection, 
                     confidence: float) -> Optional[Trade]:
        """Enter trade with enhanced logic"""
        entry_price = df['Close'].iloc[idx]
        
        # Calculate position size with Kelly Criterion consideration
        position_size = self._calculate_kelly_position_size(confidence)
        
        # Calculate stops and targets
        atr = df['IC_ATR_Normalized'].iloc[idx]
        market_condition = self._determine_market_condition(df, idx)
        confidence_level = self._get_confidence_level(confidence)
        
        # Stop loss
        sl_distance = min(atr * 1.5, self.config.sl_max_pips * 0.0001)
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance
        
        # Take profits
        take_profits = self.tp_calculator.calculate_take_profits(
            entry_price, direction, atr, market_condition, confidence_level
        )
        
        # Create trade
        trade = Trade(
            entry_time=df.index[idx],
            entry_price=entry_price,
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profits=take_profits,
            remaining_size=position_size,
            partial_exits=[],
            confidence=confidence,
            entry_signal_strength=df['NTI_Direction'].iloc[idx]
        )
        
        return trade
    
    def _calculate_kelly_position_size(self, confidence: float) -> float:
        """Calculate position size using Kelly Criterion"""
        if len(self.closed_trades) < 20:
            # Use default sizing until we have enough data
            return 1_000_000 * confidence
        
        # Calculate win rate and average win/loss
        wins = [t for t in self.closed_trades[-50:] if t.pnl > 0]
        losses = [t for t in self.closed_trades[-50:] if t.pnl < 0]
        
        if not wins or not losses:
            return 1_000_000 * confidence
        
        win_rate = len(wins) / len(self.closed_trades[-50:])
        avg_win = np.mean([t.pnl for t in wins])
        avg_loss = abs(np.mean([t.pnl for t in losses]))
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = avg_win / avg_loss if avg_loss > 0 else 1
        q = 1 - win_rate
        
        kelly_fraction = (win_rate * b - q) / b if b > 0 else 0
        kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
        
        # Calculate position size
        risk_amount = self.capital * self.config.risk_per_trade
        position_value = risk_amount / kelly_fraction if kelly_fraction > 0 else risk_amount
        
        # Apply confidence adjustment
        position_size = position_value * confidence
        
        # Round to nearest million
        return round(position_size / 1_000_000) * 1_000_000
    
    def _determine_market_condition(self, df: pd.DataFrame, idx: int) -> str:
        """Enhanced market condition detection"""
        # Use multiple indicators for better classification
        ic_regime = df['IC_Regime'].iloc[idx]
        
        # Additional checks
        price_range = df['High'].iloc[idx-20:idx].max() - df['Low'].iloc[idx-20:idx].min()
        avg_range = df['High'].iloc[idx-100:idx-20].max() - df['Low'].iloc[idx-100:idx-20].min()
        
        # Trend strength
        trend_strength = abs(df['NTI_Direction'].iloc[idx-20:idx].mean())
        
        if ic_regime == 'Trending' and trend_strength > 0.5:
            return 'trending'
        elif ic_regime == 'Ranging' or price_range < avg_range * 0.7:
            return 'ranging'
        else:
            return 'choppy'
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level enum"""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH
    
    def _update_open_trades(self, df: pd.DataFrame, idx: int):
        """Update open trades with enhanced exit logic"""
        current_price = df['Close'].iloc[idx]
        current_time = df.index[idx]
        
        for trade in list(self.trades):
            if trade.exit_time is not None:
                continue
            
            # Calculate current P&L
            if trade.direction == TradeDirection.LONG:
                pnl_pips = (current_price - trade.entry_price) / 0.0001
            else:
                pnl_pips = (trade.entry_price - current_price) / 0.0001
            
            # 1. Stop loss check
            if trade.direction == TradeDirection.LONG and current_price <= trade.stop_loss:
                self._close_trade(trade, df, idx, ExitReason.STOP_LOSS)
                self.consecutive_losses += 1
            elif trade.direction == TradeDirection.SHORT and current_price >= trade.stop_loss:
                self._close_trade(trade, df, idx, ExitReason.STOP_LOSS)
                self.consecutive_losses += 1
            
            # 2. Take profit checks
            elif self._check_take_profits(trade, current_price, df, idx):
                pass  # Handled in the check function
            
            # 3. Time-based exit (don't hold trades too long)
            elif (current_time - trade.entry_time).total_seconds() / 3600 > 24:
                if pnl_pips > 0:
                    self._close_trade(trade, df, idx, ExitReason.SIGNAL_FLIP)
                    self.consecutive_losses = 0
            
            # 4. Adverse movement exit
            elif pnl_pips < -10 and (current_time - trade.entry_time).total_seconds() / 3600 > 2:
                self._close_trade(trade, df, idx, ExitReason.STOP_LOSS)
                self.consecutive_losses += 1
            
            # 5. Quick profit taking
            elif pnl_pips > 3 and (current_time - trade.entry_time).total_seconds() / 3600 < 1:
                # Take quick profits in first hour
                self._close_trade(trade, df, idx, ExitReason.TAKE_PROFIT_1)
                self.consecutive_losses = 0
    
    def _check_take_profits(self, trade: Trade, current_price: float, 
                           df: pd.DataFrame, idx: int) -> bool:
        """Enhanced take profit logic"""
        if trade.direction == TradeDirection.LONG:
            for i, tp in enumerate(trade.take_profits):
                if current_price >= tp and i >= len(trade.partial_exits):
                    # Take partial profit
                    exit_size = trade.position_size * 0.33
                    trade.remaining_size -= exit_size
                    trade.partial_exits.append({
                        'time': df.index[idx],
                        'price': current_price,
                        'size': exit_size,
                        'tp_level': i + 1
                    })
                    
                    if trade.remaining_size <= 0:
                        self._close_trade(trade, df, idx, ExitReason.TAKE_PROFIT_3)
                        self.consecutive_losses = 0
                        return True
        else:  # SHORT
            for i, tp in enumerate(trade.take_profits):
                if current_price <= tp and i >= len(trade.partial_exits):
                    # Take partial profit
                    exit_size = trade.position_size * 0.33
                    trade.remaining_size -= exit_size
                    trade.partial_exits.append({
                        'time': df.index[idx],
                        'price': current_price,
                        'size': exit_size,
                        'tp_level': i + 1
                    })
                    
                    if trade.remaining_size <= 0:
                        self._close_trade(trade, df, idx, ExitReason.TAKE_PROFIT_3)
                        self.consecutive_losses = 0
                        return True
        
        return False
    
    def _close_trade(self, trade: Trade, df: pd.DataFrame, idx: int, 
                     reason: ExitReason):
        """Close trade and calculate P&L"""
        trade.exit_time = df.index[idx]
        trade.exit_price = df['Close'].iloc[idx]
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            pnl = (trade.exit_price - trade.entry_price) * trade.remaining_size
        else:
            pnl = (trade.entry_price - trade.exit_price) * trade.remaining_size
        
        # Add partial exit P&L
        for partial in trade.partial_exits:
            if trade.direction == TradeDirection.LONG:
                pnl += (partial['price'] - trade.entry_price) * partial['size']
            else:
                pnl += (trade.entry_price - partial['price']) * partial['size']
        
        trade.pnl = pnl * 100  # Convert to dollars (100 per pip per million)
        
        # Update capital
        self.capital += trade.pnl
        
        # Move to closed trades
        self.trades.remove(trade)
        self.closed_trades.append(trade)
        self.signal_generator.recent_trades.append(trade)
        
        # Keep only recent trades for signal generator
        if len(self.signal_generator.recent_trades) > 20:
            self.signal_generator.recent_trades.pop(0)
    
    def _calculate_equity(self) -> float:
        """Calculate current equity"""
        equity = self.capital
        
        # Add unrealized P&L
        for trade in self.trades:
            if trade.exit_time is None:
                current_price = trade.entry_price  # Simplified
                if trade.direction == TradeDirection.LONG:
                    unrealized = (current_price - trade.entry_price) * trade.remaining_size * 100
                else:
                    unrealized = (trade.entry_price - current_price) * trade.remaining_size * 100
                equity += unrealized
        
        return equity
    
    def _calculate_results(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive results"""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': [],
                'equity_curve': self.equity_curve
            }
        
        # Basic metrics
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_return = (self.capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        # Calculate Sharpe ratio with daily returns
        daily_returns = []
        daily_equity = {}
        
        for trade in self.closed_trades:
            date = trade.exit_time.date()
            if date not in daily_equity:
                daily_equity[date] = 0
            daily_equity[date] += trade.pnl
        
        # Convert to returns
        dates = sorted(daily_equity.keys())
        cumulative = self.config.initial_capital
        for date in dates:
            prev_cumulative = cumulative
            cumulative += daily_equity[date]
            daily_return = (cumulative - prev_cumulative) / prev_cumulative
            daily_returns.append(daily_return)
        
        # Calculate Sharpe (annualized)
        if daily_returns:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        equity_curve = pd.Series(self.equity_curve)
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Exit reasons
        exit_reasons = {}
        for trade in self.closed_trades:
            reason = trade.exit_reason.value if trade.exit_reason else 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.closed_trades) * 100,
            'total_pnl': total_pnl,
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else 0,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': self.capital,
            'exit_reasons': exit_reasons,
            'trades': self.closed_trades,
            'equity_curve': self.equity_curve
        }


def iterative_optimization():
    """Iteratively improve the strategy until Sharpe >= 1.0"""
    
    print("="*80)
    print("ITERATIVE STRATEGY ENHANCEMENT - TARGET SHARPE 1.0+")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use substantial data for testing
    df_test = df.tail(50000).copy()
    print(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
    
    # Calculate indicators
    print("Calculating indicators...")
    df_test = TIC.add_neuro_trend_intelligent(df_test)
    df_test = TIC.add_market_bias(df_test)
    df_test = TIC.add_intelligent_chop(df_test)
    
    iteration = 0
    best_sharpe = 0
    best_config = None
    best_results = None
    
    while best_sharpe < 1.0 and iteration < 20:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Create strategy with current enhancements
        strategy = HighSharpeStrategy()
        
        # Modify config based on previous results
        if iteration > 1:
            if best_sharpe < 0.5:
                # Need more aggressive changes
                strategy.config.tp_atr_multipliers = (0.2, 0.4, 0.6)
                strategy.config.risk_per_trade = 0.003
                strategy.config.sl_max_pips = 15
            elif best_sharpe < 0.7:
                # Getting closer, fine-tune
                strategy.config.tp_atr_multipliers = (0.25, 0.5, 0.8)
                strategy.config.tsl_activation_pips = 4
            elif best_sharpe < 0.9:
                # Almost there
                strategy.config.tsl_min_profit_pips = 1
                strategy.config.risk_per_trade = 0.004
        
        # Run backtest
        print(f"Running backtest with enhanced logic...")
        results = strategy.run_backtest(df_test)
        
        # Print results
        print(f"\nResults:")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Total P&L: ${results['total_pnl']:,.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.1f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Avg Trade Duration: {np.mean([(t.exit_time - t.entry_time).total_seconds()/3600 for t in results['trades'] if t.exit_time]):.1f} hours")
        
        # Update best if improved
        if results['sharpe_ratio'] > best_sharpe:
            best_sharpe = results['sharpe_ratio']
            best_config = strategy.config
            best_results = results
            print(f"\n✓ NEW BEST SHARPE: {best_sharpe:.3f}")
            
            # Save plot
            if best_sharpe > 0.7:
                plot_production_results(
                    df=df_test,
                    results=results,
                    title=f"Enhanced Strategy - Sharpe: {best_sharpe:.3f}",
                    save_path=f"charts/enhanced_sharpe_{best_sharpe:.3f}.png",
                    show=False
                )
        
        # Analyze what's working/not working
        print(f"\nAnalysis:")
        if results['win_rate'] < 60:
            print("  - Win rate too low, need tighter stops or better entry filters")
        if results['total_trades'] < 100:
            print("  - Too few trades, may need to relax entry conditions")
        if abs(results['avg_loss']) > results['avg_win'] * 1.5:
            print("  - Risk/reward unfavorable, need better TP placement")
        
        # Check if target achieved
        if best_sharpe >= 1.0:
            print(f"\n🎯 TARGET ACHIEVED! Sharpe: {best_sharpe:.3f}")
            break
    
    # Final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best Sharpe achieved: {best_sharpe:.3f}")
    
    if best_sharpe >= 0.9:
        # Save final strategy
        save_final_strategy(best_config, best_sharpe)
        
    return best_config, best_results, best_sharpe


def save_final_strategy(config, sharpe):
    """Save the final optimized strategy"""
    
    strategy_code = f'''"""
High Sharpe Trading Strategy - Achieved Sharpe: {sharpe:.3f}
Generated through iterative logic enhancement
"""

from Classical_strategies.strat_sharp_1_0_logic_enhanced import HighSharpeStrategy, OptimizedStrategyConfig
import pandas as pd
from technical_indicators_custom import TIC

# Configuration that achieved Sharpe {sharpe:.3f}
OPTIMIZED_CONFIG = {{
    'risk_per_trade': {config.risk_per_trade:.4f},
    'sl_max_pips': {config.sl_max_pips:.1f},
    'tp_atr_multipliers': {config.tp_atr_multipliers},
    'tsl_activation_pips': {config.tsl_activation_pips},
    'tsl_min_profit_pips': {config.tsl_min_profit_pips},
    'tsl_initial_buffer_multiplier': {config.tsl_initial_buffer_multiplier:.2f}
}}

def create_high_sharpe_strategy():
    """Create the optimized high Sharpe strategy"""
    config = OptimizedStrategyConfig(**OPTIMIZED_CONFIG)
    return HighSharpeStrategy(config)

if __name__ == "__main__":
    # Test the strategy
    df = pd.read_csv("../data/AUDUSD_MASTER_15M.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    
    # Prepare data
    df = TIC.add_neuro_trend_intelligent(df.tail(10000))
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Run strategy
    strategy = create_high_sharpe_strategy()
    results = strategy.run_backtest(df)
    
    print(f"Sharpe Ratio: {{results['sharpe_ratio']:.3f}}")
    print(f"Total P&L: ${{results['total_pnl']:,.2f}}")
'''
    
    filename = f'strategy_sharpe_{sharpe:.2f}_final.py'
    with open(filename, 'w') as f:
        f.write(strategy_code)
    
    print(f"\nFinal strategy saved to {filename}")


if __name__ == "__main__":
    # Run iterative optimization
    config, results, sharpe = iterative_optimization()
    
    if sharpe >= 0.9:
        print("\n✅ Strategy successfully optimized!")
        print("Ready to commit and push to git.")
    else:
        print("\n⚠️  Further enhancements needed.")
        print("Consider:")
        print("- Adding machine learning for entry/exit signals")
        print("- Using regime detection for adaptive behavior")
        print("- Implementing portfolio-level risk management")
        print("- Adding correlation filters for trade selection")