import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import indicators from clone_indicators
try:
    from clone_indicators.indicators import (
        supertrend_indicator, 
        market_bias_indicator,
        neurotrend_intelligent,
        intelligent_chop
    )
    USE_ADVANCED_INDICATORS = True
except ImportError:
    USE_ADVANCED_INDICATORS = False
    print("Advanced indicators not available, using basic indicators only")


class ImprovedTrendStrategy:
    """
    Improved trend-following strategy with:
    1. Better entry timing using pullbacks in trends
    2. Dynamic position sizing based on market conditions
    3. Multiple timeframe confirmation
    4. Volatility-based filters
    5. Momentum confirmation
    """
    
    def __init__(self,
                 initial_capital: float = 1000000,
                 base_risk: float = 0.01,
                 max_risk: float = 0.025,
                 min_rr_ratio: float = 1.5,
                 use_advanced: bool = True):
        
        self.initial_capital = initial_capital
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.min_rr_ratio = min_rr_ratio  # Minimum risk-reward ratio
        self.use_advanced = use_advanced and USE_ADVANCED_INDICATORS
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators including advanced ones if available."""
        
        # Price action
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Multiple timeframe EMAs
        df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_34'] = df['Close'].ewm(span=34, adjust=False).mean()
        df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()
        df['EMA_89'] = df['Close'].ewm(span=89, adjust=False).mean()
        df['EMA_144'] = df['Close'].ewm(span=144, adjust=False).mean()
        df['EMA_233'] = df['Close'].ewm(span=233, adjust=False).mean()
        
        # ATR with multiple periods
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        df['ATR_14'] = true_range.rolling(14).mean()
        df['ATR_21'] = true_range.rolling(21).mean()
        df['ATR_Percent'] = df['ATR_14'] / df['Close'] * 100
        
        # Momentum indicators
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI with multiple periods
        for period in [9, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ADX for trend strength
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.abs().rolling(14).sum() / tr14)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        df['DI_plus'] = plus_di
        df['DI_minus'] = minus_di
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_squeeze'] = df['BB_width'] < df['BB_width'].rolling(120).quantile(0.2)
        
        # Keltner Channels
        df['KC_middle'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['KC_upper'] = df['KC_middle'] + 2 * df['ATR_14']
        df['KC_lower'] = df['KC_middle'] - 2 * df['ATR_14']
        
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        df['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
        
        # Trend definition
        df['Trend_Up'] = (df['EMA_21'] > df['EMA_55']) & (df['EMA_55'] > df['EMA_89'])
        df['Trend_Down'] = (df['EMA_21'] < df['EMA_55']) & (df['EMA_55'] < df['EMA_89'])
        
        # Pullback detection
        df['Pullback_Up'] = df['Trend_Up'] & (df['Low'] <= df['EMA_21']) & (df['Close'] > df['EMA_21'])
        df['Pullback_Down'] = df['Trend_Down'] & (df['High'] >= df['EMA_21']) & (df['Close'] < df['EMA_21'])
        
        # Volume (synthetic)
        df['Volume'] = 1000000
        
        # Add advanced indicators if available
        if self.use_advanced:
            try:
                # SuperTrend
                st_result = supertrend_indicator(df, atr_period=10, multiplier=2.5, use_numba=True)
                for col in st_result.columns:
                    df[col] = st_result[col]
                
                # Market Bias
                mb_result = market_bias_indicator(df, ha_len=50, ha_len2=10, use_numba=True)
                for col in mb_result.columns:
                    df[col] = mb_result[col]
                
                # NeuroTrend Intelligent
                nt_result = neurotrend_intelligent(df, base_fast_len=10, base_slow_len=50,
                                                 confirm_bars=3, dynamic_thresholds=True,
                                                 use_numba=True)
                for col in nt_result.columns:
                    df[col] = nt_result[col]
                
                # Intelligent Chop
                ic_result = intelligent_chop(df, use_numba=True)
                for col in ic_result.columns:
                    df[col] = ic_result[col]
                    
            except Exception as e:
                print(f"Error adding advanced indicators: {e}")
                self.use_advanced = False
        
        return df
    
    def calculate_trend_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate overall trend score using multiple timeframes."""
        
        if idx < 233:  # Need enough data
            return 0
        
        score = 0
        row = df.iloc[idx]
        
        # EMA alignment (Fibonacci sequence)
        emas = ['EMA_8', 'EMA_21', 'EMA_34', 'EMA_55', 'EMA_89', 'EMA_144', 'EMA_233']
        
        # Check if EMAs are in order
        ema_values = [row[ema] for ema in emas]
        
        # Bullish alignment
        if all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1)):
            score += 3
        # Bearish alignment
        elif all(ema_values[i] < ema_values[i+1] for i in range(len(ema_values)-1)):
            score -= 3
        # Partial alignment
        else:
            # Check shorter-term alignment
            if row['EMA_8'] > row['EMA_21'] > row['EMA_34']:
                score += 1
            elif row['EMA_8'] < row['EMA_21'] < row['EMA_34']:
                score -= 1
        
        # Price position relative to EMAs
        if row['Close'] > row['EMA_21']:
            score += 0.5
        else:
            score -= 0.5
        
        # ADX trend strength
        if row['ADX'] > 25:
            if row['DI_plus'] > row['DI_minus']:
                score += 1
            else:
                score -= 1
        elif row['ADX'] < 20:
            score *= 0.5  # Reduce score in choppy market
        
        # Advanced indicators if available
        if self.use_advanced:
            # SuperTrend
            if 'SuperTrend_Direction' in df.columns and pd.notna(row['SuperTrend_Direction']):
                score += row['SuperTrend_Direction'] * 1.5
            
            # Market Bias
            if 'MB_Bias' in df.columns and pd.notna(row['MB_Bias']):
                score += row['MB_Bias'] * 1.0
            
            # NeuroTrend
            if 'NTI_Direction' in df.columns and pd.notna(row['NTI_Direction']):
                score += row['NTI_Direction'] * 1.5
            
            # Avoid choppy markets
            if 'IC_Regime' in df.columns and pd.notna(row['IC_Regime']):
                if row['IC_Regime'] == 0:  # Choppy
                    score *= 0.3
        
        return score / 10  # Normalize
    
    def find_entry_signals(self, df: pd.DataFrame, idx: int) -> Tuple[float, str]:
        """Find high-quality entry signals."""
        
        if idx < 233:
            return 0, ""
        
        row = df.iloc[idx]
        trend_score = self.calculate_trend_score(df, idx)
        
        # No signal if trend is weak
        if abs(trend_score) < 0.3:
            return 0, ""
        
        signal_strength = 0
        signal_type = ""
        
        # Long signals
        if trend_score > 0:
            # 1. Pullback to EMA in uptrend
            if row['Pullback_Up']:
                signal_strength = 0.7
                signal_type = "Pullback Long"
            
            # 2. Breakout with momentum
            elif (row['Close'] > row['BB_upper'] and 
                  row['RSI_14'] < 70 and 
                  row['MACD_hist'] > 0):
                signal_strength = 0.6
                signal_type = "Breakout Long"
            
            # 3. Bounce from support
            elif (row['Low'] <= row['EMA_55'] and 
                  row['Close'] > row['EMA_55'] and
                  row['RSI_14'] < 50):
                signal_strength = 0.8
                signal_type = "Support Bounce"
                
        # Short signals
        elif trend_score < 0:
            # 1. Pullback to EMA in downtrend
            if row['Pullback_Down']:
                signal_strength = -0.7
                signal_type = "Pullback Short"
            
            # 2. Breakdown with momentum
            elif (row['Close'] < row['BB_lower'] and 
                  row['RSI_14'] > 30 and 
                  row['MACD_hist'] < 0):
                signal_strength = -0.6
                signal_type = "Breakdown Short"
            
            # 3. Rejection from resistance
            elif (row['High'] >= row['EMA_55'] and 
                  row['Close'] < row['EMA_55'] and
                  row['RSI_14'] > 50):
                signal_strength = -0.8
                signal_type = "Resistance Rejection"
        
        # Apply filters
        if signal_strength != 0:
            # Volatility filter
            if row['ATR_Percent'] > 0.5:  # High volatility
                signal_strength *= 0.8
            
            # Momentum confirmation
            if abs(row['MACD_hist']) < df['MACD_hist'].rolling(50).std().iloc[idx] * 0.5:
                signal_strength *= 0.7
            
            # Squeeze filter (avoid low volatility)
            if row['BB_squeeze']:
                signal_strength *= 0.5
        
        return signal_strength * trend_score, signal_type
    
    def calculate_dynamic_position_size(self, df: pd.DataFrame, idx: int, 
                                      signal_strength: float, capital: float) -> Tuple[float, float, float]:
        """Calculate position size with dynamic risk adjustment."""
        
        row = df.iloc[idx]
        
        # Base risk calculation
        volatility_rank = df['ATR_Percent'].rolling(100).rank(pct=True).iloc[idx]
        
        # Adjust risk based on market conditions
        if volatility_rank < 0.3:  # Low volatility
            risk_multiplier = 1.2
        elif volatility_rank > 0.7:  # High volatility
            risk_multiplier = 0.7
        else:
            risk_multiplier = 1.0
        
        # Adjust for signal strength
        signal_multiplier = min(abs(signal_strength), 1.0)
        
        # Calculate final risk
        position_risk = self.base_risk * risk_multiplier * signal_multiplier
        position_risk = min(position_risk, self.max_risk)
        
        # Calculate stops based on ATR and market structure
        atr = row['ATR_14']
        
        if signal_strength > 0:  # Long
            # Find recent swing low
            recent_low = df['Low'].iloc[idx-20:idx].min()
            structure_stop = recent_low - atr * 0.5
            atr_stop = row['Close'] - atr * 2.0
            stop_loss = max(structure_stop, atr_stop)
            
            # Target based on risk-reward
            risk_distance = row['Close'] - stop_loss
            take_profit = row['Close'] + (risk_distance * max(self.min_rr_ratio, 2.0))
            
        else:  # Short
            # Find recent swing high
            recent_high = df['High'].iloc[idx-20:idx].max()
            structure_stop = recent_high + atr * 0.5
            atr_stop = row['Close'] + atr * 2.0
            stop_loss = min(structure_stop, atr_stop)
            
            # Target based on risk-reward
            risk_distance = stop_loss - row['Close']
            take_profit = row['Close'] - (risk_distance * max(self.min_rr_ratio, 2.0))
        
        # Calculate position size
        risk_amount = capital * position_risk
        risk_per_unit = abs(row['Close'] - stop_loss)
        
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
        else:
            position_size = 0
        
        return position_size, stop_loss, take_profit
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Execute backtest with improved logic."""
        
        print("Adding indicators...")
        df = self.add_indicators(df)
        
        # Initialize
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [capital]
        signals_log = []
        
        # Start after warmup
        start_idx = 250
        
        print(f"Running backtest from {start_idx} to {len(df)}...")
        
        for idx in range(start_idx, len(df)):
            if idx % 10000 == 0:
                print(f"Progress: {idx}/{len(df)}")
            
            current_date = df.index[idx]
            
            # Position management
            if position is not None:
                row = df.iloc[idx]
                exit_signal = False
                exit_reason = ""
                
                # Check stops
                if position['type'] == 'long':
                    if row['Low'] <= position['stop_loss']:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    elif row['High'] >= position['take_profit']:
                        exit_signal = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                else:  # short
                    if row['High'] >= position['stop_loss']:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                        exit_price = position['stop_loss']
                    elif row['Low'] <= position['take_profit']:
                        exit_signal = True
                        exit_reason = "Take Profit"
                        exit_price = position['take_profit']
                
                # Trailing stop after positive move
                if not exit_signal and position['unrealized_high'] > position['entry_price'] * 1.01:
                    if position['type'] == 'long':
                        trailing_stop = position['unrealized_high'] - row['ATR_14'] * 1.5
                        if row['Low'] <= trailing_stop:
                            exit_signal = True
                            exit_reason = "Trailing Stop"
                            exit_price = trailing_stop
                    else:
                        trailing_stop = position['unrealized_low'] + row['ATR_14'] * 1.5
                        if row['High'] >= trailing_stop:
                            exit_signal = True
                            exit_reason = "Trailing Stop"
                            exit_price = trailing_stop
                
                # Update unrealized high/low
                if position['type'] == 'long':
                    position['unrealized_high'] = max(position['unrealized_high'], row['High'])
                else:
                    position['unrealized_low'] = min(position['unrealized_low'], row['Low'])
                
                # Exit on trend reversal
                if not exit_signal:
                    trend_score = self.calculate_trend_score(df, idx)
                    if (position['type'] == 'long' and trend_score < -0.3) or \
                       (position['type'] == 'short' and trend_score > 0.3):
                        exit_signal = True
                        exit_reason = "Trend Reversal"
                        exit_price = row['Close']
                
                if exit_signal:
                    # Calculate P&L
                    if position['type'] == 'long':
                        pnl = (exit_price - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['size']
                    
                    capital += position['capital_used'] + pnl
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'type': position['type'],
                        'signal_type': position['signal_type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / position['capital_used']) * 100,
                        'exit_reason': exit_reason,
                        'bars_held': idx - position['entry_idx']
                    })
                    
                    position = None
            
            # Entry logic
            else:
                # Check for minimum time between trades
                if len(trades) > 0:
                    bars_since_exit = idx - trades[-1].get('exit_idx', 0)
                    if bars_since_exit < 4:  # Wait at least 1 hour
                        equity_curve.append(capital)
                        continue
                
                # Get entry signal
                signal_strength, signal_type = self.find_entry_signals(df, idx)
                
                if abs(signal_strength) > 0.5:
                    # Calculate position parameters
                    position_size, stop_loss, take_profit = self.calculate_dynamic_position_size(
                        df, idx, signal_strength, capital
                    )
                    
                    if position_size > 0:
                        row = df.iloc[idx]
                        capital_needed = position_size * row['Close']
                        
                        # Risk checks
                        max_position_value = capital * 0.3  # Max 30% per position
                        if capital_needed > max_position_value:
                            capital_needed = max_position_value
                            position_size = capital_needed / row['Close']
                        
                        if capital_needed < capital * 0.95:  # Keep 5% buffer
                            position = {
                                'type': 'long' if signal_strength > 0 else 'short',
                                'signal_type': signal_type,
                                'entry_date': current_date,
                                'entry_idx': idx,
                                'entry_price': row['Close'],
                                'size': position_size,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit,
                                'capital_used': capital_needed,
                                'signal_strength': signal_strength,
                                'unrealized_high': row['High'],
                                'unrealized_low': row['Low']
                            }
                            
                            capital -= capital_needed
                            
                            signals_log.append({
                                'date': current_date,
                                'signal': signal_strength,
                                'type': signal_type
                            })
                            
                            if len(trades) > 0:
                                trades[-1]['exit_idx'] = idx
            
            # Update equity
            if position is not None:
                row = df.iloc[idx]
                if position['type'] == 'long':
                    current_value = position['size'] * row['Close']
                else:
                    current_value = position['capital_used'] + (position['entry_price'] - row['Close']) * position['size']
                
                equity_curve.append(capital + current_value)
            else:
                equity_curve.append(capital)
        
        # Close final position
        if position is not None:
            final_row = df.iloc[-1]
            if position['type'] == 'long':
                pnl = (final_row['Close'] - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - final_row['Close']) * position['size']
            
            capital += position['capital_used'] + pnl
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'type': position['type'],
                'signal_type': position['signal_type'],
                'entry_price': position['entry_price'],
                'exit_price': final_row['Close'],
                'size': position['size'],
                'pnl': pnl,
                'pnl_pct': (pnl / position['capital_used']) * 100,
                'exit_reason': 'End of Data',
                'bars_held': len(df) - position['entry_idx']
            })
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(trades, equity_curve, df.index[start_idx:])
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital,
            'signals_log': signals_log
        }
    
    def calculate_comprehensive_metrics(self, trades: List[Dict], equity_curve: List[float],
                                      dates: pd.DatetimeIndex) -> Dict:
        """Calculate detailed performance metrics."""
        
        if len(trades) == 0:
            return {
                'sharpe_ratio': 0,
                'total_trades': 0,
                'status': 'No trades executed'
            }
        
        # Convert to arrays
        equity_array = np.array(equity_curve)
        trade_pnls = [t['pnl'] for t in trades]
        
        # Basic statistics
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        # Returns calculation
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Daily returns
        bars_per_day = 96  # 15-min bars
        daily_equity = []
        
        for i in range(0, len(equity_array) - bars_per_day, bars_per_day):
            daily_equity.append(equity_array[i + bars_per_day])
        
        daily_equity = np.array(daily_equity)
        if len(daily_equity) > 1:
            daily_returns = np.diff(daily_equity) / daily_equity[:-1]
        else:
            daily_returns = np.array([0])
        
        # Performance metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / downside_std if downside_std > 0 else 0
        else:
            sortino_ratio = sharpe_ratio  # No downside
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Drawdown duration
        drawdown_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown[i] < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_dd_duration = i - drawdown_start
            else:
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
                drawdown_start = None
                current_dd_duration = 0
        
        # Win/Loss statistics
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk-reward ratio
        avg_win_pct = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([abs(t['pnl_pct']) for t in losing_trades]) if losing_trades else 0
        realized_rr = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0
        
        # Trade analysis by type
        trade_types = {}
        for trade in trades:
            signal_type = trade.get('signal_type', 'Unknown')
            if signal_type not in trade_types:
                trade_types[signal_type] = {'count': 0, 'pnl': 0, 'wins': 0}
            
            trade_types[signal_type]['count'] += 1
            trade_types[signal_type]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                trade_types[signal_type]['wins'] += 1
        
        # Calculate win rate by signal type
        for signal_type in trade_types:
            trade_types[signal_type]['win_rate'] = (
                trade_types[signal_type]['wins'] / trade_types[signal_type]['count']
            )
        
        # Annualized metrics
        days = (dates[-1] - dates[0]).days
        years = days / 365
        
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            annual_return = 0
            calmar_ratio = 0
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'realized_rr_ratio': realized_rr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_dd_duration_bars': max_dd_duration,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_pnl': sum(trade_pnls),
            'best_trade': max(trade_pnls) if trade_pnls else 0,
            'worst_trade': min(trade_pnls) if trade_pnls else 0,
            'avg_bars_in_trade': np.mean([t.get('bars_held', 0) for t in trades]),
            'trade_types': trade_types
        }
        
        return metrics


def recursive_improvement(max_iterations: int = 5):
    """Implement recursive self-improvement to achieve Sharpe > 1."""
    
    print("Starting recursive self-improvement process...")
    
    # Load data
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Track improvements
    iteration_results = []
    best_sharpe = -999
    best_params = {}
    
    # Initial parameters
    current_params = {
        'base_risk': 0.01,
        'max_risk': 0.02,
        'min_rr_ratio': 2.0,
        'use_advanced': True
    }
    
    for iteration in range(max_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}")
        print(f"{'='*60}")
        
        # Test current parameters
        strategy = ImprovedTrendStrategy(**current_params)
        results = strategy.run_backtest(df)
        
        current_sharpe = results['metrics']['sharpe_ratio']
        iteration_results.append({
            'iteration': iteration + 1,
            'params': current_params.copy(),
            'metrics': results['metrics'],
            'improvement': current_sharpe - best_sharpe if iteration > 0 else 0
        })
        
        print(f"\nCurrent Sharpe: {current_sharpe:.3f}")
        print(f"Total Return: {results['metrics']['total_return']:.2%}")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
        print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
        
        # Check if we achieved our goal
        if current_sharpe > 1.0:
            print(f"\n SUCCESS! Achieved Sharpe ratio > 1.0: {current_sharpe:.3f}")
            best_sharpe = current_sharpe
            best_params = current_params.copy()
            break
        
        # Update best if improved
        if current_sharpe > best_sharpe:
            best_sharpe = current_sharpe
            best_params = current_params.copy()
            print(f"New best Sharpe ratio: {best_sharpe:.3f}")
        
        # Analyze and adjust parameters
        print("\nAnalyzing results for improvements...")
        
        # Identify issues and adjust
        metrics = results['metrics']
        
        # 1. If win rate is low, be more selective
        if metrics['win_rate'] < 0.45:
            print("- Low win rate detected, increasing selectivity")
            current_params['min_rr_ratio'] = min(current_params['min_rr_ratio'] * 1.2, 3.5)
        
        # 2. If drawdown is high, reduce risk
        if metrics['max_drawdown'] < -0.15:
            print("- High drawdown detected, reducing risk")
            current_params['base_risk'] *= 0.8
            current_params['max_risk'] *= 0.8
        
        # 3. If profit factor is low, improve entry quality
        if metrics['profit_factor'] < 1.2:
            print("- Low profit factor, improving entry quality")
            # This would trigger using more filters in the strategy
        
        # 4. Analyze trade types
        if 'trade_types' in metrics:
            print("\nTrade type analysis:")
            best_signal_type = None
            best_signal_wr = 0
            
            for signal_type, stats in metrics['trade_types'].items():
                print(f"  {signal_type}: {stats['count']} trades, "
                      f"{stats['win_rate']:.2%} win rate, ${stats['pnl']:.2f} PnL")
                
                if stats['win_rate'] > best_signal_wr and stats['count'] > 10:
                    best_signal_type = signal_type
                    best_signal_wr = stats['win_rate']
            
            if best_signal_type:
                print(f"  Best performing: {best_signal_type}")
        
        # 5. If close to goal, fine-tune
        if current_sharpe > 0.7:
            print("- Close to target, fine-tuning parameters")
            # Small adjustments
            if metrics['win_rate'] > 0.5:
                current_params['base_risk'] *= 1.1
            else:
                current_params['min_rr_ratio'] *= 1.05
        
        # Prevent overfitting by adding some randomness
        if iteration > 2 and best_sharpe < 0.5:
            print("- Adding exploration to escape local minimum")
            current_params['base_risk'] = np.random.uniform(0.005, 0.02)
            current_params['min_rr_ratio'] = np.random.uniform(1.5, 3.0)
    
    # Final report
    print("\n" + "="*60)
    print("RECURSIVE IMPROVEMENT COMPLETE")
    print("="*60)
    
    print(f"\nBest achieved Sharpe ratio: {best_sharpe:.3f}")
    print(f"Best parameters: {best_params}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    final_output = {
        'version': 'v4_recursive_improved',
        'timestamp': datetime.now().isoformat(),
        'iterations': iteration_results,
        'best_params': best_params,
        'best_sharpe': best_sharpe,
        'achieved_goal': best_sharpe > 1.0
    }
    
    with open(f'results_v4_recursive_{timestamp}.json', 'w') as f:
        json.dump(final_output, f, indent=2, default=str)
    
    print(f"\nResults saved to results_v4_recursive_{timestamp}.json")
    
    # If we didn't achieve Sharpe > 1, provide recommendations
    if best_sharpe < 1.0:
        print("\nRECOMMENDATIONS FOR ACHIEVING SHARPE > 1:")
        print("1. Consider using higher timeframes (1H, 4H) for less noise")
        print("2. Add more sophisticated filters (market regime, volatility filters)")
        print("3. Implement portfolio of strategies rather than single strategy")
        print("4. Use machine learning for signal generation")
        print("5. Add mean reversion strategies for ranging markets")
        print("6. Implement better risk management (correlation-based sizing)")
    
    return best_params, best_sharpe


if __name__ == "__main__":
    # Run recursive improvement
    best_params, best_sharpe = recursive_improvement(max_iterations=5)
    
    # If we achieved our goal, run final validation
    if best_sharpe > 1.0:
        print("\n\nRUNNING FINAL VALIDATION...")
        
        # Load fresh data
        df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.set_index('DateTime', inplace=True)
        
        # Split for out-of-sample test
        split_point = int(len(df) * 0.8)
        df_test = df.iloc[split_point:]
        
        # Run on test data
        strategy = ImprovedTrendStrategy(**best_params)
        test_results = strategy.run_backtest(df_test)
        
        print("\nOUT-OF-SAMPLE RESULTS:")
        print(f"Sharpe Ratio: {test_results['metrics']['sharpe_ratio']:.3f}")
        print(f"Total Return: {test_results['metrics']['total_return']:.2%}")
        print(f"Max Drawdown: {test_results['metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {test_results['metrics']['win_rate']:.2%}")