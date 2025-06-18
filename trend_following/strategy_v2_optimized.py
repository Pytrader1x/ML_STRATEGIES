import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
import ta
from numba import jit
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


@jit(nopython=True)
def calculate_signals_numba(close, ema_fast, ema_slow, rsi, adx, di_plus, di_minus, 
                           macd_diff, bb_upper, bb_lower, atr, psar):
    """Numba-optimized signal calculation."""
    n = len(close)
    signals = np.zeros(n)
    
    for i in range(200, n):
        score = 0.0
        
        # Trend alignment
        if close[i] > ema_fast[i] > ema_slow[i]:
            score += 2.0
        elif close[i] < ema_fast[i] < ema_slow[i]:
            score -= 2.0
        
        # MACD momentum
        if macd_diff[i] > 0:
            score += 1.5
        else:
            score -= 1.5
        
        # ADX trend strength
        if adx[i] > 25:
            if di_plus[i] > di_minus[i]:
                score += 1.5
            else:
                score -= 1.5
        
        # RSI
        if rsi[i] > 70:
            score -= 0.5
        elif rsi[i] < 30:
            score += 0.5
        
        # Bollinger Bands
        if close[i] > bb_upper[i]:
            score -= 0.5
        elif close[i] < bb_lower[i]:
            score += 0.5
        
        # PSAR
        if close[i] > psar[i]:
            score += 1.0
        else:
            score -= 1.0
        
        signals[i] = score / 8.0  # Normalize to [-1, 1]
    
    return signals


class TrendFollowingV2:
    """
    Optimized trend-following strategy with:
    1. Vectorized operations for speed
    2. Machine learning-inspired signal filtering
    3. Dynamic risk management
    4. Regime detection
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 risk_per_trade: float = 0.015,
                 min_signal_strength: float = 0.5,
                 use_ml_filter: bool = True):
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.min_signal_strength = min_signal_strength
        self.use_ml_filter = use_ml_filter
        
    def add_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators using vectorized operations."""
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Hull Moving Average
        df['HMA'] = self.calculate_hma(df['Close'], 20)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(close=df['Close'], window=14, 
                                                  smooth1=3, smooth2=3)
        df['StochRSI_K'] = stoch_rsi.stochrsi_k()
        df['StochRSI_D'] = stoch_rsi.stochrsi_d()
        
        # ATR and volatility
        atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], 
                                            close=df['Close'], window=14)
        df['ATR'] = atr.average_true_range()
        df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_width'] = bb.bollinger_wband()
        df['BB_percent'] = bb.bollinger_pband()
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], 
                                         close=df['Close'], window=20)
        df['KC_upper'] = kc.keltner_channel_hband()
        df['KC_lower'] = kc.keltner_channel_lband()
        
        # ADX
        adx = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], 
                                   close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.PSARIndicator(high=df['High'], low=df['Low'], 
                                           close=df['Close']).psar()
        
        # Supertrend
        df['Supertrend'] = self.calculate_supertrend(df, period=10, multiplier=3)
        
        # Market regime detection
        df['Volatility_Regime'] = self.detect_volatility_regime(df)
        df['Trend_Strength'] = self.calculate_trend_strength(df)
        
        # Price action patterns
        df['Hammer'] = self.detect_hammer(df)
        df['Engulfing'] = self.detect_engulfing(df)
        
        # Volume (synthetic if not available)
        if 'Volume' not in df.columns:
            df['Volume'] = np.ones(len(df)) * 1000000
        
        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], 
                                                       volume=df['Volume']).on_balance_volume()
        
        # Chaikin Money Flow
        df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(high=df['High'], low=df['Low'],
                                                        close=df['Close'], volume=df['Volume']).chaikin_money_flow()
        
        return df
    
    def calculate_hma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = series.rolling(half_period).mean()
        wma_full = series.rolling(period).mean()
        
        raw_hma = 2 * wma_half - wma_full
        hma = raw_hma.rolling(sqrt_period).mean()
        
        return hma
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, 
                           multiplier: float = 3) -> pd.Series:
        """Calculate Supertrend indicator."""
        hl_avg = (df['High'] + df['Low']) / 2
        atr = df['ATR'].rolling(period).mean()
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        for i in range(period, len(df)):
            if df['Close'].iloc[i] <= upper_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
                
            if i > period:
                if direction.iloc[i] == 1:
                    if supertrend.iloc[i] < supertrend.iloc[i-1]:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                else:
                    if supertrend.iloc[i] > supertrend.iloc[i-1]:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
        
        return supertrend
    
    def detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market volatility regime."""
        # Calculate rolling volatility percentiles
        returns = df['Returns'].fillna(0)
        rolling_vol = returns.rolling(20).std()
        
        vol_percentile = rolling_vol.rolling(100).rank(pct=True)
        
        regime = pd.Series(index=df.index, dtype=str)
        regime[vol_percentile < 0.25] = 'Low'
        regime[(vol_percentile >= 0.25) & (vol_percentile < 0.75)] = 'Medium'
        regime[vol_percentile >= 0.75] = 'High'
        
        return regime
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength score."""
        # Multiple timeframe trend alignment
        trend_score = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            score = 0
            
            if i < 200:
                trend_score.iloc[i] = 0
                continue
            
            # Check EMA alignment
            if df['EMA_10'].iloc[i] > df['EMA_21'].iloc[i] > df['EMA_50'].iloc[i]:
                score += 1
            elif df['EMA_10'].iloc[i] < df['EMA_21'].iloc[i] < df['EMA_50'].iloc[i]:
                score -= 1
            
            # ADX strength
            if df['ADX'].iloc[i] > 25:
                score += 0.5
            
            # Price vs long-term MA
            if df['Close'].iloc[i] > df['EMA_200'].iloc[i]:
                score += 0.5
            else:
                score -= 0.5
            
            trend_score.iloc[i] = score
        
        return trend_score
    
    def detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern."""
        body = abs(df['Close'] - df['Open'])
        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        
        hammer = (lower_shadow > 2 * body) & (upper_shadow < 0.1 * body)
        
        return hammer.astype(int)
    
    def detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect engulfing patterns."""
        prev_body = (df['Close'] - df['Open']).shift(1)
        curr_body = df['Close'] - df['Open']
        
        bullish_engulfing = (prev_body < 0) & (curr_body > 0) & (curr_body > abs(prev_body))
        bearish_engulfing = (prev_body > 0) & (curr_body < 0) & (abs(curr_body) > prev_body)
        
        engulfing = pd.Series(index=df.index, dtype=int)
        engulfing[bullish_engulfing] = 1
        engulfing[bearish_engulfing] = -1
        engulfing.fillna(0, inplace=True)
        
        return engulfing
    
    def generate_ml_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals using ML-inspired approach."""
        
        # Feature engineering
        features = pd.DataFrame(index=df.index)
        
        # Momentum features
        features['RSI_Signal'] = (df['RSI'] - 50) / 50
        features['MACD_Signal'] = np.sign(df['MACD_diff'])
        features['StochRSI_Signal'] = (df['StochRSI_K'] - 50) / 50
        
        # Trend features
        features['EMA_Trend'] = np.where(df['EMA_10'] > df['EMA_50'], 1, -1)
        features['Price_vs_BB'] = (df['Close'] - df['BB_middle']) / df['BB_width']
        features['ADX_Strength'] = df['ADX'] / 100
        
        # Volatility features
        features['ATR_Normalized'] = df['ATR_Percent'] / df['ATR_Percent'].rolling(50).mean()
        features['BB_Squeeze'] = df['BB_width'] / df['BB_width'].rolling(50).mean()
        
        # Pattern features
        features['Hammer'] = df['Hammer']
        features['Engulfing'] = df['Engulfing']
        
        # Combine features with weights
        weights = {
            'RSI_Signal': 0.8,
            'MACD_Signal': 1.5,
            'StochRSI_Signal': 0.5,
            'EMA_Trend': 2.0,
            'Price_vs_BB': 0.7,
            'ADX_Strength': 1.2,
            'ATR_Normalized': -0.3,  # Negative weight for high volatility
            'BB_Squeeze': 0.5,
            'Hammer': 0.3,
            'Engulfing': 0.5
        }
        
        # Calculate weighted signal
        signal = pd.Series(index=df.index, dtype=float)
        signal[:] = 0
        
        for feature, weight in weights.items():
            if feature in features.columns:
                signal += features[feature].fillna(0) * weight
        
        # Normalize to [-1, 1]
        total_weight = sum(abs(w) for w in weights.values())
        signal = signal / total_weight
        
        # Apply regime filter
        signal[df['Volatility_Regime'] == 'High'] *= 0.5  # Reduce signals in high volatility
        signal[df['ADX'] < 20] *= 0.3  # Reduce signals in choppy markets
        
        return signal
    
    def apply_risk_filters(self, signals: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Apply risk management filters to signals."""
        filtered_signals = signals.copy()
        
        # Don't trade during extreme volatility
        extreme_vol = df['ATR_Percent'] > df['ATR_Percent'].rolling(100).quantile(0.95)
        filtered_signals[extreme_vol] = 0
        
        # Reduce position size after losses (anti-martingale)
        # This would be implemented in the backtest loop
        
        # Time-based filters (avoid trading during known low liquidity periods)
        # For 15-min bars, avoid first and last hour of trading day
        
        return filtered_signals
    
    def calculate_dynamic_stops(self, df: pd.DataFrame, position_type: str, 
                              entry_idx: int) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels."""
        
        atr = df.iloc[entry_idx]['ATR']
        close = df.iloc[entry_idx]['Close']
        volatility_regime = df.iloc[entry_idx]['Volatility_Regime']
        
        # Adjust multipliers based on volatility regime
        if volatility_regime == 'Low':
            sl_mult = 1.5
            tp_mult = 4.0
        elif volatility_regime == 'Medium':
            sl_mult = 2.0
            tp_mult = 3.0
        else:  # High
            sl_mult = 2.5
            tp_mult = 2.0
        
        if position_type == 'long':
            stop_loss = close - (atr * sl_mult)
            take_profit = close + (atr * tp_mult)
        else:
            stop_loss = close + (atr * sl_mult)
            take_profit = close - (atr * tp_mult)
        
        return stop_loss, take_profit
    
    def backtest_vectorized(self, df: pd.DataFrame) -> Dict:
        """Run vectorized backtest for speed."""
        
        print("Adding indicators...")
        df = self.add_indicators_vectorized(df)
        
        print("Generating signals...")
        if self.use_ml_filter:
            signals = self.generate_ml_signals(df)
        else:
            # Use simple numba-optimized signals
            signals = calculate_signals_numba(
                df['Close'].values,
                df['EMA_10'].values,
                df['EMA_50'].values,
                df['RSI'].values,
                df['ADX'].values,
                df['DI_plus'].values,
                df['DI_minus'].values,
                df['MACD_diff'].values,
                df['BB_upper'].values,
                df['BB_lower'].values,
                df['ATR'].values,
                df['PSAR'].values
            )
            signals = pd.Series(signals, index=df.index)
        
        # Apply risk filters
        signals = self.apply_risk_filters(signals, df)
        
        # Initialize tracking
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [capital]
        
        print("Running backtest...")
        for i in range(200, len(df)):
            if i % 10000 == 0:
                print(f"Processing {i}/{len(df)} bars...")
            
            signal_strength = signals.iloc[i]
            
            # Check exit conditions for existing position
            if position is not None:
                close = df.iloc[i]['Close']
                exit_signal = False
                exit_reason = ""
                
                # Stop loss / Take profit
                if position['type'] == 'long':
                    if close <= position['stop_loss']:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                    elif close >= position['take_profit']:
                        exit_signal = True
                        exit_reason = "Take Profit"
                else:
                    if close >= position['stop_loss']:
                        exit_signal = True
                        exit_reason = "Stop Loss"
                    elif close <= position['take_profit']:
                        exit_signal = True
                        exit_reason = "Take Profit"
                
                # Signal reversal
                if not exit_signal:
                    if (position['type'] == 'long' and signal_strength < -0.3) or \
                       (position['type'] == 'short' and signal_strength > 0.3):
                        exit_signal = True
                        exit_reason = "Signal Reversal"
                
                # Time-based exit (hold for max 500 bars = ~5 days)
                if not exit_signal and i - position['entry_idx'] > 500:
                    exit_signal = True
                    exit_reason = "Time Exit"
                
                if exit_signal:
                    # Calculate P&L
                    if position['type'] == 'long':
                        pnl = (close - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - close) * position['size']
                    
                    capital += pnl + (position['entry_price'] * position['size'])
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': df.index[i],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': close,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl / (position['entry_price'] * position['size']) * 100,
                        'exit_reason': exit_reason,
                        'bars_held': i - position['entry_idx']
                    })
                    
                    position = None
            
            # Check entry conditions
            elif abs(signal_strength) >= self.min_signal_strength:
                # Skip if recently exited
                if len(trades) > 0 and i - trades[-1].get('exit_idx', 0) < 10:
                    continue
                
                position_type = 'long' if signal_strength > 0 else 'short'
                
                # Calculate position size
                close = df.iloc[i]['Close']
                atr = df.iloc[i]['ATR']
                
                if atr > 0:
                    risk_amount = capital * self.risk_per_trade
                    stop_distance = atr * 2.0
                    position_size = risk_amount / stop_distance
                    
                    # Apply Kelly adjustment
                    kelly_fraction = min(abs(signal_strength) * 0.2, 0.2)
                    position_size *= kelly_fraction
                    
                    # Check if we have enough capital
                    if position_size * close < capital * 0.9:  # Leave 10% buffer
                        stop_loss, take_profit = self.calculate_dynamic_stops(df, position_type, i)
                        
                        position = {
                            'type': position_type,
                            'entry_date': df.index[i],
                            'entry_idx': i,
                            'entry_price': close,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'signal_strength': signal_strength
                        }
                        
                        capital -= close * position_size
                        
                        if len(trades) > 0:
                            trades[-1]['exit_idx'] = i
            
            # Update equity
            if position is not None:
                close = df.iloc[i]['Close']
                if position['type'] == 'long':
                    unrealized_pnl = (close - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - close) * position['size']
                
                equity = capital + position['entry_price'] * position['size'] + unrealized_pnl
            else:
                equity = capital
            
            equity_curve.append(equity)
        
        # Close final position
        if position is not None:
            close = df.iloc[-1]['Close']
            if position['type'] == 'long':
                pnl = (close - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - close) * position['size']
            
            capital += pnl + (position['entry_price'] * position['size'])
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': close,
                'size': position['size'],
                'pnl': pnl,
                'pnl_pct': pnl / (position['entry_price'] * position['size']) * 100,
                'exit_reason': 'End of Data',
                'bars_held': len(df) - position['entry_idx']
            })
        
        # Calculate metrics
        metrics = self.calculate_advanced_metrics(trades, equity_curve, df.index[200:])
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': capital,
            'signals': signals
        }
    
    def calculate_advanced_metrics(self, trades: List[Dict], equity_curve: List[float], 
                                 dates: pd.DatetimeIndex) -> Dict:
        """Calculate comprehensive metrics including risk-adjusted returns."""
        
        if len(trades) == 0:
            return {'sharpe_ratio': 0, 'total_trades': 0}
        
        # Convert to arrays
        equity_array = np.array(equity_curve)
        
        # Calculate returns
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Daily returns (96 bars per day)
        daily_returns = []
        for i in range(0, len(returns) - 96, 96):
            daily_return = (equity_array[i + 96] - equity_array[i]) / equity_array[i]
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Core metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        days = (dates[-1] - dates[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        trade_pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        durations = [t.get('bars_held', 0) for t in trades]
        avg_duration_bars = np.mean(durations) if durations else 0
        avg_duration_hours = avg_duration_bars * 0.25  # 15-min bars
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for pnl in trade_pnls:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Ulcer Index (measures downside volatility)
        drawdown_squared = drawdown ** 2
        ulcer_index = np.sqrt(np.mean(drawdown_squared))
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(trade_pnls) if trade_pnls else 0,
            'largest_loss': min(trade_pnls) if trade_pnls else 0,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_pnl': sum(trade_pnls),
            'avg_trade_duration_hours': avg_duration_hours,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'recovery_factor': recovery_factor,
            'ulcer_index': ulcer_index,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
        return metrics
    
    def plot_results(self, df: pd.DataFrame, results: Dict, save_path: str = None):
        """Plot strategy results."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        
        # Equity curve
        axes[0].plot(results['equity_curve'], label='Equity Curve', linewidth=2)
        axes[0].set_ylabel('Equity ($)')
        axes[0].set_title(f"Strategy Performance - Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Price and signals
        axes[1].plot(df['Close'].iloc[200:], label='AUDUSD Price', alpha=0.7)
        
        # Mark trades
        trades_df = pd.DataFrame(results['trades'])
        if len(trades_df) > 0:
            long_trades = trades_df[trades_df['type'] == 'long']
            short_trades = trades_df[trades_df['type'] == 'short']
            
            for _, trade in long_trades.iterrows():
                axes[1].scatter(trade['entry_date'], trade['entry_price'], 
                              color='green', marker='^', s=100, alpha=0.7)
                axes[1].scatter(trade['exit_date'], trade['exit_price'], 
                              color='red', marker='v', s=100, alpha=0.7)
            
            for _, trade in short_trades.iterrows():
                axes[1].scatter(trade['entry_date'], trade['entry_price'], 
                              color='red', marker='v', s=100, alpha=0.7)
                axes[1].scatter(trade['exit_date'], trade['exit_price'], 
                              color='green', marker='^', s=100, alpha=0.7)
        
        axes[1].set_ylabel('Price')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Drawdown
        equity_array = np.array(results['equity_curve'])
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        
        axes[2].fill_between(range(len(drawdown)), drawdown, 0, 
                           color='red', alpha=0.3, label='Drawdown')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        plt.show()


def run_strategy_v2():
    """Run optimized strategy V2."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Data shape: {df.shape}")
    
    # Initialize strategy
    strategy = TrendFollowingV2(
        initial_capital=1000000,
        risk_per_trade=0.015,
        min_signal_strength=0.5,
        use_ml_filter=True
    )
    
    # Run backtest
    print("\nRunning Strategy V2 (Optimized)...")
    results = strategy.backtest_vectorized(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dict = {
        'version': 'v2_optimized',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'risk_per_trade': strategy.risk_per_trade,
            'min_signal_strength': strategy.min_signal_strength,
            'use_ml_filter': strategy.use_ml_filter
        },
        'metrics': results['metrics']
    }
    
    with open(f'results_v2_{timestamp}.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("STRATEGY V2 RESULTS")
    print("="*60)
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {results['metrics']['sortino_ratio']:.3f}")
    print(f"Calmar Ratio: {results['metrics']['calmar_ratio']:.3f}")
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Annual Return: {results['metrics']['annual_return']:.2%}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
    print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    print(f"Recovery Factor: {results['metrics']['recovery_factor']:.2f}")
    print("="*60)
    
    # Plot results
    try:
        strategy.plot_results(df, results, save_path=f'strategy_v2_plot_{timestamp}.png')
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return results


if __name__ == "__main__":
    run_strategy_v2()