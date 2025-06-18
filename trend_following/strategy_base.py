import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ta
from sklearn.preprocessing import StandardScaler

class TrendFollowingStrategyBase:
    """
    Base trend-following strategy using traditional indicators.
    This version focuses on robust implementation without dependencies.
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 risk_per_trade: float = 0.02,
                 max_positions: int = 1,
                 stop_loss_atr_mult: float = 2.0,
                 take_profit_atr_mult: float = 3.0,
                 min_signal_strength: float = 0.6):
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.min_signal_strength = min_signal_strength
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using ta library."""
        
        # Trend indicators
        df['EMA_10'] = ta.trend.EMAIndicator(close=df['Close'], window=10).ema_indicator()
        df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA_100'] = ta.trend.EMAIndicator(close=df['Close'], window=100).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(close=df['Close'], window=200).ema_indicator()
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_width'] = bb.bollinger_wband()
        df['BB_percent'] = bb.bollinger_pband()
        
        # ATR for volatility
        df['ATR'] = ta.volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        ).average_true_range()
        
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['DI_plus'] = adx.adx_pos()
        df['DI_minus'] = adx.adx_neg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3
        )
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # CCI - Commodity Channel Index
        df['CCI'] = ta.trend.CCIIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], window=20
        ).cci()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], lbp=14
        ).williams_r()
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(
            high=df['High'], low=df['Low'], close=df['Close'], window=20
        )
        df['KC_upper'] = kc.keltner_channel_hband()
        df['KC_lower'] = kc.keltner_channel_lband()
        df['KC_middle'] = kc.keltner_channel_mband()
        
        # Donchian Channels
        dc = ta.volatility.DonchianChannel(
            high=df['High'], low=df['Low'], close=df['Close'], window=20
        )
        df['DC_upper'] = dc.donchian_channel_hband()
        df['DC_lower'] = dc.donchian_channel_lband()
        df['DC_middle'] = dc.donchian_channel_mband()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['High'], low=df['Low'], window1=9, window2=26, window3=52
        )
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()
        df['Ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Volume indicators (create synthetic volume if not available)
        if 'Volume' not in df.columns:
            df['Volume'] = np.ones(len(df)) * 1000000
        
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        ).on_balance_volume()
        
        # Money Flow Index
        df['MFI'] = ta.volume.MFIIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=14
        ).money_flow_index()
        
        # Awesome Oscillator
        df['AO'] = ta.momentum.AwesomeOscillatorIndicator(
            high=df['High'], low=df['Low'], window1=5, window2=34
        ).awesome_oscillator()
        
        # VWAP
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Rate of Change
        df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(high=df['High'], low=df['Low'], window=25)
        df['Aroon_up'] = aroon.aroon_up()
        df['Aroon_down'] = aroon.aroon_down()
        df['Aroon_indicator'] = aroon.aroon_indicator()
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.PSARIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2
        ).psar()
        
        # Choppiness Index
        df['CHOP'] = self.calculate_choppiness_index(df, period=14)
        
        # Linear Regression Slope
        df['LR_slope'] = self.calculate_linear_regression_slope(df['Close'], period=20)
        
        # Efficiency Ratio
        df['ER'] = self.calculate_efficiency_ratio(df['Close'], period=10)
        
        return df
    
    def calculate_choppiness_index(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Choppiness Index."""
        high_low = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
        sum_atr = df['ATR'].rolling(period).sum()
        
        chop = 100 * np.log10(sum_atr / high_low) / np.log10(period)
        return chop
    
    def calculate_linear_regression_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate linear regression slope."""
        def lr_slope(values):
            if len(values) < period:
                return np.nan
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return slope
        
        return series.rolling(window=period).apply(lr_slope, raw=True)
    
    def calculate_efficiency_ratio(self, series: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio."""
        change = abs(series.diff(period))
        volatility = series.diff().abs().rolling(period).sum()
        
        er = change / volatility
        return er.fillna(0)
    
    def calculate_signal_strength(self, df: pd.DataFrame, idx: int) -> float:
        """
        Calculate trading signal strength based on multiple indicators.
        Returns a value between -1 (strong sell) and 1 (strong buy).
        """
        if idx < 200:  # Need enough data
            return 0.0
        
        signals = []
        weights = []
        
        # 1. Moving Average Trend (weight: 2.0)
        ema10 = df.iloc[idx]['EMA_10']
        ema50 = df.iloc[idx]['EMA_50']
        ema200 = df.iloc[idx]['EMA_200']
        close = df.iloc[idx]['Close']
        
        if close > ema10 > ema50 > ema200:
            signals.append(1.0)
        elif close < ema10 < ema50 < ema200:
            signals.append(-1.0)
        else:
            signals.append(0.0)
        weights.append(2.0)
        
        # 2. MACD (weight: 1.5)
        if pd.notna(df.iloc[idx]['MACD_diff']):
            macd_signal = np.sign(df.iloc[idx]['MACD_diff'])
            signals.append(macd_signal)
            weights.append(1.5)
        
        # 3. RSI (weight: 1.0)
        rsi = df.iloc[idx]['RSI']
        if pd.notna(rsi):
            if rsi > 70:
                signals.append(-0.5)  # Overbought
            elif rsi < 30:
                signals.append(0.5)   # Oversold
            else:
                signals.append((rsi - 50) / 50)  # Normalized
            weights.append(1.0)
        
        # 4. ADX Trend Strength (weight: 1.5)
        adx = df.iloc[idx]['ADX']
        if pd.notna(adx) and adx > 25:
            if df.iloc[idx]['DI_plus'] > df.iloc[idx]['DI_minus']:
                signals.append(1.0)
            else:
                signals.append(-1.0)
            weights.append(1.5)
        
        # 5. Bollinger Bands (weight: 1.0)
        bb_percent = df.iloc[idx]['BB_percent']
        if pd.notna(bb_percent):
            if bb_percent > 1:  # Above upper band
                signals.append(-0.5)
            elif bb_percent < 0:  # Below lower band
                signals.append(0.5)
            else:
                signals.append(0.0)
            weights.append(1.0)
        
        # 6. Stochastic (weight: 0.8)
        stoch_k = df.iloc[idx]['Stoch_K']
        if pd.notna(stoch_k):
            if stoch_k > 80:
                signals.append(-0.5)
            elif stoch_k < 20:
                signals.append(0.5)
            else:
                signals.append((stoch_k - 50) / 50)
            weights.append(0.8)
        
        # 7. CCI (weight: 1.0)
        cci = df.iloc[idx]['CCI']
        if pd.notna(cci):
            signals.append(np.clip(cci / 100, -1, 1))
            weights.append(1.0)
        
        # 8. Aroon (weight: 1.2)
        aroon_ind = df.iloc[idx]['Aroon_indicator']
        if pd.notna(aroon_ind):
            signals.append(aroon_ind / 100)
            weights.append(1.2)
        
        # 9. PSAR (weight: 1.5)
        psar = df.iloc[idx]['PSAR']
        if pd.notna(psar):
            if close > psar:
                signals.append(1.0)
            else:
                signals.append(-1.0)
            weights.append(1.5)
        
        # 10. Efficiency Ratio (weight: 1.0)
        er = df.iloc[idx]['ER']
        if pd.notna(er) and er > 0.3:  # High efficiency
            # Use trend direction
            lr_slope = df.iloc[idx]['LR_slope']
            if pd.notna(lr_slope):
                signals.append(np.sign(lr_slope))
                weights.append(1.0)
        
        # Calculate weighted average
        if len(signals) > 0:
            weighted_sum = sum(s * w for s, w in zip(signals, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        
        return 0.0
    
    def calculate_position_size(self, df: pd.DataFrame, idx: int, 
                              signal_strength: float, current_capital: float) -> float:
        """Calculate position size based on Kelly Criterion and ATR."""
        if pd.isna(df.iloc[idx]['ATR']) or df.iloc[idx]['ATR'] == 0:
            return 0
        
        atr = df.iloc[idx]['ATR']
        close = df.iloc[idx]['Close']
        
        # Risk amount
        risk_amount = current_capital * self.risk_per_trade
        
        # Stop loss distance
        stop_distance = atr * self.stop_loss_atr_mult
        
        # Base position size
        base_position_size = risk_amount / stop_distance
        
        # Adjust by signal strength (Kelly-inspired)
        # Use conservative Kelly fraction
        kelly_fraction = min(abs(signal_strength) * 0.25, 0.25)  # Cap at 25%
        
        # Volatility adjustment
        atr_percent = atr / close
        volatility_adjustment = 1.0 / (1.0 + atr_percent * 10)  # Reduce size in high volatility
        
        # Final position size
        position_size = base_position_size * kelly_fraction * volatility_adjustment
        
        # Cap at maximum
        max_position_value = current_capital * 0.1
        max_position_size = max_position_value / close
        
        return min(position_size, max_position_size)
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """Run backtest with improved logic."""
        # Add indicators
        print("Adding indicators...")
        df = self.add_indicators(df)
        
        # Initialize
        current_capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [current_capital]
        
        # Skip initial rows
        start_idx = 200
        
        print(f"Running backtest from index {start_idx} to {len(df)}...")
        
        for idx in range(start_idx, len(df)):
            if idx % 10000 == 0:
                print(f"Processing bar {idx}/{len(df)}...")
            
            current_date = df.index[idx]
            signal_strength = self.calculate_signal_strength(df, idx)
            
            close = df.iloc[idx]['Close']
            atr = df.iloc[idx]['ATR']
            
            # Handle existing position
            if position is not None:
                exit_signal = False
                exit_reason = ""
                
                # Check stop loss
                if position['type'] == 'long' and close <= position['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif position['type'] == 'short' and close >= position['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                
                # Check take profit
                elif position['type'] == 'long' and close >= position['take_profit']:
                    exit_signal = True
                    exit_reason = "Take Profit"
                elif position['type'] == 'short' and close <= position['take_profit']:
                    exit_signal = True
                    exit_reason = "Take Profit"
                
                # Check signal reversal
                elif (position['type'] == 'long' and signal_strength < -0.3) or \
                     (position['type'] == 'short' and signal_strength > 0.3):
                    exit_signal = True
                    exit_reason = "Signal Reversal"
                
                # Trailing stop based on ATR
                if not exit_signal and idx - position['entry_idx'] > 10:
                    if position['type'] == 'long':
                        trailing_stop = close - atr * 1.5
                        position['stop_loss'] = max(position['stop_loss'], trailing_stop)
                    else:
                        trailing_stop = close + atr * 1.5
                        position['stop_loss'] = min(position['stop_loss'], trailing_stop)
                
                if exit_signal:
                    # Calculate P&L
                    if position['type'] == 'long':
                        pnl = (close - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - close) * position['size']
                    
                    current_capital += pnl + (position['entry_price'] * position['size'])
                    
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': close,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl / (position['entry_price'] * position['size']),
                        'exit_reason': exit_reason,
                        'bars_held': idx - position['entry_idx']
                    })
                    
                    position = None
            
            # Check for new entry
            elif abs(signal_strength) >= self.min_signal_strength:
                position_type = 'long' if signal_strength > 0 else 'short'
                position_size = self.calculate_position_size(df, idx, signal_strength, current_capital)
                
                if position_size > 0 and current_capital > close * position_size:
                    # Calculate stops
                    if position_type == 'long':
                        stop_loss = close - (atr * self.stop_loss_atr_mult)
                        take_profit = close + (atr * self.take_profit_atr_mult)
                    else:
                        stop_loss = close + (atr * self.stop_loss_atr_mult)
                        take_profit = close - (atr * self.take_profit_atr_mult)
                    
                    # Enter position
                    position = {
                        'type': position_type,
                        'entry_date': current_date,
                        'entry_idx': idx,
                        'entry_price': close,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'signal_strength': signal_strength
                    }
                    
                    current_capital -= close * position_size
            
            # Update equity curve
            if position is not None:
                if position['type'] == 'long':
                    unrealized_pnl = (close - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - close) * position['size']
                
                equity = current_capital + position['entry_price'] * position['size'] + unrealized_pnl
            else:
                equity = current_capital
            
            equity_curve.append(equity)
        
        # Close final position
        if position is not None:
            close = df.iloc[-1]['Close']
            if position['type'] == 'long':
                pnl = (close - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - close) * position['size']
            
            current_capital += pnl + (position['entry_price'] * position['size'])
            
            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': close,
                'size': position['size'],
                'pnl': pnl,
                'pnl_pct': pnl / (position['entry_price'] * position['size']),
                'exit_reason': 'End of Data',
                'bars_held': len(df) - position['entry_idx']
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, equity_curve, df.index[start_idx:])
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'final_capital': current_capital
        }
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float], 
                         dates: pd.DatetimeIndex) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'sharpe_ratio': 0,
                'total_return': 0,
                'max_drawdown': 0
            }
        
        # Basic trade statistics
        pnls = [t['pnl'] for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        # Calculate returns for Sharpe
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Convert to daily returns (96 15-min bars per day)
        bars_per_day = 96
        daily_returns = []
        
        for i in range(0, len(returns) - bars_per_day, bars_per_day):
            daily_return = (equity_array[i + bars_per_day] - equity_array[i]) / equity_array[i]
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        equity_peaks = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - equity_peaks) / equity_peaks
        max_drawdown = np.min(drawdowns)
        
        # Calculate other metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Win rate and profit factor
        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        avg_trade = np.mean(pnls) if pnls else 0
        
        # Trade duration
        trade_durations = [t.get('bars_held', 0) for t in trades]
        avg_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Calmar ratio
        days = (dates[-1] - dates[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / downside_std if downside_std > 0 else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_pnl': sum(pnls),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_trade_duration': avg_duration,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }


def run_base_strategy():
    """Run the base strategy."""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Initialize strategy
    strategy = TrendFollowingStrategyBase(
        initial_capital=1000000,
        risk_per_trade=0.02,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        min_signal_strength=0.6
    )
    
    # Run backtest
    print("\nRunning Base Strategy...")
    results = strategy.backtest(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results_base_{timestamp}.json', 'w') as f:
        json.dump({
            'version': 'base',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'risk_per_trade': strategy.risk_per_trade,
                'stop_loss_atr_mult': strategy.stop_loss_atr_mult,
                'take_profit_atr_mult': strategy.take_profit_atr_mult,
                'min_signal_strength': strategy.min_signal_strength
            },
            'metrics': results['metrics'],
            'total_trades': len(results['trades'])
        }, f, indent=2)
    
    # Print metrics
    print("\n" + "="*50)
    print("BASE STRATEGY RESULTS")
    print("="*50)
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {results['metrics']['sortino_ratio']:.3f}")
    print(f"Calmar Ratio: {results['metrics']['calmar_ratio']:.3f}")
    print(f"Total Return: {results['metrics']['total_return']:.2%}")
    print(f"Annual Return: {results['metrics']['annual_return']:.2%}")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
    print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    print(f"Avg Trade Duration: {results['metrics']['avg_trade_duration']:.1f} bars")
    
    return results


if __name__ == "__main__":
    run_base_strategy()