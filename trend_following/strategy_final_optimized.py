import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class OptimizedTrendStrategy:
    """
    Final optimized strategy focusing on:
    1. High-probability setups only
    2. Strict risk management
    3. Trend continuation patterns
    4. Multi-timeframe confirmation
    """
    
    def __init__(self):
        # Optimized parameters from testing
        self.risk_per_trade = 0.01
        self.max_daily_risk = 0.02
        self.min_rr_ratio = 2.5
        self.max_correlation = 0.7
        
        # Entry filters
        self.min_trend_score = 0.7
        self.min_adx = 20
        self.max_atr_percent = 0.8
        
        # Trade management
        self.use_trailing_stop = True
        self.breakeven_threshold = 1.0  # Move to breakeven at 1R profit
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optimized indicator set."""
        
        # Core calculations
        df['Returns'] = df['Close'].pct_change()
        
        # Trend indicators - multiple timeframes
        for period in [8, 21, 55, 89, 144]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # ATR
        tr = self.calculate_true_range(df)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
        
        # Momentum
        df['ROC'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        
        # RSI with smoothing
        rsi_raw = self.calculate_rsi(df['Close'], 14)
        df['RSI'] = rsi_raw.ewm(span=3).mean()  # Smooth RSI
        
        # ADX
        df['ADX'], df['DI_plus'], df['DI_minus'] = self.calculate_adx(df, 14)
        
        # Price structure
        df['Swing_High'] = df['High'].rolling(20).max()
        df['Swing_Low'] = df['Low'].rolling(20).min()
        df['Price_Position'] = (df['Close'] - df['Swing_Low']) / (df['Swing_High'] - df['Swing_Low'])
        
        # Volatility compression
        df['BB_Width'] = self.calculate_bb_width(df, 20, 2)
        df['Volatility_Rank'] = df['ATR_Percent'].rolling(100).rank(pct=True)
        
        # Trend quality
        df['Trend_Consistency'] = self.calculate_trend_consistency(df)
        
        return df
    
    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    def calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_adx(self, df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX and directional indicators."""
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        
        plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
        minus_dm = -low_diff.where((low_diff < 0) & (-low_diff > high_diff), 0)
        
        tr = self.calculate_true_range(df)
        
        # Smooth the indicators
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    def calculate_bb_width(self, df: pd.DataFrame, period: int, std: float) -> pd.Series:
        """Calculate Bollinger Band width."""
        middle = df['Close'].rolling(period).mean()
        std_dev = df['Close'].rolling(period).std()
        
        upper = middle + std * std_dev
        lower = middle - std * std_dev
        
        return (upper - lower) / middle
    
    def calculate_trend_consistency(self, df: pd.DataFrame) -> pd.Series:
        """Measure how consistently price follows trend."""
        consistency = pd.Series(index=df.index, dtype=float)
        
        for i in range(55, len(df)):
            # Count bars where close > EMA21
            above_ema = sum(df['Close'].iloc[i-20:i] > df['EMA_21'].iloc[i-20:i])
            consistency.iloc[i] = above_ema / 20
        
        return consistency
    
    def identify_setup(self, df: pd.DataFrame, idx: int) -> Tuple[int, float, str]:
        """
        Identify high-probability trade setups.
        Returns: (direction, strength, setup_type)
        """
        
        if idx < 144:  # Need enough data
            return 0, 0, ""
        
        row = df.iloc[idx]
        prev_rows = df.iloc[idx-5:idx]
        
        # Check trend alignment
        trend_aligned_up = (
            row['EMA_8'] > row['EMA_21'] > row['EMA_55'] > row['EMA_89'] > row['EMA_144']
        )
        trend_aligned_down = (
            row['EMA_8'] < row['EMA_21'] < row['EMA_55'] < row['EMA_89'] < row['EMA_144']
        )
        
        # ADX filter
        if row['ADX'] < self.min_adx:
            return 0, 0, ""
        
        # Volatility filter
        if row['ATR_Percent'] > self.max_atr_percent:
            return 0, 0, ""
        
        direction = 0
        strength = 0
        setup_type = ""
        
        # LONG SETUPS
        if trend_aligned_up and row['DI_plus'] > row['DI_minus']:
            
            # Setup 1: Trend continuation after pullback
            if (row['Low'] <= row['EMA_21'] and 
                row['Close'] > row['EMA_21'] and
                row['RSI'] > 40 and row['RSI'] < 65):
                
                direction = 1
                strength = 0.8
                setup_type = "Pullback Buy"
            
            # Setup 2: Breakout with momentum
            elif (row['Close'] > prev_rows['High'].max() and
                  row['ROC'] > 0 and
                  row['Trend_Consistency'] > 0.7):
                
                direction = 1
                strength = 0.7
                setup_type = "Momentum Breakout"
            
            # Setup 3: Flag pattern
            elif (row['ATR_Percent'] < prev_rows['ATR_Percent'].mean() * 0.7 and
                  row['Close'] > row['EMA_8'] and
                  row['Price_Position'] > 0.6):
                
                direction = 1
                strength = 0.75
                setup_type = "Bull Flag"
        
        # SHORT SETUPS
        elif trend_aligned_down and row['DI_minus'] > row['DI_plus']:
            
            # Setup 1: Trend continuation after pullback
            if (row['High'] >= row['EMA_21'] and 
                row['Close'] < row['EMA_21'] and
                row['RSI'] < 60 and row['RSI'] > 35):
                
                direction = -1
                strength = 0.8
                setup_type = "Pullback Sell"
            
            # Setup 2: Breakdown with momentum
            elif (row['Close'] < prev_rows['Low'].min() and
                  row['ROC'] < 0 and
                  row['Trend_Consistency'] < 0.3):
                
                direction = -1
                strength = 0.7
                setup_type = "Momentum Breakdown"
            
            # Setup 3: Bear flag pattern
            elif (row['ATR_Percent'] < prev_rows['ATR_Percent'].mean() * 0.7 and
                  row['Close'] < row['EMA_8'] and
                  row['Price_Position'] < 0.4):
                
                direction = -1
                strength = 0.75
                setup_type = "Bear Flag"
        
        # Apply final quality filters
        if strength > 0:
            # Reduce strength if volatility rank is extreme
            if row['Volatility_Rank'] > 0.9 or row['Volatility_Rank'] < 0.1:
                strength *= 0.8
            
            # Boost strength if trend consistency is high
            if row['Trend_Consistency'] > 0.8:
                strength *= 1.1
        
        return direction, min(strength, 1.0), setup_type
    
    def calculate_position_parameters(self, df: pd.DataFrame, idx: int, 
                                    direction: int, capital: float) -> Dict:
        """Calculate position size and risk parameters."""
        
        row = df.iloc[idx]
        
        # Dynamic stop loss based on market structure
        atr = row['ATR']
        
        if direction == 1:  # Long
            # Find recent swing low
            recent_low = df['Low'].iloc[idx-20:idx].min()
            atr_stop = row['Close'] - (atr * 2.0)
            stop_loss = max(recent_low - atr * 0.2, atr_stop)
            
            # Ensure minimum stop distance
            min_stop = row['Close'] * 0.98  # 2% minimum
            stop_loss = min(stop_loss, min_stop)
            
        else:  # Short
            # Find recent swing high
            recent_high = df['High'].iloc[idx-20:idx].max()
            atr_stop = row['Close'] + (atr * 2.0)
            stop_loss = min(recent_high + atr * 0.2, atr_stop)
            
            # Ensure minimum stop distance
            min_stop = row['Close'] * 1.02  # 2% minimum
            stop_loss = max(stop_loss, min_stop)
        
        # Calculate position size
        risk_amount = capital * self.risk_per_trade
        stop_distance = abs(row['Close'] - stop_loss)
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = 0
        
        # Calculate take profit (minimum RR ratio)
        take_profit_distance = stop_distance * self.min_rr_ratio
        
        if direction == 1:
            take_profit = row['Close'] + take_profit_distance
        else:
            take_profit = row['Close'] - take_profit_distance
        
        return {
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'stop_distance': stop_distance
        }
    
    def manage_position(self, position: Dict, df: pd.DataFrame, idx: int) -> Tuple[bool, str, float]:
        """
        Manage open position with trailing stops and profit taking.
        Returns: (should_exit, reason, exit_price)
        """
        
        row = df.iloc[idx]
        
        # Check fixed stops
        if position['direction'] == 1:  # Long
            if row['Low'] <= position['stop_loss']:
                return True, "Stop Loss", position['stop_loss']
            elif row['High'] >= position['take_profit']:
                return True, "Take Profit", position['take_profit']
        else:  # Short
            if row['High'] >= position['stop_loss']:
                return True, "Stop Loss", position['stop_loss']
            elif row['Low'] <= position['take_profit']:
                return True, "Take Profit", position['take_profit']
        
        # Trailing stop logic
        if self.use_trailing_stop:
            bars_in_trade = idx - position['entry_idx']
            
            if position['direction'] == 1:
                # Calculate profit in R
                current_profit = (row['Close'] - position['entry_price']) / position['stop_distance']
                
                # Move to breakeven
                if current_profit >= self.breakeven_threshold and position['stop_loss'] < position['entry_price']:
                    position['stop_loss'] = position['entry_price'] + position['stop_distance'] * 0.1
                
                # Trail stop after 2R profit
                if current_profit >= 2.0:
                    trail_stop = row['Close'] - row['ATR'] * 1.5
                    position['stop_loss'] = max(position['stop_loss'], trail_stop)
                
                # Time-based trailing after 50 bars
                if bars_in_trade > 50:
                    time_trail = position['entry_price'] + (row['Close'] - position['entry_price']) * 0.5
                    position['stop_loss'] = max(position['stop_loss'], time_trail)
                    
            else:  # Short
                # Calculate profit in R
                current_profit = (position['entry_price'] - row['Close']) / position['stop_distance']
                
                # Move to breakeven
                if current_profit >= self.breakeven_threshold and position['stop_loss'] > position['entry_price']:
                    position['stop_loss'] = position['entry_price'] - position['stop_distance'] * 0.1
                
                # Trail stop after 2R profit
                if current_profit >= 2.0:
                    trail_stop = row['Close'] + row['ATR'] * 1.5
                    position['stop_loss'] = min(position['stop_loss'], trail_stop)
                
                # Time-based trailing after 50 bars
                if bars_in_trade > 50:
                    time_trail = position['entry_price'] - (position['entry_price'] - row['Close']) * 0.5
                    position['stop_loss'] = min(position['stop_loss'], time_trail)
        
        # Exit on trend change
        if position['direction'] == 1 and row['EMA_8'] < row['EMA_21']:
            return True, "Trend Change", row['Close']
        elif position['direction'] == -1 and row['EMA_8'] > row['EMA_21']:
            return True, "Trend Change", row['Close']
        
        return False, "", 0
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 1000000) -> Dict:
        """Execute backtest with the optimized strategy."""
        
        # Prepare data
        print("Preparing data...")
        df = self.prepare_data(df)
        
        # Initialize
        capital = initial_capital
        positions = []  # Can have multiple positions
        closed_trades = []
        equity_curve = [capital]
        daily_returns = []
        
        # Risk management
        daily_risk_used = 0
        last_trade_date = None
        
        # Start after warmup
        start_idx = 200
        
        print(f"Running backtest on {len(df)-start_idx} bars...")
        
        for idx in range(start_idx, len(df)):
            current_date = df.index[idx]
            
            # Reset daily risk if new day
            if last_trade_date is None or current_date.date() != last_trade_date.date():
                daily_risk_used = 0
            
            # Manage open positions
            positions_to_remove = []
            
            for i, position in enumerate(positions):
                should_exit, reason, exit_price = self.manage_position(position, df, idx)
                
                if should_exit:
                    # Calculate P&L
                    if position['direction'] == 1:
                        pnl = (exit_price - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['size']
                    
                    capital += pnl
                    
                    closed_trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'direction': position['direction'],
                        'setup_type': position['setup_type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_r': pnl / position['risk_amount'],
                        'exit_reason': reason,
                        'bars_held': idx - position['entry_idx']
                    })
                    
                    positions_to_remove.append(i)
            
            # Remove closed positions
            for i in reversed(positions_to_remove):
                positions.pop(i)
            
            # Check for new entry
            if daily_risk_used < self.max_daily_risk:
                direction, strength, setup_type = self.identify_setup(df, idx)
                
                if strength >= self.min_trend_score:
                    # Check correlation with existing positions
                    can_enter = True
                    
                    for pos in positions:
                        if pos['direction'] == direction:
                            # Already have position in same direction
                            can_enter = False
                            break
                    
                    if can_enter:
                        # Calculate position parameters
                        params = self.calculate_position_parameters(df, idx, direction, capital)
                        
                        if params['size'] > 0 and params['risk_amount'] < capital * 0.05:
                            # Enter position
                            position = {
                                'direction': direction,
                                'setup_type': setup_type,
                                'entry_date': current_date,
                                'entry_idx': idx,
                                'entry_price': df.iloc[idx]['Close'],
                                'size': params['size'],
                                'stop_loss': params['stop_loss'],
                                'take_profit': params['take_profit'],
                                'risk_amount': params['risk_amount'],
                                'stop_distance': params['stop_distance']
                            }
                            
                            positions.append(position)
                            daily_risk_used += self.risk_per_trade
                            last_trade_date = current_date
            
            # Calculate equity
            total_equity = capital
            
            for position in positions:
                current_price = df.iloc[idx]['Close']
                
                if position['direction'] == 1:
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                
                total_equity += unrealized_pnl
            
            equity_curve.append(total_equity)
            
            # Track daily returns
            if len(equity_curve) > 96:  # Full day of data
                daily_return = (equity_curve[-1] - equity_curve[-97]) / equity_curve[-97]
                daily_returns.append(daily_return)
        
        # Close remaining positions
        for position in positions:
            final_price = df.iloc[-1]['Close']
            
            if position['direction'] == 1:
                pnl = (final_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - final_price) * position['size']
            
            capital += pnl
            
            closed_trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'direction': position['direction'],
                'setup_type': position['setup_type'],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'size': position['size'],
                'pnl': pnl,
                'pnl_r': pnl / position['risk_amount'],
                'exit_reason': 'End of Data',
                'bars_held': len(df) - position['entry_idx']
            })
        
        # Calculate final metrics
        metrics = self.calculate_final_metrics(closed_trades, equity_curve, daily_returns, initial_capital)
        
        return {
            'metrics': metrics,
            'trades': closed_trades,
            'equity_curve': equity_curve,
            'final_capital': capital
        }
    
    def calculate_final_metrics(self, trades: List[Dict], equity_curve: List[float], 
                               daily_returns: List[float], initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        if len(trades) == 0:
            return {'sharpe_ratio': 0, 'total_trades': 0}
        
        # Convert to arrays
        equity_array = np.array(equity_curve)
        daily_returns_array = np.array(daily_returns)
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades
        
        # P&L statistics
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        avg_win_r = np.mean([t['pnl_r'] for t in winning_trades]) if winning_trades else 0
        avg_loss_r = np.mean([t['pnl_r'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expected value
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        expectancy_r = (win_rate * avg_win_r) + ((1 - win_rate) * avg_loss_r)
        
        # Sharpe ratio
        if len(daily_returns_array) > 1:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns_array) / np.std(daily_returns_array)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        negative_returns = daily_returns_array[daily_returns_array < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino_ratio = np.sqrt(252) * np.mean(daily_returns_array) / downside_std
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Total return
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        
        # Win/loss streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                if current_streak >= 0:
                    current_streak += 1
                    max_win_streak = max(max_win_streak, current_streak)
                else:
                    current_streak = 1
            else:
                if current_streak <= 0:
                    current_streak -= 1
                    max_loss_streak = max(max_loss_streak, abs(current_streak))
                else:
                    current_streak = -1
        
        # Setup analysis
        setup_stats = {}
        for trade in trades:
            setup = trade['setup_type']
            if setup not in setup_stats:
                setup_stats[setup] = {
                    'count': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'total_r': 0
                }
            
            setup_stats[setup]['count'] += 1
            if trade['pnl'] > 0:
                setup_stats[setup]['wins'] += 1
            setup_stats[setup]['total_pnl'] += trade['pnl']
            setup_stats[setup]['total_r'] += trade['pnl_r']
        
        # Calculate setup win rates
        for setup in setup_stats:
            stats = setup_stats[setup]
            stats['win_rate'] = stats['wins'] / stats['count']
            stats['avg_r'] = stats['total_r'] / stats['count']
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'expectancy_r': expectancy_r,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_r': avg_win_r,
            'avg_loss_r': avg_loss_r,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'setup_stats': setup_stats
        }


def main():
    """Run the final optimized strategy."""
    
    # Load data
    print("Loading AUDUSD data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    print(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Run strategy
    strategy = OptimizedTrendStrategy()
    results = strategy.run_backtest(df)
    
    # Print results
    print("\n" + "="*60)
    print("FINAL OPTIMIZED STRATEGY RESULTS")
    print("="*60)
    
    metrics = results['metrics']
    print(f"\nPerformance Metrics:")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    print(f"\nTrade Statistics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Expectancy per Trade: ${metrics['expectancy']:.2f}")
    print(f"Expectancy in R: {metrics['expectancy_r']:.2f}R")
    
    print(f"\nRisk Metrics:")
    print(f"Average Win: {metrics['avg_win_r']:.2f}R (${metrics['avg_win']:.2f})")
    print(f"Average Loss: {metrics['avg_loss_r']:.2f}R (${metrics['avg_loss']:.2f})")
    print(f"Max Win Streak: {metrics['max_win_streak']}")
    print(f"Max Loss Streak: {metrics['max_loss_streak']}")
    
    print(f"\nSetup Performance:")
    for setup, stats in metrics['setup_stats'].items():
        print(f"  {setup}: {stats['count']} trades, {stats['win_rate']:.1%} win rate, {stats['avg_r']:.2f}R avg")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output = {
        'version': 'final_optimized',
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'parameters': {
            'risk_per_trade': strategy.risk_per_trade,
            'max_daily_risk': strategy.max_daily_risk,
            'min_rr_ratio': strategy.min_rr_ratio,
            'min_trend_score': strategy.min_trend_score,
            'min_adx': strategy.min_adx
        },
        'sample_trades': results['trades'][:20] if results['trades'] else []
    }
    
    filename = f'results_final_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to {filename}")
    
    # Determine if we achieved our goal
    if metrics['sharpe_ratio'] > 1.0:
        print("\n🎉 SUCCESS! Achieved Sharpe Ratio > 1.0")
    else:
        print(f"\nSharpe ratio of {metrics['sharpe_ratio']:.3f} is below target of 1.0")
        print("Consider using higher timeframe data or combining with other strategies")
    
    return results


if __name__ == "__main__":
    results = main()