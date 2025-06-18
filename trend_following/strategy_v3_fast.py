import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
import ta
from typing import Dict, List


class FastTrendStrategy:
    """
    Fast, focused trend-following strategy optimized for performance.
    Uses fewer indicators but with better signal quality.
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 risk_per_trade: float = 0.02,
                 atr_stop_mult: float = 2.0,
                 atr_target_mult: float = 3.0):
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add only essential indicators for speed."""
        
        # Core trend indicators
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # ATR for risk management
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # MACD for momentum
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI for overbought/oversold
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
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
        
        # Bollinger Bands for volatility
        df['BB_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Simple volume proxy
        df['Volume'] = 1000000
        
        return df
    
    def calculate_signal(self, row) -> float:
        """Calculate trading signal strength."""
        signal = 0
        
        # Trend alignment (most important)
        if row['EMA_20'] > row['EMA_50'] > row['EMA_200']:
            signal += 3
        elif row['EMA_20'] < row['EMA_50'] < row['EMA_200']:
            signal -= 3
        
        # MACD momentum
        if row['MACD_hist'] > 0:
            signal += 2
        else:
            signal -= 2
        
        # ADX trend strength
        if row['ADX'] > 25:
            if row['DI_plus'] > row['DI_minus']:
                signal += 2
            else:
                signal -= 2
        
        # RSI extremes (contrarian for mean reversion)
        if row['RSI'] < 30:
            signal += 1
        elif row['RSI'] > 70:
            signal -= 1
        
        # Price vs Bollinger Bands
        price_position = (row['Close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])
        if price_position < 0.2:
            signal += 1
        elif price_position > 0.8:
            signal -= 1
        
        # Normalize to [-1, 1]
        return signal / 10
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Execute backtest with efficient logic."""
        
        # Prepare data
        print("Preparing data...")
        df = self.prepare_data(df)
        
        # Initialize variables
        capital = self.initial_capital
        position = None
        trades = []
        equity = []
        
        # Skip warmup period
        start_idx = 200
        
        print(f"Running backtest on {len(df)-start_idx} bars...")
        
        # Vectorize signal calculation for speed
        signals = df.iloc[start_idx:].apply(self.calculate_signal, axis=1)
        
        # Main loop
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            signal = signals.iloc[i-start_idx] if i-start_idx < len(signals) else 0
            
            # Position management
            if position is not None:
                # Check exit conditions
                exit_trade = False
                exit_reason = ""
                
                if position['type'] == 'long':
                    if row['Close'] <= position['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    elif row['Close'] >= position['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                    elif signal < -0.3:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                else:  # short
                    if row['Close'] >= position['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                    elif row['Close'] <= position['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                    elif signal > 0.3:
                        exit_trade = True
                        exit_reason = "Signal Reversal"
                
                # Trailing stop
                if not exit_trade and i - position['entry_idx'] > 20:
                    if position['type'] == 'long':
                        new_stop = row['Close'] - row['ATR'] * 1.5
                        position['stop_loss'] = max(position['stop_loss'], new_stop)
                    else:
                        new_stop = row['Close'] + row['ATR'] * 1.5
                        position['stop_loss'] = min(position['stop_loss'], new_stop)
                
                if exit_trade:
                    # Calculate P&L
                    if position['type'] == 'long':
                        pnl = (row['Close'] - position['entry_price']) * position['size']
                    else:
                        pnl = (position['entry_price'] - row['Close']) * position['size']
                    
                    capital += position['capital_used'] + pnl
                    
                    trades.append({
                        'entry_date': df.index[position['entry_idx']],
                        'exit_date': df.index[i],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': row['Close'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / position['capital_used']) * 100,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
            
            # Entry logic
            elif abs(signal) > 0.5:
                # Ensure minimum time between trades
                if len(trades) > 0:
                    last_exit = trades[-1]['exit_date']
                    if i < len(df) and (df.index[i] - last_exit).total_seconds() < 3600:  # 1 hour minimum
                        equity.append(capital)
                        continue
                
                # Position sizing
                risk_amount = capital * self.risk_per_trade
                stop_distance = row['ATR'] * self.atr_stop_mult
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    capital_needed = position_size * row['Close']
                    
                    # Ensure we don't use more than 50% of capital
                    if capital_needed > capital * 0.5:
                        capital_needed = capital * 0.5
                        position_size = capital_needed / row['Close']
                    
                    if capital_needed < capital:
                        # Enter position
                        position_type = 'long' if signal > 0 else 'short'
                        
                        if position_type == 'long':
                            stop_loss = row['Close'] - stop_distance
                            take_profit = row['Close'] + (row['ATR'] * self.atr_target_mult)
                        else:
                            stop_loss = row['Close'] + stop_distance
                            take_profit = row['Close'] - (row['ATR'] * self.atr_target_mult)
                        
                        position = {
                            'type': position_type,
                            'entry_idx': i,
                            'entry_price': row['Close'],
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'capital_used': capital_needed,
                            'signal': signal
                        }
                        
                        capital -= capital_needed
            
            # Track equity
            if position is not None:
                current_value = position['capital_used']
                if position['type'] == 'long':
                    current_value += (row['Close'] - position['entry_price']) * position['size']
                else:
                    current_value += (position['entry_price'] - row['Close']) * position['size']
                equity.append(capital + current_value)
            else:
                equity.append(capital)
        
        # Close any open position
        if position is not None:
            final_price = df.iloc[-1]['Close']
            if position['type'] == 'long':
                pnl = (final_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - final_price) * position['size']
            
            capital += position['capital_used'] + pnl
            
            trades.append({
                'entry_date': df.index[position['entry_idx']],
                'exit_date': df.index[-1],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'pnl': pnl,
                'pnl_pct': (pnl / position['capital_used']) * 100,
                'exit_reason': 'End of Data'
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades, equity)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity,
            'final_capital': capital
        }
    
    def calculate_metrics(self, trades: List[Dict], equity_curve: List[float]) -> Dict:
        """Calculate performance metrics."""
        
        if len(trades) == 0:
            return {'sharpe_ratio': 0, 'total_trades': 0}
        
        # Basic statistics
        pnls = [t['pnl'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades]
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        # Calculate daily returns from equity curve
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Convert to daily (96 15-min bars per day)
        daily_returns = []
        for i in range(0, len(returns) - 96, 96):
            daily_return = (equity_array[i + 96] - equity_array[i]) / equity_array[i]
            daily_returns.append(daily_return)
        
        daily_returns = np.array(daily_returns)
        
        # Sharpe ratio
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_array)
        dd = (equity_array - peak) / peak
        max_dd = np.min(dd)
        
        # Other metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': sum(pnls),
            'avg_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0
        }


def optimize_parameters():
    """Run parameter optimization to find best settings."""
    
    # Load data once
    print("Loading data for optimization...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use subset for faster optimization
    df_subset = df.iloc[-100000:]  # Last ~2 years
    
    # Parameter grid
    param_grid = {
        'risk_per_trade': [0.01, 0.015, 0.02, 0.025],
        'atr_stop_mult': [1.5, 2.0, 2.5],
        'atr_target_mult': [2.0, 3.0, 4.0]
    }
    
    best_sharpe = -999
    best_params = {}
    results_list = []
    
    print("\nRunning parameter optimization...")
    
    # Grid search
    for risk in param_grid['risk_per_trade']:
        for stop_mult in param_grid['atr_stop_mult']:
            for target_mult in param_grid['atr_target_mult']:
                # Skip if target < stop
                if target_mult <= stop_mult:
                    continue
                
                # Run strategy
                strategy = FastTrendStrategy(
                    initial_capital=1000000,
                    risk_per_trade=risk,
                    atr_stop_mult=stop_mult,
                    atr_target_mult=target_mult
                )
                
                try:
                    results = strategy.run_backtest(df_subset)
                    sharpe = results['metrics']['sharpe_ratio']
                    
                    result_entry = {
                        'params': {
                            'risk_per_trade': risk,
                            'atr_stop_mult': stop_mult,
                            'atr_target_mult': target_mult
                        },
                        'metrics': results['metrics']
                    }
                    results_list.append(result_entry)
                    
                    print(f"Risk: {risk}, Stop: {stop_mult}, Target: {target_mult} => Sharpe: {sharpe:.3f}")
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = result_entry['params']
                        
                except Exception as e:
                    print(f"Error with params {risk}, {stop_mult}, {target_mult}: {e}")
    
    # Sort results by Sharpe ratio
    results_list.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 5 PARAMETER COMBINATIONS")
    print("="*60)
    
    for i, result in enumerate(results_list[:5]):
        print(f"\nRank {i+1}:")
        print(f"Parameters: {result['params']}")
        print(f"Sharpe Ratio: {result['metrics']['sharpe_ratio']:.3f}")
        print(f"Total Return: {result['metrics']['total_return']:.2%}")
        print(f"Max Drawdown: {result['metrics']['max_drawdown']:.2%}")
        print(f"Win Rate: {result['metrics']['win_rate']:.2%}")
    
    # Run best params on full dataset
    print("\n" + "="*60)
    print("RUNNING BEST PARAMETERS ON FULL DATASET")
    print("="*60)
    
    best_strategy = FastTrendStrategy(
        initial_capital=1000000,
        **best_params
    )
    
    final_results = best_strategy.run_backtest(df)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output = {
        'version': 'v3_optimized',
        'timestamp': datetime.now().isoformat(),
        'optimization_results': results_list[:10],  # Top 10
        'best_params': best_params,
        'final_metrics': final_results['metrics'],
        'total_trades': len(final_results['trades']),
        'sample_trades': final_results['trades'][:20]  # First 20 trades
    }
    
    with open(f'results_v3_optimized_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Print final results
    print(f"\nBest Parameters: {best_params}")
    print("\nFINAL RESULTS:")
    print(f"Sharpe Ratio: {final_results['metrics']['sharpe_ratio']:.3f}")
    print(f"Total Return: {final_results['metrics']['total_return']:.2%}")
    print(f"Max Drawdown: {final_results['metrics']['max_drawdown']:.2%}")
    print(f"Total Trades: {final_results['metrics']['total_trades']}")
    print(f"Win Rate: {final_results['metrics']['win_rate']:.2%}")
    print(f"Profit Factor: {final_results['metrics']['profit_factor']:.2f}")
    print(f"Average Win: ${final_results['metrics']['avg_win']:.2f}")
    print(f"Average Loss: ${final_results['metrics']['avg_loss']:.2f}")
    
    return final_results


if __name__ == "__main__":
    optimize_parameters()