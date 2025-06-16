import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os


def download_data(symbol, start_date, end_date, interval='1h'):
    """
    Download historical data from Yahoo Finance.
    
    Parameters:
    -----------
    symbol : str
        Ticker symbol (e.g., 'SPY', 'AAPL')
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str
        Data interval ('1m', '5m', '15m', '1h', '1d')
    
    Returns:
    --------
    pd.DataFrame : OHLCV data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date, interval=interval)
    
    # Ensure column names match backtrader expectations
    data.columns = [col.capitalize() for col in data.columns]
    
    return data


def calculate_position_metrics(entry_price, stop_price, account_value, risk_percent=0.03):
    """
    Calculate position size and risk metrics.
    
    Parameters:
    -----------
    entry_price : float
        Entry price for the trade
    stop_price : float
        Stop loss price
    account_value : float
        Total account value
    risk_percent : float
        Percentage of account to risk (default 3%)
    
    Returns:
    --------
    dict : Position metrics including size, risk amount, etc.
    """
    risk_amount = account_value * risk_percent
    stop_distance = abs(entry_price - stop_price)
    
    if stop_distance == 0:
        return {
            'position_size': 0,
            'risk_amount': 0,
            'stop_distance': 0,
            'stop_distance_percent': 0
        }
    
    position_size = risk_amount / stop_distance
    stop_distance_percent = (stop_distance / entry_price) * 100
    
    return {
        'position_size': int(position_size),
        'risk_amount': risk_amount,
        'stop_distance': stop_distance,
        'stop_distance_percent': stop_distance_percent,
        'position_value': position_size * entry_price
    }


def analyze_trade_results(trades_df):
    """
    Analyze trade results and generate statistics.
    
    Parameters:
    -----------
    trades_df : pd.DataFrame
        DataFrame with trade history
    
    Returns:
    --------
    dict : Comprehensive trade statistics
    """
    if trades_df.empty:
        return {
            'total_trades': 0,
            'message': 'No trades to analyze'
        }
    
    # Basic statistics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    
    # PnL statistics
    total_pnl = trades_df['pnl'].sum()
    avg_pnl = trades_df['pnl'].mean()
    
    avg_win = winning_trades['pnl'].mean() if win_count > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if loss_count > 0 else 0
    
    largest_win = winning_trades['pnl'].max() if win_count > 0 else 0
    largest_loss = losing_trades['pnl'].min() if loss_count > 0 else 0
    
    # Risk metrics
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if loss_count > 0 else np.inf
    
    # Duration statistics
    if 'duration' in trades_df.columns:
        avg_duration = trades_df['duration'].mean()
        avg_win_duration = winning_trades['duration'].mean() if win_count > 0 else 0
        avg_loss_duration = losing_trades['duration'].mean() if loss_count > 0 else 0
    else:
        avg_duration = avg_win_duration = avg_loss_duration = None
    
    # Consecutive wins/losses
    trades_df['is_win'] = trades_df['pnl'] > 0
    
    # Calculate consecutive wins
    consecutive_wins = []
    consecutive_losses = []
    current_streak = 0
    is_win_streak = None
    
    for is_win in trades_df['is_win']:
        if is_win_streak is None:
            is_win_streak = is_win
            current_streak = 1
        elif is_win == is_win_streak:
            current_streak += 1
        else:
            if is_win_streak:
                consecutive_wins.append(current_streak)
            else:
                consecutive_losses.append(current_streak)
            is_win_streak = is_win
            current_streak = 1
    
    # Add the last streak
    if is_win_streak is not None:
        if is_win_streak:
            consecutive_wins.append(current_streak)
        else:
            consecutive_losses.append(current_streak)
    
    max_consecutive_wins = max(consecutive_wins) if consecutive_wins else 0
    max_consecutive_losses = max(consecutive_losses) if consecutive_losses else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'average_pnl': avg_pnl,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'profit_factor': profit_factor,
        'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'average_duration': avg_duration,
        'average_win_duration': avg_win_duration,
        'average_loss_duration': avg_loss_duration
    }


def generate_performance_report(backtest_results, trades_df, output_path='ADX_Strategy/performance_report.txt'):
    """
    Generate a comprehensive performance report.
    
    Parameters:
    -----------
    backtest_results : dict
        Results from backtest run
    trades_df : pd.DataFrame
        DataFrame with trade history
    output_path : str
        Path to save the report
    """
    trade_stats = analyze_trade_results(trades_df)
    
    report = []
    report.append("=" * 60)
    report.append("ADX TREND STRATEGY PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Portfolio Performance
    report.append("PORTFOLIO PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Initial Capital: ${10000:.2f}")  # Hardcoded for now
    report.append(f"Final Value: ${backtest_results['final_value']:.2f}")
    report.append(f"Total Return: {backtest_results['total_return'] * 100:.2f}%")
    report.append(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}" if backtest_results['sharpe_ratio'] else "Sharpe Ratio: N/A")
    report.append(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
    report.append("")
    
    # Trade Statistics
    report.append("TRADE STATISTICS")
    report.append("-" * 30)
    report.append(f"Total Trades: {trade_stats['total_trades']}")
    report.append(f"Winning Trades: {trade_stats['winning_trades']}")
    report.append(f"Losing Trades: {trade_stats['losing_trades']}")
    report.append(f"Win Rate: {trade_stats['win_rate']:.2f}%")
    report.append("")
    
    # Profit/Loss Analysis
    report.append("PROFIT/LOSS ANALYSIS")
    report.append("-" * 30)
    report.append(f"Total PnL: ${trade_stats['total_pnl']:.2f}")
    report.append(f"Average PnL: ${trade_stats['average_pnl']:.2f}")
    report.append(f"Average Win: ${trade_stats['average_win']:.2f}")
    report.append(f"Average Loss: ${trade_stats['average_loss']:.2f}")
    report.append(f"Largest Win: ${trade_stats['largest_win']:.2f}")
    report.append(f"Largest Loss: ${trade_stats['largest_loss']:.2f}")
    report.append("")
    
    # Risk Metrics
    report.append("RISK METRICS")
    report.append("-" * 30)
    report.append(f"Profit Factor: {trade_stats['profit_factor']:.2f}")
    report.append(f"Risk/Reward Ratio: {trade_stats['risk_reward_ratio']:.2f}")
    report.append(f"Max Consecutive Wins: {trade_stats['max_consecutive_wins']}")
    report.append(f"Max Consecutive Losses: {trade_stats['max_consecutive_losses']}")
    report.append("")
    
    # Save report
    report_text = '\n'.join(report)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {output_path}")
    
    return trade_stats


def prepare_data_for_ml(data, lookback_period=20):
    """
    Prepare data for machine learning models by creating features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data
    lookback_period : int
        Number of periods to look back for features
    
    Returns:
    --------
    pd.DataFrame : Data with ML features
    """
    df = data.copy()
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
    
    # Volatility features
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['atr'] = calculate_atr(df, period=14)
    
    # Volume features
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Technical indicators as features
    df['rsi'] = calculate_rsi(df['Close'], period=14)
    
    # Remove NaN values
    df.dropna(inplace=True)
    
    return df


def calculate_atr(data, period=14):
    """Calculate Average True Range."""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


if __name__ == '__main__':
    # Example: Download data and calculate position size
    print("Downloading sample data...")
    data = download_data('SPY', '2023-01-01', '2023-12-31', interval='1h')
    print(f"Downloaded {len(data)} rows of data")
    
    # Save sample data
    data.to_csv('ADX_Strategy/sample_data.csv')
    print("Sample data saved to sample_data.csv")
    
    # Example position calculation
    print("\nExample position calculation:")
    metrics = calculate_position_metrics(
        entry_price=450.00,
        stop_price=445.00,
        account_value=10000,
        risk_percent=0.03
    )
    
    for key, value in metrics.items():
        print(f"{key}: {value}")