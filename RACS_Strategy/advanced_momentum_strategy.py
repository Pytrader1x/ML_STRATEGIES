"""
Advanced Momentum Strategy with Risk Management
Extends the winning momentum strategy (Sharpe 1.286) with:
- ATR-based Stop Loss
- Trailing Stop Loss
- Take Profit Targets
- Position Sizing based on risk
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class AdvancedMomentumStrategy:
    """Momentum strategy with comprehensive risk management"""
    
    def __init__(self, data: pd.DataFrame,
                 # Original momentum parameters (proven winners)
                 lookback: int = 40,
                 entry_z: float = 1.5,
                 exit_z: float = 0.5,
                 # Risk management parameters
                 atr_period: int = 14,
                 sl_atr_multiplier: float = 2.0,
                 tp_atr_multiplier: float = 3.0,
                 trailing_sl_atr: float = 1.5,
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 initial_capital: float = 10000):
        """
        Parameters:
        -----------
        lookback : int
            Period for momentum calculation (default: 40 from winning strategy)
        entry_z : float
            Z-score threshold for entry (default: 1.5 from winning strategy)
        exit_z : float
            Z-score threshold for exit (default: 0.5 from winning strategy)
        atr_period : int
            Period for ATR calculation
        sl_atr_multiplier : float
            Stop loss distance in ATR units
        tp_atr_multiplier : float
            Take profit distance in ATR units
        trailing_sl_atr : float
            Trailing stop distance in ATR units
        risk_per_trade : float
            Maximum risk per trade as fraction of capital
        initial_capital : float
            Starting capital for position sizing
        """
        self.data = data.copy()
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_period = atr_period
        self.sl_atr_multiplier = sl_atr_multiplier
        self.tp_atr_multiplier = tp_atr_multiplier
        self.trailing_sl_atr = trailing_sl_atr
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital
        
        # Initialize tracking variables
        self.trades = []
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.position_size = 0
        
    def calculate_indicators(self):
        """Calculate all required indicators"""
        df = self.data
        
        # Original momentum indicators
        df['Momentum'] = df['Close'].pct_change(self.lookback)
        df['Mom_Mean'] = df['Momentum'].rolling(50).mean()
        df['Mom_Std'] = df['Momentum'].rolling(50).std()
        df['Mom_Z'] = (df['Momentum'] - df['Mom_Mean']) / df['Mom_Std']
        
        # ATR for risk management
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
        df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        df['ATR'] = df['True_Range'].rolling(self.atr_period).mean()
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def calculate_position_size(self, entry_price, stop_loss, current_capital):
        """Calculate position size based on risk management rules"""
        risk_amount = current_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk > 0:
            position_size = risk_amount / price_risk
        else:
            position_size = 0
            
        # Cap position size to prevent over-leveraging
        max_position_size = current_capital / entry_price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def run_backtest(self):
        """Run the backtest with risk management"""
        df = self.calculate_indicators()
        
        # Initialize result arrays
        df['Signal'] = 0
        df['Position'] = 0
        df['Position_Size'] = 0
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        df['Trailing_Stop'] = np.nan
        
        # Track capital for position sizing
        current_capital = self.initial_capital
        capital_history = [current_capital]
        
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            current_atr = df['ATR'].iloc[i]
            current_z = df['Mom_Z'].iloc[i]
            
            # Check if we have an open position
            if self.current_position != 0:
                # Check stop loss
                if (self.current_position > 0 and current_price <= self.stop_loss) or \
                   (self.current_position < 0 and current_price >= self.stop_loss):
                    # Stop loss hit
                    self.close_position(i, df, 'Stop Loss', current_capital)
                    
                # Check take profit
                elif (self.current_position > 0 and current_price >= self.take_profit) or \
                     (self.current_position < 0 and current_price <= self.take_profit):
                    # Take profit hit
                    self.close_position(i, df, 'Take Profit', current_capital)
                    
                # Check trailing stop
                else:
                    # Update trailing stop
                    if self.current_position > 0:
                        new_trailing_stop = current_price - (self.trailing_sl_atr * current_atr)
                        if new_trailing_stop > self.trailing_stop:
                            self.trailing_stop = new_trailing_stop
                            df.loc[df.index[i], 'Trailing_Stop'] = self.trailing_stop
                        
                        # Check if trailing stop hit
                        if current_price <= self.trailing_stop:
                            self.close_position(i, df, 'Trailing Stop', current_capital)
                            
                    else:  # Short position
                        new_trailing_stop = current_price + (self.trailing_sl_atr * current_atr)
                        if new_trailing_stop < self.trailing_stop:
                            self.trailing_stop = new_trailing_stop
                            df.loc[df.index[i], 'Trailing_Stop'] = self.trailing_stop
                        
                        # Check if trailing stop hit
                        if current_price >= self.trailing_stop:
                            self.close_position(i, df, 'Trailing Stop', current_capital)
                
                # Check momentum exit condition (original strategy)
                if self.current_position != 0 and abs(current_z) < self.exit_z:
                    self.close_position(i, df, 'Momentum Exit', current_capital)
                    
            # Check entry conditions if no position
            if self.current_position == 0:
                # Long entry
                if current_z < -self.entry_z:
                    self.stop_loss = current_price - (self.sl_atr_multiplier * current_atr)
                    self.take_profit = current_price + (self.tp_atr_multiplier * current_atr)
                    self.trailing_stop = self.stop_loss
                    
                    self.position_size = self.calculate_position_size(
                        current_price, self.stop_loss, current_capital
                    )
                    
                    if self.position_size > 0:
                        self.current_position = 1
                        self.entry_price = current_price
                        
                        df.loc[df.index[i], 'Signal'] = 1
                        df.loc[df.index[i], 'Stop_Loss'] = self.stop_loss
                        df.loc[df.index[i], 'Take_Profit'] = self.take_profit
                        df.loc[df.index[i], 'Trailing_Stop'] = self.trailing_stop
                        
                        self.trades.append({
                            'entry_date': df.index[i],
                            'entry_price': current_price,
                            'position': 'Long',
                            'size': self.position_size,
                            'stop_loss': self.stop_loss,
                            'take_profit': self.take_profit
                        })
                
                # Short entry
                elif current_z > self.entry_z:
                    self.stop_loss = current_price + (self.sl_atr_multiplier * current_atr)
                    self.take_profit = current_price - (self.tp_atr_multiplier * current_atr)
                    self.trailing_stop = self.stop_loss
                    
                    self.position_size = self.calculate_position_size(
                        current_price, self.stop_loss, current_capital
                    )
                    
                    if self.position_size > 0:
                        self.current_position = -1
                        self.entry_price = current_price
                        
                        df.loc[df.index[i], 'Signal'] = -1
                        df.loc[df.index[i], 'Stop_Loss'] = self.stop_loss
                        df.loc[df.index[i], 'Take_Profit'] = self.take_profit
                        df.loc[df.index[i], 'Trailing_Stop'] = self.trailing_stop
                        
                        self.trades.append({
                            'entry_date': df.index[i],
                            'entry_price': current_price,
                            'position': 'Short',
                            'size': self.position_size,
                            'stop_loss': self.stop_loss,
                            'take_profit': self.take_profit
                        })
            
            # Update position and capital
            df.loc[df.index[i], 'Position'] = self.current_position
            df.loc[df.index[i], 'Position_Size'] = self.position_size if self.current_position != 0 else 0
            
            # Calculate returns and update capital
            if i > 0 and df['Position'].iloc[i-1] != 0:
                price_change = (current_price - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                position_return = df['Position'].iloc[i-1] * price_change * df['Position_Size'].iloc[i-1]
                current_capital += position_return
                
            capital_history.append(current_capital)
        
        # Calculate performance metrics
        df['Capital'] = capital_history[:len(df)]
        df['Returns'] = df['Capital'].pct_change()
        df['Strategy_Returns'] = df['Returns']
        df['Cumulative_Returns'] = df['Capital'] / self.initial_capital
        
        return df
    
    def close_position(self, i, df, exit_reason, current_capital):
        """Close current position and record trade"""
        exit_price = df['Close'].iloc[i]
        
        if self.trades:
            self.trades[-1]['exit_date'] = df.index[i]
            self.trades[-1]['exit_price'] = exit_price
            self.trades[-1]['exit_reason'] = exit_reason
            
            # Calculate trade P&L
            if self.current_position > 0:
                trade_pnl = (exit_price - self.entry_price) / self.entry_price
            else:
                trade_pnl = (self.entry_price - exit_price) / self.entry_price
                
            self.trades[-1]['pnl_pct'] = trade_pnl * 100
            self.trades[-1]['pnl_amount'] = trade_pnl * self.position_size * self.entry_price
        
        # Reset position
        self.current_position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.position_size = 0
        
    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        returns = df['Strategy_Returns'].dropna()
        
        # Skip if no returns
        if len(returns) == 0 or returns.std() == 0:
            return {
                'sharpe': 0,
                'returns': 0,
                'win_rate': 0,
                'max_dd': 0,
                'trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        total_returns = (df['Capital'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Drawdown calculation
        cumulative = df['Cumulative_Returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_dd = drawdown.min()
        
        # Trade statistics
        trade_df = pd.DataFrame(self.trades)
        if len(trade_df) > 0 and 'pnl_pct' in trade_df.columns:
            winning_trades = trade_df[trade_df['pnl_pct'] > 0]
            losing_trades = trade_df[trade_df['pnl_pct'] <= 0]
            
            win_rate = len(winning_trades) / len(trade_df) * 100 if len(trade_df) > 0 else 0
            avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
            
            # Profit factor
            total_wins = winning_trades['pnl_pct'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl_pct'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Exit reason analysis
            exit_analysis = trade_df['exit_reason'].value_counts() if 'exit_reason' in trade_df.columns else pd.Series()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            exit_analysis = pd.Series()
        
        return {
            'sharpe': sharpe,
            'returns': total_returns,
            'win_rate': win_rate,
            'max_dd': abs(max_dd),
            'trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'exit_analysis': exit_analysis.to_dict() if len(exit_analysis) > 0 else {}
        }


def run_advanced_backtest(data_path='../data/AUDUSD_MASTER_15M.csv',
                         test_period='last_50000',
                         optimize_risk_params=False):
    """Run the advanced momentum strategy with risk management"""
    
    print("="*60)
    print("Advanced Momentum Strategy with Risk Management")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {data_path}")
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    # Select test period
    if test_period == 'last_50000':
        data = data[-50000:]
    elif test_period == 'last_20000':
        data = data[-20000:]
    elif test_period == 'full':
        pass
    else:
        # Custom date range
        if isinstance(test_period, tuple):
            start_date, end_date = test_period
            data = data[start_date:end_date]
    
    print(f"Testing on {len(data):,} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    if optimize_risk_params:
        # Test different risk management parameters
        results = []
        
        sl_multipliers = [1.5, 2.0, 2.5, 3.0]
        tp_multipliers = [2.0, 3.0, 4.0, 5.0]
        trailing_multipliers = [1.0, 1.5, 2.0]
        
        for sl in sl_multipliers:
            for tp in tp_multipliers:
                for trail in trailing_multipliers:
                    if tp > sl:  # TP should be larger than SL
                        strategy = AdvancedMomentumStrategy(
                            data,
                            sl_atr_multiplier=sl,
                            tp_atr_multiplier=tp,
                            trailing_sl_atr=trail
                        )
                        
                        df = strategy.run_backtest()
                        metrics = strategy.calculate_metrics(df)
                        
                        results.append({
                            'sl_atr': sl,
                            'tp_atr': tp,
                            'trail_atr': trail,
                            'sharpe': metrics['sharpe'],
                            'returns': metrics['returns'],
                            'max_dd': metrics['max_dd'],
                            'win_rate': metrics['win_rate'],
                            'profit_factor': metrics['profit_factor']
                        })
                        
                        print(f"\nSL: {sl}, TP: {tp}, Trail: {trail}")
                        print(f"Sharpe: {metrics['sharpe']:.3f}, Returns: {metrics['returns']:.1f}%")
        
        # Find best parameters
        results_df = pd.DataFrame(results)
        best_idx = results_df['sharpe'].idxmax()
        best_params = results_df.iloc[best_idx]
        
        print("\n" + "="*40)
        print("BEST RISK PARAMETERS:")
        print(f"SL ATR Multiplier: {best_params['sl_atr']}")
        print(f"TP ATR Multiplier: {best_params['tp_atr']}")
        print(f"Trailing ATR: {best_params['trail_atr']}")
        print(f"Sharpe: {best_params['sharpe']:.3f}")
        print(f"Returns: {best_params['returns']:.1f}%")
        print("="*40)
        
        # Save results
        results_df.to_csv('risk_optimization_results.csv', index=False)
        
    else:
        # Run with default parameters
        strategy = AdvancedMomentumStrategy(data)
        df = strategy.run_backtest()
        metrics = strategy.calculate_metrics(df)
        
        print("\n" + "-"*40)
        print("BACKTEST RESULTS:")
        print(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"Total Returns: {metrics['returns']:.1f}%")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Max Drawdown: {metrics['max_dd']:.1f}%")
        print(f"Total Trades: {metrics['trades']}")
        print(f"Average Win: {metrics['avg_win']:.2f}%")
        print(f"Average Loss: {metrics['avg_loss']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        if metrics['exit_analysis']:
            print("\nExit Analysis:")
            for reason, count in metrics['exit_analysis'].items():
                print(f"  {reason}: {count}")
        
        # Save trade log
        if strategy.trades:
            trade_df = pd.DataFrame(strategy.trades)
            trade_df.to_csv('advanced_strategy_trades.csv', index=False)
            print("\nTrade log saved to advanced_strategy_trades.csv")
        
        # Plot results
        plot_advanced_results(df, strategy)
        
        return df, metrics, strategy


def plot_advanced_results(df, strategy):
    """Create comprehensive plots for the advanced strategy"""
    
    fig, axes = plt.subplots(5, 1, figsize=(15, 15))
    
    # 1. Price and signals with risk levels
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], 'b-', alpha=0.5, linewidth=0.5)
    
    # Mark entries and exits
    long_entries = df[(df['Signal'] == 1)]
    short_entries = df[(df['Signal'] == -1)]
    
    if len(long_entries) > 0:
        ax1.scatter(long_entries.index, long_entries['Close'], 
                   color='green', marker='^', s=50, alpha=0.7, label='Long Entry')
    if len(short_entries) > 0:
        ax1.scatter(short_entries.index, short_entries['Close'], 
                   color='red', marker='v', s=50, alpha=0.7, label='Short Entry')
    
    # Plot stop loss and take profit levels
    for i in range(len(df)):
        if not pd.isna(df['Stop_Loss'].iloc[i]):
            ax1.plot([df.index[i], df.index[min(i+10, len(df)-1)]], 
                    [df['Stop_Loss'].iloc[i], df['Stop_Loss'].iloc[i]], 
                    'r--', alpha=0.3, linewidth=1)
        if not pd.isna(df['Take_Profit'].iloc[i]):
            ax1.plot([df.index[i], df.index[min(i+10, len(df)-1)]], 
                    [df['Take_Profit'].iloc[i], df['Take_Profit'].iloc[i]], 
                    'g--', alpha=0.3, linewidth=1)
    
    ax1.set_title('Price Action with Risk Management Levels', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ATR
    ax2 = axes[1]
    ax2.plot(df.index, df['ATR'], 'purple', alpha=0.7)
    ax2.set_title('Average True Range (ATR)', fontsize=14)
    ax2.set_ylabel('ATR')
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-Score
    ax3 = axes[2]
    ax3.plot(df.index, df['Mom_Z'], 'purple', alpha=0.7)
    ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Short Entry')
    ax3.axhline(y=-1.5, color='green', linestyle='--', alpha=0.5, label='Long Entry')
    ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax3.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.3)
    ax3.fill_between(df.index, -0.5, 0.5, alpha=0.1, color='gray', label='Exit Zone')
    ax3.set_title('Momentum Z-Score', fontsize=14)
    ax3.set_ylabel('Z-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Position with size
    ax4 = axes[3]
    ax4.fill_between(df.index, 0, df['Position'] * df['Position_Size'], 
                     where=(df['Position'] > 0), color='green', alpha=0.3, label='Long')
    ax4.fill_between(df.index, 0, df['Position'] * df['Position_Size'], 
                     where=(df['Position'] < 0), color='red', alpha=0.3, label='Short')
    ax4.set_title('Position Size Over Time', fontsize=14)
    ax4.set_ylabel('Position Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Portfolio value
    ax5 = axes[4]
    ax5.plot(df.index, df['Capital'], 'green', linewidth=2, label='Portfolio Value')
    ax5.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax5.set_title('Portfolio Value', fontsize=14)
    ax5.set_ylabel('Value ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_momentum_strategy_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create a separate plot for drawdown
    fig2, ax = plt.subplots(figsize=(15, 5))
    cumulative = df['Cumulative_Returns']
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    ax.fill_between(df.index, 0, drawdown, color='red', alpha=0.3)
    ax.plot(df.index, drawdown, 'red', linewidth=1)
    ax.set_title('Drawdown Analysis', fontsize=14)
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_strategy_drawdown.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run advanced momentum strategy with risk management')
    parser.add_argument('--optimize', action='store_true', help='Optimize risk parameters')
    parser.add_argument('--period', default='last_50000', help='Test period (last_50000, last_20000, full)')
    
    args = parser.parse_args()
    
    # Run backtest
    results = run_advanced_backtest(
        optimize_risk_params=args.optimize,
        test_period=args.period
    )
    
    print("\n" + "="*60)
    print("Advanced Strategy Backtest Complete!")
    print("="*60)
    
    # Compare with original strategy
    print("\nComparison with Original Strategy:")
    print("Original: Sharpe 1.286, No risk management")
    print("Advanced: Includes SL, TP, and Trailing Stops")
    print("\nRun with --optimize flag to find best risk parameters")