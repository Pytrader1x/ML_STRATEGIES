"""
Portfolio Backtesting Engine
Clean, efficient backtesting with proper cost accounting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import warnings
from .costs import FXCosts


@dataclass
class BacktestResult:
    """Container for backtest results"""
    trades: pd.DataFrame
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    metrics: Dict
    

class Backtest:
    """Portfolio backtesting engine with transaction costs"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_capital: float = 100000,
                 position_size: float = 0.02,  # 2% risk per trade
                 costs_model: Optional[FXCosts] = None):
        """
        Initialize backtester
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with DateTime index
        initial_capital : float
            Starting capital
        position_size : float
            Position size as fraction of capital
        costs_model : FXCosts, optional
            Transaction cost model
        """
        
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.costs_model = costs_model or FXCosts()
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate input data"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Check for lookahead bias in index
        if not self.data.index.is_monotonic_increasing:
            raise ValueError("Data index must be monotonically increasing")
    
    def run(self,
            signals: pd.Series,
            pair: str = 'EURUSD',
            stop_loss_pips: Optional[float] = None,
            take_profit_pips: Optional[float] = None,
            trailing_stop_pips: Optional[float] = None) -> BacktestResult:
        """
        Run backtest with given signals
        
        Parameters:
        -----------
        signals : pd.Series
            Trading signals (-1, 0, 1) aligned with data index
        pair : str
            Currency pair for cost calculations
        stop_loss_pips : float, optional
            Stop loss in pips
        take_profit_pips : float, optional
            Take profit in pips
        trailing_stop_pips : float, optional
            Trailing stop in pips
            
        Returns:
        --------
        BacktestResult object
        """
        
        # Align signals with data
        signals = signals.reindex(self.data.index, fill_value=0)
        
        # Initialize tracking variables
        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        positions = pd.Series(0, index=self.data.index)
        equity = self.initial_capital
        equity_curve = [equity]
        
        # Get pip size for this pair
        pip_size = self.costs_model.get_pip_size(pair)
        
        for i in range(1, len(self.data)):
            current_time = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            current_high = self.data['High'].iloc[i]
            current_low = self.data['Low'].iloc[i]
            signal = signals.iloc[i]
            
            # Check for exit conditions if in position
            if position != 0:
                exit_price = None
                exit_reason = None
                
                # Check stop loss
                if stop_loss_pips:
                    if position > 0 and current_low <= entry_price - stop_loss_pips * pip_size:
                        exit_price = entry_price - stop_loss_pips * pip_size
                        exit_reason = 'stop_loss'
                    elif position < 0 and current_high >= entry_price + stop_loss_pips * pip_size:
                        exit_price = entry_price + stop_loss_pips * pip_size
                        exit_reason = 'stop_loss'
                
                # Check take profit
                if not exit_price and take_profit_pips:
                    if position > 0 and current_high >= entry_price + take_profit_pips * pip_size:
                        exit_price = entry_price + take_profit_pips * pip_size
                        exit_reason = 'take_profit'
                    elif position < 0 and current_low <= entry_price - take_profit_pips * pip_size:
                        exit_price = entry_price - take_profit_pips * pip_size
                        exit_reason = 'take_profit'
                
                # Check signal exit/reversal
                if not exit_price and signal != position:
                    exit_price = current_price
                    exit_reason = 'signal'
                
                # Execute exit if needed
                if exit_price:
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pair': pair
                    }
                    trades.append(trade)
                    
                    # Update equity (simplified - assumes fixed position size)
                    gross_pnl = (exit_price - entry_price) * position
                    pnl_pct = gross_pnl / entry_price
                    trade_equity = equity * self.position_size * pnl_pct
                    equity += trade_equity
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    entry_time = None
            
            # Check for new entry
            if position == 0 and signal != 0:
                position = signal
                entry_price = current_price
                entry_time = current_time
            
            # Track position and equity
            positions.iloc[i] = position
            equity_curve.append(equity)
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            # Apply transaction costs
            trades_df = self.costs_model.apply_costs_to_trades(trades_df, pair)
        
        equity_curve = pd.Series(equity_curve[:-1], index=self.data.index)
        returns = equity_curve.pct_change().fillna(0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(returns, trades_df, equity_curve)
        
        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            metrics=metrics
        )
    
    def _calculate_metrics(self, 
                         returns: pd.Series, 
                         trades: pd.DataFrame,
                         equity_curve: pd.Series) -> Dict:
        """Calculate performance metrics"""
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        
        # Sharpe ratio (annualized)
        if returns.std() > 0:
            sharpe = np.sqrt(252 * 96) * returns.mean() / returns.std()  # 96 bars per day
        else:
            sharpe = 0
            
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() * 100
        
        # Trade statistics
        if len(trades) > 0:
            # Win rate
            trades['is_win'] = trades['net_return'] > 0
            win_rate = trades['is_win'].mean() * 100
            
            # Average win/loss
            wins = trades[trades['is_win']]
            losses = trades[~trades['is_win']]
            
            avg_win = wins['net_return'].mean() * 100 if len(wins) > 0 else 0
            avg_loss = losses['net_return'].mean() * 100 if len(losses) > 0 else 0
            
            # Profit factor
            total_wins = wins['net_return'].sum() if len(wins) > 0 else 0
            total_losses = abs(losses['net_return'].sum()) if len(losses) > 0 else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Cost analysis
            avg_cost = trades['total_cost_bps'].mean()
            total_cost_impact = trades['cost_impact'].sum() * 100
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_cost = 0
            total_cost_impact = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': abs(max_dd),
            'win_rate': win_rate,
            'num_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_cost_bps': avg_cost,
            'total_cost_impact_pct': total_cost_impact
        }
    
    def run_walk_forward(self,
                        signal_func,
                        signal_params: Dict,
                        pair: str,
                        train_periods: int = 252 * 96 * 3,  # 3 years
                        test_periods: int = 252 * 96,       # 1 year
                        step_periods: int = 21 * 96) -> List[BacktestResult]:
        """
        Walk-forward analysis
        
        Parameters:
        -----------
        signal_func : callable
            Signal generation function
        signal_params : dict
            Parameters for signal function
        pair : str
            Currency pair
        train_periods : int
            Training window size
        test_periods : int
            Test window size
        step_periods : int
            Step size for rolling window
            
        Returns:
        --------
        List of BacktestResult objects
        """
        
        results = []
        
        # Calculate windows
        total_periods = len(self.data)
        start_test = train_periods
        
        while start_test + test_periods <= total_periods:
            # Define windows
            train_start = start_test - train_periods
            train_end = start_test
            test_end = start_test + test_periods
            
            # Get data slices
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[start_test:test_end]
            
            # Generate signals on test data
            # (In practice, you might optimize params on train data first)
            test_prices = test_data['Close']
            signals = signal_func(test_prices, **signal_params)
            
            # Run backtest on test period
            test_backtest = Backtest(test_data, self.initial_capital, self.position_size)
            result = test_backtest.run(signals['signal'], pair)
            
            # Add window info to metrics
            result.metrics['train_start'] = train_data.index[0]
            result.metrics['train_end'] = train_data.index[-1]
            result.metrics['test_start'] = test_data.index[0]
            result.metrics['test_end'] = test_data.index[-1]
            
            results.append(result)
            
            # Move window
            start_test += step_periods
            
        return results