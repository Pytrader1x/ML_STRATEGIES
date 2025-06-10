import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from technical_indicators_custom import TIC
from technical_indicators_custom.plotting import IndicatorPlotter
from technical_indicators_custom.plotting import plot_neurotrend_market_bias_chop


@dataclass
class Trade:
    """Data class to store trade information"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    position_size: float
    stop_loss: float
    take_profits: List[float]
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    trailing_stop: Optional[float] = None
    tp_hits: int = 0
    remaining_size: float = None
    partial_pnl: float = 0.0  # Track cumulative P&L from partial exits
    tp_exit_times: List[pd.Timestamp] = None
    tp_exit_prices: List[float] = None
    partial_exits: List[Dict] = None  # Track all partial exits for visualization
    
    def __post_init__(self):
        if self.remaining_size is None:
            self.remaining_size = self.position_size
        if self.tp_exit_times is None:
            self.tp_exit_times = []
        if self.tp_exit_prices is None:
            self.tp_exit_prices = []
        if self.partial_exits is None:
            self.partial_exits = []


class Strategy_1:
    """
    Backtesting class for Strategy 1 - NeuroTrend + Market Bias + Intelligent Chop
    
    This strategy enters trades when all three indicators align:
    - Long: NeuroTrend uptrend + Bullish Market Bias + Weak/Strong Trend
    - Short: NeuroTrend downtrend + Bearish Market Bias + Weak/Strong Trend
    
    Risk Management:
    - Three-tiered take profit system (33% at each level)
    - Intelligent trailing stop loss based on ATR
    - Early exit on signal flip
    """
    
    def __init__(self, initial_capital: float = 10000, risk_per_trade: float = 0.02,
                 tp_atr_multipliers: Tuple[float, float, float] = (1.0, 2.0, 3.0),
                 sl_atr_multiplier: float = 2.0, 
                 trailing_atr_multiplier: float = 1.5,
                 tsl_activation_pips: float = 15,  # Pips profit to activate TSL
                 tsl_min_profit_pips: float = 5,   # Minimum pips to lock in
                 max_tp_percent: float = 0.01,
                 min_lot_size: float = 1000000,  # 1M AUD minimum
                 pip_value_per_million: float = 100,
                 verbose: bool = False):
        """
        Initialize the Strategy_1 backtesting class
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital for backtesting
        risk_per_trade : float
            Percentage of capital to risk per trade (e.g., 0.02 = 2%)
        tp_atr_multipliers : tuple
            ATR multipliers for the three take profit levels
        sl_atr_multiplier : float
            ATR multiplier for initial stop loss
        trailing_atr_multiplier : float
            ATR multiplier for trailing stop loss
        tsl_activation_pips : float
            Pips of profit required to activate trailing stop (default 15)
        tsl_min_profit_pips : float
            Minimum pips of profit to lock in with trailing stop (default 5)
        max_tp_percent : float
            Maximum take profit as percentage of entry price (e.g., 0.01 = 1%)
        min_lot_size : float
            Minimum trade size in units (default 1M AUD)
        pip_value_per_million : float
            P&L per pip per million units (default $100 for AUDUSD)
        verbose : bool
            Whether to print detailed trade information
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.tp_atr_multipliers = tp_atr_multipliers
        self.sl_atr_multiplier = sl_atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.tsl_activation_pips = tsl_activation_pips
        self.tsl_min_profit_pips = tsl_min_profit_pips
        self.max_tp_percent = max_tp_percent
        self.min_lot_size = min_lot_size
        self.pip_value_per_million = pip_value_per_million
        self.verbose = verbose
        
        # Trading state
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        
        # Performance metrics
        self.equity_curve = []
        self.drawdown_curve = []
        
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management rules"""
        risk_amount = self.current_capital * self.risk_per_trade
        price_risk_pips = abs(entry_price - stop_loss) * 10000  # Convert to pips for AUDUSD
        
        if price_risk_pips == 0:
            return 0
        
        # Calculate ideal position size in millions
        # risk_amount = position_size_millions * pip_value_per_million * price_risk_pips
        ideal_size_millions = risk_amount / (self.pip_value_per_million * price_risk_pips)
        
        # Round to nearest million (minimum 1M)
        position_size_millions = max(1.0, round(ideal_size_millions))
        
        # Convert back to units
        position_size = position_size_millions * self.min_lot_size
        
        # Check if we have enough capital for this trade
        # Each million requires margin (simplified: using 1% margin requirement)
        required_margin = position_size * 0.01
        if required_margin > self.current_capital:
            # Reduce position size to what we can afford
            affordable_millions = int(self.current_capital / (self.min_lot_size * 0.01))
            position_size = affordable_millions * self.min_lot_size
            
        return position_size
    
    def calculate_take_profits(self, entry_price: float, direction: str, atr: float) -> List[float]:
        """Calculate three take profit levels based on ATR"""
        take_profits = []
        
        # TP1: Closer target using first multiplier
        tp1_distance = atr * self.tp_atr_multipliers[0]
        tp1_distance = min(tp1_distance, entry_price * 0.003)  # Max 0.3% for TP1
        
        # TP2: Medium distance using second multiplier
        tp2_distance = atr * self.tp_atr_multipliers[1]
        tp2_distance = min(tp2_distance, entry_price * 0.006)  # Max 0.6% for TP2
        
        # TP3: Maximum 1% from entry
        tp3_distance = atr * self.tp_atr_multipliers[2]
        tp3_distance = min(tp3_distance, entry_price * 0.01)  # Max 1% for TP3
        
        if direction == 'long':
            take_profits = [
                entry_price + tp1_distance,
                entry_price + tp2_distance,
                entry_price + tp3_distance
            ]
        else:
            take_profits = [
                entry_price - tp1_distance,
                entry_price - tp2_distance,
                entry_price - tp3_distance
            ]
            
        return take_profits
    
    def calculate_stop_loss(self, entry_price: float, direction: str, atr: float, row: pd.Series) -> float:
        """Calculate initial stop loss based on ATR and Market Bias bar"""
        sl_distance = atr * self.sl_atr_multiplier
        
        if direction == 'long':
            # For long positions, stop loss below entry
            atr_stop = entry_price - sl_distance
            
            # Check if Market Bias bar is available
            if 'MB_l2' in row and not pd.isna(row['MB_l2']):
                # Place stop just below Market Bias bar lower level
                mb_stop = row['MB_l2'] - 0.00005  # 0.5 pip buffer below MB low
                # Use whichever is closer to entry (more conservative)
                stop_loss = max(atr_stop, mb_stop)
            else:
                stop_loss = atr_stop
        else:
            # For short positions, stop loss above entry
            atr_stop = entry_price + sl_distance
            
            # Check if Market Bias bar is available
            if 'MB_h2' in row and not pd.isna(row['MB_h2']):
                # Place stop just above Market Bias bar upper level
                mb_stop = row['MB_h2'] + 0.00005  # 0.5 pip buffer above MB high
                # Use whichever is closer to entry (more conservative)
                stop_loss = min(atr_stop, mb_stop)
            else:
                stop_loss = atr_stop
            
        return stop_loss
    
    def update_trailing_stop(self, current_price: float, trade: Trade, atr: float) -> float:
        """Update trailing stop loss - activates at 15 pips profit, locks in 5 pips minimum"""
        # Convert pip thresholds to price
        activation_pips = self.tsl_activation_pips
        min_profit_pips = self.tsl_min_profit_pips
        pip_size = 0.0001  # For AUDUSD
        
        if trade.direction == 'long':
            # Calculate profit in pips
            profit_pips = (current_price - trade.entry_price) / pip_size
            
            # Only activate trailing stop if we're 15+ pips in profit
            if profit_pips >= activation_pips:
                # Set stop to lock in minimum 5 pips profit
                min_profit_stop = trade.entry_price + (min_profit_pips * pip_size)
                
                # Also calculate ATR-based trailing stop
                atr_trailing_stop = current_price - (atr * self.trailing_atr_multiplier)
                
                # Use the higher of the two (to ensure minimum profit)
                new_trailing_stop = max(min_profit_stop, atr_trailing_stop)
                
                # Only update if new stop is higher than current
                if trade.trailing_stop is None or new_trailing_stop > trade.trailing_stop:
                    return new_trailing_stop
                    
        else:  # short
            # Calculate profit in pips
            profit_pips = (trade.entry_price - current_price) / pip_size
            
            # Only activate trailing stop if we're 15+ pips in profit
            if profit_pips >= activation_pips:
                # Set stop to lock in minimum 5 pips profit
                min_profit_stop = trade.entry_price - (min_profit_pips * pip_size)
                
                # Also calculate ATR-based trailing stop
                atr_trailing_stop = current_price + (atr * self.trailing_atr_multiplier)
                
                # Use the lower of the two (to ensure minimum profit)
                new_trailing_stop = min(min_profit_stop, atr_trailing_stop)
                
                # Only update if new stop is lower than current
                if trade.trailing_stop is None or new_trailing_stop < trade.trailing_stop:
                    return new_trailing_stop
        
        return trade.trailing_stop if trade.trailing_stop is not None else trade.stop_loss
    
    def check_entry_conditions(self, row: pd.Series) -> Optional[str]:
        """Check if entry conditions are met"""
        # Long entry conditions
        if (row['NTI_Direction'] == 1 and  # NeuroTrend uptrend
            row['MB_Bias'] == 1 and  # Bullish Market Bias
            row['IC_Regime'] in [1, 2]):  # Weak or Strong Trend
            return 'long'
        
        # Short entry conditions
        elif (row['NTI_Direction'] == -1 and  # NeuroTrend downtrend
              row['MB_Bias'] == -1 and  # Bearish Market Bias
              row['IC_Regime'] in [1, 2]):  # Weak or Strong Trend
            return 'short'
        
        return None
    
    def check_exit_conditions(self, row: pd.Series, trade: Trade) -> Tuple[bool, str]:
        """Check if exit conditions are met"""
        current_price = row['Close']
        
        # Check take profit levels - use High/Low for more accurate TP detection
        if trade.direction == 'long':
            for i, tp in enumerate(trade.take_profits):
                if i >= trade.tp_hits and row['High'] >= tp:
                    return True, f'take_profit_{i+1}'
        else:  # short
            for i, tp in enumerate(trade.take_profits):
                if i >= trade.tp_hits and row['Low'] <= tp:
                    return True, f'take_profit_{i+1}'
        
        # Check stop loss (including trailing stop)
        current_stop = trade.trailing_stop if trade.trailing_stop is not None else trade.stop_loss
        
        if trade.direction == 'long' and current_price <= current_stop:
            exit_reason = 'trailing_stop' if trade.trailing_stop is not None else 'stop_loss'
            return True, exit_reason
        elif trade.direction == 'short' and current_price >= current_stop:
            exit_reason = 'trailing_stop' if trade.trailing_stop is not None else 'stop_loss'
            return True, exit_reason
        
        # Check early exit on signal flip
        if trade.direction == 'long':
            if row['NTI_Direction'] == -1 or row['MB_Bias'] == -1:
                return True, 'signal_flip'
        else:  # short
            if row['NTI_Direction'] == 1 or row['MB_Bias'] == 1:
                return True, 'signal_flip'
        
        return False, None
    
    def execute_trade_exit(self, trade: Trade, exit_time: pd.Timestamp, 
                          exit_price: float, exit_reason: str) -> Trade:
        """Execute trade exit and calculate PnL"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        
        # Handle partial exits for take profits
        if 'take_profit' in exit_reason:
            tp_index = int(exit_reason.split('_')[-1]) - 1
            trade.tp_hits = tp_index + 1
            
            # Close 33% of original position
            exit_size = trade.position_size / 3
            trade.remaining_size -= exit_size
            
            # Calculate PnL for this partial exit
            if trade.direction == 'long':
                price_change_pips = (exit_price - trade.entry_price) * 10000
            else:
                price_change_pips = (trade.entry_price - exit_price) * 10000
                
            # Calculate P&L: millions * pip_value_per_million * pips
            millions_exited = exit_size / self.min_lot_size
            partial_pnl = millions_exited * self.pip_value_per_million * price_change_pips
            
            # Add to cumulative partial P&L
            trade.partial_pnl += partial_pnl
            self.current_capital += partial_pnl  # Update capital immediately
            
            # Record TP exit
            trade.tp_exit_times.append(exit_time)
            trade.tp_exit_prices.append(exit_price)
            
            # Record partial exit for visualization
            trade.partial_exits.append({
                'time': exit_time,
                'price': exit_price,
                'tp_level': trade.tp_hits,
                'pnl': partial_pnl,
                'size': exit_size
            })
            
            # Debug logging
            if self.verbose:
                print(f"TP{trade.tp_hits} hit at {exit_time}: Exit {millions_exited:.1f}M at {exit_price:.5f}, P&L: ${partial_pnl:.2f}")
            
            # If this is the third TP or no position left, close entire trade
            if trade.tp_hits >= 3 or trade.remaining_size <= 0:
                trade.remaining_size = 0
                # Total P&L is the sum of all partial exits
                trade.pnl = trade.partial_pnl
            else:
                # Continue with partial position
                return None  # Signal to continue the trade
                
        else:
            # Full exit (stop loss, trailing stop, or signal flip)
            if trade.direction == 'long':
                price_change_pips = (exit_price - trade.entry_price) * 10000
            else:
                price_change_pips = (trade.entry_price - exit_price) * 10000
                
            remaining_millions = trade.remaining_size / self.min_lot_size
            remaining_pnl = remaining_millions * self.pip_value_per_million * price_change_pips
            
            # Total P&L includes any partial profits already taken
            trade.pnl = trade.partial_pnl + remaining_pnl
            trade.remaining_size = 0
            
            # Update capital with remaining P&L
            self.current_capital += remaining_pnl
        
        # Calculate percentage return based on original position size
        original_value = trade.entry_price * trade.position_size
        trade.pnl_percent = (trade.pnl / original_value) * 100
        
        return trade
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run the backtest on the provided DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data and calculated indicators
            
        Returns:
        --------
        dict
            Dictionary containing backtest results and performance metrics
        """
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'NTI_Direction', 
                        'MB_Bias', 'IC_Regime', 'IC_ATR_Normalized']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.trades = []
        self.current_trade = None
        self.equity_curve = [self.initial_capital]
        
        # Iterate through the DataFrame
        for idx in range(1, len(df)):
            current_row = df.iloc[idx]
            current_time = df.index[idx]
            
            # Update equity curve
            self.equity_curve.append(self.current_capital)
            
            # If we have an open trade, check exit conditions
            if self.current_trade is not None:
                # Update trailing stop
                atr = current_row['IC_ATR_Normalized']
                self.current_trade.trailing_stop = self.update_trailing_stop(
                    current_row['Close'], self.current_trade, atr
                )
                
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(current_row, self.current_trade)
                
                if should_exit:
                    exit_price = current_row['Close']
                    completed_trade = self.execute_trade_exit(
                        self.current_trade, current_time, exit_price, exit_reason
                    )
                    
                    if completed_trade is not None:
                        # Trade fully closed
                        self.trades.append(self.current_trade)
                        self.current_trade = None
                    # else: partial exit, trade continues
            
            # If no open trade, check entry conditions
            elif self.current_trade is None:
                entry_signal = self.check_entry_conditions(current_row)
                
                if entry_signal is not None:
                    # Calculate trade parameters
                    entry_price = current_row['Close']
                    atr = current_row['IC_ATR_Normalized']
                    
                    # Calculate stop loss and position size
                    stop_loss = self.calculate_stop_loss(entry_price, entry_signal, atr, current_row)
                    position_size = self.calculate_position_size(entry_price, stop_loss)
                    
                    # Calculate take profit levels
                    take_profits = self.calculate_take_profits(entry_price, entry_signal, atr)
                    
                    # Create new trade
                    self.current_trade = Trade(
                        entry_time=current_time,
                        entry_price=entry_price,
                        direction=entry_signal,
                        position_size=position_size,
                        stop_loss=stop_loss,
                        take_profits=take_profits
                    )
        
        # Close any remaining open trade at the end
        if self.current_trade is not None:
            last_row = df.iloc[-1]
            last_time = df.index[-1]
            exit_price = last_row['Close']
            
            self.execute_trade_exit(
                self.current_trade, last_time, exit_price, 'end_of_data'
            )
            self.trades.append(self.current_trade)
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics()
        results['trades'] = self.trades
        results['equity_curve'] = self.equity_curve
        
        return results
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Win/Loss metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Maximum drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # Sharpe ratio (simplified - daily returns)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Exit reason breakdown
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = 0
            exit_reasons[reason] += 1
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'exit_reasons': exit_reasons,
            'final_capital': self.current_capital
        }


# ============================================================================
# Combined NeuroTrend + Market Bias + Intelligent Chop Plot
# ============================================================================

def plot_neurotrend_market_bias_chop(df: pd.DataFrame,
                                    title: Optional[str] = None,
                                    figsize: Tuple[int, int] = (16, 12),
                                    save_path: Optional[Union[str, Path]] = None,
                                    show: bool = True,
                                    show_indicators: bool = True,
                                    show_chop_subplots: bool = True,
                                    use_chop_background: bool = False,
                                    single_plot_height_ratio: float = 0.5,
                                    main_height_ratio: float = 4,
                                    chop_height_ratio: float = 3,
                                    simplified_regime_colors: bool = False,
                                    trend_color: Union[str, Tuple[float, float, float]] = '#4ECDC4',
                                    range_color: Union[str, Tuple[float, float, float]] = '#95A5A6',
                                    trades: Optional[List[Dict]] = None,
                                    show_pnl: bool = True,
                                    performance_metrics: Optional[Dict] = None) -> plt.Figure:
    """
    Create a combined plot with NeuroTrend Intelligent and Market Bias overlay on top,
    and optionally Intelligent Chop indicator below, with trade visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data and indicator columns
    title : str, optional
        Chart title (auto-generated if None)
    figsize : tuple, default=(16, 12)
        Figure size (width, height). When show_chop_subplots=False, height is 
        automatically adjusted by single_plot_height_ratio
    save_path : str or Path, optional
        Path to save the chart
    show : bool, default=True
        Whether to display the chart
    show_indicators : bool, default=True
        Whether to show the indicators subplot (ADX, Choppiness, Efficiency)
    show_chop_subplots : bool, default=True
        Whether to show the Intelligent Chop subplots at all
    use_chop_background : bool, default=False
        If True and show_chop_subplots is False, use Intelligent Chop regime colors 
        as background in the top plot instead of NeuroTrend colors
    single_plot_height_ratio : float, default=0.5
        Height ratio to apply when show_chop_subplots=False. For example, 0.5 means
        the figure height will be half of the original figsize[1]
    main_height_ratio : float, default=4
        Height ratio for the main (top) plot when showing multiple sections
    chop_height_ratio : float, default=3
        Height ratio for the Intelligent Chop section when showing subplots
    simplified_regime_colors : bool, default=False
        If True, uses only two colors: trend_color for Strong/Weak Trend, 
        range_color for Quiet Range/Volatile Chop
    trend_color : str or tuple, default='#4ECDC4' (turquoise)
        Color for trending regimes when simplified_regime_colors=True.
        Can be matplotlib color string ('red', 'blue') or RGB tuple (0.2, 0.4, 0.6)
    range_color : str or tuple, default='#95A5A6' (grey)
        Color for ranging/choppy regimes when simplified_regime_colors=True.
        Can be matplotlib color string or RGB tuple
    trades : list of dict, optional
        List of trade dictionaries with keys: entry_time, exit_time, entry_price, 
        exit_price, direction, exit_reason
    show_pnl : bool, default=True
        Whether to show the P&L subplot below the main chart
    performance_metrics : dict, optional
        Dictionary with performance metrics to display in a table
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    
    # Color scheme
    COLORS = {
        'bg': '#131722',
        'grid': '#363c4e',
        'text': '#d1d4dc',
        'bullish': '#26A69A',
        'bearish': '#EF5350',
        'neutral': '#FFB74D',
        'strong_trend': '#81C784',
        'weak_trend': '#FFF176',
        'quiet_range': '#64B5F6',
        'volatile_chop': '#EF5350',
        'white': '#ffffff'
    }
    
    # Trade marker colors
    TRADE_COLORS = {
        'long_entry': '#1E88E5',  # Bright Blue
        'short_entry': '#FFD600',  # Bright Yellow
        'take_profit': '#43A047',  # Green
        'stop_loss': '#E53935',  # Red
        'trailing_stop': '#9C27B0',  # Purple
        'signal_flip': '#FF8F00',  # Orange
        'end_of_data': '#9E9E9E'  # Grey
    }
    
    # Apply dark theme
    plt.style.use('dark_background')
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    
    # Adjust figsize if not showing chop subplots
    if not show_chop_subplots:
        # Single plot mode
        if show_pnl and trades:
            # Add P&L subplot below the main chart
            adjusted_height = figsize[1] * 0.7  # Increase height to accommodate P&L
            fig = plt.figure(figsize=(figsize[0], adjusted_height), constrained_layout=False)
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
            ax_price = fig.add_subplot(gs[0])
            ax_pnl = fig.add_subplot(gs[1], sharex=ax_price)
            axes_list = [ax_price, ax_pnl]
        else:
            # No P&L subplot
            adjusted_height = figsize[1] * single_plot_height_ratio
            fig = plt.figure(figsize=(figsize[0], adjusted_height), constrained_layout=False)
            ax_price = fig.add_subplot(111)
            ax_pnl = None
            axes_list = [ax_price]
        ax_chop_price = None
        ax_regime = None
        ax_indicators = None
    else:
        # Multiple subplots mode
        if show_pnl and trades:
            # Create 3 main sections: price, chop, and P&L
            gs = fig.add_gridspec(3, 1, height_ratios=[main_height_ratio, chop_height_ratio, 1], hspace=0.15)
            
            # Top section: NeuroTrend + Market Bias
            ax_price = fig.add_subplot(gs[0])
            
            # Middle section: Intelligent Chop (create sub-grid)
            if show_indicators:
                gs_chop = gs[1].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
                ax_chop_price = fig.add_subplot(gs_chop[0], sharex=ax_price)
                ax_regime = fig.add_subplot(gs_chop[1], sharex=ax_price)
                ax_indicators = fig.add_subplot(gs_chop[2], sharex=ax_price)
                axes_list = [ax_price, ax_chop_price, ax_regime, ax_indicators]
            else:
                gs_chop = gs[1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
                ax_chop_price = fig.add_subplot(gs_chop[0], sharex=ax_price)
                ax_regime = fig.add_subplot(gs_chop[1], sharex=ax_price)
                ax_indicators = None
                axes_list = [ax_price, ax_chop_price, ax_regime]
            
            # Bottom section: P&L
            ax_pnl = fig.add_subplot(gs[2], sharex=ax_price)
            axes_list.append(ax_pnl)
        else:
            # No P&L subplot - original logic
            gs = fig.add_gridspec(2, 1, height_ratios=[main_height_ratio, chop_height_ratio], hspace=0.15)
            
            # Top section: NeuroTrend + Market Bias
            ax_price = fig.add_subplot(gs[0])
            
            # Bottom section: Intelligent Chop (create sub-grid)
            if show_indicators:
                gs_chop = gs[1].subgridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
                ax_chop_price = fig.add_subplot(gs_chop[0], sharex=ax_price)
                ax_regime = fig.add_subplot(gs_chop[1], sharex=ax_price)
                ax_indicators = fig.add_subplot(gs_chop[2], sharex=ax_price)
                axes_list = [ax_price, ax_chop_price, ax_regime, ax_indicators]
            else:
                gs_chop = gs[1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
                ax_chop_price = fig.add_subplot(gs_chop[0], sharex=ax_price)
                ax_regime = fig.add_subplot(gs_chop[1], sharex=ax_price)
                ax_indicators = None
                axes_list = [ax_price, ax_chop_price, ax_regime]
            ax_pnl = None
    
    # Set background color
    fig.patch.set_facecolor(COLORS['bg'])
    for ax in axes_list:
        ax.set_facecolor(COLORS['bg'])
    
    # === TOP SECTION: NeuroTrend + Market Bias ===
    x_pos = np.arange(len(df))
    
    # Determine NeuroTrend type and columns
    if 'NTI_Direction' in df.columns:
        # NeuroTrend Intelligent
        direction_col = 'NTI_Direction'
        confidence_col = 'NTI_Confidence'
        fast_ema_col = 'NTI_FastEMA'
        slow_ema_col = 'NTI_SlowEMA'
    elif 'NT3_Direction' in df.columns:
        # NeuroTrend 3-State
        direction_col = 'NT3_Direction'
        confidence_col = 'NT3_Confidence'
        fast_ema_col = 'NTI_FastEMA' if 'NTI_FastEMA' in df.columns else None
        slow_ema_col = 'NTI_SlowEMA' if 'NTI_SlowEMA' in df.columns else None
    else:
        raise ValueError("No NeuroTrend Intelligent data found in DataFrame")
    
    # Plot background coloring - either NeuroTrend or Intelligent Chop
    if not show_chop_subplots and use_chop_background and 'IC_RegimeName' in df.columns:
        # Use Intelligent Chop regime colors as background
        if simplified_regime_colors:
            # Simplified binary colors: trend vs range
            REGIME_COLORS = {
                'Strong Trend': trend_color,
                'Weak Trend': trend_color,
                'Quiet Range': range_color,
                'Volatile Chop': range_color,
                'Transitional': range_color
            }
        else:
            # Original 4-color scheme
            REGIME_COLORS = {
                'Strong Trend': '#81C784',   # Pastel Green
                'Weak Trend': '#FFF176',     # Pastel Yellow
                'Quiet Range': '#64B5F6',    # Pastel Blue
                'Volatile Chop': '#FFCDD2',  # Light Pastel Red
                'Transitional': '#E0E0E0'    # Light Grey
            }
        
        for i in range(len(df)):
            regime = df['IC_RegimeName'].iloc[i]
            color = REGIME_COLORS.get(regime, COLORS['neutral'])
            ax_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                            color=color, alpha=0.4, ec='none')
    else:
        # Use NeuroTrend background coloring
        for i in range(len(df)):
            direction = df[direction_col].iloc[i]
            if direction == 1:
                color = COLORS['bullish']
            elif direction == -1:
                color = COLORS['bearish']
            else:
                color = COLORS['neutral']
            ax_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                            color=color, alpha=0.15, ec='none')
    
    # Plot candlesticks
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    for idx in range(len(df)):
        close_price = closes[idx]
        open_price = opens[idx]
        high_price = highs[idx]
        low_price = lows[idx]
        
        color = COLORS['bullish'] if close_price >= open_price else COLORS['bearish']
        
        # Wicks
        ax_price.plot([x_pos[idx], x_pos[idx]], [low_price, high_price], 
                     color=color, linewidth=1, alpha=0.8)
        
        # Body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        if body_height < (df['Close'].mean() * 0.0001):
            body_height = df['Close'].mean() * 0.0001
        
        ax_price.add_patch(Rectangle((x_pos[idx] - 0.3, body_bottom), 0.6, body_height, 
                                    facecolor=color, edgecolor=color, alpha=0.8))
    
    # Plot Market Bias overlay if available
    if 'MB_Bias' in df.columns:
        valid_mask = ~(df['MB_o2'].isna() | df['MB_c2'].isna())
        mb_bias = df['MB_Bias'].values
        mb_o2 = df['MB_o2'].values
        mb_c2 = df['MB_c2'].values
        mb_h2 = df['MB_h2'].values
        mb_l2 = df['MB_l2'].values
        
        for i in np.where(valid_mask)[0]:
            mb_color = COLORS['bullish'] if mb_bias[i] == 1 else COLORS['bearish']
            ax_price.plot([x_pos[i], x_pos[i]], [mb_l2[i], mb_h2[i]], 
                         color=mb_color, linewidth=10, alpha=0.3, solid_capstyle='round')
            body_bottom = min(mb_o2[i], mb_c2[i])
            body_top = max(mb_o2[i], mb_c2[i])
            body_height = body_top - body_bottom
            if body_height < (df['Close'].mean() * 0.0001):
                body_height = df['Close'].mean() * 0.0001
            ax_price.add_patch(Rectangle((x_pos[i] - 0.4, body_bottom), 0.8, body_height, 
                                        facecolor=mb_color, edgecolor='none', alpha=0.4))
    
    # Plot NeuroTrend EMAs if available
    if fast_ema_col and slow_ema_col and fast_ema_col in df.columns and slow_ema_col in df.columns:
        fast_ema = df[fast_ema_col].values
        slow_ema = df[slow_ema_col].values
        valid_mask = ~np.isnan(fast_ema)
        
        if np.any(valid_mask):
            # Plot EMAs with color based on direction
            for i in range(len(df) - 1):
                if valid_mask[i] and valid_mask[i + 1]:
                    direction = df[direction_col].iloc[i]
                    if direction == 1:
                        color = COLORS['bullish']
                    elif direction == -1:
                        color = COLORS['bearish']
                    else:
                        color = COLORS['neutral']
                    
                    ax_price.plot([x_pos[i], x_pos[i + 1]], 
                                 [fast_ema[i], fast_ema[i + 1]], 
                                 color=color, linewidth=2, alpha=0.9)
                    ax_price.plot([x_pos[i], x_pos[i + 1]], 
                                 [slow_ema[i], slow_ema[i + 1]], 
                                 color=color, linewidth=2, alpha=0.7, linestyle='--')
    
    # Plot trades if provided
    if trades:
        # Convert trades to dictionaries if they're Trade objects
        trade_list = []
        for trade in trades:
            if hasattr(trade, '__dict__'):
                # Convert Trade object to dictionary, including all necessary fields
                trade_dict = {
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'direction': trade.direction,
                    'exit_reason': trade.exit_reason,
                    'take_profits': trade.take_profits,
                    'stop_loss': trade.stop_loss,
                    'tp_hits': trade.tp_hits,
                    'partial_pnl': trade.partial_pnl,
                    'partial_exits': trade.partial_exits
                }
                trade_list.append(trade_dict)
            else:
                trade_list.append(trade)
        
        # Plot each trade (use original trades list to preserve all attributes)
        for idx, trade in enumerate(trades):
            # Find x positions for entry and exit
            entry_idx = None
            exit_idx = None
            
            # Get attributes from trade object or dict
            entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
            exit_time = trade.exit_time if hasattr(trade, 'exit_time') else trade.get('exit_time')
            entry_price = trade.entry_price if hasattr(trade, 'entry_price') else trade.get('entry_price')
            exit_price = trade.exit_price if hasattr(trade, 'exit_price') else trade.get('exit_price')
            direction = trade.direction if hasattr(trade, 'direction') else trade.get('direction')
            exit_reason = trade.exit_reason if hasattr(trade, 'exit_reason') else trade.get('exit_reason')
            
            for i, timestamp in enumerate(df.index):
                if timestamp == entry_time:
                    entry_idx = i
                if timestamp == exit_time:
                    exit_idx = i
            
            if entry_idx is not None:
                # Plot entry marker
                if direction == 'long':
                    ax_price.scatter(x_pos[entry_idx], entry_price, 
                                   marker='^', s=200, color=TRADE_COLORS['long_entry'], 
                                   edgecolor='white', linewidth=2, zorder=5, 
                                   label='Long Entry' if 'Long Entry' not in [l.get_label() for l in ax_price.get_children()] else '')
                else:  # short
                    ax_price.scatter(x_pos[entry_idx], entry_price, 
                                   marker='v', s=200, color=TRADE_COLORS['short_entry'], 
                                   edgecolor='white', linewidth=2, zorder=5,
                                   label='Short Entry' if 'Short Entry' not in [l.get_label() for l in ax_price.get_children()] else '')
            
            if exit_idx is not None:
                # Determine exit color based on reason
                if 'take_profit' in exit_reason:
                    exit_color = TRADE_COLORS['take_profit']
                    exit_label = 'Take Profit'
                elif exit_reason == 'stop_loss':
                    exit_color = TRADE_COLORS['stop_loss']
                    exit_label = 'Stop Loss'
                elif exit_reason == 'trailing_stop':
                    exit_color = TRADE_COLORS['trailing_stop']
                    exit_label = 'Trailing SL'
                elif exit_reason == 'signal_flip':
                    exit_color = TRADE_COLORS['signal_flip']
                    exit_label = 'Signal Exit'
                else:
                    exit_color = TRADE_COLORS['end_of_data']
                    exit_label = 'End of Data'
                
                # Plot exit marker
                ax_price.scatter(x_pos[exit_idx], exit_price, 
                               marker='x', s=200, color=exit_color, 
                               linewidth=3, zorder=5,
                               label=exit_label if exit_label not in [l.get_label() for l in ax_price.get_children()] else '')
                
                # Calculate pips for exit annotation
                if direction == 'long':
                    exit_pips = (exit_price - entry_price) * 10000
                else:  # short
                    exit_pips = (entry_price - exit_price) * 10000
                
                # Add pip annotation next to exit marker
                pip_color = '#43A047' if exit_pips > 0 else '#E53935'  # Green for profit, red for loss
                ax_price.text(x_pos[exit_idx] + 0.5, exit_price, 
                            f'{exit_pips:+.0f}p', 
                            fontsize=8, color=pip_color, 
                            va='center', ha='left', 
                            bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor=COLORS['bg'], 
                                    edgecolor=pip_color, 
                                    alpha=0.8))
                
                # Draw connecting line
                if entry_idx is not None:
                    ax_price.plot([x_pos[entry_idx], x_pos[exit_idx]], 
                                [entry_price, exit_price], 
                                color='white', linestyle='--', alpha=0.5, linewidth=1)
            
            # Draw TP and SL levels for each trade
            if entry_idx is not None:
                # Get trade details - handle both Trade objects and dicts
                tp_levels = trade.take_profits if hasattr(trade, 'take_profits') else trade.get('take_profits', [])
                sl_level = trade.stop_loss if hasattr(trade, 'stop_loss') else trade.get('stop_loss', None)
                
                # Determine the range to draw the levels
                if exit_idx is not None:
                    level_end = exit_idx
                else:
                    level_end = min(entry_idx + 50, len(df) - 1)  # Draw for 50 bars or until end
                
                # Draw TP levels
                if tp_levels:
                    tp_colors = ['#90EE90', '#3CB371', '#228B22']  # Light to dark green
                    for i, tp in enumerate(tp_levels[:3]):  # Max 3 TPs
                        if tp is not None:
                            ax_price.plot([x_pos[entry_idx], x_pos[level_end]], [tp, tp], 
                                        color=tp_colors[i], linestyle=':', alpha=0.6, linewidth=1,
                                        label=f'TP{i+1}' if f'TP{i+1}' not in [l.get_label() for l in ax_price.get_children()] else '')
                            
                            # Add small text label at the start
                            ax_price.text(x_pos[entry_idx] + 1, tp, f'TP{i+1}', 
                                        fontsize=7, color=tp_colors[i], 
                                        va='center', ha='left', alpha=0.8)
                
                # Draw SL level
                if sl_level is not None:
                    ax_price.plot([x_pos[entry_idx], x_pos[level_end]], [sl_level, sl_level], 
                                color='#FF6B6B', linestyle=':', alpha=0.6, linewidth=1,
                                label='Stop Loss' if 'Stop Loss Level' not in [l.get_label() for l in ax_price.get_children()] else '')
                    
                    # Add small text label at the start
                    ax_price.text(x_pos[entry_idx] + 1, sl_level, 'SL', 
                                fontsize=7, color='#FF6B6B', 
                                va='center', ha='left', alpha=0.8)
                
                # Plot partial exits if any
                partial_exits = trade.partial_exits if hasattr(trade, 'partial_exits') else trade.get('partial_exits', [])
                if partial_exits:
                    tp_exit_colors = ['#90EE90', '#3CB371', '#228B22']  # Light to dark green for TP1, TP2, TP3
                    for partial_exit in partial_exits:
                        # Find the x position for this partial exit
                        partial_exit_time = partial_exit['time']
                        partial_exit_idx = None
                        for i, timestamp in enumerate(df.index):
                            if timestamp == partial_exit_time:
                                partial_exit_idx = i
                                break
                        
                        if partial_exit_idx is not None:
                            tp_level = partial_exit['tp_level']
                            exit_color = tp_exit_colors[min(tp_level - 1, 2)]  # TP1=0, TP2=1, TP3=2
                            
                            # Plot partial exit marker (smaller than full exit)
                            ax_price.scatter(x_pos[partial_exit_idx], partial_exit['price'], 
                                           marker='o', s=100, color=exit_color, 
                                           edgecolor='white', linewidth=1.5, zorder=5,
                                           label=f'TP{tp_level} Exit' if f'TP{tp_level} Exit' not in [l.get_label() for l in ax_price.get_children()] else '')
                            
                            # Calculate pips for partial exit
                            if direction == 'long':
                                partial_pips = (partial_exit['price'] - entry_price) * 10000
                            else:  # short
                                partial_pips = (entry_price - partial_exit['price']) * 10000
                            
                            # Add pip annotation for partial exit
                            ax_price.text(x_pos[partial_exit_idx] + 0.5, partial_exit['price'], 
                                        f'+{partial_pips:.0f}p', 
                                        fontsize=7, color=exit_color, 
                                        va='center', ha='left', 
                                        bbox=dict(boxstyle='round,pad=0.2', 
                                                facecolor=COLORS['bg'], 
                                                edgecolor=exit_color, 
                                                alpha=0.8))
                            
                            # Draw thin line from entry to partial exit
                            ax_price.plot([x_pos[entry_idx], x_pos[partial_exit_idx]], 
                                        [entry_price, partial_exit['price']], 
                                        color=exit_color, linestyle='-', alpha=0.3, linewidth=1)
    
    # Add confidence text
    if confidence_col in df.columns:
        latest_confidence = df[confidence_col].iloc[-1]
        ax_price.text(0.02, 0.98, f'Confidence: {latest_confidence:.1%}', 
                     transform=ax_price.transAxes, fontsize=10, 
                     color=COLORS['text'], va='top', 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'], alpha=0.8))
    
    # Add legend with deduplication
    handles, labels = ax_price.get_legend_handles_labels()
    
    # Create unique legend entries
    legend_dict = {}
    for handle, label in zip(handles, labels):
        if label and label not in legend_dict:
            legend_dict[label] = handle
    
    # Add standard indicator legends if not already present
    if 'NTI_Direction' in df.columns or 'NT3_Direction' in df.columns:
        if 'Uptrend' not in legend_dict:
            legend_dict['Uptrend'] = Line2D([0], [0], color=COLORS['bullish'], lw=2)
        if 'Downtrend' not in legend_dict:
            legend_dict['Downtrend'] = Line2D([0], [0], color=COLORS['bearish'], lw=2)
        if 'NT3_Direction' in df.columns and 'Neutral' not in legend_dict:
            legend_dict['Neutral'] = Line2D([0], [0], color=COLORS['neutral'], lw=2)
    
    if 'MB_Bias' in df.columns:
        if 'Bullish Bias' not in legend_dict:
            legend_dict['Bullish Bias'] = mpatches.Patch(color=COLORS['bullish'], alpha=0.4)
        if 'Bearish Bias' not in legend_dict:
            legend_dict['Bearish Bias'] = mpatches.Patch(color=COLORS['bearish'], alpha=0.4)
    
    # Create ordered legend
    ordered_labels = ['Uptrend', 'Downtrend', 'Neutral', 'Bullish Bias', 'Bearish Bias',
                      'Long Entry', 'Short Entry', 'Take Profit', 'Stop Loss', 
                      'Trailing SL', 'Signal Exit', 'End of Data']
    
    final_handles = []
    final_labels = []
    for label in ordered_labels:
        if label in legend_dict:
            final_handles.append(legend_dict[label])
            final_labels.append(label)
    
    ax_price.legend(handles=final_handles, labels=final_labels, 
                   loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    
    ax_price.set_xlim(-1, len(df))
    ax_price.set_ylabel('Price', fontsize=11, color=COLORS['text'])
    ax_price.grid(True, alpha=0.3, color=COLORS['grid'])
    # Hide x-axis labels if we have subplots below
    ax_price.tick_params(colors=COLORS['text'], labelbottom=not (show_chop_subplots or (show_pnl and trades)))
    
    # === P&L SUBPLOT ===
    if show_pnl and trades and 'ax_pnl' in locals() and ax_pnl is not None:
        # Calculate cumulative P&L
        cumulative_pnl = [0]  # Start at 0
        pnl_times = [0]  # Start at beginning
        
        # Sort trades by exit time
        sorted_trades = sorted([t for t in trades if hasattr(t, 'exit_time') or 'exit_time' in t], 
                             key=lambda x: x.exit_time if hasattr(x, 'exit_time') else x['exit_time'])
        
        for trade in sorted_trades:
            # Get trade P&L
            if hasattr(trade, 'pnl'):
                trade_pnl = trade.pnl
                exit_time = trade.exit_time
            else:
                trade_pnl = trade.get('pnl', 0)
                exit_time = trade.get('exit_time')
            
            if exit_time is not None and trade_pnl is not None:
                # Find the index of this exit time
                exit_idx = None
                for i, timestamp in enumerate(df.index):
                    if timestamp == exit_time:
                        exit_idx = i
                        break
                
                if exit_idx is not None:
                    cumulative_pnl.append(cumulative_pnl[-1] + trade_pnl)
                    pnl_times.append(exit_idx)
        
        # Extend to end of chart
        if pnl_times[-1] < len(df) - 1:
            pnl_times.append(len(df) - 1)
            cumulative_pnl.append(cumulative_pnl[-1])
        
        # Plot cumulative P&L as a step chart
        ax_pnl.step(pnl_times, cumulative_pnl, where='post', color='#FFD700', linewidth=2, alpha=0.9)
        ax_pnl.fill_between(pnl_times, 0, cumulative_pnl, step='post', alpha=0.3, 
                           color='#43A047' if cumulative_pnl[-1] >= 0 else '#E53935')
        
        # Add zero line
        ax_pnl.axhline(y=0, color=COLORS['white'], linestyle='-', linewidth=1, alpha=0.5)
        
        # Format P&L axis
        ax_pnl.set_ylabel('Cumulative P&L ($)', fontsize=10, color=COLORS['text'])
        ax_pnl.set_xlim(-1, len(df))
        ax_pnl.grid(True, alpha=0.2, color=COLORS['grid'])
        
        # Add P&L stats text
        final_pnl = cumulative_pnl[-1]
        max_pnl = max(cumulative_pnl)
        min_pnl = min(cumulative_pnl)
        ax_pnl.text(0.02, 0.95, f'Final P&L: ${final_pnl:.2f}', 
                   transform=ax_pnl.transAxes, fontsize=9, 
                   color='#43A047' if final_pnl >= 0 else '#E53935', va='top', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg'], alpha=0.8))
        ax_pnl.text(0.02, 0.75, f'Max: ${max_pnl:.2f}', 
                   transform=ax_pnl.transAxes, fontsize=8, 
                   color=COLORS['text'], va='top')
        ax_pnl.text(0.02, 0.55, f'Min: ${min_pnl:.2f}', 
                   transform=ax_pnl.transAxes, fontsize=8, 
                   color=COLORS['text'], va='top')
        
        # Hide x-axis labels if we have more subplots below
        ax_pnl.tick_params(colors=COLORS['text'], labelbottom=not show_chop_subplots)
    
    # === BOTTOM SECTION: Intelligent Chop ===
    if show_chop_subplots:
        if 'IC_RegimeName' not in df.columns:
            ax_chop_price.text(0.5, 0.5, 'Intelligent Chop data not available', 
                              transform=ax_chop_price.transAxes, 
                              ha='center', va='center', fontsize=12, color=COLORS['text'])
        else:
            # Regime color mapping
            if simplified_regime_colors:
                # Simplified binary colors: trend vs range
                REGIME_COLORS = {
                    'Strong Trend': trend_color,
                    'Weak Trend': trend_color,
                    'Quiet Range': range_color,
                    'Volatile Chop': range_color,
                    'Transitional': range_color
                }
            else:
                # Original 4-color scheme
                REGIME_COLORS = {
                    'Strong Trend': '#81C784',
                    'Weak Trend': '#FFF176',
                    'Quiet Range': '#64B5F6',
                    'Volatile Chop': '#FFCDD2',
                    'Transitional': '#E0E0E0'
                }
            
            # Plot regime background
            for i in range(len(df)):
                regime = df['IC_RegimeName'].iloc[i]
                color = REGIME_COLORS.get(regime, COLORS['white'])
                ax_chop_price.axvspan(x_pos[i] - 0.5, x_pos[i] + 0.5, 
                                     color=color, alpha=0.4, ec='none')
            
            # Plot simple price line
            ax_chop_price.plot(x_pos, closes, color=COLORS['white'], linewidth=1.5, alpha=0.9)
            
            # Regime bars
            regime_numeric = df['IC_Regime'].values
            colors_regime = []
            heights = []
            
            for i in range(len(df)):
                regime_name = df['IC_RegimeName'].iloc[i]
                colors_regime.append(REGIME_COLORS.get(regime_name, COLORS['white']))
                
                if regime_numeric[i] == 2:  # Strong Trend
                    heights.append(1.0)
                elif regime_numeric[i] == 1:  # Weak Trend
                    heights.append(0.7)
                elif regime_numeric[i] == 3:  # Quiet Range
                    heights.append(0.5)
                else:  # Choppy
                    heights.append(-0.8)
            
            ax_regime.bar(x_pos, heights, color=colors_regime, alpha=0.8, width=0.9)
            ax_regime.set_ylabel('Regime', fontsize=10, color=COLORS['text'])
            ax_regime.set_ylim(-1, 1.2)
            ax_regime.axhline(y=0, color=COLORS['white'], linestyle='-', alpha=0.3)
            ax_regime.grid(True, alpha=0.2, color=COLORS['grid'])
            # Only hide x-axis labels if indicators subplot is shown
            ax_regime.tick_params(colors=COLORS['text'], labelbottom=not show_indicators)
            
            # Technical indicators (only if show_indicators is True)
            if show_indicators and all(col in df.columns for col in ['IC_ADX', 'IC_ChoppinessIndex', 'IC_EfficiencyRatio']):
                ax_indicators.plot(x_pos, df['IC_ADX'], color='#FFB74D', linewidth=2, label='ADX', alpha=0.9)
                ax_indicators.plot(x_pos, df['IC_ChoppinessIndex'], color='#64B5F6', linewidth=2, label='Choppiness', alpha=0.9)
                ax_indicators.plot(x_pos, df['IC_EfficiencyRatio'] * 100, color='#81C784', linewidth=2, label='Efficiency', alpha=0.9)
                
                ax_indicators.axhline(y=25, color=COLORS['white'], linestyle='--', alpha=0.3)
                ax_indicators.axhline(y=61.8, color=COLORS['white'], linestyle='--', alpha=0.3)
                ax_indicators.set_ylabel('Indicators', fontsize=10, color=COLORS['text'])
                ax_indicators.set_ylim(0, 100)
                ax_indicators.legend(loc='upper left', fontsize=8, framealpha=0.9)
                ax_indicators.grid(True, alpha=0.2, color=COLORS['grid'])
                ax_indicators.tick_params(colors=COLORS['text'])
            
            # Chop legend
            if simplified_regime_colors:
                # Simplified legend with only two categories
                chop_legend = [
                    mpatches.Patch(color=trend_color, alpha=0.6, label='Trending'),
                    mpatches.Patch(color=range_color, alpha=0.6, label='Ranging/Choppy')
                ]
                ax_chop_price.legend(handles=chop_legend, loc='upper right', fontsize=8, 
                                    framealpha=0.9, ncol=2)
            else:
                # Original 4-category legend
                chop_legend = [
                    mpatches.Patch(color='#81C784', alpha=0.6, label='Strong Trend'),
                    mpatches.Patch(color='#FFF176', alpha=0.6, label='Weak Trend'),
                    mpatches.Patch(color='#64B5F6', alpha=0.6, label='Quiet Range'),
                    mpatches.Patch(color='#FFCDD2', alpha=0.6, label='Volatile Chop')
                ]
                ax_chop_price.legend(handles=chop_legend, loc='upper right', fontsize=8, 
                                    framealpha=0.9, ncol=2)
            
            ax_chop_price.set_xlim(-1, len(df))
            ax_chop_price.set_ylabel('Price', fontsize=10, color=COLORS['text'])
            ax_chop_price.grid(True, alpha=0.3, color=COLORS['grid'])
            ax_chop_price.tick_params(colors=COLORS['text'], labelbottom=False)
    
    # Format x-axis on the bottom-most axis
    if isinstance(df.index, pd.DatetimeIndex):
        # Determine which is the bottom-most axis
        if not show_chop_subplots:
            if show_pnl and trades and 'ax_pnl' in locals() and ax_pnl is not None:
                bottom_ax = ax_pnl
            else:
                bottom_ax = ax_price
        elif show_indicators and ax_indicators is not None:
            bottom_ax = ax_indicators
        elif ax_regime is not None:
            bottom_ax = ax_regime
        else:
            bottom_ax = ax_price
        
        # Create x-axis positions for plotting (0 to len-1)
        x_dates = df.index
        n_ticks = min(8, len(df))  # Limit number of ticks for readability
        tick_positions = np.linspace(0, len(df)-1, n_ticks, dtype=int)
        
        # Set ticks and labels
        bottom_ax.set_xticks(tick_positions)
        tick_labels = [x_dates[i].strftime('%Y-%m-%d') for i in tick_positions]
        bottom_ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add title
    if title is None:
        symbol = df.index.name if df.index.name else "Market"
        title = f"{symbol} - NeuroTrend Intelligent with Market Bias & Intelligent Chop Analysis"
    ax_price.set_title(title, fontsize=14, color=COLORS['text'], pad=30)
    
    # Add performance metrics table if provided
    if performance_metrics:
        # Extract key metrics
        metrics_text = []
        
        # First row
        if 'win_rate' in performance_metrics:
            metrics_text.append(f"Win Rate: {performance_metrics['win_rate']:.1f}%")
        if 'sharpe_ratio' in performance_metrics:
            metrics_text.append(f"Sharpe: {performance_metrics['sharpe_ratio']:.2f}")
        if 'profit_factor' in performance_metrics:
            metrics_text.append(f"PF: {performance_metrics['profit_factor']:.2f}")
        
        # Second row
        if 'total_pnl' in performance_metrics:
            metrics_text.append(f"P&L: ${performance_metrics['total_pnl']:,.0f}")
        if 'total_return' in performance_metrics:
            metrics_text.append(f"Return: {performance_metrics['total_return']:.2f}%")
        if 'max_drawdown' in performance_metrics:
            metrics_text.append(f"DD: {performance_metrics['max_drawdown']:.2f}%")
        
        # Create two rows
        row1 = "  |  ".join(metrics_text[:3]) if len(metrics_text) >= 3 else "  |  ".join(metrics_text)
        row2 = "  |  ".join(metrics_text[3:6]) if len(metrics_text) > 3 else ""
        
        # Add metrics text below title
        y_pos = 0.99
        if row1:
            ax_price.text(0.5, y_pos, row1, 
                         transform=ax_price.transAxes, 
                         fontsize=10, 
                         color=COLORS['text'], 
                         ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.4', 
                                 facecolor=COLORS['bg'], 
                                 edgecolor=COLORS['grid'],
                                 alpha=0.9))
            y_pos -= 0.04
        
        if row2:
            ax_price.text(0.5, y_pos, row2, 
                         transform=ax_price.transAxes, 
                         fontsize=10, 
                         color=COLORS['text'], 
                         ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.4', 
                                 facecolor=COLORS['bg'], 
                                 edgecolor=COLORS['grid'],
                                 alpha=0.9))
    
    # Style spines
    for ax in axes_list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.spines['left'].set_color(COLORS['grid'])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Chart saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# Initialize the plotter
plotter = IndicatorPlotter()

# Load your OHLC data
df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')

# Convert DateTime column to datetime and set as index
df['DateTime'] = pd.to_datetime(df['DateTime'])
df.set_index('DateTime', inplace=True)

# Use a longer timeframe for testing - 5000 rows (about 50 days of 15-minute data)
sample_size = 50_000
max_start = len(df) - sample_size
# Use a fixed seed for reproducibility
np.random.seed(42)
random_start = np.random.randint(0, max_start)
df_analysis = df.iloc[random_start:random_start + sample_size].copy()

print(f"Random sample starting from index {random_start} ({df.index[random_start]})")


# Calculate indicators
print("Calculating NeuroTrend Intelligent...")
df_analysis = TIC.add_neuro_trend_intelligent(
    df_analysis,  
    base_fast=10,
    base_slow=50,
    confirm_bars=3
)   

print("Calculating Market Bias...")
df_analysis = df_analysis = TIC.add_market_bias(df_analysis, ha_len=300, ha_len2=30)  

print("Calculating Intelligent Chop...")
df_analysis = TIC.add_intelligent_chop(df_analysis)
 



# Run backtest
print("\nRunning Strategy_1 backtest...")
strategy = Strategy_1(
    initial_capital=100000,  # $100k for realistic 1M trades
    risk_per_trade=0.02,
    tp_atr_multipliers=(0.8, 1.5, 2.5),  # Staggered multipliers for TP1 (close), TP2 (medium), TP3 (far)
    sl_atr_multiplier=2.0,
    trailing_atr_multiplier=1.2,
    max_tp_percent=0.01,  # Not used anymore as we have specific limits per TP
    min_lot_size=1000000,  # 1M minimum
    pip_value_per_million=100,  # $100 per pip per million
    verbose=False  # Disable for cleaner output
)

# Run the backtest
results = strategy.run_backtest(df_analysis)

# Print performance metrics
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Total Trades: {results['total_trades']}")
print(f"Winning Trades: {results['winning_trades']}")
print(f"Losing Trades: {results['losing_trades']}")
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Total P&L: ${results['total_pnl']:.2f}")
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Average Win: ${results['avg_win']:.2f}")
print(f"Average Loss: ${results['avg_loss']:.2f}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Final Capital: ${results['final_capital']:.2f}")

print("\nExit Reasons Breakdown:")
for reason, count in results['exit_reasons'].items():
    print(f"  {reason}: {count}")

# Print partial exits info
print("\nPartial Exits Summary:")
trades_with_partials = 0
total_partials = 0
for trade in results['trades']:
    if trade.partial_exits:
        trades_with_partials += 1
        total_partials += len(trade.partial_exits)
        print(f"Trade {trades_with_partials}: {len(trade.partial_exits)} partial exits")
        for pe in trade.partial_exits:
            print(f"  - TP{pe['tp_level']} at {pe['time']}: ${pe['pnl']:.2f}")

print(f"\nTotal trades with partial exits: {trades_with_partials}")
print(f"Total partial exits: {total_partials}")

# Create the combined plot with trades
print("\nCreating combined plot with trades...")

fig = plot_neurotrend_market_bias_chop(
    df_analysis,
    title="Strategy_1 Backtest Results - AUDUSD 15M",
    figsize=(20, 16),  # Base size
    save_path='charts/strategy1_backtest_results.png',
    show=True,
    show_chop_subplots=False,
    use_chop_background=True,
    single_plot_height_ratio=0.4,  # Results in 20x3.2
    simplified_regime_colors=True,
    trend_color='#2ECC71',  # Hex color
    range_color='#95A5A6',   # Hex color
    trades=results['trades'],  # Pass the trades for visualization
    performance_metrics=results  # Pass the performance metrics
)

print("\nBacktest complete!")



