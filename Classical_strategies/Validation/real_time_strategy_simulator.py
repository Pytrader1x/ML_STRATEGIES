"""
Real-Time Strategy Simulator
Simulates the strategy running in live market conditions with data streaming one row at a time
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy_code.Prod_strategy import (
    OptimizedProdStrategy, OptimizedStrategyConfig, 
    TradeDirection, ExitReason, Trade
)
from real_time_data_generator import RealTimeDataGenerator


def annualization_factor(timestamps: List[datetime]) -> float:
    """
    Automatically detect data frequency and return appropriate annualization factor.
    
    Args:
        timestamps: List of datetime timestamps from the data
        
    Returns:
        Square root of periods per year for Sharpe ratio calculation
    """
    if len(timestamps) < 2:
        return np.sqrt(252)  # Default to daily
    
    # Convert to numpy array for easier manipulation
    ts_array = pd.to_datetime(timestamps)
    
    # Calculate median time difference in seconds
    time_diffs = np.diff(ts_array).astype('timedelta64[s]').astype(float)
    median_seconds = np.median(time_diffs)
    
    if median_seconds == 0:
        return np.sqrt(252)  # Fallback to daily
    
    # Calculate periods per year based on median interval
    seconds_per_year = 365.25 * 24 * 3600
    periods_per_year = seconds_per_year / median_seconds
    
    # Common timeframes for validation
    if 840 <= median_seconds <= 960:  # 14-16 minutes
        print(f"Detected 15-minute bars: {periods_per_year:.0f} periods/year")
        return np.sqrt(252 * 96)  # 96 fifteen-minute bars per trading day
    elif 3300 <= median_seconds <= 3900:  # 55-65 minutes
        print(f"Detected hourly bars: {periods_per_year:.0f} periods/year")
        return np.sqrt(252 * 24)  # 24 hourly bars per trading day
    elif 14000 <= median_seconds <= 15000:  # ~4 hours
        print(f"Detected 4-hour bars: {periods_per_year:.0f} periods/year")
        return np.sqrt(252 * 6)  # 6 four-hour bars per trading day
    elif 82800 <= median_seconds <= 93600:  # ~23-26 hours
        print(f"Detected daily bars: {periods_per_year:.0f} periods/year")
        return np.sqrt(252)  # Daily bars
    else:
        print(f"Detected custom interval: {median_seconds:.0f} seconds, {periods_per_year:.0f} periods/year")
        return np.sqrt(periods_per_year)


@dataclass
class RealTimeTradeEvent:
    """Records a real-time trading event"""
    timestamp: datetime
    row_index: int
    event_type: str  # 'entry', 'exit', 'update', 'signal'
    price: float
    direction: Optional[str] = None
    reason: Optional[str] = None
    pnl: Optional[float] = None
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[List[float]] = None
    indicators: Optional[Dict] = None
    capital: Optional[float] = None


class RealTimeStrategySimulator:
    """
    Simulates strategy execution in real-time market conditions
    """
    
    def __init__(self, config: OptimizedStrategyConfig):
        self.config = config
        self.strategy = OptimizedProdStrategy(config)
        self.events = []
        self.trade_history = []
        self.current_row_data = None
        self.simulation_start_time = None
        
        # Real-time state tracking
        self.capital_history = []
        self.indicator_history = []
        self.decision_log = []
        
    def run_real_time_simulation(self, currency_pair: str = 'AUDUSD', 
                                rows_to_simulate: int = 8000,
                                start_date: str = None,
                                start_idx: int = None,
                                verbose: bool = True) -> Dict[str, Any]:
        """
        Run a real-time simulation of the strategy
        
        Args:
            currency_pair: Currency pair to trade
            rows_to_simulate: Number of data rows to process
            start_date: Optional start date (YYYY-MM-DD format)
            verbose: Whether to print detailed progress
            
        Returns:
            Simulation results dictionary
        """
        
        print(f"\\n{'='*60}")
        print(f"REAL-TIME STRATEGY SIMULATION")
        print(f"Currency: {currency_pair}")
        print(f"Rows to simulate: {rows_to_simulate:,}")
        print(f"Initial capital: ${self.config.initial_capital:,.0f}")
        print(f"{'='*60}")
        
        # Initialize data generator
        data_generator = RealTimeDataGenerator(currency_pair)
        
        # Get sample period
        start_idx, end_idx = data_generator.get_sample_period(
            start_date=start_date, 
            rows=rows_to_simulate,
            start_idx=start_idx
        )
        
        # Reset strategy state
        self.strategy.reset()
        self.events = []
        self.trade_history = []
        self.capital_history = [self.config.initial_capital]
        self.indicator_history = []
        self.decision_log = []
        self.simulation_start_time = datetime.now()
        
        row_count = 0
        
        # Process data stream row by row
        for data_point in data_generator.stream_data(start_idx, end_idx):
            row_count += 1
            self.current_row_data = data_point
            
            # Extract current market data
            current_row = pd.Series(data_point['data'])
            current_time = pd.to_datetime(data_point['current_time'])
            current_price = data_point['price']
            
            # Log current market state
            self._log_market_state(data_point, verbose and row_count % 500 == 0)
            
            # Process current trade if exists
            if self.strategy.current_trade is not None:
                self._process_open_trade(current_row, current_time, verbose)
            
            # Check for new entry opportunities
            elif self.strategy.current_trade is None:
                self._check_entry_opportunity(current_row, current_time, verbose)
            
            # Update capital history
            current_capital = self._calculate_current_equity(current_price)
            self.capital_history.append(current_capital)
            
            # Store indicator state
            self.indicator_history.append({
                'row_index': data_point['row_index'],
                'timestamp': current_time,
                'NTI_Direction': current_row.get('NTI_Direction', 0),
                'MB_Bias': current_row.get('MB_Bias', 0),
                'IC_Regime': current_row.get('IC_Regime', 3),
                'IC_ATR_Normalized': current_row.get('IC_ATR_Normalized', 0.0001),
                'price': current_price
            })
            
            # Progress reporting
            if verbose and row_count % 1000 == 0:
                self._print_progress_report(row_count, rows_to_simulate)
        
        # Close any remaining trade
        if self.strategy.current_trade is not None:
            self._close_final_trade(verbose)
        
        # Calculate final results
        results = self._calculate_simulation_results(row_count, data_generator, start_idx, end_idx)
        
        if verbose:
            self._print_final_results(results)
        
        return results
    
    def _log_market_state(self, data_point: Dict, should_print: bool = False):
        """Log current market state and indicators"""
        indicators = {
            'NTI': data_point['data'].get('NTI_Direction', 0),
            'MB': data_point['data'].get('MB_Bias', 0), 
            'IC': data_point['data'].get('IC_Regime', 3),
            'ATR': data_point['data'].get('IC_ATR_Normalized', 0.0001)
        }
        
        if should_print:
            print(f"\\nRow {data_point['row_index']:5d} | {data_point['current_time']} | "
                  f"Price: {data_point['price']:.5f}")
            print(f"  Indicators: NTI={indicators['NTI']:2.0f} | "
                  f"MB={indicators['MB']:2.0f} | IC={indicators['IC']:1.0f} | "
                  f"ATR={indicators['ATR']:.5f}")
            print(f"  Capital: ${self.capital_history[-1]:,.0f} | "
                  f"Trades: {len(self.strategy.trades)}")
    
    def _process_open_trade(self, current_row: pd.Series, current_time: pd.Timestamp, verbose: bool):
        """Process existing open trade"""
        if verbose and self.config.debug_decisions:
            current_pnl = self.strategy.pnl_calculator.calculate_pnl(
                self.strategy.current_trade.entry_price, current_row['Close'],
                self.strategy.current_trade.remaining_size, self.strategy.current_trade.direction
            )[0]
            
            print(f"  🔄 OPEN TRADE: {self.strategy.current_trade.direction.value} from "
                  f"{self.strategy.current_trade.entry_price:.5f}")
            print(f"     Current P&L: ${current_pnl:,.0f} | SL: {self.strategy.current_trade.stop_loss:.5f}")
        
        # Record trade update event
        current_pnl = self.strategy.pnl_calculator.calculate_pnl(
            self.strategy.current_trade.entry_price, current_row['Close'],
            self.strategy.current_trade.remaining_size, self.strategy.current_trade.direction
        )[0]
        
        self.events.append(RealTimeTradeEvent(
            timestamp=current_time,
            row_index=self.current_row_data['row_index'],
            event_type='update',
            price=current_row['Close'],
            direction=self.strategy.current_trade.direction.value,
            pnl=current_pnl,
            capital=self.strategy.current_capital + current_pnl,
            indicators={
                'NTI': current_row.get('NTI_Direction', 0),
                'MB': current_row.get('MB_Bias', 0),
                'IC': current_row.get('IC_Regime', 3)
            }
        ))
        
        # Check for partial profit taking
        if self.strategy.signal_generator.check_partial_profit_conditions(current_row, self.strategy.current_trade):
            trade_id = id(self.strategy.current_trade)
            if trade_id not in self.strategy.partial_profit_taken:
                if verbose:
                    print(f"  ✅ PARTIAL PROFIT: Taking {self.config.partial_profit_size_percent*100:.0f}% "
                          f"profit at {current_row['Close']:.5f}")
                
                # Execute partial profit
                exit_price = current_row['Close']
                completed = self.strategy._execute_partial_exit(
                    self.strategy.current_trade, current_time, exit_price,
                    self.config.partial_profit_size_percent
                )
                self.strategy.partial_profit_taken[trade_id] = True
                
                # Record partial exit event
                self.events.append(RealTimeTradeEvent(
                    timestamp=current_time,
                    row_index=self.current_row_data['row_index'],
                    event_type='exit',
                    price=exit_price,
                    direction=self.strategy.current_trade.direction.value,
                    reason='partial_profit',
                    pnl=self.strategy.current_trade.partial_pnl,
                    capital=self.strategy.current_capital
                ))
                
                if completed:
                    if verbose:
                        print(f"  🏁 TRADE COMPLETED via partial profit")
                    self.strategy.trades.append(self.strategy.current_trade)
                    self.strategy.current_trade = None
                    return
        
        # Update trailing stop
        atr = current_row['IC_ATR_Normalized']
        old_trailing_stop = self.strategy.current_trade.trailing_stop
        new_trailing_stop = self.strategy._update_trailing_stop(
            current_row['Close'], self.strategy.current_trade, atr
        )
        if new_trailing_stop is not None:
            self.strategy.current_trade.trailing_stop = new_trailing_stop
            if verbose and self.config.debug_decisions and new_trailing_stop != old_trailing_stop:
                print(f"  📈 TRAILING STOP updated: {old_trailing_stop} → {new_trailing_stop:.5f}")
        
        # Check exit conditions
        should_exit, exit_reason, exit_percent = self.strategy.signal_generator.check_exit_conditions(
            current_row, self.strategy.current_trade, current_time
        )
        
        if should_exit:
            if verbose:
                print(f"  🚪 EXIT SIGNAL: {exit_reason.value} ({exit_percent*100:.0f}% of position)")
            
            # Determine exit price
            exit_price = self.strategy._get_exit_price(current_row, self.strategy.current_trade, exit_reason)
            
            # Calculate exit P&L
            final_pnl = self.strategy.pnl_calculator.calculate_pnl(
                self.strategy.current_trade.entry_price, exit_price,
                self.strategy.current_trade.remaining_size * exit_percent, 
                self.strategy.current_trade.direction
            )[0]
            
            if verbose:
                print(f"     Exit price: {exit_price:.5f} | Exit P&L: ${final_pnl:,.0f}")
            
            # Execute exit
            if exit_percent < 1.0:
                completed_trade = self.strategy._execute_partial_exit(
                    self.strategy.current_trade, current_time, exit_price,
                    exit_percent, exit_reason
                )
            else:
                completed_trade = self.strategy._execute_full_exit(
                    self.strategy.current_trade, current_time, exit_price, exit_reason
                )
            
            # Record exit event
            self.events.append(RealTimeTradeEvent(
                timestamp=current_time,
                row_index=self.current_row_data['row_index'],
                event_type='exit',
                price=exit_price,
                direction=self.strategy.current_trade.direction.value,
                reason=exit_reason.value,
                pnl=final_pnl,
                capital=self.strategy.current_capital
            ))
            
            if completed_trade is not None:
                if verbose:
                    print(f"  🏁 TRADE COMPLETED: Final P&L ${completed_trade.pnl:,.0f} | "
                          f"Total trades: {len(self.strategy.trades) + 1}")
                self.strategy.trades.append(self.strategy.current_trade)
                self.strategy.current_trade = None
    
    def _check_entry_opportunity(self, current_row: pd.Series, current_time: pd.Timestamp, verbose: bool):
        """Check for new trade entry opportunities"""
        entry_result = self.strategy.signal_generator.check_entry_conditions(current_row)
        
        if entry_result is not None:
            direction, is_relaxed = entry_result
            
            if verbose:
                entry_type = "RELAXED" if is_relaxed else "STANDARD"
                print(f"  🎯 ENTRY SIGNAL: {entry_type} {direction.value}")
                print(f"     Conditions: NTI={current_row['NTI_Direction']}, "
                      f"MB={current_row['MB_Bias']}, IC={current_row['IC_Regime']}")
            
            # Create new trade
            self.strategy.current_trade = self.strategy._create_new_trade(
                current_time, current_row, direction, is_relaxed
            )
            
            if verbose:
                size_millions = self.strategy.current_trade.position_size / self.config.min_lot_size
                print(f"     Entry: {self.strategy.current_trade.entry_price:.5f} | Size: {size_millions:.1f}M")
                print(f"     SL: {self.strategy.current_trade.stop_loss:.5f} | "
                      f"TPs: {[f'{tp:.5f}' for tp in self.strategy.current_trade.take_profits]}")
            
            # Record entry event
            self.events.append(RealTimeTradeEvent(
                timestamp=current_time,
                row_index=self.current_row_data['row_index'],
                event_type='entry',
                price=self.strategy.current_trade.entry_price,
                direction=direction.value,
                reason='standard_entry' if not is_relaxed else 'relaxed_entry',
                position_size=self.strategy.current_trade.position_size,
                stop_loss=self.strategy.current_trade.stop_loss,
                take_profit=self.strategy.current_trade.take_profits,
                capital=self.strategy.current_capital,
                indicators={
                    'NTI': current_row['NTI_Direction'],
                    'MB': current_row['MB_Bias'],
                    'IC': current_row['IC_Regime']
                }
            ))
        else:
            # Log why no entry was taken (occasionally)
            if verbose and self.config.debug_decisions and self.current_row_data['row_index'] % 1000 == 0:
                print(f"  ⏸️  NO ENTRY: NTI={current_row['NTI_Direction']}, "
                      f"MB={current_row['MB_Bias']}, IC={current_row['IC_Regime']}")
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L"""
        equity = self.strategy.current_capital
        
        if self.strategy.current_trade is not None:
            unrealized_pnl = self.strategy.pnl_calculator.calculate_pnl(
                self.strategy.current_trade.entry_price, current_price,
                self.strategy.current_trade.remaining_size, self.strategy.current_trade.direction
            )[0]
            equity += unrealized_pnl
        
        return equity
    
    def _close_final_trade(self, verbose: bool):
        """Close any remaining trade at end of simulation"""
        if self.strategy.current_trade is not None:
            last_data = self.current_row_data
            last_price = last_data['price']
            last_time = pd.to_datetime(last_data['current_time'])
            
            if verbose:
                print(f"🏁 CLOSING FINAL TRADE at end of simulation: {last_price:.5f}")
            
            completed_trade = self.strategy._execute_full_exit(
                self.strategy.current_trade, last_time, last_price, ExitReason.END_OF_DATA
            )
            
            if completed_trade:
                self.strategy.trades.append(self.strategy.current_trade)
                
                # Record final exit event
                self.events.append(RealTimeTradeEvent(
                    timestamp=last_time,
                    row_index=last_data['row_index'],
                    event_type='exit',
                    price=last_price,
                    direction=self.strategy.current_trade.direction.value,
                    reason='end_of_data',
                    pnl=completed_trade.pnl,
                    capital=self.strategy.current_capital
                ))
    
    def _print_progress_report(self, current_row: int, total_rows: int):
        """Print simulation progress"""
        progress = (current_row / total_rows) * 100
        current_capital = self.capital_history[-1] if self.capital_history else self.config.initial_capital
        total_trades = len(self.strategy.trades)
        
        print(f"\\nProgress: {progress:5.1f}% ({current_row:,}/{total_rows:,}) | "
              f"Capital: ${current_capital:,.0f} | Trades: {total_trades}")
    
    def _calculate_simulation_results(self, rows_processed: int, data_generator=None, start_idx=None, end_idx=None) -> Dict[str, Any]:
        """Calculate final simulation results"""
        final_capital = self.capital_history[-1] if self.capital_history else self.config.initial_capital
        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital * 100
        
        # Basic trade statistics
        total_trades = len(self.strategy.trades)
        winning_trades = [t for t in self.strategy.trades if t.pnl > 0]
        losing_trades = [t for t in self.strategy.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate Sharpe ratio from equity curve
        if len(self.capital_history) > 1 and len(self.indicator_history) > 1:
            # Create a time series of capital values with timestamps
            timestamps = [ih['timestamp'] for ih in self.indicator_history]
            
            # Ensure we have matching lengths
            min_length = min(len(timestamps), len(self.capital_history))
            
            # Create DataFrame for easier manipulation
            equity_df = pd.DataFrame({
                'timestamp': timestamps[:min_length],
                'capital': self.capital_history[:min_length]
            })
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
            
            # Aggregate to daily returns to remove intraday serial correlation
            # Use last value of each day
            daily_equity = equity_df.resample('D').last().dropna()
            
            if len(daily_equity) > 1:
                # Calculate daily returns
                daily_returns = daily_equity['capital'].pct_change().dropna()
                
                # Calculate Sharpe with standard 252 trading days
                if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
                    # Annual Sharpe = daily_mean / daily_std * sqrt(252)
                    sharpe_ratio = daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
                else:
                    sharpe_ratio = 0
            else:
                # Not enough daily data, fallback to bar-level calculation
                returns = np.diff(self.capital_history) / self.capital_history[:-1]
                if len(returns) > 1 and np.std(returns, ddof=1) > 0:
                    # Detect data frequency for annualization
                    ann_factor = annualization_factor(timestamps)
                    sharpe_ratio = np.mean(returns) / np.std(returns, ddof=1) * ann_factor
                else:
                    sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        if len(self.capital_history) > 1:
            equity_array = np.array(self.capital_history)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max * 100
            max_drawdown = abs(np.min(drawdown))  # Convert to positive magnitude
        else:
            max_drawdown = 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return {
            'simulation_summary': {
                'rows_processed': rows_processed,
                'simulation_duration': datetime.now() - self.simulation_start_time,
                'events_recorded': len(self.events),
                'data_generator': data_generator,  # Store the generator
                'start_idx': start_idx,
                'end_idx': end_idx
            },
            'performance_metrics': {
                'initial_capital': self.config.initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_pnl': final_capital - self.config.initial_capital,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor
            },
            'trade_statistics': {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'detailed_data': {
                'trades': self.strategy.trades,
                'events': self.events,
                'capital_history': self.capital_history,
                'indicator_history': self.indicator_history
            }
        }
    
    def _print_final_results(self, results: Dict[str, Any]):
        """Print final simulation results"""
        sim = results['simulation_summary']
        perf = results['performance_metrics']
        trade = results['trade_statistics']
        
        print(f"\\n{'='*60}")
        print(f"REAL-TIME SIMULATION COMPLETED")
        print(f"{'='*60}")
        print(f"Simulation Duration: {sim['simulation_duration']}")
        print(f"Rows Processed: {sim['rows_processed']:,}")
        print(f"Events Recorded: {sim['events_recorded']:,}")
        
        print(f"\\n📊 PERFORMANCE METRICS:")
        print(f"  Initial Capital: ${perf['initial_capital']:,.0f}")
        print(f"  Final Capital:   ${perf['final_capital']:,.0f}")
        print(f"  Total Return:    {perf['total_return']:+.2f}%")
        print(f"  Total P&L:       ${perf['total_pnl']:+,.0f}")
        print(f"  Sharpe Ratio:    {perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:    {perf['max_drawdown']:.2f}%")
        print(f"  Profit Factor:   {perf['profit_factor']:.2f}")
        
        print(f"\\n🎯 TRADE STATISTICS:")
        print(f"  Total Trades:    {trade['total_trades']}")
        print(f"  Winning Trades:  {trade['winning_trades']}")
        print(f"  Losing Trades:   {trade['losing_trades']}")
        print(f"  Win Rate:        {trade['win_rate']:.1f}%")
        print(f"  Average Win:     ${trade['avg_win']:,.0f}")
        print(f"  Average Loss:    ${trade['avg_loss']:,.0f}")
        
        print(f"\\n✅ Real-time simulation completed successfully!")


if __name__ == "__main__":
    # Test the real-time simulator
    from strategy_code.Prod_strategy import OptimizedStrategyConfig
    
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,
        debug_decisions=False,  # Set to True for detailed logging
        verbose=False
    )
    
    simulator = RealTimeStrategySimulator(config)
    results = simulator.run_real_time_simulation(
        currency_pair='AUDUSD',
        rows_to_simulate=1000,  # Small test
        verbose=True
    )