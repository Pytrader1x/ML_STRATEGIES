import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ADX_Strategy import ADXTrendStrategy


class TradeAnalyzer(bt.Analyzer):
    """Custom analyzer to track detailed trade statistics."""
    
    def __init__(self):
        self.trades = []
        self.current_trade = {}
        
    def notify_trade(self, trade):
        if trade.isclosed:
            exit_price = trade.price + (trade.pnl / trade.size if trade.size != 0 else 0)
            self.trades.append({
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'entry_price': trade.price,
                'exit_price': exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'commission': trade.commission,
                'duration': trade.dtclose - trade.dtopen,
            })
            
    def get_analysis(self):
        return self.trades


def run_backtest(
    data_path=None,
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_cash=10000,
    commission=0.001,
    plot=True,
    **strategy_params
):
    """
    Run backtest for the ADX Trend Strategy.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with OHLCV data
    start_date : str
        Start date for backtest
    end_date : str
        End date for backtest
    initial_cash : float
        Starting capital
    commission : float
        Commission rate (0.001 = 0.1%)
    plot : bool
        Whether to plot results
    **strategy_params : dict
        Additional strategy parameters
    """
    
    # Create cerebro instance
    cerebro = bt.Cerebro()
    
    # Add strategy with parameters
    cerebro.addstrategy(ADXTrendStrategy, **strategy_params)
    
    # Load data
    if data_path and os.path.exists(data_path):
        # Load from CSV
        dataframe = pd.read_csv(data_path)
        
        # Handle different datetime column names
        datetime_col = None
        for col in ['DateTime', 'Date', 'datetime', 'date']:
            if col in dataframe.columns:
                datetime_col = col
                break
        
        if datetime_col:
            dataframe[datetime_col] = pd.to_datetime(dataframe[datetime_col])
            dataframe.set_index(datetime_col, inplace=True)
        else:
            # If no datetime column, assume index is already datetime
            dataframe.index = pd.to_datetime(dataframe.index)
        
        # Ensure column names are capitalized for backtrader
        dataframe.columns = [col.capitalize() for col in dataframe.columns]
        
        # Add Volume column if missing (required by backtrader)
        if 'Volume' not in dataframe.columns:
            dataframe['Volume'] = 0
        
        # Filter by date range
        dataframe = dataframe[
            (dataframe.index >= pd.to_datetime(start_date)) & 
            (dataframe.index <= pd.to_datetime(end_date))
        ]
        
        if dataframe.empty:
            print(f"Warning: No data found between {start_date} and {end_date}")
            if len(dataframe.index) > 0:
                print(f"Data range available: {dataframe.index.min()} to {dataframe.index.max()}")
            return None
        
        data = bt.feeds.PandasData(
            dataname=dataframe,
            datetime=None,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            openinterest=None
        )
    else:
        # Fallback to sample data generation
        print("No data file provided. Using sample data...")
        
        # Generate sample data
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        np.random.seed(42)
        
        # Generate realistic OHLC data
        close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
        
        dataframe = pd.DataFrame({
            'Open': close_prices + np.random.uniform(-0.5, 0.5, len(dates)),
            'High': close_prices + np.random.uniform(0, 1, len(dates)),
            'Low': close_prices + np.random.uniform(-1, 0, len(dates)),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        data = bt.feeds.PandasData(dataname=dataframe)
    
    # Add data to cerebro
    cerebro.adddata(data)
    
    # Set initial cash and commission
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(TradeAnalyzer, _name='custom_trades')
    
    # Print starting conditions
    print(f'Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    
    # Run backtest
    results = cerebro.run()
    strat = results[0]
    
    # Print ending conditions
    print(f'Ending Portfolio Value: ${cerebro.broker.getvalue():.2f}')
    
    # Get analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    custom_trades = strat.analyzers.custom_trades.get_analysis()
    
    # Print performance metrics
    print('\n=== PERFORMANCE METRICS ===')
    print(f'Total Return: {returns["rtot"] * 100:.2f}%')
    print(f'Annualized Return: {returns["rnorm100"]:.2f}%')
    print(f'Sharpe Ratio: {sharpe.get("sharperatio", "N/A")}')
    print(f'Max Drawdown: {drawdown["max"]["drawdown"]:.2f}%')
    print(f'Max Drawdown Duration: {drawdown["max"]["len"]} days')
    
    print('\n=== TRADE STATISTICS ===')
    print(f'Total Trades: {trades["total"]["total"]}')
    print(f'Winning Trades: {trades["won"]["total"] if "won" in trades else 0}')
    print(f'Losing Trades: {trades["lost"]["total"] if "lost" in trades else 0}')
    
    if trades["total"]["total"] > 0:
        win_rate = (trades["won"]["total"] / trades["total"]["total"] * 100) if "won" in trades else 0
        print(f'Win Rate: {win_rate:.2f}%')
        
        if "won" in trades and trades["won"]["total"] > 0:
            avg_win = trades["won"]["pnl"]["average"]
            print(f'Average Win: ${avg_win:.2f}')
            
        if "lost" in trades and trades["lost"]["total"] > 0:
            avg_loss = trades["lost"]["pnl"]["average"]
            print(f'Average Loss: ${avg_loss:.2f}')
            
            if "won" in trades and trades["won"]["total"] > 0:
                profit_factor = abs(trades["won"]["pnl"]["total"] / trades["lost"]["pnl"]["total"])
                print(f'Profit Factor: {profit_factor:.2f}')
    
    # Export trade details
    if custom_trades:
        trades_df = pd.DataFrame(custom_trades)
        trades_df.to_csv('trade_history.csv', index=False)
        print(f'\nTrade history exported to trade_history.csv')
    
    # Plot if requested
    if plot:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    
    return {
        'final_value': cerebro.broker.getvalue(),
        'total_return': returns['rtot'],
        'sharpe_ratio': sharpe.get('sharperatio', None),
        'max_drawdown': drawdown['max']['drawdown'],
        'trades': custom_trades
    }


def optimize_strategy(
    data_path=None,
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_cash=10000,
    commission=0.001
):
    """
    Optimize strategy parameters using grid search.
    """
    cerebro = bt.Cerebro()
    
    # Add strategy with parameter ranges for optimization
    cerebro.optstrategy(
        ADXTrendStrategy,
        adx_threshold=range(40, 61, 5),  # Test ADX thresholds from 40 to 60
        williams_period=range(10, 21, 2),  # Test Williams periods from 10 to 20
        sma_period=range(40, 61, 10),  # Test SMA periods from 40 to 60
        printlog=False  # Disable logging during optimization
    )
    
    # Load data (same logic as run_backtest)
    if data_path and os.path.exists(data_path):
        dataframe = pd.read_csv(data_path)
        
        # Handle different datetime column names
        datetime_col = None
        for col in ['DateTime', 'Date', 'datetime', 'date']:
            if col in dataframe.columns:
                datetime_col = col
                break
        
        if datetime_col:
            dataframe[datetime_col] = pd.to_datetime(dataframe[datetime_col])
            dataframe.set_index(datetime_col, inplace=True)
        else:
            dataframe.index = pd.to_datetime(dataframe.index)
        
        # Ensure column names are capitalized
        dataframe.columns = [col.capitalize() for col in dataframe.columns]
        
        # Add Volume column if missing
        if 'Volume' not in dataframe.columns:
            dataframe['Volume'] = 0
        
        # Filter by date range
        dataframe = dataframe[
            (dataframe.index >= pd.to_datetime(start_date)) & 
            (dataframe.index <= pd.to_datetime(end_date))
        ]
        
        data = bt.feeds.PandasData(dataname=dataframe)
    else:
        # Generate sample data
        print("No data file provided. Using sample data...")
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        np.random.seed(42)
        close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
        
        dataframe = pd.DataFrame({
            'Open': close_prices + np.random.uniform(-0.5, 0.5, len(dates)),
            'High': close_prices + np.random.uniform(0, 1, len(dates)),
            'Low': close_prices + np.random.uniform(-1, 0, len(dates)),
            'Close': close_prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        data = bt.feeds.PandasData(dataname=dataframe)
    
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzer
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    print('Running optimization...')
    results = cerebro.run()
    
    # Collect optimization results
    opt_results = []
    for result in results:
        strat = result[0]
        sharpe = strat.analyzers.sharpe.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        
        opt_results.append({
            'adx_threshold': strat.params.adx_threshold,
            'williams_period': strat.params.williams_period,
            'sma_period': strat.params.sma_period,
            'sharpe_ratio': sharpe.get('sharperatio', 0),
            'total_return': returns['rtot'],
            'final_value': strat.broker.getvalue()
        })
    
    # Sort by Sharpe ratio
    opt_results_df = pd.DataFrame(opt_results)
    opt_results_df = opt_results_df.sort_values('sharpe_ratio', ascending=False)
    
    print('\n=== TOP 10 PARAMETER COMBINATIONS ===')
    print(opt_results_df.head(10))
    
    # Save results
    opt_results_df.to_csv('ADX_Strategy/optimization_results.csv', index=False)
    print('\nOptimization results saved to optimization_results.csv')
    
    return opt_results_df


if __name__ == '__main__':
    # Example usage
    print('=== ADX TREND STRATEGY BACKTEST ===\n')
    
    # Run single backtest
    results = run_backtest(
        start_date='2021-01-01',
        end_date='2023-12-31',
        initial_cash=10000,
        commission=0.001,
        plot=False,  # Set to True to see chart
        printlog=True
    )
    
    # Uncomment to run optimization
    # print('\n\n=== RUNNING PARAMETER OPTIMIZATION ===\n')
    # optimization_results = optimize_strategy(
    #     start_date='2021-01-01',
    #     end_date='2023-12-31'
    # )