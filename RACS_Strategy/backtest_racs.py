import pandas as pd
import numpy as np
import backtrader as bt
import sys
import os
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tic import TIC
from racs_strategy import RACSStrategy


class PandasData_Custom(bt.feeds.PandasData):
    """Custom Pandas Data Feed to include our indicator columns"""
    lines = (
        # Intelligent Chop (only numeric columns)
        'IC_Regime', 'IC_Confidence', 'IC_ADX', 
        'IC_ChoppinessIndex', 'IC_BandWidth', 'IC_ATR_Normalized', 
        'IC_EfficiencyRatio', 'IC_BB_Upper', 'IC_BB_Lower',
        
        # Market Bias
        'MB_Bias', 'MB_ha_avg',
        
        # NeuroTrend Intelligent (only numeric columns)
        'NTI_Direction', 'NTI_Confidence', 'NTI_SlopePower',
        'NTI_ReversalRisk', 'NTI_StallDetected',
        
        # SuperTrend
        'SuperTrend_Direction', 'SuperTrend_Line',
        
        # Support/Resistance
        'SR_FractalHighs', 'SR_FractalLows',
    )
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', -1),
        ('openinterest', -1),
        
        # Map our custom indicators (only numeric columns)
        ('IC_Regime', -1),
        ('IC_Confidence', -1),
        ('IC_ADX', -1),
        ('IC_ChoppinessIndex', -1),
        ('IC_BandWidth', -1),
        ('IC_ATR_Normalized', -1),
        ('IC_EfficiencyRatio', -1),
        ('IC_BB_Upper', -1),
        ('IC_BB_Lower', -1),
        
        ('MB_Bias', -1),
        ('MB_ha_avg', -1),
        
        ('NTI_Direction', -1),
        ('NTI_Confidence', -1),
        ('NTI_SlopePower', -1),
        ('NTI_ReversalRisk', -1),
        ('NTI_StallDetected', -1),
        
        ('SuperTrend_Direction', -1),
        ('SuperTrend_Line', -1),
        
        ('SR_FractalHighs', -1),
        ('SR_FractalLows', -1),
    )


def prepare_data(csv_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load data and add all required indicators"""
    print("Loading data...")
    df = pd.read_csv(csv_path, parse_dates=['DateTime'], index_col='DateTime')
    
    # Filter date range if specified
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Add all indicators
    print("Adding Intelligent Chop indicator...")
    df = TIC.add_intelligent_chop(df, inplace=True)
    
    print("Adding Market Bias indicator...")
    df = TIC.add_market_bias(df, inplace=True)
    
    print("Adding NeuroTrend Intelligent indicator...")
    df = TIC.add_neuro_trend_intelligent(df, inplace=True)
    
    print("Adding SuperTrend indicator...")
    df = TIC.add_super_trend(df, inplace=True)
    
    print("Adding Fractal Support/Resistance...")
    df = TIC.add_fractal_sr(df, inplace=True)
    
    # Drop any NaN rows from indicator initialization
    df = df.dropna()
    
    print(f"Data shape after indicators: {df.shape}")
    
    return df


def run_backtest(df: pd.DataFrame, 
                initial_cash: float = 10000,
                commission: float = 0.001,
                print_trades: bool = True) -> dict:
    """Run the RACS backtest and return performance metrics"""
    
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(RACSStrategy, 
                       account_size=initial_cash,
                       max_positions=3)
    
    # Create data feed
    data = PandasData_Custom(dataname=df)
    cerebro.adddata(data)
    
    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                       riskfreerate=0.01, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # Run backtest
    print(f"\nStarting backtest with ${initial_cash:,.2f} initial capital...")
    results = cerebro.run()
    strat = results[0]
    
    # Extract performance metrics
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_cash - 1) * 100
    
    # Get analyzer results
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()
    
    # Calculate additional metrics
    trade_analysis = trades.total
    won_trades = trades.won.total if hasattr(trades.won, 'total') else 0
    lost_trades = trades.lost.total if hasattr(trades.lost, 'total') else 0
    
    win_rate = (won_trades / trade_analysis.total * 100) if trade_analysis.total > 0 else 0
    
    # Average win/loss
    avg_win = trades.won.pnl.average if hasattr(trades.won.pnl, 'average') else 0
    avg_loss = abs(trades.lost.pnl.average) if hasattr(trades.lost.pnl, 'average') else 0
    
    # Risk/reward ratio
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    # Compile results
    results_dict = {
        'initial_capital': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio.get('sharperatio', 0),
        'max_drawdown': drawdown.max.drawdown,
        'max_drawdown_duration': drawdown.max.len,
        'total_trades': trade_analysis.total,
        'won_trades': won_trades,
        'lost_trades': lost_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'risk_reward_ratio': risk_reward,
        'sqn': sqn.get('sqn', 0),
        'annual_return': returns.get('rnorm100', 0),
        'trades_log': strat.trade_log if hasattr(strat, 'trade_log') else []
    }
    
    return results_dict


def print_performance_report(results: dict):
    """Print a formatted performance report"""
    print("\n" + "="*60)
    print("RACS STRATEGY PERFORMANCE REPORT")
    print("="*60)
    
    print(f"\nCAPITAL:")
    print(f"  Initial Capital:     ${results['initial_capital']:,.2f}")
    print(f"  Final Value:         ${results['final_value']:,.2f}")
    print(f"  Total Return:        {results['total_return']:.2f}%")
    print(f"  Annual Return:       {results['annual_return']:.2f}%")
    
    print(f"\nRISK METRICS:")
    print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:        {results['max_drawdown']:.2f}%")
    print(f"  Max DD Duration:     {results['max_drawdown_duration']} periods")
    print(f"  System Quality (SQN):{results['sqn']:.2f}")
    
    print(f"\nTRADE STATISTICS:")
    print(f"  Total Trades:        {results['total_trades']}")
    print(f"  Winning Trades:      {results['won_trades']}")
    print(f"  Losing Trades:       {results['lost_trades']}")
    print(f"  Win Rate:            {results['win_rate']:.1f}%")
    print(f"  Avg Win:             ${results['avg_win']:.2f}")
    print(f"  Avg Loss:            ${results['avg_loss']:.2f}")
    print(f"  Risk/Reward Ratio:   {results['risk_reward_ratio']:.2f}")
    
    print("\n" + "="*60)
    
    # Interpret SQN
    sqn = results['sqn']
    if sqn < 1.6:
        sqn_rating = "Poor"
    elif sqn < 1.9:
        sqn_rating = "Below Average"
    elif sqn < 2.4:
        sqn_rating = "Average"
    elif sqn < 2.9:
        sqn_rating = "Good"
    elif sqn < 5.0:
        sqn_rating = "Excellent"
    else:
        sqn_rating = "Superb"
    
    print(f"\nSystem Quality Rating: {sqn_rating}")
    
    # Overall assessment
    if results['sharpe_ratio'] > 2.0 and results['win_rate'] > 60 and results['max_drawdown'] < 10:
        print("\nStrategy Assessment: EXCELLENT - Meets all key performance targets")
    elif results['sharpe_ratio'] > 1.5 and results['win_rate'] > 50 and results['max_drawdown'] < 15:
        print("\nStrategy Assessment: GOOD - Solid performance with room for optimization")
    else:
        print("\nStrategy Assessment: NEEDS IMPROVEMENT - Consider parameter optimization")


def main():
    """Main function to run the backtest"""
    # Configuration
    csv_path = "../data/AUDUSD_MASTER_15M.csv"
    start_date = "2020-01-01"  # Optional: filter start date
    end_date = None  # Optional: filter end date
    initial_cash = 10000
    commission = 0.001  # 0.1%
    
    # Check if data file exists
    if not os.path.exists(csv_path):
        print(f"Error: Data file not found at {csv_path}")
        return
    
    try:
        # Prepare data with indicators
        df = prepare_data(csv_path, start_date, end_date)
        
        # Run backtest
        results = run_backtest(df, initial_cash, commission)
        
        # Print performance report
        print_performance_report(results)
        
        # Save detailed trade log
        if results['trades_log']:
            trades_df = pd.DataFrame(results['trades_log'])
            trades_df.to_csv('RACS_Strategy/backtest_trades.csv', index=False)
            print(f"\nDetailed trade log saved to: RACS_Strategy/backtest_trades.csv")
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()