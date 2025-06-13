"""
Test Enhanced React Chart for AUDUSD with all features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategy_code.react_integration import ReactChartIntegration

def generate_enhanced_audusd_data():
    """Generate comprehensive AUDUSD data with all indicators and trades"""
    
    # Create 3 months of 15-minute data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    # Generate realistic AUDUSD price data
    np.random.seed(42)
    n_periods = len(dates)
    
    # AUDUSD typically trades 0.65-0.70
    returns = np.random.normal(0.00003, 0.0012, n_periods)
    price = 0.6850
    prices = []
    
    for ret in returns:
        price *= (1 + ret)
        price = max(0.6400, min(0.7200, price))  # Keep in realistic range
        prices.append(price)
    
    # Create DataFrame
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
    
    # Add noise for high/low
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.0006, n_periods))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.0006, n_periods))
    
    # Add NeuroTrend Indicators
    df['NTI_FastEMA'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['NTI_SlowEMA'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['NTI_Direction'] = np.where(df['NTI_FastEMA'] > df['NTI_SlowEMA'], 1, -1)
    df['NTI_Confidence'] = np.random.uniform(0.65, 0.95, n_periods)
    
    # Add Market Bias
    df['MB_Bias'] = np.random.choice([1, -1], n_periods, p=[0.52, 0.48])
    df['MB_o2'] = df['Open'] * (1 + np.random.uniform(-0.0002, 0.0002, n_periods))
    df['MB_h2'] = df['High'] * (1 + np.random.uniform(0, 0.0003, n_periods))
    df['MB_l2'] = df['Low'] * (1 - np.random.uniform(0, 0.0003, n_periods))
    df['MB_c2'] = df['Close'] * (1 + np.random.uniform(-0.0002, 0.0002, n_periods))
    
    # Add Intelligent Chop
    regimes = ['Strong Trend', 'Weak Trend', 'Quiet Range', 'Volatile Chop', 'Transitional']
    regime_probs = [0.2, 0.25, 0.3, 0.15, 0.1]
    df['IC_Regime'] = np.random.choice([0, 1, 2, 3, 4], n_periods, p=regime_probs)
    df['IC_RegimeName'] = df['IC_Regime'].map({
        0: 'Strong Trend', 1: 'Weak Trend', 2: 'Quiet Range', 
        3: 'Volatile Chop', 4: 'Transitional'
    })
    
    # Generate comprehensive trades
    trades = []
    trade_times = sorted(np.random.choice(range(200, n_periods-200), 15, replace=False))
    
    for i, entry_idx in enumerate(trade_times):
        # Trade parameters
        direction = 'long' if df['NTI_Direction'].iloc[entry_idx] == 1 else 'short'
        entry_time = dates[entry_idx]
        entry_price = df['Close'].iloc[entry_idx]
        
        # Position size (1M to 3M)
        position_size = np.random.choice([1000000, 2000000, 3000000])
        
        # Set TP levels
        if direction == 'long':
            tp1 = entry_price + 0.0010  # 10 pips
            tp2 = entry_price + 0.0020  # 20 pips
            tp3 = entry_price + 0.0035  # 35 pips
            sl = entry_price - 0.0015   # 15 pips
        else:
            tp1 = entry_price - 0.0010
            tp2 = entry_price - 0.0020
            tp3 = entry_price - 0.0035
            sl = entry_price + 0.0015
        
        # Simulate trade outcome
        trade_duration = np.random.randint(20, 150)
        exit_idx = min(entry_idx + trade_duration, n_periods - 1)
        
        # Determine exit scenario
        exit_scenario = np.random.choice(['tp1_partial', 'tp2_partial', 'tp3_full', 
                                        'stop_loss', 'trailing_stop', 'signal_flip'], 
                                        p=[0.25, 0.20, 0.15, 0.20, 0.10, 0.10])
        
        partial_exits = []
        total_pnl = 0
        
        if exit_scenario == 'tp1_partial':
            # Exit 1/3 at TP1
            tp1_idx = entry_idx + np.random.randint(10, 30)
            if tp1_idx < exit_idx:
                partial_exit = {
                    'time': dates[tp1_idx],
                    'price': tp1,
                    'size': position_size / 3,
                    'tp_level': 1,
                    'pnl': (tp1 - entry_price) * position_size / 3 * 100000 if direction == 'long' 
                           else (entry_price - tp1) * position_size / 3 * 100000
                }
                partial_exits.append(partial_exit)
                total_pnl += partial_exit['pnl']
            
            # Exit remaining at trailing stop
            exit_price = entry_price + np.random.uniform(0.0005, 0.0015) if direction == 'long' else \
                        entry_price - np.random.uniform(0.0005, 0.0015)
            remaining_pnl = (exit_price - entry_price) * position_size * 2/3 * 100000 if direction == 'long' \
                           else (entry_price - exit_price) * position_size * 2/3 * 100000
            total_pnl += remaining_pnl
            exit_reason = 'trailing_stop'
            
        elif exit_scenario == 'tp2_partial':
            # Exit 1/3 at TP1, 1/3 at TP2
            tp1_idx = entry_idx + np.random.randint(10, 30)
            tp2_idx = entry_idx + np.random.randint(30, 60)
            
            if tp1_idx < exit_idx:
                partial_exit = {
                    'time': dates[tp1_idx],
                    'price': tp1,
                    'size': position_size / 3,
                    'tp_level': 1,
                    'pnl': (tp1 - entry_price) * position_size / 3 * 100000 if direction == 'long' 
                           else (entry_price - tp1) * position_size / 3 * 100000
                }
                partial_exits.append(partial_exit)
                total_pnl += partial_exit['pnl']
            
            if tp2_idx < exit_idx:
                partial_exit = {
                    'time': dates[tp2_idx],
                    'price': tp2,
                    'size': position_size / 3,
                    'tp_level': 2,
                    'pnl': (tp2 - entry_price) * position_size / 3 * 100000 if direction == 'long' 
                           else (entry_price - tp2) * position_size / 3 * 100000
                }
                partial_exits.append(partial_exit)
                total_pnl += partial_exit['pnl']
            
            # Exit remaining
            exit_price = tp2 + np.random.uniform(-0.0002, 0.0003)
            remaining_pnl = (exit_price - entry_price) * position_size / 3 * 100000 if direction == 'long' \
                           else (entry_price - exit_price) * position_size / 3 * 100000
            total_pnl += remaining_pnl
            exit_reason = 'take_profit'
            
        elif exit_scenario == 'tp3_full':
            # Full exit at TP3
            exit_price = tp3
            total_pnl = (exit_price - entry_price) * position_size * 100000 if direction == 'long' \
                       else (entry_price - exit_price) * position_size * 100000
            exit_reason = 'take_profit'
            
        elif exit_scenario == 'stop_loss':
            # Stop loss hit
            exit_price = sl
            total_pnl = (exit_price - entry_price) * position_size * 100000 if direction == 'long' \
                       else (entry_price - exit_price) * position_size * 100000
            exit_reason = 'stop_loss'
            
        else:
            # Trailing stop or signal flip
            exit_price = df['Close'].iloc[exit_idx]
            total_pnl = (exit_price - entry_price) * position_size * 100000 if direction == 'long' \
                       else (entry_price - exit_price) * position_size * 100000
            exit_reason = exit_scenario
        
        trade = {
            'entry_time': entry_time,
            'exit_time': dates[exit_idx],
            'entry_price': entry_price,
            'exit_price': df['Close'].iloc[exit_idx],
            'direction': direction,
            'exit_reason': exit_reason,
            'take_profits': [tp1, tp2, tp3],
            'stop_loss': sl,
            'partial_exits': partial_exits,
            'position_size': position_size,
            'pnl': total_pnl,
            'pnl_pct': (total_pnl / (position_size / 10)) * 100  # Percentage based on position
        }
        trades.append(trade)
    
    # Calculate metrics
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    total_trades = len(trades)
    total_pnl = sum(t['pnl'] for t in trades)
    
    results = {
        'symbol': 'AUDUSD',
        'trades': trades,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'sharpe_ratio': 1.45,
        'max_drawdown': 0.068,
        'total_return': total_pnl / 1000000,  # Return on 1M account
        'profit_factor': 1.52,
        'total_pnl': total_pnl
    }
    
    return df, results


def main():
    """Test enhanced React chart with AUDUSD data"""
    print("Enhanced AUDUSD React Chart Test")
    print("================================\n")
    
    # Generate comprehensive data
    print("Generating enhanced AUDUSD data with all features...")
    df, results = generate_enhanced_audusd_data()
    
    print(f"✅ Generated {len(df)} rows of AUDUSD data")
    print(f"✅ Generated {len(results['trades'])} trades with partial exits")
    print(f"✅ Includes NeuroTrend, Market Bias, and Intelligent Chop indicators\n")
    
    # Display some statistics
    print("Trading Statistics:")
    print(f"  Symbol: {results['symbol']}")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Total P&L: ${results['total_pnl']:,.2f}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.1f}%\n")
    
    # Count partial exits
    partial_count = sum(len(t['partial_exits']) for t in results['trades'])
    print(f"Trade Details:")
    print(f"  Trades with partial exits: {sum(1 for t in results['trades'] if t['partial_exits'])}")
    print(f"  Total partial exits: {partial_count}")
    print(f"  Average position size: ${np.mean([t['position_size'] for t in results['trades']]):,.0f}\n")
    
    # Export for React
    integration = ReactChartIntegration()
    output_path = integration.export_only(df, results)
    print(f"✅ Enhanced chart data exported to: {output_path}\n")
    
    print("To view the enhanced AUDUSD chart:")
    print("  1. cd react_chart")
    print("  2. npm run dev")
    print("  3. Open http://localhost:5173")
    print("  4. Make sure 'Enhanced Chart' is checked")
    print("\nThe enhanced chart includes:")
    print("  - Regime background coloring")
    print("  - TP/SL level visualization")
    print("  - Partial exit markers")
    print("  - Position size chart")
    print("  - Cumulative P&L chart")
    print("  - Comprehensive performance metrics")


if __name__ == "__main__":
    main()