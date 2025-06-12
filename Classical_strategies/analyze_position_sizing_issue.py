"""
Analyze Position Sizing Issue - Check if strategy is compounding
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
import sys
sys.path.append('..')
from technical_indicators_custom import TIC


def analyze_position_sizing():
    """Check if position sizing is growing (compounding)"""
    print("Loading AUDUSD data...")
    df = pd.read_csv('../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent data
    df = df['2023-01-01':'2024-12-31']
    
    # Add indicators
    df = TIC.add_neuro_trend_intelligent(df)
    df = TIC.add_market_bias(df)
    df = TIC.add_intelligent_chop(df)
    
    # Create strategy
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,
        sl_max_pips=10.0,
        verbose=False
    )
    strategy = OptimizedProdStrategy(config)
    
    # Run backtest
    print("\nRunning backtest...")
    results = strategy.run_backtest(df)
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {results['total_return']:.1f}%")
    print(f"Total P&L: ${results['total_pnl']:,.2f}")
    print(f"Final Capital: ${results.get('final_capital', 100000):,.2f}")
    
    # Analyze position sizes
    if results['trades']:
        trades = results['trades']
        print(f"\nAnalyzing {len(trades)} trades...")
        
        # Get position sizes over time
        position_sizes = []
        for i, trade in enumerate(trades):
            position_sizes.append({
                'trade_num': i + 1,
                'date': trade.entry_time,
                'position_size': trade.position_size,
                'pnl': trade.pnl
            })
        
        # Convert to DataFrame
        pos_df = pd.DataFrame(position_sizes)
        
        # Check if position sizes are growing
        print(f"\nPosition Size Analysis:")
        print(f"First 10 trades avg size: {pos_df.head(10)['position_size'].mean():,.0f}")
        print(f"Last 10 trades avg size: {pos_df.tail(10)['position_size'].mean():,.0f}")
        
        # Calculate cumulative P&L
        pos_df['cumulative_pnl'] = pos_df['pnl'].cumsum()
        pos_df['cumulative_capital'] = 100000 + pos_df['cumulative_pnl']
        
        # Expected position size if not compounding
        expected_size = 200_000  # 0.2% of 100k with 10 pip SL
        
        print(f"\nExpected position size (no compounding): {expected_size:,}")
        print(f"\nActual position sizes:")
        print(f"  Min: {pos_df['position_size'].min():,.0f}")
        print(f"  Max: {pos_df['position_size'].max():,.0f}")
        print(f"  Mean: {pos_df['position_size'].mean():,.0f}")
        
        # Show sample of trades
        print(f"\nSample trades showing position size growth:")
        sample_indices = np.linspace(0, len(pos_df)-1, 10, dtype=int)
        for idx in sample_indices:
            row = pos_df.iloc[idx]
            print(f"Trade {row['trade_num']:4d}: {row['date']} - "
                  f"Size: {row['position_size']:>10,.0f} - "
                  f"Capital: ${row['cumulative_capital']:>10,.0f}")
        
        # Check if this is compounding
        size_ratio = pos_df.iloc[-1]['position_size'] / pos_df.iloc[0]['position_size']
        capital_ratio = pos_df.iloc[-1]['cumulative_capital'] / 100000
        
        print(f"\nCompounding Analysis:")
        print(f"Position size increased by: {size_ratio:.1f}x")
        print(f"Capital increased by: {capital_ratio:.1f}x")
        print(f"Match? {'YES - COMPOUNDING DETECTED!' if abs(size_ratio - capital_ratio) < 0.5 else 'NO'}")
        
        # Save detailed analysis
        pos_df.to_csv('position_sizing_analysis.csv', index=False)
        print(f"\nDetailed analysis saved to position_sizing_analysis.csv")
        
        return pos_df


if __name__ == "__main__":
    analyze_position_sizing()