"""
Test script to verify consecutive wins/losses calculations
"""

import pandas as pd
import numpy as np
from run_Strategy import calculate_trade_statistics, create_config_1_ultra_tight_risk, load_and_prepare_data
import sys
sys.path.append('..')
from technical_indicators_custom import TIC


def test_consecutive_calculations():
    """Test the consecutive wins/losses calculation with known data"""
    
    print("="*80)
    print("TESTING CONSECUTIVE WINS/LOSSES CALCULATION")
    print("="*80)
    
    # Test Case 1: Simple synthetic data
    print("\n1. Testing with synthetic trade data:")
    
    # Create mock results with known pattern
    mock_trades = [
        type('Trade', (), {'pnl': 100}),    # Win
        type('Trade', (), {'pnl': 150}),    # Win
        type('Trade', (), {'pnl': -50}),    # Loss
        type('Trade', (), {'pnl': 200}),    # Win
        type('Trade', (), {'pnl': 175}),    # Win
        type('Trade', (), {'pnl': 125}),    # Win
        type('Trade', (), {'pnl': -75}),    # Loss
        type('Trade', (), {'pnl': -100}),   # Loss
        type('Trade', (), {'pnl': 50}),     # Win
    ]
    
    mock_results = {
        'trades': mock_trades,
        'total_trades': len(mock_trades),
        'win_rate': 66.7
    }
    
    # Calculate statistics
    stats = calculate_trade_statistics(mock_results)
    
    print(f"   Pattern: W W L W W W L L W")
    print(f"   Expected max consecutive wins: 3")
    print(f"   Calculated max consecutive wins: {stats['max_consecutive_wins']}")
    print(f"   Expected max consecutive losses: 2")
    print(f"   Calculated max consecutive losses: {stats['max_consecutive_losses']}")
    print(f"   Total wins: {stats['num_wins']}")
    print(f"   Total losses: {stats['num_losses']}")
    
    # Test Case 2: Test with actual strategy
    print("\n2. Testing with real strategy on AUDUSD data:")
    
    try:
        # Load data
        df = load_and_prepare_data('AUDUSD', data_path='../data')
        
        # Use a small sample
        sample_df = df.iloc[:5000].copy()  # About 1 month of data
        
        # Create strategy
        strategy = create_config_1_ultra_tight_risk()
        
        # Run backtest
        print("   Running backtest...")
        results = strategy.run_backtest(sample_df)
        
        # Calculate trade statistics
        trade_stats = calculate_trade_statistics(results)
        
        print(f"\n   Results from real data:")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Win rate: {results['win_rate']:.1f}%")
        print(f"   Max consecutive wins: {trade_stats['max_consecutive_wins']}")
        print(f"   Max consecutive losses: {trade_stats['max_consecutive_losses']}")
        print(f"   Average consecutive wins: {trade_stats['avg_consecutive_wins']:.1f}")
        print(f"   Average consecutive losses: {trade_stats['avg_consecutive_losses']:.1f}")
        
        # Show actual win/loss pattern for first 20 trades
        if 'trades' in results and len(results['trades']) > 0:
            print("\n   First 20 trades pattern:")
            pattern = ""
            for i, trade in enumerate(results['trades'][:20]):
                if hasattr(trade, 'pnl'):
                    pattern += "W" if trade.pnl > 0 else "L"
                else:
                    pattern += "?"
                if (i + 1) % 10 == 0:
                    pattern += " "
            print(f"   {pattern}")
            
    except Exception as e:
        print(f"   Error running real strategy test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Case 3: Edge cases
    print("\n3. Testing edge cases:")
    
    # All wins
    all_wins = {'trades': [type('Trade', (), {'pnl': 100}) for _ in range(10)]}
    stats_all_wins = calculate_trade_statistics(all_wins)
    print(f"   All wins (10 trades):")
    print(f"   - Max consecutive wins: {stats_all_wins['max_consecutive_wins']} (expected: 10)")
    print(f"   - Max consecutive losses: {stats_all_wins['max_consecutive_losses']} (expected: 0)")
    
    # All losses
    all_losses = {'trades': [type('Trade', (), {'pnl': -100}) for _ in range(10)]}
    stats_all_losses = calculate_trade_statistics(all_losses)
    print(f"   All losses (10 trades):")
    print(f"   - Max consecutive wins: {stats_all_losses['max_consecutive_wins']} (expected: 0)")
    print(f"   - Max consecutive losses: {stats_all_losses['max_consecutive_losses']} (expected: 10)")
    
    # Empty trades
    empty_trades = {'trades': []}
    stats_empty = calculate_trade_statistics(empty_trades)
    print(f"   Empty trades:")
    print(f"   - Max consecutive wins: {stats_empty['max_consecutive_wins']} (expected: 0)")
    print(f"   - Max consecutive losses: {stats_empty['max_consecutive_losses']} (expected: 0)")


if __name__ == "__main__":
    test_consecutive_calculations()