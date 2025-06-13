#!/usr/bin/env python3
"""
Quick test script to verify React chart data export works correctly
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.append('.')
sys.path.append('strategy_code')

from strategy_code.chart_data_exporter import ChartDataExporter

def test_export():
    """Test the chart data export with sample data"""
    
    print("Testing React chart data export...")
    
    # Create sample trading data
    dates = pd.date_range('2023-01-01', periods=100, freq='15T')
    
    # Generate realistic OHLC data
    np.random.seed(42)
    prices = 1.0 + np.cumsum(np.random.randn(100) * 0.001)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': prices + np.abs(np.random.randn(100) * 0.002),
        'Low': prices - np.abs(np.random.randn(100) * 0.002),
        'Close': prices + np.random.randn(100) * 0.001,
        # Add some dummy indicators
        'NTI_FastEMA': prices * 1.001,
        'NTI_SlowEMA': prices * 0.999,
        'NTI_Direction': np.random.choice([-1, 0, 1], 100),
        'NTI_Confidence': np.random.uniform(0, 1, 100),
        'MB_Bias': np.random.choice([-1, 0, 1], 100),
        'IC_Regime': np.random.choice([0, 1, 2], 100),
        'IC_RegimeName': np.random.choice(['Trend', 'Range', 'Chop'], 100)
    }, index=dates)
    
    # Create sample results
    results = {
        'trades': [],
        'symbol': 'TEST',
        'total_pnl': 1000,
        'total_return': 10.5,
        'sharpe_ratio': 1.5,
        'win_rate': 65.5,
        'max_drawdown': -2.5,
        'profit_factor': 1.8,
        'total_trades': 20
    }
    
    # Export to JSON
    output_path = '../react_chart/public/test_chart_data.json'
    chart_data = ChartDataExporter.export_to_json(df, results, output_path)
    
    # Verify timestamps
    ohlc = chart_data['ohlc']
    times = [point['time'] for point in ohlc]
    
    print(f"✅ Exported {len(ohlc)} OHLC data points")
    print(f"✅ First timestamp: {times[0]}")
    print(f"✅ Last timestamp: {times[-1]}")
    print(f"✅ Time intervals: {[times[i] - times[i-1] for i in range(1, min(6, len(times)))]}")
    print(f"✅ All ascending: {all(times[i] > times[i-1] for i in range(1, len(times)))}")
    print(f"✅ All unique: {len(set(times)) == len(times)}")
    
    # Check indicators
    indicators = chart_data['indicators']
    print(f"✅ NeuroTrend indicators: {len(indicators['neurotrend']['fast_ema'])} points")
    print(f"✅ Market Bias indicators: {len(indicators['market_bias']['bias'])} points")
    print(f"✅ Intelligent Chop indicators: {len(indicators['intelligent_chop']['regime'])} points")
    
    print(f"\n🎉 Export successful! File saved to: {output_path}")
    print("📊 You can now view the chart at: http://localhost:5173")
    print("\nTo start the React server:")
    print("  cd ../react_chart")
    print("  npm run dev")
    
    return True

if __name__ == '__main__':
    test_export()