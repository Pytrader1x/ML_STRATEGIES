#!/usr/bin/env python3
"""
Debug script to test timestamp generation and verify React chart compatibility
"""

import sys
import json
import pandas as pd
import numpy as np

sys.path.append('strategy_code')
from chart_data_exporter import ChartDataExporter

def test_timestamp_generation():
    """Test timestamp generation with small dataset"""
    
    print("🔍 Testing timestamp generation...")
    
    # Create small test dataset 
    dates = pd.date_range('2023-01-01 00:00:00', periods=20, freq='15T')
    df = pd.DataFrame({
        'Open': np.random.uniform(1.0, 1.1, 20),
        'High': np.random.uniform(1.05, 1.15, 20),
        'Low': np.random.uniform(0.95, 1.05, 20),
        'Close': np.random.uniform(1.0, 1.1, 20)
    }, index=dates)
    
    print(f"📊 Created DataFrame with {len(df)} rows")
    print(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test export
    results = {'trades': [], 'symbol': 'TEST'}
    chart_data = ChartDataExporter.export_to_json(df, results)
    
    # Analyze timestamps
    times = [point['time'] for point in chart_data['ohlc']]
    
    print(f"\n⏰ Timestamp Analysis:")
    print(f"   Total points: {len(times)}")
    print(f"   First timestamp: {times[0]} ({pd.Timestamp(times[0], unit='s')})")
    print(f"   Last timestamp: {times[-1]} ({pd.Timestamp(times[-1], unit='s')})")
    
    # Check for duplicates
    unique_times = set(times)
    duplicates = len(times) - len(unique_times)
    print(f"   Unique timestamps: {len(unique_times)}")
    print(f"   Duplicate timestamps: {duplicates}")
    
    # Check ordering
    is_ascending = all(times[i] > times[i-1] for i in range(1, len(times)))
    print(f"   All ascending: {is_ascending}")
    
    # Check intervals
    intervals = [times[i] - times[i-1] for i in range(1, len(times))]
    print(f"   Intervals: {intervals[:5]}... (showing first 5)")
    print(f"   All 900s: {all(interval == 900 for interval in intervals)}")
    
    if duplicates == 0 and is_ascending:
        print("✅ Timestamps are perfect!")
        
        # Save test file
        output_path = '../react_chart/public/debug_chart_data.json'
        ChartDataExporter.export_to_json(df, results, output_path)
        print(f"💾 Saved debug chart data to: {output_path}")
        
        return True
    else:
        print("❌ Timestamp issues detected!")
        
        # Show problem timestamps
        for i in range(1, len(times)):
            if times[i] <= times[i-1]:
                print(f"   Problem at index {i}: {times[i]} <= {times[i-1]}")
        
        return False

def check_current_chart_data():
    """Check the current chart_data.json file"""
    
    print("\n🔍 Checking current chart_data.json...")
    
    try:
        with open('../react_chart/public/chart_data.json', 'r') as f:
            data = json.load(f)
        
        times = [point['time'] for point in data['ohlc']]
        
        print(f"📊 Current chart data:")
        print(f"   Total points: {len(times)}")
        print(f"   First 10 timestamps: {times[:10]}")
        
        # Find duplicates
        seen = set()
        duplicates = []
        for i, t in enumerate(times):
            if t in seen:
                duplicates.append((i, t))
            seen.add(t)
        
        if duplicates:
            print(f"❌ Found {len(duplicates)} duplicate timestamps:")
            for i, t in duplicates[:5]:  # Show first 5
                print(f"   Index {i}: timestamp {t}")
        else:
            print("✅ No duplicate timestamps found")
            
        # Check ordering
        non_ascending = []
        for i in range(1, len(times)):
            if times[i] <= times[i-1]:
                non_ascending.append((i, times[i-1], times[i]))
        
        if non_ascending:
            print(f"❌ Found {len(non_ascending)} ordering issues:")
            for i, prev_t, curr_t in non_ascending[:5]:  # Show first 5
                print(f"   Index {i}: {curr_t} <= {prev_t}")
        else:
            print("✅ All timestamps properly ordered")
            
    except FileNotFoundError:
        print("❌ chart_data.json not found")
    except Exception as e:
        print(f"❌ Error reading chart_data.json: {e}")

if __name__ == '__main__':
    print("🚀 Starting timestamp debugging...\n")
    
    # Test with fresh data
    test_result = test_timestamp_generation()
    
    # Check existing data
    check_current_chart_data()
    
    if test_result:
        print("\n🎉 Debug data generated successfully!")
        print("🌐 You can test the debug data by:")
        print("   1. cd ../react_chart")
        print("   2. npm run dev")
        print("   3. Open http://localhost:5173")
        print("   4. Load debug_chart_data.json to test")
    else:
        print("\n⚠️  Timestamp generation still has issues!")