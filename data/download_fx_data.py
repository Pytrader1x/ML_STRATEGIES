#!/usr/bin/env python3
"""
FX Data Download Script
Run this script from the data directory to download historical FX data.

Usage:
    cd /Users/williamsmith/Python_local_Mac/Ml_Strategies/data
    python download_fx_data.py
"""

import fx_data_downloader as fx
from datetime import datetime

# Currency pairs to download
pairs = [
    'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 
    'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 
    'NZDUSD', 'USDCAD'
]

# Data directory - current directory since script is in data/
data_dir = '.'

# Start year - changed to 2010 for more historical data
start_year = 2010

print(f"Starting FX data download at {datetime.now()}")
print(f"Downloading data from {start_year} to present for {len(pairs)} currency pairs")
print("-" * 60)

# Download 1-minute data for each pair
for i, pair in enumerate(pairs, 1):
    print(f"\n[{i}/{len(pairs)}] Processing {pair}...")
    
    try:
        # Download or update 1-minute data
        print(f"  Downloading 1-minute data...")
        fx.download_fx_data(pair, start_year=start_year, data_dir=data_dir)
        print(f"  ✓ Downloaded {pair} 1-minute data")
        
        # Create 15-minute resampled version
        print(f"  Creating 15-minute resampled data...")
        fx.create_resampled_data(pair, '15M', data_dir=data_dir)
        print(f"  ✓ Created {pair} 15-minute data")
        
    except Exception as e:
        print(f"  ✗ Error processing {pair}: {str(e)}")
        continue

print(f"\n{'='*60}")
print(f"Download complete at {datetime.now()}")
print(f"\nData files created in current directory")
print("\nFor each pair you should have:")
print("  - {PAIR}_MASTER.csv (1-minute data)")
print("  - {PAIR}_MASTER_15M.csv (15-minute resampled data)")