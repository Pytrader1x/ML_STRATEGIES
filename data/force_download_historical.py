#!/usr/bin/env python3
"""
Force download historical AUDUSD data from earlier years
"""

import fx_data_downloader as fx
import pandas as pd
import os
from datetime import datetime

# Delete existing master file to force re-download
master_file = "AUDUSD_MASTER.csv"
if os.path.exists(master_file):
    # Make backup first
    backup = f"AUDUSD_MASTER_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.rename(master_file, backup)
    print(f"Backed up existing file to: {backup}")

print("Forcing download of AUDUSD from 2010...")
print("-" * 60)

try:
    # Force download from 2010
    fx.download_fx_data('AUDUSD', start_year=2010, data_dir='.')
    
    # Check what we got
    if os.path.exists(master_file):
        df = pd.read_csv(master_file)
        print(f"\nDownloaded data:")
        print(f"- Rows: {len(df):,}")
        print(f"- Date range: {df['DateTime'].min()} to {df['DateTime'].max()}")
        
        # Create 15-minute data
        print("\nCreating 15-minute data...")
        fx.create_resampled_data('AUDUSD', '15M', data_dir='.')
        
        # Check 15-min data
        df_15m = pd.read_csv("AUDUSD_MASTER_15M.csv")
        print(f"\n15-minute data:")
        print(f"- Rows: {len(df_15m):,}")
        print(f"- Date range: {df_15m['DateTime'].min()} to {df_15m['DateTime'].max()}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nThe data source (histdata.com) might only have data from 2018 onwards.")
    print("\nFor earlier data, you would need to:")
    print("1. Use a paid data provider (e.g., Refinitiv, Bloomberg)")
    print("2. Download manually from Dukascopy")
    print("3. Use MetaTrader historical data export")