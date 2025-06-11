#!/usr/bin/env python3
"""
Download AUDUSD historical data from 2010
Run this script from the data directory.
"""

import fx_data_downloader as fx
from datetime import datetime
import pandas as pd
import os

# Just download AUDUSD
pair = 'AUDUSD'
data_dir = '.'
start_year = 2015  # Try 2015 as histdata might have more recent years only

print(f"Starting AUDUSD historical data download at {datetime.now()}")
print(f"Downloading data from {start_year} to present")
print("-" * 60)

# Check current data first
existing_file = f"{pair}_MASTER_15M.csv"
if os.path.exists(existing_file):
    df_existing = pd.read_csv(existing_file)
    print(f"\nCurrent {existing_file}:")
    print(f"- Rows: {len(df_existing):,}")
    print(f"- Date range: {df_existing['DateTime'].min()} to {df_existing['DateTime'].max()}")
    
    # Backup
    backup_file = f"{pair}_MASTER_15M_pre2010_{datetime.now().strftime('%Y%m%d')}.csv"
    df_existing.to_csv(backup_file, index=False)
    print(f"- Backup saved: {backup_file}")

try:
    # Download 1-minute data
    print(f"\nDownloading {pair} 1-minute data from {start_year}...")
    print("This may take 10-30 minutes...")
    fx.download_fx_data(pair, start_year=start_year, data_dir=data_dir)
    print(f"✓ Downloaded {pair} 1-minute data")
    
    # Create 15-minute resampled version
    print(f"\nCreating 15-minute resampled data...")
    fx.create_resampled_data(pair, '15M', data_dir=data_dir)
    print(f"✓ Created {pair} 15-minute data")
    
    # Check new data
    if os.path.exists(existing_file):
        df_new = pd.read_csv(existing_file)
        print(f"\nUpdated {existing_file}:")
        print(f"- Rows: {len(df_new):,}")
        print(f"- Date range: {df_new['DateTime'].min()} to {df_new['DateTime'].max()}")
        
        if 'df_existing' in locals():
            rows_added = len(df_new) - len(df_existing)
            print(f"- New rows added: {rows_added:,}")
    
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    print("\nIf download fails, try:")
    print("1. Using start_year=2015 instead")
    print("2. Running during off-peak hours")
    print("3. Using a VPN if geo-blocked")

print(f"\n{'='*60}")
print(f"Script completed at {datetime.now()}")