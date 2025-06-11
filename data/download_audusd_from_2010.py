#!/usr/bin/env python3
"""
Download AUDUSD data from 2010 using fx_data_downloader
"""

import fx_data_downloader as fx
import pandas as pd
from datetime import datetime
import os

def download_and_merge_audusd():
    """Download AUDUSD from 2010 and merge with existing data"""
    
    print("=" * 80)
    print("AUDUSD HISTORICAL DATA DOWNLOAD (2010-Present)")
    print("=" * 80)
    
    # Parameters
    pair = 'AUDUSD'
    start_year = 2010
    data_dir = '.'
    
    # Check existing data first
    existing_file = f"{pair}_MASTER_15M.csv"
    existing_1m_file = f"{pair}_MASTER.csv"
    
    if os.path.exists(existing_file):
        print(f"\nExisting {existing_file} found")
        df_existing = pd.read_csv(existing_file)
        df_existing['DateTime'] = pd.to_datetime(df_existing['DateTime'])
        print(f"Current data range: {df_existing['DateTime'].min()} to {df_existing['DateTime'].max()}")
        print(f"Current rows: {len(df_existing):,}")
        
        # Backup existing file
        backup_file = f"{pair}_MASTER_15M_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_existing.to_csv(backup_file, index=False)
        print(f"Backup saved: {backup_file}")
    
    try:
        # Download 1-minute data from 2010
        print(f"\nDownloading {pair} 1-minute data from {start_year}...")
        print("This may take 10-30 minutes depending on your connection...")
        
        fx.download_fx_data(pair, start_year=start_year, data_dir=data_dir)
        print(f"✓ Downloaded {pair} 1-minute data")
        
        # Create 15-minute resampled version
        print(f"\nCreating 15-minute resampled data...")
        fx.create_resampled_data(pair, '15M', data_dir=data_dir)
        print(f"✓ Created {pair} 15-minute data")
        
        # Check the new data
        if os.path.exists(existing_file):
            df_new = pd.read_csv(existing_file)
            df_new['DateTime'] = pd.to_datetime(df_new['DateTime'])
            print(f"\nNew data range: {df_new['DateTime'].min()} to {df_new['DateTime'].max()}")
            print(f"New rows: {len(df_new):,}")
            
            # Calculate how much historical data was added
            if 'df_existing' in locals():
                new_start = df_new['DateTime'].min()
                old_start = df_existing['DateTime'].min()
                if new_start < old_start:
                    years_added = (old_start - new_start).days / 365.25
                    rows_added = len(df_new) - len(df_existing)
                    print(f"\nAdded {years_added:.1f} years of historical data")
                    print(f"Added {rows_added:,} new rows")
                else:
                    print("\nNo earlier data was available from the source")
        
        # Verify data quality
        print("\n" + "-"*60)
        print("DATA QUALITY CHECK")
        print("-"*60)
        
        if os.path.exists(existing_file):
            df = pd.read_csv(existing_file)
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.any():
                print("Missing values found:")
                print(missing[missing > 0])
            else:
                print("✓ No missing values")
            
            # Check price consistency
            invalid_hl = (df['High'] < df['Low']).sum()
            if invalid_hl > 0:
                print(f"⚠ WARNING: {invalid_hl} rows where High < Low")
            else:
                print("✓ Price consistency check passed")
            
            # Show sample data
            print("\nSample data (first 5 rows):")
            print(df.head())
            
            print("\nSample data (last 5 rows):")
            print(df.tail())
        
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        
        print("\n" + "-"*60)
        print("TROUBLESHOOTING")
        print("-"*60)
        print("1. Check your internet connection")
        print("2. The data source might be temporarily unavailable")
        print("3. Try again in a few minutes")
        print("4. If the problem persists, try downloading from 2015 instead:")
        print("   fx.download_fx_data('AUDUSD', start_year=2015)")
        
        return False

if __name__ == "__main__":
    success = download_and_merge_audusd()
    
    if success:
        print("\nNext steps:")
        print("1. Run the Monte Carlo analysis with the extended dataset")
        print("2. Compare strategy performance across different time periods")
        print("3. Test robustness during major events (2008 crisis, COVID, etc.)")