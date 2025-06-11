#!/usr/bin/env python3
"""
Download AUDUSD data from 2010 onwards using yfinance
Updates existing AUDUSD_MASTER_15M.csv with historical data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np

def download_audusd_historical():
    """Download AUDUSD data from 2010 and merge with existing data"""
    
    print("="*80)
    print("AUDUSD HISTORICAL DATA DOWNLOAD (2010-Present)")
    print("="*80)
    
    # Define parameters
    symbol = "AUDUSD=X"  # Yahoo Finance symbol for AUD/USD
    start_date = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {symbol} from {start_date} to {end_date}")
    
    try:
        # Download data with 15-minute intervals
        # Note: Yahoo Finance might not have 15-minute data going back to 2010
        # We'll try 1-hour first and then resample if needed
        print("\nAttempting to download 1-hour data (will resample to 15-min)...")
        
        ticker = yf.Ticker(symbol)
        df_hourly = ticker.history(start=start_date, end=end_date, interval="1h")
        
        if df_hourly.empty:
            print("No hourly data available. Trying daily data...")
            df_daily = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df_daily.empty:
                print("ERROR: No data available from Yahoo Finance")
                return False
            else:
                print(f"Downloaded daily data: {len(df_daily)} rows")
                # For daily data, we can't really resample to 15-min meaningfully
                print("WARNING: Only daily data available - cannot create meaningful 15-min data")
                return False
        
        print(f"Downloaded hourly data: {len(df_hourly)} rows")
        print(f"Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
        
        # Convert to 15-minute data by forward-filling
        # This is not ideal but gives us more granular data
        print("\nResampling to 15-minute intervals...")
        df_15min = df_hourly.resample('15min').ffill()
        
        # Prepare dataframe in the same format as existing data
        df_new = pd.DataFrame()
        df_new['DateTime'] = df_15min.index
        df_new['Open'] = df_15min['Open']
        df_new['High'] = df_15min['High']
        df_new['Low'] = df_15min['Low']
        df_new['Close'] = df_15min['Close']
        df_new['Volume'] = df_15min['Volume'].fillna(0)
        
        # Reset index
        df_new = df_new.reset_index(drop=True)
        
        # Load existing data
        existing_file = "AUDUSD_MASTER_15M.csv"
        if os.path.exists(existing_file):
            print(f"\nLoading existing {existing_file}...")
            df_existing = pd.read_csv(existing_file)
            df_existing['DateTime'] = pd.to_datetime(df_existing['DateTime'])
            
            print(f"Existing data: {len(df_existing)} rows")
            print(f"Existing range: {df_existing['DateTime'].min()} to {df_existing['DateTime'].max()}")
            
            # Find overlap and merge
            existing_min_date = df_existing['DateTime'].min()
            new_data_before = df_new[df_new['DateTime'] < existing_min_date]
            
            print(f"\nNew historical data before existing: {len(new_data_before)} rows")
            
            if len(new_data_before) > 0:
                # Combine datasets
                df_combined = pd.concat([new_data_before, df_existing], ignore_index=True)
                df_combined = df_combined.sort_values('DateTime').reset_index(drop=True)
                
                # Remove duplicates
                df_combined = df_combined.drop_duplicates(subset=['DateTime'], keep='last')
                
                print(f"Combined data: {len(df_combined)} rows")
                print(f"Combined range: {df_combined['DateTime'].min()} to {df_combined['DateTime'].max()}")
                
                # Save backup
                backup_file = f"AUDUSD_MASTER_15M_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df_existing.to_csv(backup_file, index=False)
                print(f"\nBackup saved: {backup_file}")
                
                # Save updated file
                df_combined.to_csv(existing_file, index=False)
                print(f"Updated {existing_file} successfully!")
                
                # Calculate statistics
                years_added = (existing_min_date - df_combined['DateTime'].min()).days / 365.25
                print(f"\nAdded {years_added:.1f} years of historical data")
                print(f"Total dataset now spans {(df_combined['DateTime'].max() - df_combined['DateTime'].min()).days / 365.25:.1f} years")
                
                return True
            else:
                print("\nNo new historical data to add (existing data already covers this period)")
                return True
                
        else:
            # No existing file, save new data
            print(f"\nNo existing file found. Saving new data to {existing_file}")
            df_new.to_csv(existing_file, index=False)
            print(f"Saved {len(df_new)} rows")
            return True
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTrying alternative approach with Alpha Vantage or other sources...")
        return False

def download_with_alpha_vantage():
    """Alternative method using Alpha Vantage (requires API key)"""
    print("\n" + "="*60)
    print("ALTERNATIVE: Alpha Vantage Download")
    print("="*60)
    
    # Note: This requires an API key
    print("To use Alpha Vantage:")
    print("1. Get a free API key from: https://www.alphavantage.co/support/#api-key")
    print("2. Install: pip install alpha-vantage")
    print("3. Use their FX_INTRADAY function for 15-min data")
    print("\nHowever, free tier is limited to:")
    print("- 5 API requests per minute")
    print("- 500 requests per day")
    print("- Limited historical data (usually last 2 months for intraday)")
    
    return False

def verify_data_quality(df):
    """Verify the quality of downloaded data"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing)
    
    # Check for zero/negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                print(f"WARNING: {invalid} zero/negative values in {col}")
    
    # Check for price consistency
    if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
        invalid_hl = (df['High'] < df['Low']).sum()
        if invalid_hl > 0:
            print(f"WARNING: {invalid_hl} rows where High < Low")
    
    return True

if __name__ == "__main__":
    # Try yfinance first
    success = download_audusd_historical()
    
    if not success:
        # Provide alternative options
        print("\n" + "="*60)
        print("ALTERNATIVE DATA SOURCES")
        print("="*60)
        print("\n1. Dukascopy:")
        print("   - URL: https://www.dukascopy.com/swiss/english/marketwatch/historical/")
        print("   - Provides tick data from 2003")
        print("   - Free but requires manual download")
        print("\n2. FXCM:")
        print("   - URL: https://www.fxcm.com/markets/data-download/")
        print("   - Historical data available")
        print("   - Requires account")
        print("\n3. MetaTrader:")
        print("   - Export from MT4/MT5 if you have access")
        print("   - Most complete historical data")
        print("\n4. Quandl:")
        print("   - Some FX data available")
        print("   - Requires API key")
        
        # Try Alpha Vantage info
        download_with_alpha_vantage()
    
    print("\n" + "="*80)
    print("Download script completed")
    print("="*80)