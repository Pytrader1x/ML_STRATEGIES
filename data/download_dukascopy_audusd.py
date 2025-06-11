#!/usr/bin/env python3
"""
Download AUDUSD historical data from Dukascopy
This provides high-quality tick data that can be resampled to any timeframe
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import struct
import lzma
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class DukascopyDownloader:
    def __init__(self):
        self.base_url = "https://datafeed.dukascopy.com/datafeed"
        self.symbol = "AUDUSD"
        self.point_value = 0.00001  # 5 decimal places for AUDUSD
        
    def get_tick_url(self, date, hour):
        """Generate URL for tick data file"""
        year = date.year
        month = date.month - 1  # Dukascopy uses 0-based months
        day = date.day
        
        return f"{self.base_url}/{self.symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    
    def download_hour(self, date, hour):
        """Download one hour of tick data"""
        url = self.get_tick_url(date, hour)
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return self.parse_ticks(response.content, date, hour)
            else:
                return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def parse_ticks(self, data, date, hour):
        """Parse binary tick data"""
        try:
            # Decompress LZMA data
            decompressed = lzma.decompress(data)
            
            # Parse binary format: time(4), ask(4), bid(4), ask_vol(4), bid_vol(4)
            ticks = []
            for i in range(0, len(decompressed), 20):
                if i + 20 <= len(decompressed):
                    chunk = decompressed[i:i+20]
                    time_delta, ask, bid, ask_vol, bid_vol = struct.unpack('>iiiff', chunk)
                    
                    # Calculate timestamp
                    timestamp = datetime(date.year, date.month, date.day, hour) + timedelta(milliseconds=time_delta)
                    
                    # Convert to prices
                    ask_price = ask * self.point_value
                    bid_price = bid * self.point_value
                    
                    ticks.append({
                        'DateTime': timestamp,
                        'Bid': bid_price,
                        'Ask': ask_price,
                        'BidVolume': bid_vol,
                        'AskVolume': ask_vol
                    })
            
            return pd.DataFrame(ticks)
        except:
            return pd.DataFrame()
    
    def download_day(self, date):
        """Download all hours for a specific day"""
        day_data = []
        
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = []
            for hour in range(24):
                future = executor.submit(self.download_hour, date, hour)
                futures.append(future)
            
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    day_data.append(df)
        
        if day_data:
            return pd.concat(day_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def resample_to_ohlc(self, tick_df, timeframe='15min'):
        """Convert tick data to OHLC format"""
        if tick_df.empty:
            return pd.DataFrame()
        
        # Calculate mid price
        tick_df['Mid'] = (tick_df['Bid'] + tick_df['Ask']) / 2
        
        # Set DateTime as index
        tick_df.set_index('DateTime', inplace=True)
        
        # Resample to OHLC
        ohlc = tick_df['Mid'].resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        
        # Add volume (sum of bid and ask volumes)
        volume = (tick_df['BidVolume'] + tick_df['AskVolume']).resample(timeframe).sum()
        ohlc['Volume'] = volume
        
        # Remove empty rows
        ohlc = ohlc.dropna()
        
        return ohlc
    
    def download_range(self, start_date, end_date):
        """Download data for a date range"""
        print(f"Downloading {self.symbol} from {start_date} to {end_date}")
        
        all_data = []
        current_date = start_date
        
        # Create progress bar
        total_days = (end_date - start_date).days + 1
        pbar = tqdm(total=total_days, desc="Downloading days")
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                df = self.download_day(current_date)
                if not df.empty:
                    all_data.append(df)
                    pbar.set_postfix({'date': current_date.strftime('%Y-%m-%d'), 'ticks': len(df)})
            
            current_date += timedelta(days=1)
            pbar.update(1)
        
        pbar.close()
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

def main():
    """Download AUDUSD data from 2010 and merge with existing data"""
    print("="*80)
    print("DUKASCOPY AUDUSD HISTORICAL DATA DOWNLOAD")
    print("="*80)
    
    downloader = DukascopyDownloader()
    
    # For demonstration, download just last month
    # (Full download from 2010 would take several hours)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\nDownloading tick data for last 30 days...")
    print("(For full 2010-present download, modify start_date in the script)")
    
    # Download tick data
    tick_data = downloader.download_range(start_date, end_date)
    
    if not tick_data.empty:
        print(f"\nDownloaded {len(tick_data):,} ticks")
        
        # Resample to 15-minute
        print("\nResampling to 15-minute OHLC...")
        ohlc_15min = downloader.resample_to_ohlc(tick_data, '15min')
        
        print(f"Created {len(ohlc_15min):,} 15-minute bars")
        
        # Prepare in standard format
        df_new = ohlc_15min.reset_index()
        df_new.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Load and merge with existing data
        existing_file = "AUDUSD_MASTER_15M.csv"
        if os.path.exists(existing_file):
            print(f"\nLoading existing {existing_file}...")
            df_existing = pd.read_csv(existing_file)
            df_existing['DateTime'] = pd.to_datetime(df_existing['DateTime'])
            
            # Merge datasets
            df_combined = pd.concat([df_new, df_existing], ignore_index=True)
            df_combined = df_combined.sort_values('DateTime').drop_duplicates(subset=['DateTime'], keep='last')
            
            # Save backup
            backup_file = f"AUDUSD_MASTER_15M_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_existing.to_csv(backup_file, index=False)
            print(f"Backup saved: {backup_file}")
            
            # Save updated file
            df_combined.to_csv(existing_file, index=False)
            print(f"Updated {existing_file} with {len(df_new)} new bars")
        else:
            # Save new file
            df_new.to_csv(existing_file, index=False)
            print(f"Saved {existing_file} with {len(df_new)} bars")
    else:
        print("\nNo data downloaded")
    
    print("\n" + "="*60)
    print("FULL HISTORICAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nTo download from 2010:")
    print("1. Modify the script to set: start_date = datetime(2010, 1, 1)")
    print("2. Be patient - this will download ~14 years of tick data")
    print("3. Estimated time: 2-4 hours depending on connection")
    print("4. Estimated size: ~50GB of tick data → ~500MB of 15-min data")
    print("\nAlternatively, use the Dukascopy web interface:")
    print("https://www.dukascopy.com/swiss/english/marketwatch/historical/")

if __name__ == "__main__":
    main()