#!/usr/bin/env python3
"""
Alternative Data Sources - Specialized downloaders for tick data and alternative providers
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import argparse

# Try to import required libraries
try:
    import requests
    import lzma
    import struct
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    DUKASCOPY_AVAILABLE = True
except ImportError:
    DUKASCOPY_AVAILABLE = False
    print("Warning: Dependencies for Dukascopy not available (requests, lzma, tqdm)")

try:
    import yfinance as yf
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    print("Warning: yfinance not installed")


class DukascopyDownloader:
    """Download high-quality tick data from Dukascopy"""
    
    BASE_URL = "https://datafeed.dukascopy.com/datafeed"
    
    INSTRUMENTS = {
        'AUDUSD': 'AUDUSD',
        'GBPUSD': 'GBPUSD', 
        'EURUSD': 'EURUSD',
        'NZDUSD': 'NZDUSD',
        'USDCAD': 'USDCAD',
        'USDJPY': 'USDJPY',
        'GBPJPY': 'GBPJPY',
        'EURJPY': 'EURJPY',
        'AUDJPY': 'AUDJPY',
        'CADJPY': 'CADJPY',
        'CHFJPY': 'CHFJPY',
        'AUDNZD': 'AUDNZD',
        'EURGBP': 'EURGBP'
    }
    
    def __init__(self, output_dir: str = '.'):
        self.output_dir = output_dir
        
    def _download_hour(self, instrument: str, year: int, month: int, 
                      day: int, hour: int) -> Optional[bytes]:
        """Download one hour of tick data"""
        url = f"{self.BASE_URL}/{instrument}/{year:04d}/{month-1:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.content
            return None
        except Exception:
            return None
    
    def _decompress_ticks(self, data: bytes) -> List[Tuple[int, float, float, float, float]]:
        """Decompress tick data from Dukascopy format"""
        try:
            # Decompress LZMA data
            decompressed = lzma.decompress(data)
            
            # Parse binary format
            ticks = []
            offset = 0
            while offset < len(decompressed):
                # Read tick: time(4), ask(4), bid(4), ask_vol(4), bid_vol(4)
                if offset + 20 > len(decompressed):
                    break
                    
                tick_data = struct.unpack('>IIIfI', decompressed[offset:offset+20])
                time_delta, ask, bid, ask_vol, bid_vol = tick_data
                
                # Convert to proper values
                ask = ask / 100000
                bid = bid / 100000
                
                ticks.append((time_delta, ask, bid, ask_vol, bid_vol))
                offset += 20
                
            return ticks
        except Exception as e:
            print(f"Error decompressing ticks: {e}")
            return []
    
    def download_period(self, instrument: str, start_date: datetime, 
                       end_date: datetime) -> pd.DataFrame:
        """Download tick data for a period"""
        
        if not DUKASCOPY_AVAILABLE:
            raise ImportError("Dukascopy dependencies not available")
            
        all_ticks = []
        current_date = start_date
        
        # Calculate total hours for progress bar
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        
        with tqdm(total=total_hours, desc=f"Downloading {instrument}") as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                while current_date <= end_date:
                    # Submit download tasks
                    future = executor.submit(
                        self._download_hour,
                        instrument,
                        current_date.year,
                        current_date.month,
                        current_date.day,
                        current_date.hour
                    )
                    futures.append((future, current_date))
                    
                    current_date += timedelta(hours=1)
                    
                    # Process completed futures
                    if len(futures) >= 100:
                        for future, dt in futures[:50]:
                            data = future.result()
                            if data:
                                ticks = self._decompress_ticks(data)
                                base_time = int(dt.timestamp() * 1000)
                                
                                for tick in ticks:
                                    time_ms = base_time + tick[0]
                                    all_ticks.append({
                                        'Time': pd.Timestamp(time_ms, unit='ms'),
                                        'Ask': tick[1],
                                        'Bid': tick[2],
                                        'AskVolume': tick[3],
                                        'BidVolume': tick[4]
                                    })
                            pbar.update(1)
                        futures = futures[50:]
                
                # Process remaining futures
                for future, dt in futures:
                    data = future.result()
                    if data:
                        ticks = self._decompress_ticks(data)
                        base_time = int(dt.timestamp() * 1000)
                        
                        for tick in ticks:
                            time_ms = base_time + tick[0]
                            all_ticks.append({
                                'Time': pd.Timestamp(time_ms, unit='ms'),
                                'Ask': tick[1],
                                'Bid': tick[2],
                                'AskVolume': tick[3],
                                'BidVolume': tick[4]
                            })
                    pbar.update(1)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ticks)
        if not df.empty:
            df = df.set_index('Time').sort_index()
            
        return df
    
    def ticks_to_ohlc(self, tick_df: pd.DataFrame, period: str = '15T') -> pd.DataFrame:
        """Convert tick data to OHLC format"""
        if tick_df.empty:
            return pd.DataFrame()
            
        # Calculate mid price
        tick_df['Mid'] = (tick_df['Ask'] + tick_df['Bid']) / 2
        
        # Resample to OHLC
        ohlc = tick_df['Mid'].resample(period).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        
        # Add volume (sum of ask and bid volumes)
        volume = (tick_df['AskVolume'] + tick_df['BidVolume']).resample(period).sum()
        ohlc['Volume'] = volume
        
        # Remove any rows with NaN
        ohlc = ohlc.dropna()
        
        return ohlc
    
    def download_and_save(self, pair: str, start_year: int = 2015, 
                         save_ticks: bool = False) -> bool:
        """Download and save data for a currency pair"""
        
        if pair not in self.INSTRUMENTS:
            print(f"Instrument {pair} not supported by Dukascopy")
            return False
            
        instrument = self.INSTRUMENTS[pair]
        start_date = datetime(start_year, 1, 1)
        end_date = datetime.now()
        
        print(f"Downloading {pair} tick data from Dukascopy...")
        
        try:
            # Download tick data
            tick_df = self.download_period(instrument, start_date, end_date)
            
            if tick_df.empty:
                print(f"No data downloaded for {pair}")
                return False
                
            print(f"Downloaded {len(tick_df)} ticks")
            
            # Save tick data if requested
            if save_ticks:
                tick_file = os.path.join(self.output_dir, f"{pair}_TICKS.csv")
                tick_df.to_csv(tick_file)
                print(f"Saved tick data to {tick_file}")
            
            # Convert to OHLC
            print("Converting to 1-minute OHLC...")
            ohlc_1m = self.ticks_to_ohlc(tick_df, '1T')
            
            print("Converting to 15-minute OHLC...")
            ohlc_15m = self.ticks_to_ohlc(tick_df, '15T')
            
            # Save OHLC data
            file_1m = os.path.join(self.output_dir, f"{pair}_MASTER.csv")
            file_15m = os.path.join(self.output_dir, f"{pair}_MASTER_15M.csv")
            
            ohlc_1m.to_csv(file_1m)
            ohlc_15m.to_csv(file_15m)
            
            print(f"Saved {len(ohlc_1m)} 1-minute bars to {file_1m}")
            print(f"Saved {len(ohlc_15m)} 15-minute bars to {file_15m}")
            
            return True
            
        except Exception as e:
            print(f"Error downloading {pair}: {e}")
            return False


class YahooFinanceDownloader:
    """Download data from Yahoo Finance with enhanced features"""
    
    def __init__(self, output_dir: str = '.'):
        self.output_dir = output_dir
        
    def download_pair(self, pair: str, start_year: int = 2010,
                     merge_with_existing: bool = True) -> bool:
        """Download FX data from Yahoo Finance"""
        
        if not YAHOO_AVAILABLE:
            raise ImportError("yfinance not installed")
            
        # Convert pair format for Yahoo
        ticker = f"{pair[:3]}{pair[3:]}=X"
        
        print(f"Downloading {pair} from Yahoo Finance...")
        
        try:
            # Try hourly data first
            start_date = f"{start_year}-01-01"
            df_hourly = yf.download(ticker, start=start_date, interval='1h', 
                                   progress=True, auto_adjust=True)
            
            if df_hourly.empty:
                print("Hourly data not available, trying daily...")
                df_daily = yf.download(ticker, start=start_date, interval='1d',
                                     progress=True, auto_adjust=True)
                
                if df_daily.empty:
                    print(f"No data available for {pair}")
                    return False
                    
                # Interpolate daily to hourly
                df_hourly = df_daily.resample('1H').interpolate(method='linear')
                
            # Resample to 15 minutes
            df_15m = df_hourly.resample('15T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            print(f"Downloaded {len(df_15m)} 15-minute bars")
            
            # Handle merge with existing data
            file_15m = os.path.join(self.output_dir, f"{pair}_MASTER_15M.csv")
            
            if merge_with_existing and os.path.exists(file_15m):
                # Backup existing file
                backup_path = f"{file_15m}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(file_15m, backup_path)
                print(f"Backed up existing file to {backup_path}")
                
                # Load existing data
                existing_df = pd.read_csv(backup_path, index_col='Time', parse_dates=True)
                
                # Merge with new data
                merged_df = pd.concat([df_15m, existing_df])
                merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
                merged_df = merged_df.sort_index()
                
                print(f"Merged with existing data: {len(existing_df)} + {len(df_15m)} = {len(merged_df)} rows")
                df_15m = merged_df
            
            # Save data
            df_15m.to_csv(file_15m)
            print(f"Saved to {file_15m}")
            
            # Data quality check
            self._check_data_quality(df_15m, pair)
            
            return True
            
        except Exception as e:
            print(f"Error downloading {pair} from Yahoo: {e}")
            return False
    
    def _check_data_quality(self, df: pd.DataFrame, pair: str):
        """Check data quality and print warnings"""
        # Check for gaps
        expected_freq = pd.Timedelta('15 minutes')
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs > expected_freq * 2]
        
        if len(gaps) > 0:
            print(f"Warning: Found {len(gaps)} time gaps in data")
            if len(gaps) <= 5:
                for gap_time, gap_size in gaps.items():
                    print(f"  Gap at {gap_time}: {gap_size}")
        
        # Check for suspicious price movements
        price_changes = df['Close'].pct_change().abs()
        large_moves = price_changes[price_changes > 0.05]  # 5% moves
        
        if len(large_moves) > 0:
            print(f"Warning: Found {len(large_moves)} price movements > 5%")
            
        # Check data coverage
        date_range = df.index.max() - df.index.min()
        expected_rows = date_range.total_seconds() / (15 * 60)  # 15 minutes in seconds
        coverage = len(df) / expected_rows * 100
        
        print(f"Data coverage: {coverage:.1f}% (accounting for weekends/holidays)")


def main():
    parser = argparse.ArgumentParser(description='Alternative FX data sources')
    parser.add_argument('source', choices=['dukascopy', 'yahoo'],
                        help='Data source to use')
    parser.add_argument('--pair', type=str, required=True,
                        help='Currency pair (e.g., AUDUSD)')
    parser.add_argument('--start-year', type=int, default=2015,
                        help='Start year for download')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for data files')
    parser.add_argument('--save-ticks', action='store_true',
                        help='Save raw tick data (Dukascopy only)')
    parser.add_argument('--no-merge', action='store_true',
                        help='Do not merge with existing data (Yahoo only)')
    
    args = parser.parse_args()
    
    if args.source == 'dukascopy':
        downloader = DukascopyDownloader(args.output_dir)
        success = downloader.download_and_save(args.pair, args.start_year, args.save_ticks)
    else:  # yahoo
        downloader = YahooFinanceDownloader(args.output_dir)
        success = downloader.download_pair(args.pair, args.start_year, 
                                         not args.no_merge)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()