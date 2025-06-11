#!/usr/bin/env python3
"""
FX Data Manager - Consolidated script for downloading and managing forex data
Combines functionality from multiple download scripts into a single, maintainable solution
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import argparse

# Try to import fx_data_downloader
try:
    from fx_data_downloader import download_fx_data
except ImportError:
    print("Warning: fx_data_downloader not found. Some functionality may be limited.")
    download_fx_data = None

# Try to import alternative data sources
try:
    import yfinance as yf
except ImportError:
    yf = None
    print("Warning: yfinance not installed. Yahoo Finance downloads unavailable.")

try:
    import requests
    import lzma
    import struct
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    DUKASCOPY_AVAILABLE = True
except ImportError:
    DUKASCOPY_AVAILABLE = False
    print("Warning: Dependencies for Dukascopy downloads not available.")


class FXDataManager:
    """Unified FX data download and management class"""
    
    CURRENCY_PAIRS = [
        'AUDUSD', 'GBPUSD', 'EURUSD', 'NZDUSD', 'USDCAD', 'USDJPY',
        'GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'AUDNZD', 'EURGBP'
    ]
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        
    def backup_file(self, filepath: str) -> Optional[str]:
        """Create a backup of existing file"""
        if os.path.exists(filepath):
            backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(filepath, backup_path)
            print(f"Created backup: {backup_path}")
            return backup_path
        return None
    
    def verify_data_quality(self, df: pd.DataFrame, pair: str) -> Dict[str, any]:
        """Verify data quality and return statistics"""
        stats = {
            'pair': pair,
            'rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.index.duplicated().sum()
        }
        
        # Check for price consistency
        if 'Close' in df.columns:
            price_changes = df['Close'].pct_change().dropna()
            stats['max_price_change'] = price_changes.abs().max()
            stats['suspicious_spikes'] = (price_changes.abs() > 0.1).sum()  # >10% changes
        
        return stats
    
    def download_pair(self, pair: str, start_year: int = 2015, 
                     source: str = 'fx_downloader', force: bool = False) -> bool:
        """Download data for a single currency pair"""
        
        # File paths
        file_1m = os.path.join(self.data_dir, f"{pair}_MASTER.csv")
        file_15m = os.path.join(self.data_dir, f"{pair}_MASTER_15M.csv")
        
        # Backup existing files if force download
        if force and os.path.exists(file_1m):
            self.backup_file(file_1m)
            os.remove(file_1m)
            if os.path.exists(file_15m):
                os.remove(file_15m)
        
        # Check if files already exist
        if not force and os.path.exists(file_15m):
            print(f"{pair} data already exists. Use --force to re-download.")
            return True
        
        print(f"\nDownloading {pair} from {start_year} using {source}...")
        
        if source == 'fx_downloader' and download_fx_data:
            return self._download_fx_downloader(pair, start_year)
        elif source == 'yahoo' and yf:
            return self._download_yahoo(pair, start_year)
        elif source == 'dukascopy' and DUKASCOPY_AVAILABLE:
            return self._download_dukascopy(pair, start_year)
        else:
            print(f"Source '{source}' not available for {pair}")
            return False
    
    def _download_fx_downloader(self, pair: str, start_year: int) -> bool:
        """Download using fx_data_downloader"""
        try:
            download_fx_data(pair, start_year)
            
            # Create 15M data if not exists
            file_1m = os.path.join(self.data_dir, f"{pair}_MASTER.csv")
            file_15m = os.path.join(self.data_dir, f"{pair}_MASTER_15M.csv")
            
            if os.path.exists(file_1m) and not os.path.exists(file_15m):
                print(f"Creating 15-minute data for {pair}...")
                df = pd.read_csv(file_1m, index_col='Time', parse_dates=True)
                df_15m = df.resample('15T').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                df_15m.to_csv(file_15m)
                
            # Verify data
            if os.path.exists(file_15m):
                df = pd.read_csv(file_15m, index_col='Time', parse_dates=True)
                stats = self.verify_data_quality(df, pair)
                print(f"Downloaded {stats['rows']} rows for {pair}")
                print(f"Date range: {stats['date_range']}")
                return True
                
        except Exception as e:
            print(f"Error downloading {pair}: {e}")
            return False
            
    def _download_yahoo(self, pair: str, start_year: int) -> bool:
        """Download using yfinance (Yahoo Finance)"""
        try:
            # Format pair for Yahoo Finance
            ticker = f"{pair[:3]}{pair[3:]}=X"
            
            # Download hourly data
            start_date = f"{start_year}-01-01"
            df = yf.download(ticker, start=start_date, interval='1h', progress=True)
            
            if df.empty:
                print(f"No hourly data available for {pair}, trying daily...")
                df = yf.download(ticker, start=start_date, interval='1d', progress=True)
                if not df.empty:
                    # Interpolate to hourly
                    df = df.resample('1H').interpolate(method='linear')
            
            if not df.empty:
                # Resample to 15 minutes
                df_15m = df.resample('15T').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                # Save file
                file_15m = os.path.join(self.data_dir, f"{pair}_MASTER_15M.csv")
                df_15m.to_csv(file_15m)
                
                stats = self.verify_data_quality(df_15m, pair)
                print(f"Downloaded {stats['rows']} rows for {pair} from Yahoo Finance")
                return True
                
        except Exception as e:
            print(f"Error downloading {pair} from Yahoo: {e}")
            return False
            
    def _download_dukascopy(self, pair: str, start_year: int) -> bool:
        """Download tick data from Dukascopy"""
        # Implementation would go here - keeping structure for now
        print(f"Dukascopy download for {pair} not yet implemented in consolidated version")
        return False
    
    def download_all(self, pairs: Optional[List[str]] = None, 
                     start_year: int = 2015, source: str = 'fx_downloader',
                     force: bool = False) -> Dict[str, bool]:
        """Download multiple currency pairs"""
        if pairs is None:
            pairs = self.CURRENCY_PAIRS
            
        results = {}
        for pair in pairs:
            success = self.download_pair(pair, start_year, source, force)
            results[pair] = success
            
        # Summary
        successful = sum(1 for v in results.values() if v)
        print(f"\n{'='*50}")
        print(f"Download Summary: {successful}/{len(pairs)} pairs successful")
        if successful < len(pairs):
            failed = [p for p, v in results.items() if not v]
            print(f"Failed pairs: {', '.join(failed)}")
            
        return results
    
    def merge_with_existing(self, pair: str, new_data_file: str) -> bool:
        """Merge new data with existing master file"""
        try:
            file_15m = os.path.join(self.data_dir, f"{pair}_MASTER_15M.csv")
            
            # Backup existing
            if os.path.exists(file_15m):
                self.backup_file(file_15m)
                existing_df = pd.read_csv(file_15m, index_col='Time', parse_dates=True)
            else:
                existing_df = pd.DataFrame()
            
            # Load new data
            new_df = pd.read_csv(new_data_file, index_col='Time', parse_dates=True)
            
            # Merge
            if not existing_df.empty:
                merged_df = pd.concat([new_df, existing_df])
                merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
                merged_df = merged_df.sort_index()
            else:
                merged_df = new_df
            
            # Save
            merged_df.to_csv(file_15m)
            
            stats = self.verify_data_quality(merged_df, pair)
            print(f"Merged data for {pair}: {stats['rows']} total rows")
            print(f"Date range: {stats['date_range']}")
            
            return True
            
        except Exception as e:
            print(f"Error merging data for {pair}: {e}")
            return False
    
    def filter_data_by_year(self, year: int, output_dir: str = 'git_data') -> Dict[str, bool]:
        """Filter data to keep only specific year for git storage"""
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        for pair in self.CURRENCY_PAIRS:
            try:
                file_1m = os.path.join(self.data_dir, f"{pair}_MASTER.csv")
                if os.path.exists(file_1m):
                    df = pd.read_csv(file_1m, index_col='Time', parse_dates=True)
                    df_filtered = df[df.index.year >= year]
                    
                    output_file = os.path.join(output_dir, f"{pair}_MASTER_{year}.csv")
                    df_filtered.to_csv(output_file)
                    
                    print(f"{pair}: Filtered {len(df)} -> {len(df_filtered)} rows")
                    results[pair] = True
                else:
                    results[pair] = False
                    
            except Exception as e:
                print(f"Error filtering {pair}: {e}")
                results[pair] = False
                
        return results
    
    def clean_backups(self, days_to_keep: int = 7) -> int:
        """Clean old backup files"""
        cleaned = 0
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for file in os.listdir(self.data_dir):
            if '.backup_' in file:
                filepath = os.path.join(self.data_dir, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_time < cutoff_date:
                    os.remove(filepath)
                    cleaned += 1
                    print(f"Removed old backup: {file}")
                    
        return cleaned


def main():
    parser = argparse.ArgumentParser(description='FX Data Manager - Download and manage forex data')
    parser.add_argument('command', choices=['download', 'download-all', 'merge', 'filter', 'clean'],
                        help='Command to execute')
    parser.add_argument('--pair', type=str, help='Currency pair (e.g., AUDUSD)')
    parser.add_argument('--pairs', nargs='+', help='Multiple currency pairs')
    parser.add_argument('--start-year', type=int, default=2015, help='Start year for download')
    parser.add_argument('--source', choices=['fx_downloader', 'yahoo', 'dukascopy'], 
                        default='fx_downloader', help='Data source')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    parser.add_argument('--data-dir', default='.', help='Data directory')
    parser.add_argument('--year', type=int, help='Year to filter (for filter command)')
    parser.add_argument('--days', type=int, default=7, help='Days to keep backups (for clean command)')
    parser.add_argument('--merge-file', type=str, help='File to merge with existing data')
    
    args = parser.parse_args()
    
    manager = FXDataManager(data_dir=args.data_dir)
    
    if args.command == 'download':
        if not args.pair:
            print("Error: --pair required for download command")
            sys.exit(1)
        success = manager.download_pair(args.pair, args.start_year, args.source, args.force)
        sys.exit(0 if success else 1)
        
    elif args.command == 'download-all':
        results = manager.download_all(args.pairs, args.start_year, args.source, args.force)
        sys.exit(0 if all(results.values()) else 1)
        
    elif args.command == 'merge':
        if not args.pair or not args.merge_file:
            print("Error: --pair and --merge-file required for merge command")
            sys.exit(1)
        success = manager.merge_with_existing(args.pair, args.merge_file)
        sys.exit(0 if success else 1)
        
    elif args.command == 'filter':
        if not args.year:
            print("Error: --year required for filter command")
            sys.exit(1)
        results = manager.filter_data_by_year(args.year)
        sys.exit(0 if all(results.values()) else 1)
        
    elif args.command == 'clean':
        cleaned = manager.clean_backups(args.days)
        print(f"Cleaned {cleaned} old backup files")
        sys.exit(0)


if __name__ == "__main__":
    main()