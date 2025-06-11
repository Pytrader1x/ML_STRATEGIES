"""
Convert Kraken OHLCVT data to standard format and resample to 15-minute intervals
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def convert_kraken_to_standard_format(input_file, output_file):
    """
    Convert Kraken CSV format to standard OHLC format with 15-minute intervals
    
    Kraken format: timestamp,open,high,low,close,volume,trades
    Target format: DateTime,Open,High,Low,Close
    """
    
    print(f"Converting {input_file} to standard format...")
    
    # Read the CSV file
    # Kraken data has no header, columns are: timestamp,open,high,low,close,volume,trades
    df = pd.read_csv(input_file, header=None, 
                     names=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades'])
    
    # Convert Unix timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Set DateTime as index
    df.set_index('DateTime', inplace=True)
    
    # Create OHLC data
    ohlc_df = pd.DataFrame({
        'Open': df['open'],
        'High': df['high'],
        'Low': df['low'],
        'Close': df['close'],
        'Volume': df['volume']
    })
    
    print(f"Original data: {len(ohlc_df)} rows")
    print(f"Date range: {ohlc_df.index[0]} to {ohlc_df.index[-1]}")
    
    # Resample to 15-minute intervals
    # For OHLC data, we use specific aggregation rules
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Resample to 15-minute intervals
    resampled_df = ohlc_df.resample('15min').agg(ohlc_dict)
    
    # Remove rows where all OHLC values are NaN
    resampled_df = resampled_df.dropna(how='all')
    
    # Forward fill any remaining NaN values
    resampled_df = resampled_df.ffill()
    
    # Reset index to make DateTime a column
    resampled_df.reset_index(inplace=True)
    
    # Drop Volume column as it's not in the target format
    resampled_df = resampled_df[['DateTime', 'Open', 'High', 'Low', 'Close']]
    
    # Format DateTime to match target format
    resampled_df['DateTime'] = resampled_df['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    resampled_df.to_csv(output_file, index=False)
    
    print(f"Converted data: {len(resampled_df)} rows")
    print(f"Saved to: {output_file}")
    
    # Display sample
    print("\nSample of converted data:")
    print(resampled_df.head())
    
    return resampled_df


def process_all_crypto_files():
    """Process all crypto files in the directory"""
    
    import os
    
    # Get all CSV files with _1.csv pattern (1-minute data)
    crypto_files = [f for f in os.listdir('.') if f.endswith('_1.csv')]
    
    for file in crypto_files:
        # Extract currency pair name
        pair = file.replace('_1.csv', '')
        
        # Convert to standard format
        output_file = f"{pair}_MASTER_15M.csv"
        
        try:
            convert_kraken_to_standard_format(file, output_file)
            print(f"\n✅ Successfully converted {file}")
        except Exception as e:
            print(f"\n❌ Error converting {file}: {e}")
    
    print("\n✅ All conversions complete!")


def main():
    """Main function"""
    
    print("="*60)
    print("KRAKEN DATA CONVERTER")
    print("Converting to standard 15-minute OHLC format")
    print("="*60)
    
    # Process all crypto files
    process_all_crypto_files()
    
    # Additional check for specific ETHUSD file
    if 'ETHUSD_MASTER_15M.csv' in os.listdir('.'):
        print("\n" + "="*60)
        print("ETHUSD DATA READY FOR STRATEGY TESTING")
        print("="*60)
        
        # Read and display statistics
        eth_df = pd.read_csv('ETHUSD_MASTER_15M.csv')
        print(f"\nETHUSD Statistics:")
        print(f"Total rows: {len(eth_df):,}")
        print(f"Date range: {eth_df['DateTime'].iloc[0]} to {eth_df['DateTime'].iloc[-1]}")
        print(f"Price range: ${eth_df['Low'].min():.2f} - ${eth_df['High'].max():.2f}")


if __name__ == "__main__":
    import os
    main()