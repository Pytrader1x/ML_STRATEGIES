"""
Data Quality Control and Integrity Checks
Ensures FX data meets quality standards before backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import pyarrow.parquet as pq
import hashlib
from datetime import datetime, timezone
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Comprehensive data quality control for FX time series"""
    
    def __init__(self, 
                 max_missing_pct: float = 0.01,  # Max 1% missing bars
                 spike_threshold: float = 8.0,    # 8 sigma for spike detection
                 max_spread_pips: float = 50.0,   # Max reasonable spread
                 cache_dir: Path = Path('.cache')):
        self.max_missing_pct = max_missing_pct
        self.spike_threshold = spike_threshold
        self.max_spread_pips = max_spread_pips
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # FX-specific parameters
        self.pip_sizes = {
            'JPY': 0.01,  # JPY pairs
            'default': 0.0001  # All other pairs
        }
        
    def check_data_integrity(self, 
                           df: pd.DataFrame, 
                           pair: str,
                           timeframe: str = '15T') -> Dict:
        """Run comprehensive data quality checks"""
        
        logger.info(f"Running QC for {pair} ({len(df)} bars)")
        
        results = {
            'pair': pair,
            'total_bars': len(df),
            'date_range': f"{df.index[0]} to {df.index[-1]}",
            'checks': {},
            'passed': True,
            'data_hash': self._calculate_hash(df)
        }
        
        # 1. Check missing bars
        missing_check = self._check_missing_bars(df, timeframe)
        results['checks']['missing_bars'] = missing_check
        if missing_check['missing_pct'] > self.max_missing_pct:
            results['passed'] = False
            
        # 2. Check duplicate timestamps
        dup_check = self._check_duplicates(df)
        results['checks']['duplicates'] = dup_check
        if dup_check['duplicate_count'] > 0:
            results['passed'] = False
            
        # 3. Check for extreme spikes
        spike_check = self._check_spikes(df, pair)
        results['checks']['spikes'] = spike_check
        if spike_check['spike_count'] > 0:
            results['passed'] = False
            
        # 4. Check spread sanity
        spread_check = self._check_spreads(df, pair)
        results['checks']['spreads'] = spread_check
        if spread_check['max_spread_pips'] > self.max_spread_pips:
            results['passed'] = False
            
        # 5. Check data types and schema
        schema_check = self._check_schema(df)
        results['checks']['schema'] = schema_check
        if not schema_check['valid']:
            results['passed'] = False
            
        # 6. Check timezone (should be UTC)
        tz_check = self._check_timezone(df)
        results['checks']['timezone'] = tz_check
        if not tz_check['is_utc']:
            results['passed'] = False
            
        return results
    
    def _check_missing_bars(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Check for missing time bars"""
        
        # Create expected date range
        expected_range = pd.date_range(
            start=df.index[0], 
            end=df.index[-1], 
            freq=timeframe
        )
        
        # Find missing timestamps
        missing = expected_range.difference(df.index)
        
        # Filter out weekends (FX market closed)
        missing = missing[~missing.weekday.isin([5, 6])]  # Remove Sat, Sun
        
        missing_pct = len(missing) / len(expected_range) * 100
        
        return {
            'missing_count': len(missing),
            'missing_pct': missing_pct,
            'first_missing': str(missing[0]) if len(missing) > 0 else None,
            'last_missing': str(missing[-1]) if len(missing) > 0 else None
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate timestamps"""
        
        duplicates = df.index.duplicated()
        dup_count = duplicates.sum()
        
        return {
            'duplicate_count': int(dup_count),
            'duplicate_timestamps': df.index[duplicates].tolist() if dup_count > 0 else []
        }
    
    def _check_spikes(self, df: pd.DataFrame, pair: str) -> Dict:
        """Check for extreme price spikes"""
        
        pip_size = self._get_pip_size(pair)
        
        # Calculate returns
        returns = df['Close'].pct_change()
        
        # Rolling statistics
        rolling_mean = returns.rolling(100, min_periods=50).mean()
        rolling_std = returns.rolling(100, min_periods=50).std()
        
        # Z-scores
        z_scores = (returns - rolling_mean) / rolling_std
        
        # Find spikes
        spikes = abs(z_scores) > self.spike_threshold
        spike_indices = df.index[spikes & ~z_scores.isna()]
        
        # Calculate spike sizes in pips
        spike_details = []
        for idx in spike_indices[:10]:  # First 10 spikes
            loc = df.index.get_loc(idx)
            if loc > 0:
                prev_close = df['Close'].iloc[loc-1]
                curr_close = df['Close'].iloc[loc]
                spike_pips = abs(curr_close - prev_close) / pip_size
                spike_details.append({
                    'timestamp': str(idx),
                    'spike_pips': round(spike_pips, 1),
                    'z_score': round(z_scores.iloc[loc], 2)
                })
        
        return {
            'spike_count': int(spikes.sum()),
            'spike_details': spike_details,
            'max_z_score': round(abs(z_scores).max(), 2) if not z_scores.isna().all() else 0
        }
    
    def _check_spreads(self, df: pd.DataFrame, pair: str) -> Dict:
        """Check bid-ask spreads are reasonable"""
        
        pip_size = self._get_pip_size(pair)
        
        if 'Bid' in df.columns and 'Ask' in df.columns:
            spreads = df['Ask'] - df['Bid']
            spread_pips = spreads / pip_size
            
            return {
                'avg_spread_pips': round(spread_pips.mean(), 2),
                'max_spread_pips': round(spread_pips.max(), 2),
                'negative_spreads': int((spreads < 0).sum())
            }
        else:
            # If no bid/ask, estimate from high/low
            estimated_spread = (df['High'] - df['Low']) * 0.25  # Rough estimate
            spread_pips = estimated_spread / pip_size
            
            return {
                'avg_spread_pips': round(spread_pips.mean(), 2),
                'max_spread_pips': round(spread_pips.max(), 2),
                'negative_spreads': 0,
                'note': 'Estimated from High-Low'
            }
    
    def _check_schema(self, df: pd.DataFrame) -> Dict:
        """Check data types and required columns"""
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Check data types
        expected_types = {
            'Open': np.float64,
            'High': np.float64,
            'Low': np.float64,
            'Close': np.float64,
            'Volume': (np.int64, np.float64)
        }
        
        type_issues = []
        for col, expected in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if isinstance(expected, tuple):
                    if actual_type not in expected:
                        type_issues.append(f"{col}: expected {expected}, got {actual_type}")
                else:
                    if actual_type != expected:
                        type_issues.append(f"{col}: expected {expected}, got {actual_type}")
        
        # Check OHLC consistency
        ohlc_issues = []
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_high = (df['High'] < df['Low']).sum()
            if invalid_high > 0:
                ohlc_issues.append(f"High < Low in {invalid_high} bars")
                
            invalid_range = ((df['Close'] > df['High']) | (df['Close'] < df['Low'])).sum()
            if invalid_range > 0:
                ohlc_issues.append(f"Close outside High-Low in {invalid_range} bars")
        
        return {
            'valid': len(missing_columns) == 0 and len(type_issues) == 0 and len(ohlc_issues) == 0,
            'missing_columns': missing_columns,
            'type_issues': type_issues,
            'ohlc_issues': ohlc_issues
        }
    
    def _check_timezone(self, df: pd.DataFrame) -> Dict:
        """Check if data is in UTC"""
        
        if hasattr(df.index, 'tz'):
            is_utc = df.index.tz is not None and df.index.tz.zone == 'UTC'
            current_tz = df.index.tz.zone if df.index.tz else 'None'
        else:
            is_utc = False
            current_tz = 'No timezone info'
            
        return {
            'is_utc': is_utc,
            'current_timezone': current_tz
        }
    
    def _get_pip_size(self, pair: str) -> float:
        """Get pip size for currency pair"""
        if 'JPY' in pair:
            return self.pip_sizes['JPY']
        return self.pip_sizes['default']
    
    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataframe for caching"""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def clean_and_cache(self, 
                       df: pd.DataFrame, 
                       pair: str,
                       qc_results: Dict) -> Optional[Path]:
        """Clean data and cache as Parquet if it passes QC"""
        
        if not qc_results['passed']:
            logger.warning(f"Data for {pair} failed QC, not caching")
            return None
            
        # Convert to UTC if needed
        if not qc_results['checks']['timezone']['is_utc']:
            logger.info(f"Converting {pair} to UTC")
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
        
        # Enforce schema
        df = self._enforce_schema(df)
        
        # Cache as Parquet
        cache_file = self.cache_dir / f"{pair}_{qc_results['data_hash'][:8]}.parquet"
        df.to_parquet(cache_file, engine='pyarrow', compression='snappy')
        
        # Save QC results
        qc_file = self.cache_dir / f"{pair}_{qc_results['data_hash'][:8]}_qc.json"
        with open(qc_file, 'w') as f:
            json.dump(qc_results, f, indent=2)
            
        logger.info(f"Cached clean data for {pair} to {cache_file}")
        return cache_file
    
    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce consistent data types"""
        
        # Ensure float types for OHLC
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
                
        # Ensure volume is numeric
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            
        return df


def run_qc_on_all_pairs(data_dir: Path = Path('../data'),
                       output_file: Path = Path('data_qc_report.json')) -> Dict:
    """Run QC on all currency pair files"""
    
    qc = DataQualityChecker()
    results = {}
    
    # Find all 15M CSV files
    csv_files = list(data_dir.glob('*_MASTER_15M.csv'))
    logger.info(f"Found {len(csv_files)} files to check")
    
    for csv_file in csv_files:
        pair = csv_file.stem.replace('_MASTER_15M', '')
        logger.info(f"\nProcessing {pair}...")
        
        try:
            # Load data
            df = pd.read_csv(csv_file, parse_dates=['DateTime'], index_col='DateTime')
            
            # Run QC
            qc_results = qc.check_data_integrity(df, pair)
            results[pair] = qc_results
            
            # Cache if passed
            if qc_results['passed']:
                qc.clean_and_cache(df, pair, qc_results)
                
        except Exception as e:
            logger.error(f"Error processing {pair}: {str(e)}")
            results[pair] = {
                'error': str(e),
                'passed': False
            }
    
    # Summary statistics
    summary = {
        'total_pairs': len(results),
        'passed': sum(1 for r in results.values() if r.get('passed', False)),
        'failed': sum(1 for r in results.values() if not r.get('passed', False)),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'results': results
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Print summary
    print(f"\nData QC Summary:")
    print(f"Total pairs checked: {summary['total_pairs']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    
    if summary['failed'] > 0:
        print("\nFailed pairs:")
        for pair, result in results.items():
            if not result.get('passed', False):
                print(f"  - {pair}: {result.get('error', 'QC checks failed')}")
                
    return summary


if __name__ == "__main__":
    # Run QC on all pairs
    summary = run_qc_on_all_pairs()
    
    # Exit with error code if any failures
    if summary['failed'] > 0:
        exit(1)