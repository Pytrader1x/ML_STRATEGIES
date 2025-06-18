"""
Performance Benchmarking
Ensures backtesting meets performance requirements
"""

import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path
import json
from typing import Dict, List
import logging
from datetime import datetime
import multiprocessing as mp

import sys
sys.path.append('..')

from quantlab import momentum, Backtest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark backtesting performance"""
    
    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.target_time_5y_1min = 120  # Target: 5 years of 1-min data in 120 seconds
        
    def generate_synthetic_data(self, 
                              n_bars: int,
                              freq: str = '15T') -> pd.DataFrame:
        """Generate synthetic OHLCV data for benchmarking"""
        
        # Create timestamps
        end_date = pd.Timestamp.now()
        if freq == '1T':
            # 1-minute bars
            dates = pd.date_range(end=end_date, periods=n_bars, freq='1T')
        else:
            # 15-minute bars
            dates = pd.date_range(end=end_date, periods=n_bars, freq='15T')
            
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0, 0.0001, n_bars)  # 1 bp vol
        close_prices = 1.3000 * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close
        noise = np.random.uniform(-0.0001, 0.0001, n_bars)
        open_prices = close_prices * (1 + noise)
        
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(noise))
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(noise))
        
        volume = np.random.uniform(1000, 5000, n_bars)
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        }, index=dates)
    
    def benchmark_signal_generation(self) -> Dict:
        """Benchmark signal generation performance"""
        
        logger.info("Benchmarking signal generation...")
        
        # Test different data sizes
        test_sizes = [
            (10000, '10K bars (1 month)'),
            (50000, '50K bars (6 months)'),
            (100000, '100K bars (1 year)'),
            (500000, '500K bars (5 years)')
        ]
        
        results = []
        
        for n_bars, desc in test_sizes:
            # Generate data
            data = self.generate_synthetic_data(n_bars)
            
            # Time signal generation
            start_time = time.time()
            start_cpu = time.process_time()
            
            signals = momentum(
                data['Close'],
                lookback=40,
                entry_z=1.5,
                exit_z=0.5
            )
            
            end_time = time.time()
            end_cpu = time.process_time()
            
            # Calculate metrics
            wall_time = end_time - start_time
            cpu_time = end_cpu - start_cpu
            bars_per_second = n_bars / wall_time
            
            results.append({
                'n_bars': n_bars,
                'description': desc,
                'wall_time': wall_time,
                'cpu_time': cpu_time,
                'bars_per_second': bars_per_second,
                'efficiency': cpu_time / wall_time  # CPU utilization
            })
            
            logger.info(f"{desc}: {wall_time:.3f}s ({bars_per_second:.0f} bars/sec)")
            
        return results
    
    def benchmark_backtest(self) -> Dict:
        """Benchmark full backtest performance"""
        
        logger.info("Benchmarking backtest engine...")
        
        # Test sizes
        test_configs = [
            (50000, '15T', '50K 15-min bars'),
            (100000, '15T', '100K 15-min bars'),
            (60*24*252*5, '1T', '5 years 1-min bars')  # 1.89M bars
        ]
        
        results = []
        
        for n_bars, freq, desc in test_configs:
            # Generate data
            logger.info(f"Testing {desc}...")
            data = self.generate_synthetic_data(n_bars, freq)
            
            # Generate signals
            signals = momentum(data['Close'])
            
            # Time backtest
            start_time = time.time()
            start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            
            backtest = Backtest(data)
            result = backtest.run(signals['signal'], 'EURUSD')
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Metrics
            wall_time = end_time - start_time
            memory_used = end_memory - start_memory
            bars_per_second = n_bars / wall_time
            trades_processed = len(result.trades)
            
            results.append({
                'test': desc,
                'n_bars': n_bars,
                'frequency': freq,
                'wall_time': wall_time,
                'memory_mb': memory_used,
                'bars_per_second': bars_per_second,
                'trades': trades_processed,
                'sharpe': result.metrics['sharpe_ratio']
            })
            
            logger.info(f"  Time: {wall_time:.2f}s, Memory: {memory_used:.1f}MB, "
                       f"Speed: {bars_per_second:.0f} bars/sec")
            
            # Check if 5-year 1-min target is met
            if freq == '1T' and n_bars > 1000000:
                if wall_time <= self.target_time_5y_1min:
                    logger.info(f"  ✓ PASSED: Target met ({wall_time:.1f}s < {self.target_time_5y_1min}s)")
                else:
                    logger.warning(f"  ✗ FAILED: Target not met ({wall_time:.1f}s > {self.target_time_5y_1min}s)")
                    
        return results
    
    def benchmark_parallel_processing(self) -> Dict:
        """Benchmark parallel processing capabilities"""
        
        logger.info(f"Benchmarking parallel processing ({self.cpu_count} cores)...")
        
        # Generate shared data
        data = self.generate_synthetic_data(100000)
        
        # Test different currency pairs in parallel
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
                 'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY']
        
        # Sequential processing
        start_seq = time.time()
        seq_results = []
        
        for pair in pairs:
            signals = momentum(data['Close'])
            backtest = Backtest(data)
            result = backtest.run(signals['signal'], pair)
            seq_results.append(result.metrics['sharpe_ratio'])
            
        seq_time = time.time() - start_seq
        
        # Parallel processing
        start_par = time.time()
        
        with mp.Pool(processes=min(self.cpu_count, len(pairs))) as pool:
            par_results = pool.starmap(
                self._process_pair,
                [(data, pair) for pair in pairs]
            )
            
        par_time = time.time() - start_par
        
        speedup = seq_time / par_time
        efficiency = speedup / min(self.cpu_count, len(pairs))
        
        results = {
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'cpu_count': self.cpu_count,
            'pairs_tested': len(pairs)
        }
        
        logger.info(f"Sequential: {seq_time:.2f}s")
        logger.info(f"Parallel: {par_time:.2f}s")
        logger.info(f"Speedup: {speedup:.2f}x (Efficiency: {efficiency:.1%})")
        
        return results
    
    def _process_pair(self, data: pd.DataFrame, pair: str) -> float:
        """Process single pair for parallel benchmark"""
        signals = momentum(data['Close'])
        backtest = Backtest(data)
        result = backtest.run(signals['signal'], pair)
        return result.metrics['sharpe_ratio']
    
    def benchmark_memory_usage(self) -> Dict:
        """Profile memory usage patterns"""
        
        logger.info("Benchmarking memory usage...")
        
        process = psutil.Process(os.getpid())
        results = []
        
        test_sizes = [10000, 50000, 100000, 500000]
        
        for n_bars in test_sizes:
            # Force garbage collection
            import gc
            gc.collect()
            
            start_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate data
            data = self.generate_synthetic_data(n_bars)
            data_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate signals
            signals = momentum(data['Close'])
            signals_memory = process.memory_info().rss / 1024 / 1024
            
            # Run backtest
            backtest = Backtest(data)
            result = backtest.run(signals['signal'], 'EURUSD')
            backtest_memory = process.memory_info().rss / 1024 / 1024
            
            results.append({
                'n_bars': n_bars,
                'data_mb': data_memory - start_memory,
                'signals_mb': signals_memory - data_memory,
                'backtest_mb': backtest_memory - signals_memory,
                'total_mb': backtest_memory - start_memory,
                'mb_per_million_bars': (backtest_memory - start_memory) / (n_bars / 1e6)
            })
            
            # Clean up
            del data, signals, backtest, result
            gc.collect()
            
        return results
    
    def create_benchmark_report(self, output_dir: Path):
        """Create comprehensive benchmark report"""
        
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Running comprehensive performance benchmarks...")
        
        # Run all benchmarks
        signal_bench = self.benchmark_signal_generation()
        backtest_bench = self.benchmark_backtest()
        parallel_bench = self.benchmark_parallel_processing()
        memory_bench = self.benchmark_memory_usage()
        
        # Check if performance targets are met
        targets_met = {
            'signal_generation': all(r['bars_per_second'] > 1000000 for r in signal_bench),
            'backtest_5y_1min': any(r['frequency'] == '1T' and r['wall_time'] <= 120 
                                   for r in backtest_bench),
            'parallel_efficiency': parallel_bench['efficiency'] > 0.7,
            'memory_efficiency': all(r['mb_per_million_bars'] < 500 for r in memory_bench)
        }
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_count': self.cpu_count,
                'total_memory_gb': psutil.virtual_memory().total / 1024**3,
                'python_version': sys.version
            },
            'targets_met': targets_met,
            'all_targets_met': all(targets_met.values()),
            'benchmarks': {
                'signal_generation': signal_bench,
                'backtest': backtest_bench,
                'parallel': parallel_bench,
                'memory': memory_bench
            }
        }
        
        # Save results
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Create summary plot
        self._create_benchmark_plots(summary, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        print(f"CPU Cores: {self.cpu_count}")
        print(f"Total Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print("\nTargets Met:")
        for target, met in targets_met.items():
            status = "✓ PASS" if met else "✗ FAIL"
            print(f"  {target}: {status}")
        print(f"\nOverall: {'✓ ALL TARGETS MET' if summary['all_targets_met'] else '✗ SOME TARGETS MISSED'}")
        
        return summary
    
    def _create_benchmark_plots(self, summary: Dict, output_dir: Path):
        """Create benchmark visualization"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Signal generation scaling
        ax1 = axes[0, 0]
        signal_data = pd.DataFrame(summary['benchmarks']['signal_generation'])
        ax1.plot(signal_data['n_bars'], signal_data['bars_per_second'], 'b-o')
        ax1.set_xlabel('Number of Bars')
        ax1.set_ylabel('Bars per Second')
        ax1.set_title('Signal Generation Performance')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Backtest performance
        ax2 = axes[0, 1]
        backtest_data = pd.DataFrame(summary['benchmarks']['backtest'])
        x = range(len(backtest_data))
        ax2.bar(x, backtest_data['wall_time'])
        ax2.set_xticks(x)
        ax2.set_xticklabels(backtest_data['test'], rotation=45, ha='right')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Backtest Performance')
        ax2.axhline(y=120, color='red', linestyle='--', alpha=0.5, label='Target: 120s')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Memory usage
        ax3 = axes[1, 0]
        memory_data = pd.DataFrame(summary['benchmarks']['memory'])
        ax3.plot(memory_data['n_bars'], memory_data['total_mb'], 'g-o', label='Total')
        ax3.plot(memory_data['n_bars'], memory_data['data_mb'], 'b--', label='Data')
        ax3.plot(memory_data['n_bars'], memory_data['backtest_mb'], 'r--', label='Backtest')
        ax3.set_xlabel('Number of Bars')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Parallel efficiency
        ax4 = axes[1, 1]
        parallel = summary['benchmarks']['parallel']
        labels = ['Sequential', 'Parallel']
        times = [parallel['sequential_time'], parallel['parallel_time']]
        colors = ['red', 'green']
        bars = ax4.bar(labels, times, color=colors, alpha=0.7)
        
        # Add speedup text
        ax4.text(0.5, max(times) * 0.8, 
                f"Speedup: {parallel['speedup']:.1f}x\nEfficiency: {parallel['efficiency']:.1%}",
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                                   facecolor="yellow", alpha=0.5))
        
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title(f'Parallel Processing ({parallel["cpu_count"]} cores)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    summary = benchmark.create_benchmark_report(Path('benchmark_output'))
    
    # Exit with error code if targets not met
    if not summary['all_targets_met']:
        exit(1)