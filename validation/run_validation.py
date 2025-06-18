"""
Main Validation Runner
Orchestrates the complete validation pipeline
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import logging
from datetime import datetime
import argparse
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import validation modules
from data_integrity.data_qc import DataQualityChecker, run_qc_on_all_pairs
from single_pair_validation.walk_forward import WalkForwardValidator, create_tearsheet
from robustness.sensitivity_analysis import RobustnessAnalyzer
from capacity.impact_analysis import CapacityAnalyzer
from benchmarks.performance_benchmark import PerformanceBenchmark
from reporting.report_generator import ReportGenerator

# Import strategy modules
sys.path.append(str(Path(__file__).parent.parent / 'RACS_Strategy'))
from ultimate_optimizer import AdvancedBacktest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationPipeline:
    """Complete validation pipeline for trading strategies"""
    
    def __init__(self, 
                 data_dir: Path = Path('../data'),
                 output_dir: Path = Path('validation_output'),
                 strategy_params: Dict = None):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Default strategy parameters (winning momentum strategy)
        self.strategy_params = strategy_params or {
            'lookback': 40,
            'entry_z': 1.5,
            'exit_z': 0.5
        }
        
        self.results = {}
        
    def run_complete_validation(self, 
                              currency_pairs: List[str] = None,
                              skip_qc: bool = False,
                              skip_benchmarks: bool = False) -> Dict:
        """
        Run complete validation pipeline
        
        Parameters:
        -----------
        currency_pairs : list
            Currency pairs to test (None = all available)
        skip_qc : bool
            Skip data quality checks
        skip_benchmarks : bool
            Skip performance benchmarks
            
        Returns:
        --------
        Dict with all validation results
        """
        
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE STRATEGY VALIDATION")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Data Quality Control
        if not skip_qc:
            logger.info("\n📊 Step 1: Data Quality Control")
            qc_results = self._run_data_qc()
            self.results['data_qc'] = qc_results
        else:
            logger.info("\n⏭️  Skipping data QC")
            
        # Step 2: Multi-Currency Backtest
        logger.info("\n💹 Step 2: Multi-Currency Backtesting")
        if currency_pairs is None:
            currency_pairs = self._get_available_pairs()
        backtest_results = self._run_multi_currency_backtest(currency_pairs)
        self.results['multi_currency'] = backtest_results
        
        # Step 3: Walk-Forward Validation (on best pair)
        logger.info("\n🔄 Step 3: Walk-Forward Validation")
        best_pair = self._get_best_pair(backtest_results)
        wf_results = self._run_walk_forward(best_pair)
        self.results['walk_forward'] = wf_results
        
        # Step 4: Robustness Testing
        logger.info("\n🛡️  Step 4: Robustness Analysis")
        robustness_results = self._run_robustness_tests(best_pair)
        self.results['robustness'] = robustness_results
        
        # Step 5: Capacity Analysis
        logger.info("\n📈 Step 5: Capacity & Impact Analysis")
        capacity_results = self._run_capacity_analysis(best_pair)
        self.results['capacity'] = capacity_results
        
        # Step 6: Performance Benchmarks
        if not skip_benchmarks:
            logger.info("\n⚡ Step 6: Performance Benchmarking")
            benchmark_results = self._run_benchmarks()
            self.results['benchmarks'] = benchmark_results
        else:
            logger.info("\n⏭️  Skipping performance benchmarks")
            
        # Step 7: Generate Reports
        logger.info("\n📄 Step 7: Generating Reports")
        report_path = self._generate_reports()
        self.results['report_path'] = report_path
        
        # Save all results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _run_data_qc(self) -> Dict:
        """Run data quality checks"""
        try:
            qc_report_path = self.output_dir / 'data_qc_report.json'
            qc_summary = run_qc_on_all_pairs(self.data_dir, qc_report_path)
            
            logger.info(f"✅ Data QC complete: {qc_summary['passed']}/{qc_summary['total_pairs']} passed")
            
            if qc_summary['failed'] > 0:
                logger.warning(f"⚠️  {qc_summary['failed']} pairs failed QC checks")
                
            return qc_summary
        except Exception as e:
            logger.error(f"❌ Data QC failed: {e}")
            return {'error': str(e)}
    
    def _run_multi_currency_backtest(self, currency_pairs: List[str]) -> List[Dict]:
        """Run backtest on multiple currency pairs"""
        
        results = []
        
        for pair in currency_pairs:
            logger.info(f"  Testing {pair}...")
            
            try:
                # Load data
                data_path = self.data_dir / f"{pair}_MASTER_15M.csv"
                if not data_path.exists():
                    logger.warning(f"  ⚠️  Data not found for {pair}")
                    continue
                    
                data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
                
                # Run backtest on full data
                backtester = AdvancedBacktest(data)
                result = backtester.strategy_momentum(**self.strategy_params)
                
                results.append({
                    'pair': pair,
                    'sharpe': result['sharpe'],
                    'returns': result['returns'],
                    'win_rate': result['win_rate'],
                    'max_dd': result['max_dd'],
                    'trades': result['trades'],
                    'years': len(data) / (252 * 96)  # Approximate
                })
                
                logger.info(f"    Sharpe: {result['sharpe']:.3f}, Returns: {result['returns']:.1f}%")
                
            except Exception as e:
                logger.error(f"  ❌ Error testing {pair}: {e}")
                
        return results
    
    def _run_walk_forward(self, pair: str) -> Dict:
        """Run walk-forward validation"""
        
        try:
            data_path = self.data_dir / f"{pair}_MASTER_15M.csv"
            data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
            
            # Use last 7 years for walk-forward
            data = data[-7*252*96:]
            
            validator = WalkForwardValidator(data, pair)
            results = validator.run_walk_forward(optimize=True, n_trials=50)
            
            # Create tearsheet
            tearsheet_dir = self.output_dir / 'walk_forward'
            tearsheet_dir.mkdir(exist_ok=True)
            create_tearsheet(results, tearsheet_dir / f'{pair}_walk_forward.png')
            
            logger.info(f"✅ Walk-forward complete: Mean Sharpe = {results['aggregate_metrics']['mean_sharpe']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Walk-forward failed: {e}")
            return {'error': str(e)}
    
    def _run_robustness_tests(self, pair: str) -> Dict:
        """Run robustness analysis"""
        
        try:
            data_path = self.data_dir / f"{pair}_MASTER_15M.csv"
            data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
            
            # Use recent data for speed
            data = data[-100000:]
            
            analyzer = RobustnessAnalyzer(data, pair)
            robustness_dir = self.output_dir / 'robustness'
            results = analyzer.create_robustness_report(robustness_dir)
            
            logger.info(f"✅ Robustness analysis complete")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Robustness analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_capacity_analysis(self, pair: str) -> Dict:
        """Run capacity and impact analysis"""
        
        try:
            data_path = self.data_dir / f"{pair}_MASTER_15M.csv"
            data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
            
            # Use recent data
            data = data[-100000:]
            
            analyzer = CapacityAnalyzer(data, pair)
            capacity_dir = self.output_dir / 'capacity'
            results = analyzer.create_capacity_report(capacity_dir)
            
            logger.info(f"✅ Capacity analysis complete: Max size = ${results['max_viable_size_mm']:.1f}M")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Capacity analysis failed: {e}")
            return {'error': str(e)}
    
    def _run_benchmarks(self) -> Dict:
        """Run performance benchmarks"""
        
        try:
            benchmark = PerformanceBenchmark()
            benchmark_dir = self.output_dir / 'benchmarks'
            results = benchmark.create_benchmark_report(benchmark_dir)
            
            if results['all_targets_met']:
                logger.info("✅ All performance targets met")
            else:
                logger.warning("⚠️  Some performance targets not met")
                
            return results
            
        except Exception as e:
            logger.error(f"❌ Benchmarking failed: {e}")
            return {'error': str(e)}
    
    def _generate_reports(self) -> Path:
        """Generate final reports"""
        
        try:
            generator = ReportGenerator(self.output_dir)
            
            # Prepare results for report
            strategy_results = {
                'multi_currency': self.results.get('multi_currency', [])
            }
            
            validation_results = {
                'walk_forward': self.results.get('walk_forward', {}),
                'robustness': self.results.get('robustness', {}),
                'capacity': self.results.get('capacity', {})
            }
            
            currency_pairs = [r['pair'] for r in self.results.get('multi_currency', [])]
            
            report_path = generator.generate_comprehensive_report(
                strategy_results,
                validation_results,
                currency_pairs
            )
            
            logger.info(f"✅ Reports generated: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return None
    
    def _get_available_pairs(self) -> List[str]:
        """Get list of available currency pairs"""
        
        pairs = []
        for csv_file in self.data_dir.glob('*_MASTER_15M.csv'):
            pair = csv_file.stem.replace('_MASTER_15M', '')
            pairs.append(pair)
            
        return sorted(pairs)
    
    def _get_best_pair(self, results: List[Dict]) -> str:
        """Get best performing currency pair"""
        
        if not results:
            return 'AUDUSD'  # Default
            
        best = max(results, key=lambda x: x.get('sharpe', 0))
        return best['pair']
    
    def _save_results(self):
        """Save all results to JSON"""
        
        results_path = self.output_dir / 'validation_results.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        logger.info(f"📁 Results saved to {results_path}")
    
    def _print_summary(self):
        """Print validation summary"""
        
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        # Multi-currency results
        mc_results = self.results.get('multi_currency', [])
        if mc_results:
            df = pd.DataFrame(mc_results)
            print(f"\nMulti-Currency Results ({len(mc_results)} pairs):")
            print(f"Average Sharpe: {df['sharpe'].mean():.3f}")
            print(f"Best: {df.loc[df['sharpe'].idxmax(), 'pair']} (Sharpe: {df['sharpe'].max():.3f})")
            print(f"Worst: {df.loc[df['sharpe'].idxmin(), 'pair']} (Sharpe: {df['sharpe'].min():.3f})")
            
        # Walk-forward
        wf = self.results.get('walk_forward', {})
        if wf and 'aggregate_metrics' in wf:
            print(f"\nWalk-Forward Validation:")
            print(f"Mean OOS Sharpe: {wf['aggregate_metrics']['mean_sharpe']:.3f}")
            print(f"P-value: {wf['aggregate_metrics'].get('sharpe_pvalue', 'N/A')}")
            
        # Capacity
        cap = self.results.get('capacity', {})
        if cap:
            print(f"\nCapacity Analysis:")
            print(f"Max viable size: ${cap.get('max_viable_size_mm', 0):.1f}M")
            
        print("\n" + "="*80)
        
        # Overall verdict
        avg_sharpe = df['sharpe'].mean() if mc_results else 0
        if avg_sharpe > 1.5:
            print("✅ VERDICT: Strategy is EXCELLENT - ready for production")
        elif avg_sharpe > 1.0:
            print("✅ VERDICT: Strategy is GOOD - suitable for live trading")
        elif avg_sharpe > 0.5:
            print("⚠️  VERDICT: Strategy is MODERATE - needs refinement")
        else:
            print("❌ VERDICT: Strategy is WEAK - not recommended")
            
        print("="*80)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Run comprehensive strategy validation')
    parser.add_argument('--pairs', nargs='+', help='Currency pairs to test')
    parser.add_argument('--skip-qc', action='store_true', help='Skip data quality checks')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip performance benchmarks')
    parser.add_argument('--output', default='validation_output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick validation (fewer tests)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ValidationPipeline(
        output_dir=Path(args.output)
    )
    
    # If quick mode, only test a few pairs
    if args.quick:
        currency_pairs = ['AUDUSD', 'EURUSD', 'GBPUSD']
    else:
        currency_pairs = args.pairs
    
    # Run validation
    results = pipeline.run_complete_validation(
        currency_pairs=currency_pairs,
        skip_qc=args.skip_qc,
        skip_benchmarks=args.skip_benchmarks
    )
    
    # Exit with appropriate code
    avg_sharpe = 0
    mc_results = results.get('multi_currency', [])
    if mc_results:
        avg_sharpe = pd.DataFrame(mc_results)['sharpe'].mean()
        
    if avg_sharpe < 0.5:
        exit(1)  # Strategy failed validation
    else:
        exit(0)  # Strategy passed validation


if __name__ == "__main__":
    main()