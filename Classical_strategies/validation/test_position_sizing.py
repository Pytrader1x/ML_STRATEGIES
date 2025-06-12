"""
Test Position Sizing Issue - Why are we seeing 1M, 3M, 5M positions?
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig, RiskManager
from technical_indicators_custom import TIC
import warnings

warnings.filterwarnings('ignore')

def test_position_sizing_logic():
    """Test the position sizing calculation directly"""
    print("="*80)
    print("POSITION SIZING LOGIC TEST")
    print("="*80)
    
    # Test the RiskManager directly
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.002,
        intelligent_sizing=False,  # Should give fixed 1M
        min_lot_size=1_000_000
    )
    
    risk_manager = RiskManager(config)
    
    # Test cases
    test_cases = [
        # (entry_price, stop_loss, capital, is_relaxed, confidence, expected_size)
        (1.0000, 0.9990, 100_000, False, 50.0, 1_000_000),  # Normal
        (1.0000, 0.9990, 100_000, True, 50.0, 1_000_000),   # Relaxed
        (1.0000, 0.9990, 100_000, False, 80.0, 1_000_000),  # High confidence
        (1.0000, 0.9990, 100_000, False, 20.0, 1_000_000),  # Low confidence
    ]
    
    print("\nTesting with intelligent_sizing=False (should all be 1M):")
    for entry, sl, capital, relaxed, conf, expected in test_cases:
        size = risk_manager.calculate_position_size(entry, sl, capital, relaxed, conf)
        print(f"Relaxed={relaxed}, Conf={conf}: Size={size:,.0f} ({'PASS' if size == expected else 'FAIL'})")
    
    # Now test with intelligent sizing enabled
    config.intelligent_sizing = True
    config.size_multipliers = [0.5, 1.0, 3.0, 5.0]  # These might explain 3M, 5M
    config.confidence_thresholds = [25, 50, 75]
    
    risk_manager = RiskManager(config)
    
    print("\nTesting with intelligent_sizing=True:")
    for conf in [10, 30, 60, 90]:
        size_mult, tp_mult = risk_manager.get_position_size_multiplier(conf)
        size = risk_manager.calculate_position_size(1.0, 0.999, 100_000, False, conf)
        print(f"Confidence {conf}: Multiplier={size_mult}, Size={size/1_000_000:.0f}M")
    
    # Check if config has these values
    print("\nChecking default config values:")
    default_config = OptimizedStrategyConfig()
    print(f"intelligent_sizing: {default_config.intelligent_sizing}")
    print(f"size_multipliers: {default_config.size_multipliers}")
    print(f"confidence_thresholds: {default_config.confidence_thresholds}")


def test_position_sizing_in_backtest():
    """Test position sizing in actual backtest"""
    print("\n" + "="*80)
    print("POSITION SIZING IN BACKTEST TEST")
    print("="*80)
    
    # Load real data
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Use recent data
    test_df = df['2023-01-01':'2023-03-31'].copy()
    test_df = TIC.add_neuro_trend_intelligent(test_df)
    test_df = TIC.add_market_bias(test_df)
    test_df = TIC.add_intelligent_chop(test_df)
    
    # Test with different configs
    configs_to_test = [
        ("Default", OptimizedStrategyConfig()),
        ("Intelligent OFF", OptimizedStrategyConfig(intelligent_sizing=False)),
        ("Intelligent ON", OptimizedStrategyConfig(
            intelligent_sizing=True,
            size_multipliers=[1.0, 1.0, 3.0, 5.0],
            confidence_thresholds=[25, 50, 75]
        )),
    ]
    
    for name, config in configs_to_test:
        print(f"\n{name} Configuration:")
        config.verbose = True
        strategy = OptimizedProdStrategy(config)
        
        # Capture logs to analyze position sizes
        import io
        import logging
        
        # Create string buffer to capture logs
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        # Get the strategy logger
        logger = logging.getLogger('strategy_code.Prod_strategy')
        logger.addHandler(handler)
        
        # Run backtest
        results = strategy.run_backtest(test_df)
        
        # Analyze logs
        log_contents = log_capture.getvalue()
        
        # Extract position sizes from logs
        import re
        pattern = r"TRADE:.*with (\d+)M"
        sizes = re.findall(pattern, log_contents)
        
        if sizes:
            size_counts = {}
            for size in sizes:
                size_counts[size] = size_counts.get(size, 0) + 1
            
            print(f"Position sizes found:")
            for size, count in sorted(size_counts.items()):
                print(f"  {size}M: {count} trades")
        else:
            print("No position sizes found in logs")
        
        # Clean up
        logger.removeHandler(handler)
        log_capture.close()


def test_confidence_calculation():
    """Test how confidence is calculated"""
    print("\n" + "="*80)
    print("CONFIDENCE CALCULATION TEST")
    print("="*80)
    
    # This would need to examine how confidence is calculated in the strategy
    # Looking at the source, it seems to come from the chop index
    
    # Load some data to check IC values
    df = pd.read_csv('../../data/AUDUSD_MASTER_15M.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    sample = df['2023-01-01':'2023-01-07'].copy()
    sample = TIC.add_neuro_trend_intelligent(sample)
    sample = TIC.add_market_bias(sample)
    sample = TIC.add_intelligent_chop(sample)
    
    # Check IC_Chop values (confidence source)
    print("\nIC_Chop value distribution:")
    ic_chop = sample['IC_Chop'].dropna()
    print(f"Min: {ic_chop.min():.1f}")
    print(f"Max: {ic_chop.max():.1f}")
    print(f"Mean: {ic_chop.mean():.1f}")
    print(f"Std: {ic_chop.std():.1f}")
    
    # Confidence buckets
    print("\nConfidence distribution:")
    print(f"< 25: {(ic_chop < 25).sum()} bars")
    print(f"25-50: {((ic_chop >= 25) & (ic_chop < 50)).sum()} bars")
    print(f"50-75: {((ic_chop >= 50) & (ic_chop < 75)).sum()} bars")
    print(f">= 75: {(ic_chop >= 75).sum()} bars")


def main():
    test_position_sizing_logic()
    test_position_sizing_in_backtest()
    test_confidence_calculation()
    
    print("\n" + "="*80)
    print("POSITION SIZING TEST COMPLETE")
    print("="*80)
    print("\n🔍 Key Finding: The varying position sizes (1M, 3M, 5M) are likely")
    print("   due to intelligent_sizing being enabled with multipliers [0.5, 1, 3, 5]")
    print("   based on the IC_Chop confidence indicator.")


if __name__ == "__main__":
    main()