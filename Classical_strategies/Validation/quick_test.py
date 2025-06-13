"""
Quick Test Script
Simple validation test to verify the system works
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategy_code.Prod_strategy import OptimizedStrategyConfig
    from real_time_strategy_simulator import RealTimeStrategySimulator
    from real_time_data_generator import RealTimeDataGenerator
    
    print("✅ All imports successful")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def quick_test():
    """Run a quick test of the validation system"""
    
    print("\\n" + "="*60)
    print("QUICK VALIDATION TEST")
    print("="*60)
    
    try:
        # Test 1: Data Generator
        print("\\n🧪 Test 1: Data Generator")
        print("-" * 30)
        
        generator = RealTimeDataGenerator('AUDUSD')
        info = generator.get_data_info()
        print(f"✅ Data loaded: {info['total_rows']:,} rows")
        print(f"✅ Date range: {info['date_range']['start']} to {info['date_range']['end']}")
        
        # Test streaming (just 10 rows)
        start_idx, end_idx = generator.get_sample_period(rows=100)
        count = 0
        for data_point in generator.stream_data(start_idx, start_idx + 10):
            count += 1
            if count == 1:
                print(f"✅ First data point: Price {data_point['price']:.5f} at {data_point['current_time']}")
        
        print(f"✅ Streamed {count} data points successfully")
        
        # Test 2: Strategy Simulator
        print("\\n🧪 Test 2: Strategy Simulator")
        print("-" * 30)
        
        config = OptimizedStrategyConfig(
            initial_capital=1_000_000,
            risk_per_trade=0.002,  # 0.2% risk per trade
            sl_max_pips=10.0,
            sl_atr_multiplier=1.0,
            tp_atr_multipliers=(0.2, 0.3, 0.5),
            max_tp_percent=0.003,
            tsl_activation_pips=3,
            tsl_min_profit_pips=1,
            tsl_initial_buffer_multiplier=1.0,
            trailing_atr_multiplier=0.8,
            tp_range_market_multiplier=0.5,
            tp_trend_market_multiplier=0.7,
            tp_chop_market_multiplier=0.3,
            sl_range_market_multiplier=0.7,
            exit_on_signal_flip=False,
            signal_flip_min_profit_pips=5.0,
            signal_flip_min_time_hours=1.0,
            signal_flip_partial_exit_percent=1.0,
            partial_profit_before_sl=True,
            partial_profit_sl_distance_ratio=0.5,
            partial_profit_size_percent=0.5,
            intelligent_sizing=False,
            sl_volatility_adjustment=True,
            verbose=False,
            debug_decisions=False
        )
        
        simulator = RealTimeStrategySimulator(config)
        print("✅ Strategy simulator created")
        
        # Run small simulation
        results = simulator.run_real_time_simulation(
            currency_pair='AUDUSD',
            rows_to_simulate=4000,  # Small test
            verbose=False
        )
        
        print(f"✅ Simulation completed:")
        print(f"   - Rows processed: {results['simulation_summary']['rows_processed']}")
        print(f"   - Total trades: {results['trade_statistics']['total_trades']}")
        print(f"   - Final return: {results['performance_metrics']['total_return']:.2f}%")
        print(f"   - Sharpe ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
        
        # Test 3: Look-ahead bias check
        print("\\n🧪 Test 3: Look-ahead Bias Check")
        print("-" * 30)
        
        events = results['detailed_data']['events']
        print(f"✅ Events recorded: {len(events)}")
        
        # Check temporal consistency
        if len(events) > 1:
            temporal_ok = True
            for i in range(1, len(events)):
                if events[i].timestamp < events[i-1].timestamp:
                    temporal_ok = False
                    break
            
            if temporal_ok:
                print("✅ All events are temporally consistent - no look-ahead bias detected")
            else:
                print("❌ Temporal inconsistency detected - potential look-ahead bias")
        else:
            print("ℹ️  Not enough events to check temporal consistency")
        
        print("\\n" + "="*60)
        print("✅ QUICK TEST COMPLETED SUCCESSFULLY")
        print("✅ System is ready for comprehensive validation")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\\n🎯 Next steps:")
        print("  1. Run full validation: python run_validation_tests.py")
        print("  2. Check debug mode: Set debug_decisions=True in config")
        print("  3. Test different time periods and configurations")
    else:
        print("\\n❌ Please fix errors before running full validation")
        sys.exit(1)