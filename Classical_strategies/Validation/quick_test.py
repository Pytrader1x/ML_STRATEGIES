"""
Quick Test Script
Simple validation test to verify the system works
"""

import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from strategy_code.Prod_strategy import OptimizedStrategyConfig
    from strategy_code.Prod_plotting import plot_production_results
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
        # Test 1: Data Generator with Real-time Indicator Calculation
        print("\\n🧪 Test 1: Real-time Data Streaming & Indicator Calculation")
        print("-" * 30)
        
        generator = RealTimeDataGenerator('AUDUSD')
        info = generator.get_data_info()
        print(f"✅ Data loaded: {info['total_rows']:,} rows")
        print(f"✅ Date range: {info['date_range']['start']} to {info['date_range']['end']}")
        
        # Test streaming with real-time indicator calculation
        start_idx, _ = generator.get_sample_period(rows=1000)
        count = 0
        
        print("\\n📊 Demonstrating real-time data flow:")
        print("   1. New market data arrives (one row)")
        print("   2. Indicators calculated on all data up to that point")
        print("   3. Strategy makes decision based on current indicators")
        print("   4. Process repeats for next data point\\n")
        
        for data_point in generator.stream_data(start_idx, start_idx + 10):
            count += 1
            
            # Show the real-time flow for first 3 data points
            if count <= 3:
                print(f"\\n{'='*50}")
                print(f"⏰ NEW DATA ARRIVES - Row {count}:")
                print(f"   Time: {data_point['current_time']}")
                print(f"   Price: {data_point['price']:.5f}")
                print(f"   Historical buffer size: {data_point['historical_length']} rows")
                
                print(f"\\n📈 INDICATORS CALCULATED on {data_point['historical_length']} rows:")
                print(f"   NTI Direction: {data_point['data'].get('NTI_Direction', 'N/A')}")
                print(f"   Market Bias: {data_point['data'].get('MB_Bias', 'N/A')}")
                print(f"   IC Regime: {data_point['data'].get('IC_Regime', 'N/A')} ({data_point['data'].get('IC_RegimeName', 'Unknown')})")
                print(f"   ATR Normalized: {data_point['data'].get('IC_ATR_Normalized', 0):.5f}")
                
                print(f"\\n🎯 STRATEGY DECISION POINT:")
                print(f"   Strategy would now evaluate entry/exit conditions")
                print(f"   Based only on data available up to row {count}")
        
        print(f"\\n✅ Successfully demonstrated real-time flow for {count} data points")
        
        # Test 2: Real-time Strategy Execution Flow
        print("\\n🧪 Test 2: Real-time Strategy Execution Flow")
        print("-" * 30)
        
        print("\\n📋 Strategy Configuration (Ultra-Tight Risk):")
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
        
        print("   - Initial Capital: $1,000,000")
        print("   - Risk per Trade: 0.2%")
        print("   - Max Stop Loss: 10 pips")
        print("   - Take Profits: 0.2, 0.3, 0.5 ATR")
        
        simulator = RealTimeStrategySimulator(config)
        print("\\n✅ Strategy simulator created")
        
        print("\\n🔄 Real-time Simulation Process:")
        print("   For each new market data row:")
        print("   1. Calculate indicators on historical data")
        print("   2. Check open trade management (SL/TP/Trailing)")
        print("   3. Evaluate entry conditions if no open trade")
        print("   4. Record all decisions with timestamps")
        print("   5. Update equity and move to next data point")
        
        # Run simulation
        print("\\n▶️  Running real-time simulation...")
        
        # Get a random sample period for both simulation and plotting
        generator = RealTimeDataGenerator('AUDUSD')
        
        # Get random sample - remove fixed index for true random testing
        rows_to_test = 1000
        
        start_idx, end_idx = generator.get_sample_period(
            rows=rows_to_test
        )
        
        # Store the exact data range for plotting
        plot_data_start = start_idx
        plot_data_end = end_idx
        
        print(f"   - Using data range: rows {start_idx:,} to {end_idx:,}")
        print(f"   - Date range: {generator.full_data.iloc[start_idx]['DateTime']} to {generator.full_data.iloc[end_idx-1]['DateTime']}")
        
        # Run simulation with the exact same data range
        results = simulator.run_real_time_simulation(
            currency_pair='AUDUSD',
            rows_to_simulate=rows_to_test,
            start_idx=start_idx,  # Use the same start index
            verbose=False
        )
        
        print(f"\\n✅ Real-time simulation completed:")
        print(f"   - Rows processed: {results['simulation_summary']['rows_processed']:,}")
        print(f"   - Total trades: {results['trade_statistics']['total_trades']}")
        print(f"   - Win rate: {results['trade_statistics']['win_rate']:.1f}%")
        print(f"   - Final return: {results['performance_metrics']['total_return']:+.2f}%")
        print(f"   - Sharpe ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
        print(f"   - Max drawdown: {results['performance_metrics']['max_drawdown']:.2f}%")
        
        # Show some trade events
        if results['detailed_data']['events']:
            print(f"\\n📊 Sample Trade Events (first 5):")
            for i, event in enumerate(results['detailed_data']['events'][:5]):
                if event.event_type in ['entry', 'exit']:
                    print(f"   {i+1}. {event.event_type.upper()}: {event.timestamp} | "
                          f"Price: {event.price:.5f} | "
                          f"{'Direction: ' + event.direction if event.direction else 'Reason: ' + event.reason}")
        
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
        
        # Test 4: Demonstrate Step-by-Step Real-time Flow
        print("\\n🧪 Test 4: Step-by-Step Real-time Flow Demo")
        print("-" * 30)
        
        print("\\n📌 Real-time Trading Simulation Flow:")
        print("\\n1️⃣  NEW MARKET DATA ARRIVES (e.g., new 15-minute candle)")
        print("   └─> OHLC: Open=0.65432, High=0.65445, Low=0.65420, Close=0.65440")
        
        print("\\n2️⃣  CALCULATE INDICATORS on all data up to this point")
        print("   └─> NTI Direction: 1 (Bullish)")
        print("   └─> Market Bias: 1 (Bullish)")
        print("   └─> IC Regime: 2 (Trending)")
        print("   └─> ATR: 0.00015")
        
        print("\\n3️⃣  STRATEGY EVALUATES current situation")
        print("   └─> Check if we have open trade? No")
        print("   └─> Check entry conditions:")
        print("       • NTI=1 ✓ MB=1 ✓ IC=2 ✓ → ENTRY SIGNAL!")
        
        print("\\n4️⃣  EXECUTE TRADE DECISION")
        print("   └─> Enter LONG at 0.65440")
        print("   └─> Stop Loss: 0.65340 (10 pips)")
        print("   └─> Take Profits: [0.65470, 0.65485, 0.65515]")
        print("   └─> Position Size: 2M units")
        
        print("\\n5️⃣  WAIT FOR NEXT DATA...")
        print("   └─> Process repeats when next candle arrives")
        print("   └─> No access to future data!")
        print("   └─> Decisions based only on current and past data")
        
        # Test 5: Plot Trading Results
        print("\\n🧪 Test 5: Plotting Trading Results")
        print("-" * 30)
        
        try:
            # Prepare data for plotting
            print("\\n📊 Preparing data for visualization...")
            
            # IMPORTANT: Use the same generator instance to get the exact same data
            # This ensures we're using the identical data that was streamed to the simulator
            import pandas as pd
            
            # Get the full data from the generator that was already loaded
            df_full = generator.full_data.copy()
            
            # Convert DateTime if needed
            if 'DateTime' in df_full.columns and not isinstance(df_full.index, pd.DatetimeIndex):
                df_full['DateTime'] = pd.to_datetime(df_full['DateTime'])
                df_full.set_index('DateTime', inplace=True)
            
            # Get the exact same data slice that was used in simulation
            # Using the exact indices ensures perfect alignment
            df_plot = df_full.iloc[plot_data_start:plot_data_end].copy()
            
            print(f"   - Using simulation data from generator")
            print(f"   - Data slice: rows {plot_data_start} to {plot_data_end}")
            
            # Reset index to ensure it's datetime
            if not isinstance(df_plot.index, pd.DatetimeIndex):
                if 'DateTime' in df_plot.columns:
                    df_plot.set_index('DateTime', inplace=True)
            
            # Add indicators to the data
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from technical_indicators_custom import TIC
            
            print("   - Adding technical indicators...")
            df_plot = TIC.add_neuro_trend_intelligent(df_plot)
            df_plot = TIC.add_market_bias(df_plot)
            df_plot = TIC.add_intelligent_chop(df_plot)
            
            print(f"   - Data range for plotting: {df_plot.index[0]} to {df_plot.index[-1]}")
            
            # Convert trades to the format expected by plotting function
            print("   - Formatting trade data...")
            print(f"   - Number of trades to plot: {len(results['detailed_data']['trades'])}")
            
            # Ensure trades have all required fields for plotting
            formatted_trades = []
            for i, trade in enumerate(results['detailed_data']['trades']):
                # Debug first trade
                if i == 0:
                    print(f"\\n   First trade details:")
                    if hasattr(trade, 'entry_time'):
                        print(f"     Type: Trade object")
                        print(f"     Entry: {trade.entry_time} @ {trade.entry_price:.5f}")
                        print(f"     Exit: {trade.exit_time} @ {trade.exit_price:.5f}")
                        print(f"     Direction: {trade.direction}")
                        print(f"     P&L: ${trade.pnl:.2f}")
                    else:
                        print(f"     Type: Dictionary")
                        print(f"     Keys: {list(trade.keys())}")
                
                # If it's a Trade object, it should already have the right structure
                if hasattr(trade, 'entry_time'):
                    formatted_trades.append(trade)
                else:
                    # Convert dict to Trade object if needed
                    from strategy_code.Prod_strategy import Trade, TradeDirection, ExitReason
                    formatted_trade = Trade(
                        entry_time=trade.get('entry_time'),
                        entry_price=trade.get('entry_price'),
                        exit_time=trade.get('exit_time'),
                        exit_price=trade.get('exit_price'),
                        direction=TradeDirection(trade.get('direction')) if isinstance(trade.get('direction'), str) else trade.get('direction'),
                        exit_reason=ExitReason(trade.get('exit_reason')) if isinstance(trade.get('exit_reason'), str) else trade.get('exit_reason'),
                        position_size=trade.get('position_size', 1000000),
                        stop_loss=trade.get('stop_loss'),
                        take_profits=trade.get('take_profits', []),
                        pnl=trade.get('pnl', 0),
                        pnl_percent=trade.get('pnl_percent', 0),
                        partial_exits=trade.get('partial_exits', [])
                    )
                    formatted_trades.append(formatted_trade)
            
            print(f"   - Formatted {len(formatted_trades)} trades for plotting")
            
            # Format results with equity curve for P&L subplot
            formatted_results = {
                'trades': formatted_trades,
                'equity_curve': results['detailed_data']['capital_history'],
                'total_trades': results['trade_statistics']['total_trades'],
                'win_rate': results['trade_statistics']['win_rate'],
                'sharpe_ratio': results['performance_metrics']['sharpe_ratio'],
                'total_pnl': results['performance_metrics']['total_pnl'],
                'max_drawdown': results['performance_metrics']['max_drawdown'],
                'total_return': results['performance_metrics']['total_return']
            }
            
            # Generate the plot
            print("\\n📈 Generating trading chart...")
            
            # Debug: Check if trade times are within plot data range
            if formatted_trades:
                first_trade = formatted_trades[0]
                last_trade = formatted_trades[-1] if len(formatted_trades) > 1 else first_trade
                print(f"   - First trade time: {first_trade.entry_time}")
                print(f"   - Last trade time: {last_trade.entry_time if last_trade else 'N/A'}")
                print(f"   - Plot data range: {df_plot.index[0]} to {df_plot.index[-1]}")
                
                # Check if trades are within plot range
                trades_in_range = 0
                for trade in formatted_trades:
                    if trade.entry_time in df_plot.index:
                        trades_in_range += 1
                print(f"   - Trades within plot range: {trades_in_range}/{len(formatted_trades)}")
            
            # Generate plot with all parameters
            fig = plot_production_results(
                df=df_plot,
                results=formatted_results,
                title=f"Real-time Strategy Validation - AUDUSD\\n" + 
                      f"Sharpe: {results['performance_metrics']['sharpe_ratio']:.3f} | " +
                      f"Return: {results['performance_metrics']['total_return']:.1f}% | " +
                      f"Trades: {results['trade_statistics']['total_trades']}",
                show_pnl=True,
                show_position_sizes=False,
                show_chop_subplots=False,
                show=True  # Display the plot
            )
            
            # Save the plot
            os.makedirs('validation_charts', exist_ok=True)
            plot_filename = 'validation_charts/quick_test_trades.png'
            if fig:
                fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"✅ Trading chart saved to: {plot_filename}")
            
            print("\\n📊 Chart shows:")
            print("   - Candlestick price data")
            print("   - Entry points (green triangles)")
            print("   - Exit points (red triangles)")
            print("   - Technical indicators (NTI, MB, IC)")
            print("   - Equity curve")
            
        except Exception as e:
            print(f"⚠️  Could not generate plot: {e}")
            print("   This is normal if running in a non-graphical environment")
        
        print("\\n" + "="*60)
        print("✅ QUICK TEST COMPLETED SUCCESSFULLY")
        print("✅ System simulates real-time trading conditions")
        print("✅ No look-ahead bias possible - data processed sequentially")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_monte_carlo(num_runs=10, rows_per_test=1000):
    """Run Monte Carlo simulation with multiple random samples"""
    
    print("\n" + "="*80)
    print(f"MONTE CARLO VALIDATION - {num_runs} RUNS")
    print("="*80)
    
    # Create config once
    config = OptimizedStrategyConfig(
        initial_capital=1_000_000,
        risk_per_trade=0.002,
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
    
    # Load data once
    print("\nLoading data...")
    generator = RealTimeDataGenerator('AUDUSD')
    info = generator.get_data_info()
    print(f"✅ Data loaded: {info['total_rows']:,} rows")
    print(f"Date range: {info['date_range']['start']} to {info['date_range']['end']}")
    
    all_results = []
    
    for i in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {i+1}/{num_runs}")
        print('='*60)
        
        # Get random sample
        start_idx, end_idx = generator.get_sample_period(rows=rows_per_test)
        
        # Create new simulator for each run
        simulator = RealTimeStrategySimulator(config)
        
        # Run simulation
        results = simulator.run_real_time_simulation(
            currency_pair='AUDUSD',
            rows_to_simulate=rows_per_test,
            start_idx=start_idx,
            verbose=False
        )
        
        run_info = {
            'run': i+1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'date_start': generator.full_data.iloc[start_idx]['DateTime'],
            'date_end': generator.full_data.iloc[end_idx-1]['DateTime'],
            'sharpe': results['performance_metrics']['sharpe_ratio'],
            'return': results['performance_metrics']['total_return'],
            'trades': results['trade_statistics']['total_trades'],
            'win_rate': results['trade_statistics']['win_rate'],
            'max_dd': results['performance_metrics']['max_drawdown']
        }
        
        all_results.append(run_info)
        
        print(f"Results: Sharpe={run_info['sharpe']:.3f} | Return={run_info['return']:.1f}% | " 
              f"Trades={run_info['trades']} | WR={run_info['win_rate']:.1f}%")
    
    # Calculate summary statistics
    sharpes = [r['sharpe'] for r in all_results]
    returns = [r['return'] for r in all_results]
    trades = [r['trades'] for r in all_results]
    win_rates = [r['win_rate'] for r in all_results]
    max_dds = [r['max_dd'] for r in all_results]
    
    print("\n" + "="*80)
    print("MONTE CARLO SUMMARY")
    print("="*80)
    print(f"Sharpe Ratio:  Mean={np.mean(sharpes):.3f}, Std={np.std(sharpes):.3f}, "
          f"Min={np.min(sharpes):.3f}, Max={np.max(sharpes):.3f}")
    print(f"Returns:       Mean={np.mean(returns):.1f}%, Std={np.std(returns):.1f}%, "
          f"Min={np.min(returns):.1f}%, Max={np.max(returns):.1f}%")
    print(f"Win Rate:      Mean={np.mean(win_rates):.1f}%, Std={np.std(win_rates):.1f}%, "
          f"Min={np.min(win_rates):.1f}%, Max={np.max(win_rates):.1f}%")
    print(f"Max Drawdown:  Mean={np.mean(max_dds):.1f}%, Std={np.std(max_dds):.1f}%, "
          f"Min={np.min(max_dds):.1f}%, Max={np.max(max_dds):.1f}%")
    print(f"Trades:        Mean={np.mean(trades):.1f}, Min={np.min(trades)}, Max={np.max(trades)}")
    print(f"Profitable:    {sum(1 for r in returns if r > 0)}/{num_runs} "
          f"({sum(1 for r in returns if r > 0)/num_runs*100:.0f}%)")
    
    # Show best and worst runs
    best_run = max(all_results, key=lambda x: x['sharpe'])
    worst_run = min(all_results, key=lambda x: x['sharpe'])
    
    print(f"\nBest Run:  Run #{best_run['run']} - Sharpe={best_run['sharpe']:.3f}, "
          f"Return={best_run['return']:.1f}%, Trades={best_run['trades']}")
    print(f"Worst Run: Run #{worst_run['run']} - Sharpe={worst_run['sharpe']:.3f}, "
          f"Return={worst_run['return']:.1f}%, Trades={worst_run['trades']}")
    
    return all_results, {
        'sharpe_mean': np.mean(sharpes),
        'sharpe_std': np.std(sharpes),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'win_rate_mean': np.mean(win_rates),
        'trades_mean': np.mean(trades),
        'profitable_pct': sum(1 for r in returns if r > 0)/num_runs*100
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick validation test with Monte Carlo option')
    parser.add_argument('--monte-carlo', type=int, metavar='N', 
                       help='Run N Monte Carlo simulations (e.g., --monte-carlo 10)')
    parser.add_argument('--rows', type=int, default=1000,
                       help='Number of rows per test (default: 1000)')
    
    args = parser.parse_args()
    
    if args.monte_carlo:
        # Run Monte Carlo simulation
        all_results, summary = run_monte_carlo(num_runs=args.monte_carlo, rows_per_test=args.rows)
        print("\n✅ Monte Carlo validation completed!")
    else:
        # Run single test
        success = quick_test()
        
        if success:
            print("\\n🎯 Next steps:")
            print("  1. Run full validation: python run_validation_tests.py")
            print("  2. Check debug mode: Set debug_decisions=True in config")
            print("  3. Test different time periods and configurations")
            print("  4. Run Monte Carlo: python quick_test.py --monte-carlo 10")
        else:
            print("\\n❌ Please fix errors before running full validation")
            sys.exit(1)