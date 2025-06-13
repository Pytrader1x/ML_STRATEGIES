# Real-Time Strategy Validation System

This validation system simulates live market conditions by processing data one row at a time, ensuring the strategy never has access to future information and can detect look-ahead bias.

## 🎯 Purpose

- **Real-time simulation**: Process market data row-by-row as if trading live
- **Look-ahead bias detection**: Ensure no future data is used in decisions
- **Strategy validation**: Test trading logic under realistic conditions
- **Performance verification**: Compare results across different time periods and configurations

## 📁 Files

### Core Components

1. **`real_time_data_generator.py`**
   - Streams market data one row at a time
   - Calculates indicators incrementally on growing dataset
   - Simulates live market data feed

2. **`real_time_strategy_simulator.py`**
   - Executes trading strategy in real-time conditions
   - Records all trading decisions and events
   - Tracks equity curve and performance metrics

3. **`run_validation_tests.py`**
   - Comprehensive test suite
   - Multiple validation scenarios
   - Automated reporting

## 🚀 Quick Start

### Run Basic Validation
```bash
cd Validation/
python run_validation_tests.py
```

### Run Individual Components

#### Test Data Generator
```bash
python real_time_data_generator.py
```

#### Test Strategy Simulator
```bash
python real_time_strategy_simulator.py
```

## 🧪 Test Scenarios

The validation suite runs 5 comprehensive tests:

### 1. Time Period Testing
- Tests strategy across different market periods
- Random sampling for robustness
- Recent vs historical performance

### 2. Configuration Comparison
- Ultra-tight risk management vs scalping
- Different risk parameters
- Entry/exit logic variations

### 3. Real-time vs Batch Processing
- Validates row-by-row processing
- Ensures temporal consistency
- Performance characteristic analysis

### 4. Look-ahead Bias Detection
- Debug mode with detailed logging
- Temporal decision analysis
- Suspicious pattern identification

### 5. Indicator Consistency
- Incremental indicator calculation
- Value range validation
- Distribution analysis

## 📊 Output

### Validation Results
- `validation_results/` - Detailed JSON results
- `validation_charts/` - Performance visualizations
- Summary reports with key findings

### Example Output Structure
```
validation_results/
├── AUDUSD_validation_20250613_143052.json
├── AUDUSD_summary_20250613_143052.txt
└── ...

validation_charts/
├── performance_comparison.png
├── equity_curves.png
└── ...
```

## 🔍 Key Features

### Real-time Simulation
- **Row-by-row processing**: Data streamed one candle at a time
- **Progressive indicators**: Calculated only on available history
- **No future access**: Guaranteed prevention of look-ahead bias
- **Event logging**: Complete audit trail of all decisions

### Look-ahead Bias Prevention
- ✅ **Temporal consistency**: All decisions chronologically ordered
- ✅ **Limited data access**: Only past/current data available
- ✅ **Progressive calculation**: Indicators built incrementally
- ✅ **Event timestamping**: Complete decision audit trail

### Performance Validation
- 📈 **Multiple time periods**: Test robustness across market conditions
- ⚙️ **Configuration testing**: Compare strategy parameters
- 📊 **Statistical analysis**: Sharpe ratio, drawdown, win rates
- 🎯 **Trade-level analysis**: Individual trade performance

## 🛠️ Customization

### Test Different Configurations
```python
# In run_validation_tests.py
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,  # Adjust risk
    sl_max_pips=10.0,      # Adjust stop loss
    debug_decisions=True   # Enable detailed logging
)
```

### Test Different Time Periods
```python
# In run_validation_tests.py
periods = [
    {'name': 'Custom_Period', 'start_date': '2023-01-01', 'rows': 5000},
    {'name': 'Random_Sample', 'start_date': None, 'rows': 3000}
]
```

### Enable Debug Mode
```python
config = OptimizedStrategyConfig(
    debug_decisions=True,  # Detailed trade logging
    verbose=True          # Progress reporting
)
```

## 📝 Example Usage

### Basic Validation Run
```python
from run_validation_tests import ValidationTestSuite

# Initialize test suite
test_suite = ValidationTestSuite()

# Run comprehensive validation
results = test_suite.run_comprehensive_validation('AUDUSD')

# Access results
performance = results['time_periods']['Recent_2024']['performance']
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
```

### Custom Real-time Simulation
```python
from real_time_strategy_simulator import RealTimeStrategySimulator
from strategy_code.Prod_strategy import OptimizedStrategyConfig

# Create configuration
config = OptimizedStrategyConfig(
    initial_capital=1_000_000,
    risk_per_trade=0.002,
    debug_decisions=True
)

# Run simulation
simulator = RealTimeStrategySimulator(config)
results = simulator.run_real_time_simulation(
    currency_pair='AUDUSD',
    rows_to_simulate=5000,
    verbose=True
)
```

## 🔧 Advanced Features

### Event Analysis
```python
# Access detailed event history
events = results['detailed_data']['events']
for event in events:
    print(f"{event.timestamp}: {event.event_type} at {event.price}")
```

### Indicator History
```python
# Analyze indicator progression
indicators = results['detailed_data']['indicator_history']
nti_values = [ind['NTI_Direction'] for ind in indicators]
```

### Trade-by-Trade Analysis
```python
# Individual trade performance
trades = results['detailed_data']['trades']
for trade in trades:
    print(f"Trade: {trade.direction.value} P&L: ${trade.pnl:,.0f}")
```

## ⚠️ Important Notes

1. **Memory Usage**: Large simulations (>10k rows) may use significant memory
2. **Processing Time**: Real-time simulation is slower than batch processing
3. **Indicator Warmup**: First ~200 rows used for indicator initialization
4. **Path Dependencies**: Run from Validation/ directory for proper imports

## 🎯 Validation Checklist

- ✅ No look-ahead bias detected
- ✅ Temporal consistency maintained
- ✅ Indicators calculated progressively
- ✅ Strategy logic verified
- ✅ Performance metrics realistic
- ✅ Multiple time periods tested
- ✅ Configuration robustness confirmed

## 📞 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the Validation directory
cd Classical_strategies/Validation/
python run_validation_tests.py
```

**Data Path Issues**
```python
# The system auto-detects data paths, but you can specify:
generator = RealTimeDataGenerator('AUDUSD', data_path='../../data/AUDUSD_MASTER_15M.csv')
```

**Memory Issues**
```python
# Reduce sample size for testing
results = simulator.run_real_time_simulation(rows_to_simulate=1000)
```

This validation system provides comprehensive testing to ensure your trading strategy works correctly in real market conditions without any look-ahead bias! 🚀