# Trading Strategy React Chart Visualizer

Interactive web-based charts for visualizing trading strategy results using React and TradingView's lightweight-charts library.

## Features

- **Interactive Candlestick Charts**: OHLC data with zoom, pan, and crosshair
- **Trade Markers**: Visual indicators for entry/exit points with color coding
- **Technical Indicators**: NeuroTrend EMAs, Market Bias overlays
- **Performance Metrics**: Real-time display of key strategy metrics
- **Multiple Views**: Toggle between different chart components
- **Dark Theme**: Professional trading interface design

## Quick Start

### Standalone Mode

```bash
# Navigate to react_chart directory
cd react_chart

# Install dependencies (first time only)
npm install

# Start development server
npm run dev

# Open http://localhost:5173 in your browser
```

### Integration with Python Strategy

1. **Export and Launch** (recommended):
```bash
python run_Strategy.py --show-react
```

2. **Export Only** (for later viewing):
```bash
python run_Strategy.py --export-react
```

3. **Custom Port**:
```bash
python run_Strategy.py --show-react --react-port 3000
```

## Data Format

The app expects JSON data in the following format:

```json
{
  "metadata": {
    "symbol": "EURUSD",
    "timeframe": "15M",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  },
  "ohlc": [
    {
      "time": 1704067200000,
      "open": 1.0950,
      "high": 1.0955,
      "low": 1.0945,
      "close": 1.0952
    }
  ],
  "indicators": {
    "neurotrend": {
      "fast_ema": [],
      "slow_ema": []
    }
  },
  "trades": [],
  "metrics": {
    "total_trades": 150,
    "win_rate": 0.65,
    "sharpe_ratio": 1.85
  }
}
```

## Manual Data Upload

You can also upload JSON files directly through the web interface:

1. Click "Upload JSON" button
2. Select your exported chart_data.json file
3. Chart will update automatically

## Development

### Project Structure

```
react_chart/
├── src/
│   ├── components/
│   │   ├── TradingChart.jsx    # Main chart component
│   │   └── TradingChart.css    # Chart styles
│   ├── App.jsx                 # Main application
│   ├── App.css                 # App styles
│   └── index.css              # Global styles
├── public/
│   └── chart_data.json        # Default data location
└── package.json
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Customization

### Adding New Indicators

Edit `TradingChart.jsx` to add new indicator series:

```javascript
// Add your indicator series
const myIndicatorSeries = chart.addLineSeries({
  color: '#FF5733',
  lineWidth: 2,
  title: 'My Indicator'
});

// Set the data
myIndicatorSeries.setData(myIndicatorData);
```

### Modifying Chart Appearance

Update the chart configuration in `TradingChart.jsx`:

```javascript
const chart = createChart(chartContainerRef.current, {
  layout: {
    background: { color: '#131722' },
    textColor: '#d1d4dc',
  },
  // Add your customizations here
});
```

## Troubleshooting

### Chart not loading?

1. Check if data file exists at `public/chart_data.json`
2. Verify JSON format is valid
3. Check browser console for errors

### Server won't start?

1. Ensure Node.js is installed (v14+)
2. Run `npm install` to install dependencies
3. Check if port 5173 is available

### Integration issues?

1. Ensure `chart_data_exporter.py` is in the correct location
2. Check Python imports in `run_Strategy.py`
3. Verify React app directory path

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Multiple timeframe support
- [ ] Advanced technical indicators
- [ ] Trade analytics dashboard
- [ ] Export chart as image
- [ ] Mobile responsive design

## License

Part of the Trading Strategy System
