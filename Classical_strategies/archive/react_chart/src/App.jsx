import { useState, useEffect } from 'react';
import TradingChart from './components/TradingChart';
import SimpleChart from './components/SimpleChart';
import EnhancedTradingChart from './components/EnhancedTradingChart';
import './App.css';

function App() {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [useEnhancedChart, setUseEnhancedChart] = useState(true); // Default to enhanced

  useEffect(() => {
    loadChartData();
  }, []);

  const loadChartData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Try to load data from different sources
      let data = null;
      
      // 1. Check if data path is provided via environment variable
      const dataPath = import.meta.env.VITE_DATA_PATH;
      if (dataPath) {
        const response = await fetch(dataPath);
        if (response.ok) {
          data = await response.json();
        }
      }
      
      // 2. Try to load from default location
      if (!data) {
        try {
          const response = await fetch('/chart_data.json');
          if (response.ok) {
            data = await response.json();
          }
        } catch (e) {
          console.log('No data at default location');
        }
      }
      
      // 3. Try to load from API endpoint (if Python server is running)
      if (!data) {
        try {
          const response = await fetch('http://localhost:5000/api/chart-data');
          if (response.ok) {
            data = await response.json();
          }
        } catch (e) {
          console.log('No API server running');
        }
      }
      
      // 4. Load sample data for demo
      if (!data) {
        data = generateSampleData();
      }
      
      setChartData(data);
    } catch (err) {
      setError(err.message);
      console.error('Error loading chart data:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateSampleData = () => {
    // Generate sample data for demonstration
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    const numDays = 90;
    
    const ohlc = [];
    const fastEma = [];
    const slowEma = [];
    const positionSizes = [];
    const pnl = [];
    
    let basePrice = 100;
    let cumulativePnl = 0;
    
    for (let i = 0; i < numDays; i++) {
      const time = now - (numDays - i) * dayMs;
      const volatility = 0.02;
      
      // Generate OHLC
      const open = basePrice + (Math.random() - 0.5) * volatility * basePrice;
      const close = open + (Math.random() - 0.5) * volatility * basePrice;
      const high = Math.max(open, close) + Math.random() * volatility * basePrice * 0.5;
      const low = Math.min(open, close) - Math.random() * volatility * basePrice * 0.5;
      
      ohlc.push({ time, open, high, low, close });
      
      // Generate indicators
      const fastValue = close * (1 + (Math.random() - 0.5) * 0.01);
      const slowValue = close * (1 + (Math.random() - 0.5) * 0.005);
      
      fastEma.push({ time, value: fastValue });
      slowEma.push({ time, value: slowValue });
      
      // Generate position sizes and P&L
      const position = Math.random() > 0.7 ? Math.random() * 2 - 1 : 0;
      positionSizes.push({ time, value: position });
      
      const dailyPnl = position * (close - open);
      cumulativePnl += dailyPnl;
      pnl.push({ time, value: cumulativePnl });
      
      basePrice = close;
    }
    
    // Generate sample trades
    const trades = [
      {
        entry_time: now - 60 * dayMs,
        exit_time: now - 55 * dayMs,
        entry_price: 98.5,
        exit_price: 101.2,
        direction: 'LONG',
        exit_reason: 'TAKE_PROFIT',
        pnl: 2.7,
        pnl_pct: 2.74
      },
      {
        entry_time: now - 45 * dayMs,
        exit_time: now - 42 * dayMs,
        entry_price: 102.8,
        exit_price: 101.5,
        direction: 'SHORT',
        exit_reason: 'STOP_LOSS',
        pnl: -1.3,
        pnl_pct: -1.26
      },
      {
        entry_time: now - 30 * dayMs,
        exit_time: now - 25 * dayMs,
        entry_price: 100.2,
        exit_price: 103.8,
        direction: 'LONG',
        exit_reason: 'TAKE_PROFIT',
        pnl: 3.6,
        pnl_pct: 3.59
      }
    ];
    
    return {
      metadata: {
        symbol: 'SAMPLE',
        timeframe: '1D',
        start_date: new Date(now - numDays * dayMs).toISOString().split('T')[0],
        end_date: new Date(now).toISOString().split('T')[0],
        total_rows: numDays
      },
      ohlc,
      indicators: {
        neurotrend: {
          fast_ema: fastEma,
          slow_ema: slowEma,
          direction: []
        },
        market_bias: {
          bias: [],
          o2: [],
          h2: [],
          l2: [],
          c2: []
        },
        intelligent_chop: {
          regime: [],
          regime_name: []
        }
      },
      trades,
      performance: {
        position_sizes: positionSizes,
        cumulative_pnl: pnl,
        returns: []
      },
      metrics: {
        total_trades: 3,
        winning_trades: 2,
        losing_trades: 1,
        win_rate: 0.667,
        sharpe_ratio: 1.25,
        max_drawdown: 0.15,
        total_return: 0.052,
        profit_factor: 1.85
      }
    };
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target.result);
          setChartData(data);
          setError(null);
        } catch (err) {
          setError('Invalid JSON file');
        }
      };
      reader.readAsText(file);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Trading Strategy Visualizer</h1>
        <div className="header-controls">
          <button onClick={loadChartData} className="reload-button">
            Reload Data
          </button>
          <label className="file-upload">
            Upload JSON
            <input type="file" accept=".json" onChange={handleFileUpload} />
          </label>
          <label className="chart-toggle">
            <input 
              type="checkbox" 
              checked={useEnhancedChart}
              onChange={(e) => setUseEnhancedChart(e.target.checked)}
            />
            Enhanced Chart
          </label>
        </div>
      </header>

      <main className="app-main">
        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading chart data...</p>
          </div>
        )}

        {error && (
          <div className="error-container">
            <p>Error: {error}</p>
            <p>Using sample data for demonstration</p>
          </div>
        )}

        {!loading && chartData && (
          <>
            {useEnhancedChart ? (
              <EnhancedTradingChart data={chartData} />
            ) : (
              <TradingChart data={chartData} />
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App
