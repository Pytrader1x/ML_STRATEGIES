import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import './EnhancedTradingChart.css';

const EnhancedTradingChart = ({ data }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const positionChartRef = useRef();
  const pnlChartRef = useRef();
  
  const [selectedOptions, setSelectedOptions] = useState({
    useChopBackground: true,
    simplifiedRegimeColors: true,
    showPnl: true,
    showPositionSizes: false,
    showTradeMarkers: true,
    showTpSlLevels: true,
    showPartialExits: true
  });
  
  const [chartError, setChartError] = useState(null);

  // Color configuration matching Python
  const COLORS = {
    bg: '#131722',
    grid: '#363c4e',
    text: '#d1d4dc',
    bullish: '#26A69A',
    bearish: '#EF5350',
    neutral: '#FFB74D',
    strongTrend: '#81C784',
    weakTrend: '#FFF176',
    quietRange: '#64B5F6',
    volatileChop: '#EF5350',
    white: '#ffffff'
  };

  const TRADE_COLORS = {
    longEntry: '#1E88E5',
    shortEntry: '#FFD600',
    takeProfit: '#43A047',
    tp1Pullback: '#4CAF50',
    stopLoss: '#E53935',
    trailingStop: '#9C27B0',
    signalFlip: '#FF8F00',
    endOfData: '#9E9E9E'
  };

  const REGIME_COLORS_SIMPLE = {
    'Strong Trend': '#2ECC71',
    'Weak Trend': '#2ECC71',
    'Quiet Range': '#95A5A6',
    'Volatile Chop': '#95A5A6',
    'Transitional': '#95A5A6'
  };

  const REGIME_COLORS_FULL = {
    'Strong Trend': '#81C784',
    'Weak Trend': '#FFF176',
    'Quiet Range': '#64B5F6',
    'Volatile Chop': '#FFCDD2',
    'Transitional': '#E0E0E0'
  };

  useEffect(() => {
    if (!data || !chartContainerRef.current) return;

    try {
      setChartError(null);

      // Create main chart
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 600,
        layout: {
          background: { color: COLORS.bg },
          textColor: COLORS.text,
        },
        grid: {
          vertLines: { color: COLORS.grid },
          horzLines: { color: COLORS.grid },
        },
        rightPriceScale: {
          borderColor: '#485c7b',
          autoScale: true,
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        },
        timeScale: {
          borderColor: '#485c7b',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      chartRef.current = chart;

      // Add candlestick series
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: COLORS.bullish,
        downColor: COLORS.bearish,
        borderUpColor: COLORS.bullish,
        borderDownColor: COLORS.bearish,
        wickUpColor: COLORS.bullish,
        wickDownColor: COLORS.bearish,
      });

      // Convert and set OHLC data
      const ohlcData = data.ohlc
        .filter(candle => candle.open !== null && candle.high !== null && 
                         candle.low !== null && candle.close !== null)
        .map(candle => ({
          time: Math.floor(candle.time / 1000),
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close
        }));

      if (ohlcData.length > 0) {
        candlestickSeries.setData(ohlcData);
      }

      // Add background coloring based on regime or trend
      if (selectedOptions.useChopBackground && data.indicators.intelligent_chop?.regime_name?.length > 0) {
        addRegimeBackground(chart, data);
      } else {
        addTrendBackground(chart, data);
      }

      // Add Market Bias overlay
      if (data.indicators.market_bias?.bias?.length > 0) {
        addMarketBiasOverlay(chart, data);
      }

      // Add NeuroTrend EMAs
      if (data.indicators.neurotrend?.fast_ema?.length > 0) {
        addNeuroTrendEMAs(chart, data);
      }

      // Add trade markers and levels
      if (selectedOptions.showTradeMarkers && data.trades?.length > 0) {
        addTradeMarkers(candlestickSeries, data);
        if (selectedOptions.showTpSlLevels) {
          addTradeLevels(chart, data);
        }
        if (selectedOptions.showPartialExits) {
          addPartialExitMarkers(candlestickSeries, data);
        }
      }

      // Handle resize
      const handleResize = () => {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
      };
    } catch (error) {
      console.error('Chart error:', error);
      setChartError(error.message);
    }
  }, [data, selectedOptions]);

  // Position sizes chart
  useEffect(() => {
    if (!selectedOptions.showPositionSizes || !data || !positionChartRef.current) return;

    try {
      const positionChart = createChart(positionChartRef.current, {
        width: positionChartRef.current.clientWidth,
        height: 200,
        layout: {
          background: { color: COLORS.bg },
          textColor: COLORS.text,
        },
        grid: {
          vertLines: { color: COLORS.grid },
          horzLines: { color: COLORS.grid },
        },
        rightPriceScale: {
          borderColor: '#485c7b',
        },
        timeScale: {
          borderColor: '#485c7b',
          visible: true,
        },
      });

      const positionSeries = positionChart.addHistogramSeries({
        color: '#2196F3',
        priceScaleId: 'right',
      });

      // Build position size data from trades
      const positionData = buildPositionSizeData(data);
      if (positionData.length > 0) {
        positionSeries.setData(positionData);
      }

      // Add zero line
      const zeroLine = positionChart.addLineSeries({
        color: COLORS.white,
        lineWidth: 1,
        priceScaleId: 'right',
      });
      zeroLine.setData([
        { time: positionData[0]?.time || 0, value: 0 },
        { time: positionData[positionData.length - 1]?.time || 0, value: 0 }
      ]);

      // Sync with main chart
      if (chartRef.current) {
        chartRef.current.timeScale().subscribeVisibleTimeRangeChange(timeRange => {
          positionChart.timeScale().setVisibleRange(timeRange);
        });
      }

      const handleResize = () => {
        positionChart.applyOptions({
          width: positionChartRef.current.clientWidth,
        });
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        positionChart.remove();
      };
    } catch (error) {
      console.error('Position chart error:', error);
    }
  }, [selectedOptions.showPositionSizes, data]);

  // P&L chart
  useEffect(() => {
    if (!selectedOptions.showPnl || !data || !pnlChartRef.current) return;

    try {
      const pnlChart = createChart(pnlChartRef.current, {
        width: pnlChartRef.current.clientWidth,
        height: 250,
        layout: {
          background: { color: COLORS.bg },
          textColor: COLORS.text,
        },
        grid: {
          vertLines: { color: COLORS.grid },
          horzLines: { color: COLORS.grid },
        },
        rightPriceScale: {
          borderColor: '#485c7b',
        },
        timeScale: {
          borderColor: '#485c7b',
          visible: true,
        },
      });

      const pnlSeries = pnlChart.addLineSeries({
        color: '#FFD700',
        lineWidth: 2,
      });

      const pnlAreaSeries = pnlChart.addAreaSeries({
        topColor: data.performance.cumulative_pnl?.slice(-1)[0]?.value >= 0 ? 
                  'rgba(67, 160, 71, 0.3)' : 'rgba(229, 57, 53, 0.3)',
        bottomColor: 'rgba(0, 0, 0, 0)',
        lineColor: '#FFD700',
        lineWidth: 2,
      });

      // Build cumulative P&L data
      const pnlData = buildCumulativePnlData(data);
      if (pnlData.length > 0) {
        pnlSeries.setData(pnlData);
        pnlAreaSeries.setData(pnlData);
      }

      // Add zero line
      const zeroLine = pnlChart.addLineSeries({
        color: COLORS.white,
        lineWidth: 1,
      });
      zeroLine.setData([
        { time: pnlData[0]?.time || 0, value: 0 },
        { time: pnlData[pnlData.length - 1]?.time || 0, value: 0 }
      ]);

      // Sync with main chart
      if (chartRef.current) {
        chartRef.current.timeScale().subscribeVisibleTimeRangeChange(timeRange => {
          pnlChart.timeScale().setVisibleRange(timeRange);
        });
      }

      const handleResize = () => {
        pnlChart.applyOptions({
          width: pnlChartRef.current.clientWidth,
        });
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        pnlChart.remove();
      };
    } catch (error) {
      console.error('P&L chart error:', error);
    }
  }, [selectedOptions.showPnl, data]);

  // Helper functions
  const addRegimeBackground = (chart, data) => {
    const regimeColors = selectedOptions.simplifiedRegimeColors ? 
                        REGIME_COLORS_SIMPLE : REGIME_COLORS_FULL;
    
    const markers = [];
    let currentRegime = null;
    let regimeStart = null;

    data.indicators.intelligent_chop.regime_name.forEach((point, idx) => {
      if (point.value !== currentRegime) {
        if (currentRegime !== null && regimeStart !== null) {
          markers.push({
            time: regimeStart,
            position: 'inBar',
            color: regimeColors[currentRegime] || COLORS.neutral,
            shape: 'circle',
            size: 0,
          });
        }
        currentRegime = point.value;
        regimeStart = Math.floor(point.time / 1000);
      }
    });
  };

  const addTrendBackground = (chart, data) => {
    // Implementation for trend-based background coloring
    if (!data.indicators.neurotrend?.direction) return;
    
    // This would add background coloring based on trend direction
  };

  const addMarketBiasOverlay = (chart, data) => {
    // Create custom series for market bias overlay
    // This requires custom rendering which lightweight-charts doesn't directly support
    // We'll use colored bars to simulate the effect
  };

  const addNeuroTrendEMAs = (chart, data) => {
    // Fast EMA
    const fastEMASeries = chart.addLineSeries({
      color: COLORS.bullish,
      lineWidth: 2,
      priceScaleId: 'right',
    });

    // Slow EMA
    const slowEMASeries = chart.addLineSeries({
      color: COLORS.bullish,
      lineWidth: 2,
      lineStyle: 2, // Dashed
      priceScaleId: 'right',
    });

    const fastEMAData = data.indicators.neurotrend.fast_ema
      .filter(point => point.value !== null)
      .map(point => ({
        time: Math.floor(point.time / 1000),
        value: point.value
      }));

    const slowEMAData = data.indicators.neurotrend.slow_ema
      .filter(point => point.value !== null)
      .map(point => ({
        time: Math.floor(point.time / 1000),
        value: point.value
      }));

    if (fastEMAData.length > 0) {
      fastEMASeries.setData(fastEMAData);
    }
    if (slowEMAData.length > 0) {
      slowEMASeries.setData(slowEMAData);
    }
  };

  const addTradeMarkers = (candlestickSeries, data) => {
    const markers = [];
    
    data.trades.forEach(trade => {
      // Entry marker
      markers.push({
        time: Math.floor(trade.entry_time / 1000),
        position: trade.direction === 'LONG' || trade.direction === 'long' ? 'belowBar' : 'aboveBar',
        color: trade.direction === 'LONG' || trade.direction === 'long' ? 
               TRADE_COLORS.longEntry : TRADE_COLORS.shortEntry,
        shape: trade.direction === 'LONG' || trade.direction === 'long' ? 'arrowUp' : 'arrowDown',
        text: trade.direction === 'LONG' || trade.direction === 'long' ? 'L' : 'S',
      });

      // Exit marker
      const exitColorMap = {
        'TAKE_PROFIT': TRADE_COLORS.takeProfit,
        'take_profit': TRADE_COLORS.takeProfit,
        'tp1_pullback': TRADE_COLORS.tp1Pullback,
        'STOP_LOSS': TRADE_COLORS.stopLoss,
        'stop_loss': TRADE_COLORS.stopLoss,
        'TRAILING_STOP': TRADE_COLORS.trailingStop,
        'trailing_stop': TRADE_COLORS.trailingStop,
        'SIGNAL_FLIP': TRADE_COLORS.signalFlip,
        'signal_flip': TRADE_COLORS.signalFlip,
        'END_OF_DATA': TRADE_COLORS.endOfData,
        'end_of_data': TRADE_COLORS.endOfData
      };

      markers.push({
        time: Math.floor(trade.exit_time / 1000),
        position: 'aboveBar',
        color: exitColorMap[trade.exit_reason] || TRADE_COLORS.endOfData,
        shape: 'circle',
        text: 'X',
      });
    });

    candlestickSeries.setMarkers(markers);
  };

  const addTradeLevels = (chart, data) => {
    data.trades.forEach(trade => {
      // Add TP levels
      if (trade.take_profits && trade.take_profits.length > 0) {
        const tpColors = ['#90EE90', '#3CB371', '#228B22'];
        trade.take_profits.slice(0, 3).forEach((tp, idx) => {
          if (tp !== null) {
            const tpSeries = chart.addLineSeries({
              color: tpColors[idx],
              lineWidth: 1,
              lineStyle: 3, // Dotted
              priceScaleId: 'right',
              crosshairMarkerVisible: false,
            });
            
            const entryTime = Math.floor(trade.entry_time / 1000);
            const exitTime = Math.floor(trade.exit_time / 1000);
            
            tpSeries.setData([
              { time: entryTime, value: tp },
              { time: exitTime, value: tp }
            ]);
          }
        });
      }

      // Add SL level
      if (trade.stop_loss) {
        const slSeries = chart.addLineSeries({
          color: '#FF6B6B',
          lineWidth: 1,
          lineStyle: 3, // Dotted
          priceScaleId: 'right',
          crosshairMarkerVisible: false,
        });
        
        const entryTime = Math.floor(trade.entry_time / 1000);
        const exitTime = Math.floor(trade.exit_time / 1000);
        
        slSeries.setData([
          { time: entryTime, value: trade.stop_loss },
          { time: exitTime, value: trade.stop_loss }
        ]);
      }
    });
  };

  const addPartialExitMarkers = (candlestickSeries, data) => {
    const partialMarkers = [];
    const tpExitColors = ['#90EE90', '#3CB371', '#228B22'];
    
    data.trades.forEach(trade => {
      if (trade.partial_exits && trade.partial_exits.length > 0) {
        trade.partial_exits.forEach(exit => {
          const tpLevel = exit.tp_level || 1;
          partialMarkers.push({
            time: Math.floor(exit.time / 1000),
            position: 'inBar',
            color: tpExitColors[Math.min(tpLevel - 1, 2)],
            shape: 'circle',
            text: `TP${tpLevel}`,
          });
        });
      }
    });

    if (partialMarkers.length > 0) {
      // Merge with existing markers
      const existingMarkers = candlestickSeries.markers() || [];
      candlestickSeries.setMarkers([...existingMarkers, ...partialMarkers]);
    }
  };

  const buildPositionSizeData = (data) => {
    const positionData = [];
    const ohlcTimes = data.ohlc.map(c => Math.floor(c.time / 1000));
    const positions = new Array(ohlcTimes.length).fill(0);
    
    // Build position timeline
    data.trades.forEach(trade => {
      const entryTime = Math.floor(trade.entry_time / 1000);
      const exitTime = Math.floor(trade.exit_time / 1000);
      const direction = trade.direction === 'LONG' || trade.direction === 'long' ? 1 : -1;
      const size = (trade.position_size || 1000000) / 1000000;
      
      const entryIdx = ohlcTimes.findIndex(t => t >= entryTime);
      const exitIdx = ohlcTimes.findIndex(t => t >= exitTime);
      
      if (entryIdx !== -1 && exitIdx !== -1) {
        for (let i = entryIdx; i < exitIdx && i < positions.length; i++) {
          positions[i] += size * direction;
        }
      }
    });
    
    // Convert to chart data
    positions.forEach((pos, idx) => {
      positionData.push({
        time: ohlcTimes[idx],
        value: pos,
        color: pos > 0 ? '#43A047' : pos < 0 ? '#E53935' : COLORS.grid
      });
    });
    
    return positionData;
  };

  const buildCumulativePnlData = (data) => {
    const pnlData = [];
    let cumPnl = 0;
    const ohlcTimes = data.ohlc.map(c => Math.floor(c.time / 1000));
    
    // Create time-indexed P&L events
    const pnlEvents = [];
    
    data.trades.forEach(trade => {
      // Add partial exits
      if (trade.partial_exits) {
        trade.partial_exits.forEach(exit => {
          pnlEvents.push({
            time: Math.floor(exit.time / 1000),
            pnl: exit.pnl || 0
          });
        });
      }
      
      // Add final exit
      if (trade.exit_time && trade.pnl) {
        const partialSum = trade.partial_exits ? 
          trade.partial_exits.reduce((sum, exit) => sum + (exit.pnl || 0), 0) : 0;
        const remainingPnl = trade.pnl - partialSum;
        
        if (remainingPnl !== 0) {
          pnlEvents.push({
            time: Math.floor(trade.exit_time / 1000),
            pnl: remainingPnl
          });
        }
      }
    });
    
    // Sort events by time
    pnlEvents.sort((a, b) => a.time - b.time);
    
    // Build cumulative P&L line
    let eventIdx = 0;
    ohlcTimes.forEach(time => {
      while (eventIdx < pnlEvents.length && pnlEvents[eventIdx].time <= time) {
        cumPnl += pnlEvents[eventIdx].pnl;
        eventIdx++;
      }
      pnlData.push({ time, value: cumPnl });
    });
    
    return pnlData;
  };

  // Calculate statistics
  const calculateStats = () => {
    if (!data) return {};
    
    const stats = {
      totalRows: data.ohlc.length,
      timeframe: data.metadata.timeframe,
      startDate: data.metadata.start_date,
      endDate: data.metadata.end_date,
      confidence: data.indicators.confidence?.slice(-1)[0]?.value || null
    };
    
    return stats;
  };

  const stats = calculateStats();

  return (
    <div className="enhanced-trading-chart-container">
      <div className="chart-header">
        <div className="chart-title-section">
          <h2>Production Strategy Results</h2>
          {stats.confidence && (
            <div className="confidence-badge">
              Confidence: {(stats.confidence * 100).toFixed(1)}%
            </div>
          )}
        </div>
        
        <div className="chart-info">
          <span>Symbol: {data?.metadata?.symbol}</span>
          <span className="separator">|</span>
          <span>Timeframe: {data?.metadata?.timeframe}</span>
          <span className="separator">|</span>
          <span>Period: {data?.metadata?.start_date} to {data?.metadata?.end_date}</span>
          <span className="separator">|</span>
          <span>Rows: {stats.totalRows?.toLocaleString()}</span>
        </div>
      </div>
      
      <div className="chart-controls">
        <div className="control-group">
          <h4>Background</h4>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.useChopBackground}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                useChopBackground: e.target.checked
              })}
            />
            Use Regime Colors
          </label>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.simplifiedRegimeColors}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                simplifiedRegimeColors: e.target.checked
              })}
            />
            Simplified Colors
          </label>
        </div>
        
        <div className="control-group">
          <h4>Trade Display</h4>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.showTradeMarkers}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                showTradeMarkers: e.target.checked
              })}
            />
            Trade Markers
          </label>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.showTpSlLevels}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                showTpSlLevels: e.target.checked
              })}
            />
            TP/SL Levels
          </label>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.showPartialExits}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                showPartialExits: e.target.checked
              })}
            />
            Partial Exits
          </label>
        </div>
        
        <div className="control-group">
          <h4>Additional Charts</h4>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.showPositionSizes}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                showPositionSizes: e.target.checked
              })}
            />
            Position Sizes
          </label>
          <label>
            <input
              type="checkbox"
              checked={selectedOptions.showPnl}
              onChange={(e) => setSelectedOptions({
                ...selectedOptions,
                showPnl: e.target.checked
              })}
            />
            Cumulative P&L
          </label>
        </div>
      </div>

      {chartError ? (
        <div className="error-container">
          <p>Chart Error: {chartError}</p>
        </div>
      ) : (
        <>
          <div ref={chartContainerRef} className="main-chart-container" />
          {selectedOptions.showPositionSizes && (
            <div className="subchart-wrapper">
              <h3>Position Sizes (M)</h3>
              <div ref={positionChartRef} className="position-chart-container" />
            </div>
          )}
          {selectedOptions.showPnl && (
            <div className="subchart-wrapper">
              <h3>Cumulative P&L ($)</h3>
              <div ref={pnlChartRef} className="pnl-chart-container" />
              {data?.metrics && (
                <div className="pnl-stats">
                  <span className={data.metrics.total_pnl >= 0 ? 'positive' : 'negative'}>
                    Final P&L: ${data.metrics.total_pnl?.toFixed(2) || '0.00'}
                  </span>
                </div>
              )}
            </div>
          )}
        </>
      )}
      
      {data?.metrics && (
        <div className="performance-metrics">
          <div className="metrics-row">
            <div className="metric">
              <span className="metric-label">Win Rate:</span>
              <span className="metric-value">{(data.metrics.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Sharpe:</span>
              <span className="metric-value">{data.metrics.sharpe_ratio?.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">PF:</span>
              <span className="metric-value">{data.metrics.profit_factor?.toFixed(2)}</span>
            </div>
          </div>
          <div className="metrics-row">
            <div className="metric">
              <span className="metric-label">P&L:</span>
              <span className="metric-value">${data.metrics.total_pnl?.toLocaleString()}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Return:</span>
              <span className="metric-value">{(data.metrics.total_return * 100).toFixed(2)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">DD:</span>
              <span className="metric-value">{(data.metrics.max_drawdown * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      )}
      
      <div className="chart-legend">
        <div className="legend-section">
          <h4>Trend Indicators</h4>
          <div className="legend-item">
            <div className="legend-line bullish"></div>
            <span>Uptrend</span>
          </div>
          <div className="legend-item">
            <div className="legend-line bearish"></div>
            <span>Downtrend</span>
          </div>
        </div>
        
        <div className="legend-section">
          <h4>Trade Markers</h4>
          <div className="legend-item">
            <div className="legend-marker long-entry">▲</div>
            <span>Long Entry</span>
          </div>
          <div className="legend-item">
            <div className="legend-marker short-entry">▼</div>
            <span>Short Entry</span>
          </div>
          <div className="legend-item">
            <div className="legend-marker take-profit">✕</div>
            <span>Take Profit</span>
          </div>
          <div className="legend-item">
            <div className="legend-marker stop-loss">✕</div>
            <span>Stop Loss</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedTradingChart;