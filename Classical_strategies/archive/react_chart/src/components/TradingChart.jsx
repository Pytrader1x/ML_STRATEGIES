import React, { useEffect, useRef, useState } from 'react';
import { createChart } from 'lightweight-charts';
import './TradingChart.css';

const TradingChart = ({ data }) => {
  const chartContainerRef = useRef();
  const chartRef = useRef();
  const positionChartRef = useRef();
  const [selectedIndicators, setSelectedIndicators] = useState({
    neurotrend: true,
    marketBias: true,
    trades: true,
    positionSizes: false,
    pnl: false
  });
  const [chartError, setChartError] = useState(null);

  useEffect(() => {
    if (!data || !chartContainerRef.current) return;

    try {
      setChartError(null);

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 600,
      layout: {
        background: { color: '#131722' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#363c4e' },
        horzLines: { color: '#363c4e' },
      },
      rightPriceScale: {
        borderColor: '#485c7b',
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
      upColor: '#26A69A',
      downColor: '#EF5350',
      borderUpColor: '#26A69A',
      borderDownColor: '#EF5350',
      wickUpColor: '#26A69A',
      wickDownColor: '#EF5350',
    });

    // Convert timestamp to seconds for lightweight-charts
    const ohlcData = data.ohlc
      .filter(candle => candle.open !== null && candle.high !== null && candle.low !== null && candle.close !== null)
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

    // Add NeuroTrend EMAs if enabled
    if (selectedIndicators.neurotrend && data.indicators.neurotrend.fast_ema.length > 0) {
      const fastEMASeries = chart.addLineSeries({
        color: '#2196F3',
        lineWidth: 2,
        title: 'Fast EMA',
      });

      const slowEMASeries = chart.addLineSeries({
        color: '#FF9800',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        title: 'Slow EMA',
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
    }

    // Add trade markers
    if (selectedIndicators.trades && data.trades.length > 0) {
      const markers = [];
      
      data.trades.forEach(trade => {
        // Entry marker
        markers.push({
          time: Math.floor(trade.entry_time / 1000),
          position: 'belowBar',
          color: trade.direction === 'LONG' ? '#1E88E5' : '#FFD600',
          shape: 'arrowUp',
          text: trade.direction === 'LONG' ? 'L' : 'S',
        });

        // Exit marker
        const exitColor = {
          'TAKE_PROFIT': '#43A047',
          'STOP_LOSS': '#E53935',
          'TRAILING_STOP': '#9C27B0',
          'SIGNAL_FLIP': '#FF8F00',
          'END_OF_DATA': '#9E9E9E'
        };

        markers.push({
          time: Math.floor(trade.exit_time / 1000),
          position: 'aboveBar',
          color: exitColor[trade.exit_reason] || '#9E9E9E',
          shape: 'arrowDown',
          text: 'X',
        });
      });

      candlestickSeries.setMarkers(markers);
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
  }, [data, selectedIndicators]);

  // Add position size chart if enabled
  useEffect(() => {
    if (!selectedIndicators.positionSizes || !data || !positionChartRef.current) return;

    try {
      const positionChart = createChart(positionChartRef.current, {
        width: positionChartRef.current.clientWidth,
        height: 200,
        layout: {
          background: { color: '#131722' },
          textColor: '#d1d4dc',
        },
        grid: {
          vertLines: { color: '#363c4e' },
          horzLines: { color: '#363c4e' },
        },
      });

      const positionSeries = positionChart.addHistogramSeries({
        color: '#2196F3',
        title: 'Position Size',
      });

      const positionData = data.performance.position_sizes
        .filter(point => point.value !== null)
        .map(point => ({
          time: Math.floor(point.time / 1000),
          value: point.value
        }));

      if (positionData.length > 0) {
        positionSeries.setData(positionData);
      }

      // Sync with main chart
      if (chartRef.current) {
        chartRef.current.timeScale().subscribeVisibleTimeRangeChange(timeRange => {
          positionChart.timeScale().setVisibleRange(timeRange);
        });
      }

      // Handle resize
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
  }, [selectedIndicators.positionSizes, data]);

  return (
    <div className="trading-chart-container">
      <div className="chart-header">
        <h2>Trading Strategy Chart</h2>
        <div className="chart-info">
          <span>Symbol: {data?.metadata?.symbol}</span>
          <span>Timeframe: {data?.metadata?.timeframe}</span>
          <span>Period: {data?.metadata?.start_date} to {data?.metadata?.end_date}</span>
        </div>
      </div>
      
      <div className="indicator-controls">
        <label>
          <input
            type="checkbox"
            checked={selectedIndicators.neurotrend}
            onChange={(e) => setSelectedIndicators({
              ...selectedIndicators,
              neurotrend: e.target.checked
            })}
          />
          NeuroTrend EMAs
        </label>
        <label>
          <input
            type="checkbox"
            checked={selectedIndicators.trades}
            onChange={(e) => setSelectedIndicators({
              ...selectedIndicators,
              trades: e.target.checked
            })}
          />
          Trade Markers
        </label>
        <label>
          <input
            type="checkbox"
            checked={selectedIndicators.positionSizes}
            onChange={(e) => setSelectedIndicators({
              ...selectedIndicators,
              positionSizes: e.target.checked
            })}
          />
          Position Sizes
        </label>
        <label>
          <input
            type="checkbox"
            checked={selectedIndicators.pnl}
            onChange={(e) => setSelectedIndicators({
              ...selectedIndicators,
              pnl: e.target.checked
            })}
          />
          P&L Curve
        </label>
      </div>

      {chartError ? (
        <div className="error-container">
          <p>Chart Error: {chartError}</p>
        </div>
      ) : (
        <>
          <div ref={chartContainerRef} className="chart-container" />
          {selectedIndicators.positionSizes && (
            <div ref={positionChartRef} className="chart-container position-chart" />
          )}
        </>
      )}
      
      {data?.metrics && (
        <div className="performance-metrics">
          <h3>Performance Metrics</h3>
          <div className="metrics-grid">
            <div className="metric">
              <span className="metric-label">Total Trades:</span>
              <span className="metric-value">{data.metrics.total_trades}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Win Rate:</span>
              <span className="metric-value">{(data.metrics.win_rate * 100).toFixed(2)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Sharpe Ratio:</span>
              <span className="metric-value">{data.metrics.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div className="metric">
              <span className="metric-label">Max Drawdown:</span>
              <span className="metric-value">{(data.metrics.max_drawdown * 100).toFixed(2)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Total Return:</span>
              <span className="metric-value">{(data.metrics.total_return * 100).toFixed(2)}%</span>
            </div>
            <div className="metric">
              <span className="metric-label">Profit Factor:</span>
              <span className="metric-value">{data.metrics.profit_factor.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingChart;