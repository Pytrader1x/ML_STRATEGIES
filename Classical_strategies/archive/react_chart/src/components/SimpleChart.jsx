import React, { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

const SimpleChart = () => {
  const chartContainerRef = useRef();

  useEffect(() => {
    const chart = createChart(chartContainerRef.current, {
      width: 800,
      height: 400,
      layout: {
        background: { color: '#131722' },
        textColor: '#d1d4dc',
      },
    });

    const candleSeries = chart.addCandleSeries();
    
    const data = [
      { time: '2024-01-01', open: 100, high: 102, low: 99, close: 101 },
      { time: '2024-01-02', open: 101, high: 103, low: 100, close: 102 },
      { time: '2024-01-03', open: 102, high: 104, low: 101, close: 103 },
      { time: '2024-01-04', open: 103, high: 105, low: 102, close: 104 },
      { time: '2024-01-05', open: 104, high: 106, low: 103, close: 105 },
    ];
    
    candleSeries.setData(data);

    return () => {
      chart.remove();
    };
  }, []);

  return <div ref={chartContainerRef} />;
};

export default SimpleChart;