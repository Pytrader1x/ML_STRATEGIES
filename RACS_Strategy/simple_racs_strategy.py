"""
Simplified RACS Strategy for Fast Backtesting

This is a streamlined version focused on speed and reliability.
"""

import backtrader as bt
import numpy as np


class SimpleRACSStrategy(bt.Strategy):
    """Simplified RACS Strategy"""
    
    params = (
        ('risk_pct', 0.01),
        ('confidence_threshold', 60),
        ('slope_threshold', 20),
        ('stop_loss_atr', 1.0),
        ('take_profit_atr', 2.0),
        ('max_positions', 2),
    )
    
    def __init__(self):
        # Track basic metrics
        self.order = None
        self.trades = 0
        
        # Access data columns by index in the dataframe
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        
        # We'll use simple indicators
        self.atr = bt.indicators.ATR(self.datas[0], period=14)
        self.sma_fast = bt.indicators.SMA(self.datas[0], period=10)
        self.sma_slow = bt.indicators.SMA(self.datas[0], period=50)
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            self.trades += 1
            
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
    
    def next(self):
        # Skip if we have pending orders
        if self.order:
            return
        
        # Check position count
        if len(self.getpositions()) >= self.p.max_positions:
            return
        
        # Get current position
        position = self.position
        
        # Simple regime detection using moving averages and RSI
        trend_up = self.sma_fast[0] > self.sma_slow[0]
        trend_strength = abs(self.sma_fast[0] - self.sma_slow[0]) / self.dataclose[0] * 100
        
        # Exit logic
        if position:
            if position.size > 0:  # Long position
                # Exit if trend reverses or stop hit
                if not trend_up or self.dataclose[0] < position.price - (self.p.stop_loss_atr * self.atr[0]):
                    self.close()
            else:  # Short position
                if trend_up or self.dataclose[0] > position.price + (self.p.stop_loss_atr * self.atr[0]):
                    self.close()
            return
        
        # Entry logic - simplified
        if trend_strength > self.p.slope_threshold / 10:  # Scale slope threshold
            
            # Additional filters
            if self.rsi[0] < 30 and trend_up:  # Oversold in uptrend
                # Calculate position size
                risk_amount = self.broker.getvalue() * self.p.risk_pct
                stop_distance = self.p.stop_loss_atr * self.atr[0]
                size = risk_amount / stop_distance if stop_distance > 0 else 0
                
                if size > 0:
                    self.order = self.buy(size=size)
                    
            elif self.rsi[0] > 70 and not trend_up:  # Overbought in downtrend
                risk_amount = self.broker.getvalue() * self.p.risk_pct
                stop_distance = self.p.stop_loss_atr * self.atr[0]
                size = risk_amount / stop_distance if stop_distance > 0 else 0
                
                if size > 0:
                    self.order = self.sell(size=size)