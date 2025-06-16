import backtrader as bt
import backtrader.indicators as btind


class ADXTrendStrategy(bt.Strategy):
    """
    ADX-based trend-following scalping strategy.
    
    This strategy uses:
    - DMI/ADX to identify strong trends
    - Williams %R to find pullback entry points
    - SMA for dynamic stop-loss
    - Recent highs/lows for take-profit targets
    """
    
    params = (
        ('adx_period', 14),          # ADX calculation period
        ('adx_threshold', 50),        # Minimum ADX value for strong trend
        ('williams_period', 14),      # Williams %R period
        ('williams_oversold', -80),   # Oversold threshold
        ('williams_overbought', -20), # Overbought threshold
        ('sma_period', 50),          # SMA period for stop-loss
        ('tp_lookback', 30),         # Lookback period for take-profit
        ('risk_percent', 0.03),      # Risk 3% per trade
        ('printlog', True),          # Print log messages
    )
    
    def __init__(self):
        # Price reference
        self.dataclose = self.datas[0].close
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        
        # Indicators
        self.dmi = btind.DirectionalMovementIndex(
            self.datas[0], 
            period=self.params.adx_period
        )
        self.adx = self.dmi.adx
        self.plus_di = self.dmi.plusDI
        self.minus_di = self.dmi.minusDI
        
        self.williams = btind.WilliamsR(
            self.datas[0],
            period=self.params.williams_period
        )
        
        self.sma = btind.SimpleMovingAverage(
            self.datas[0],
            period=self.params.sma_period
        )
        
        # Track highest high and lowest low
        self.highest_high = btind.Highest(
            self.datahigh,
            period=self.params.tp_lookback
        )
        self.lowest_low = btind.Lowest(
            self.datalow,
            period=self.params.tp_lookback
        )
        
        # Order tracking
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.stop_order = None
        self.limit_order = None
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                    f'Cost: {order.executed.value:.2f}, '
                    f'Comm: {order.executed.comm:.2f}'
                )
                
            self.bar_executed = len(self)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            
        # Reset order reference
        if order == self.order:
            self.order = None
        elif order == self.stop_order:
            self.stop_order = None
        elif order == self.limit_order:
            self.limit_order = None
            
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
            
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
    def calculate_position_size(self, stop_distance):
        """Calculate position size based on risk management rules."""
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.params.risk_percent
        
        # Calculate position size based on stop distance
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            
            # Limit position size to avoid over-leveraging
            max_position = account_value / self.dataclose[0] * 0.9  # Max 90% of account
            position_size = min(position_size, max_position)
            
            # Ensure minimum position size
            if position_size < 1:
                return 0
                
            return int(position_size)
        return 0
        
    def check_long_entry(self):
        """Check if long entry conditions are met."""
        return (
            self.plus_di[0] > self.minus_di[0] and  # DI+ > DI-
            self.adx[0] > self.params.adx_threshold and  # ADX > 50
            self.williams[0] < self.params.williams_oversold  # Williams %R < -80
        )
        
    def check_short_entry(self):
        """Check if short entry conditions are met."""
        return (
            self.minus_di[0] > self.plus_di[0] and  # DI- > DI+
            self.adx[0] > self.params.adx_threshold and  # ADX > 50
            self.williams[0] > self.params.williams_overbought  # Williams %R > -20
        )
        
    def next(self):
        # Log current bar info
        self.log(f'Close: {self.dataclose[0]:.2f}')
        
        # Check if we have pending orders
        if self.order:
            return
            
        # Check position invalidation
        if self.position:
            if self.position.size > 0:  # Long position
                if self.check_short_entry():
                    self.log('LONG POSITION INVALIDATED - Closing')
                    self.close()
                    # Cancel existing stop/limit orders
                    if self.stop_order:
                        self.cancel(self.stop_order)
                    if self.limit_order:
                        self.cancel(self.limit_order)
                    return
            elif self.position.size < 0:  # Short position
                if self.check_long_entry():
                    self.log('SHORT POSITION INVALIDATED - Closing')
                    self.close()
                    # Cancel existing stop/limit orders
                    if self.stop_order:
                        self.cancel(self.stop_order)
                    if self.limit_order:
                        self.cancel(self.limit_order)
                    return
                    
        # Entry logic - only if not in position
        if not self.position:
            # Check for long entry
            if self.check_long_entry():
                # Calculate position size
                stop_price = self.sma[0]
                stop_distance = self.dataclose[0] - stop_price
                
                if stop_distance > 0:
                    size = self.calculate_position_size(stop_distance)
                    if size > 0:
                        self.log(f'LONG ENTRY SIGNAL - Size: {size}')
                        self.order = self.buy(size=size)
                        
            # Check for short entry
            elif self.check_short_entry():
                # Calculate position size
                stop_price = self.sma[0]
                stop_distance = stop_price - self.dataclose[0]
                
                if stop_distance > 0:
                    size = self.calculate_position_size(stop_distance)
                    if size > 0:
                        self.log(f'SHORT ENTRY SIGNAL - Size: {size}')
                        self.order = self.sell(size=size)
                        
        # Exit management - set stop loss and take profit after entry
        else:
            if not self.stop_order and not self.limit_order:
                if self.position.size > 0:  # Long position
                    # Stop loss at SMA
                    self.stop_order = self.sell(
                        exectype=bt.Order.Stop,
                        price=self.sma[0],
                        size=self.position.size
                    )
                    # Take profit at highest high
                    self.limit_order = self.sell(
                        exectype=bt.Order.Limit,
                        price=self.highest_high[0],
                        size=self.position.size
                    )
                    self.log(f'LONG STOPS SET - SL: {self.sma[0]:.2f}, TP: {self.highest_high[0]:.2f}')
                    
                elif self.position.size < 0:  # Short position
                    # Stop loss at SMA
                    self.stop_order = self.buy(
                        exectype=bt.Order.Stop,
                        price=self.sma[0],
                        size=abs(self.position.size)
                    )
                    # Take profit at lowest low
                    self.limit_order = self.buy(
                        exectype=bt.Order.Limit,
                        price=self.lowest_low[0],
                        size=abs(self.position.size)
                    )
                    self.log(f'SHORT STOPS SET - SL: {self.sma[0]:.2f}, TP: {self.lowest_low[0]:.2f}')
                    
    def log(self, txt, dt=None):
        """Logging function for strategy."""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
            
    def stop(self):
        """Called when strategy stops."""
        self.log(f'Strategy Ending Value: {self.broker.getvalue():.2f}', dt=None)