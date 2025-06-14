"""
Enhanced check_exit_conditions to handle multiple TP hits in same candle

This is a proposed enhancement to allow multiple TP exits within the same candle.
"""

def check_exit_conditions_enhanced(self, row: pd.Series, trade: Trade, current_time: pd.Timestamp) -> List[Tuple[bool, ExitReason, float]]:
    """
    Enhanced version that returns a list of all exit conditions hit in this candle
    
    Returns:
        List of tuples: (should_exit, exit_reason, exit_percent)
    """
    exits = []
    
    # Get current price data
    high = row['High']
    low = row['Low']
    close = row['Close']
    
    # Check all TP levels that haven't been hit yet
    if trade.direction == TradeDirection.LONG:
        # For long trades, check if high touched any TP levels
        for i, tp in enumerate(trade.take_profits):
            if tp is not None and i + 1 > trade.tp_hits:  # Only check unhit TPs
                if high >= tp:
                    exits.append((True, getattr(ExitReason, f'TAKE_PROFIT_{i+1}'), 1.0))
                    
        # Check stop loss
        if low <= trade.stop_loss:
            # If SL is hit, it overrides all TP exits (worst case)
            return [(True, ExitReason.STOP_LOSS, 1.0)]
            
        # Check trailing stop
        if trade.trailing_stop is not None and low <= trade.trailing_stop:
            # If TSL is hit after TPs, we need to determine order
            # For simplicity, process TPs first, then TSL on remaining
            if not exits:  # No TP hits
                return [(True, ExitReason.TRAILING_STOP, 1.0)]
                
    else:  # SHORT trade
        # For short trades, check if low touched any TP levels
        for i, tp in enumerate(trade.take_profits):
            if tp is not None and i + 1 > trade.tp_hits:
                if low <= tp:
                    exits.append((True, getattr(ExitReason, f'TAKE_PROFIT_{i+1}'), 1.0))
                    
        # Check stop loss
        if high >= trade.stop_loss:
            return [(True, ExitReason.STOP_LOSS, 1.0)]
            
        # Check trailing stop
        if trade.trailing_stop is not None and high >= trade.trailing_stop:
            if not exits:
                return [(True, ExitReason.TRAILING_STOP, 1.0)]
    
    # If we have multiple TP exits, sort them in order (TP1, TP2, TP3)
    if exits:
        exits.sort(key=lambda x: x[1].value)
        return exits
    
    # Check other exit conditions (signal flip, etc.)
    # ... rest of original exit logic ...
    
    return []  # No exits


def enhanced_backtest_loop_section(self):
    """
    Enhanced section of the backtest loop that handles multiple exits per candle
    """
    # ... earlier code ...
    
    # Check exit conditions - now returns a list
    exit_conditions = self.signal_generator.check_exit_conditions_enhanced(
        current_row, self.current_trade, current_time
    )
    
    # Process each exit in order
    for should_exit, exit_reason, exit_percent in exit_conditions:
        if not should_exit:
            continue
            
        # Check if trade still has remaining position
        if self.current_trade.remaining_size <= 0:
            break
            
        if self.config.debug_decisions:
            print(f"  🚪 EXIT SIGNAL {exit_reason.value}: {exit_percent*100:.0f}% of position")
        
        # Determine exit price based on exit reason
        exit_price = self._get_exit_price(current_row, self.current_trade, exit_reason)
        
        # Execute exit
        if 'take_profit' in exit_reason.value:
            completed_trade = self._execute_full_exit(
                self.current_trade, current_time, exit_price, exit_reason
            )
        elif exit_percent < 1.0:
            completed_trade = self._execute_partial_exit(
                self.current_trade, current_time, exit_price, 
                exit_percent, exit_reason
            )
        else:
            completed_trade = self._execute_full_exit(
                self.current_trade, current_time, exit_price, exit_reason
            )
        
        # If trade is completed, break the loop
        if completed_trade is not None:
            if self.config.debug_decisions:
                print(f"  🏁 TRADE COMPLETED: Final P&L ${completed_trade.pnl:,.0f}")
            self.trades.append(self.current_trade)
            self.current_trade = None
            break
        else:
            # Trade continues with reduced position
            if self.config.debug_decisions:
                print(f"  ↪️  Continuing with {self.current_trade.remaining_size/1e6:.2f}M remaining")


# Example of how to determine exit order within a candle
def determine_exit_order(high, low, entry_price, tp_levels, sl_level, direction):
    """
    Try to determine the most likely order of exits within a candle
    
    This is a simplified approach. In reality, we don't know the exact
    price path within a candle, so we make assumptions:
    1. If close is near high, assume upward movement (hit TPs before SL for longs)
    2. If close is near low, assume downward movement (hit SL before TPs for longs)
    3. Otherwise, use a conservative approach (SL before TPs)
    """
    # ... implementation ...
    pass