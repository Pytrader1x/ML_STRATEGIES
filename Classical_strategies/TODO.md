# TODO List - Classical Trading Strategies

## Completed Tasks ✅

### June 15, 2025

1. **[DONE] Fix TP2 marker visibility issue**
   - TP2 markers were not showing when TP2 and TP3 occurred at same timestamp
   - Fixed duplicate detection logic in Prod_plotting.py

2. **[DONE] Validate P&L calculation integrity**
   - Created comprehensive validation scripts
   - Confirmed no double counting or P&L inflation
   - Fixed missing PartialExit records for stop loss exits

3. **[DONE] Fix final exit marker display**
   - Fixed 0M position display issue
   - Fixed $0 P&L display for final exits
   - Fixed negative total P&L display (sign formatting)
   - Added proper TP1 pullback label display

4. **[DONE] Fix total P&L calculation**
   - Added missing 'pnl' field to trade dict conversion
   - Total P&L now displays correctly (e.g., +1.2k instead of +1.4k)

5. **[DONE] Fix TP1 pullback logic**
   - Prevented TP1 pullback from triggering in same candle as TP exit
   - Ensures proper intra-candle price movement handling

6. **[DONE] Clean up debug files**
   - Removed temporary test files
   - Organized remaining files
   - Updated documentation

7. **[DONE] Add intrabar stop-loss feature**
   - Added `intrabar_stop_on_touch` configuration parameter
   - Implemented logic to check high/low for stop loss touches
   - Maintains backward compatibility with default close-only behavior

## Future Enhancements 🚀

### High Priority
- [ ] Add support for multiple currency pairs in single backtest
- [ ] Implement real-time trading interface
- [ ] Add machine learning optimization for parameters
- [ ] Create web-based dashboard for results visualization

### Medium Priority
- [ ] Add more sophisticated risk management (Kelly Criterion)
- [ ] Implement correlation-based position sizing
- [ ] Add market regime detection for dynamic parameters
- [ ] Create automated parameter optimization framework

### Low Priority
- [ ] Add more chart customization options
- [ ] Implement additional exit strategies
- [ ] Create performance attribution analysis
- [ ] Add transaction cost modeling

## Known Issues 🐛

- None currently reported

## Notes 📝

- All P&L calculations have been validated and are working correctly
- Position tracking through partial exits is accurate
- Chart displays show correct values for all exit types
- TP exit priority is properly handled (higher TPs take precedence)