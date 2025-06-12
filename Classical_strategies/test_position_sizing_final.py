"""
Final test of position sizing with different scenarios
"""

from strategy_code.Prod_strategy import OptimizedStrategyConfig, RiskManager

def test_position_sizing_scenarios():
    """Test position sizing with various scenarios"""
    
    print("FINAL POSITION SIZING TEST")
    print("="*40)
    
    # Create config with DISABLED intelligent sizing to avoid complications
    config = OptimizedStrategyConfig(
        initial_capital=100_000,
        risk_per_trade=0.02,  # 2% risk
        min_lot_size=1_000_000,
        pip_value_per_million=100,
        intelligent_sizing=False,  # DISABLE to test base calculation
        relaxed_mode=False
    )
    
    risk_manager = RiskManager(config)
    
    test_scenarios = [
        # entry, stop_loss, expected_risk_pct, description
        (1.0000, 0.9980, 2.0, "20 pip stop"),
        (0.7500, 0.7450, 2.0, "50 pip stop"), 
        (1.2000, 1.2010, 2.0, "10 pip stop"),
        (0.8000, 0.7990, 2.0, "10 pip stop different price"),
        (1.1000, 1.0950, 2.0, "50 pip stop different price"),
    ]
    
    all_correct = True
    
    for entry, sl, expected_risk_pct, desc in test_scenarios:
        position_size = risk_manager.calculate_position_size(entry, sl, 100_000)
        
        # Calculate actual risk
        sl_distance_pips = abs(entry - sl) / 0.0001
        actual_risk = (sl_distance_pips * config.pip_value_per_million * position_size / config.min_lot_size)
        actual_risk_pct = actual_risk / 100_000 * 100
        
        print(f"\\nTest: {desc}")
        print(f"  Entry: {entry}, SL: {sl}")
        print(f"  Stop distance: {sl_distance_pips:.1f} pips")
        print(f"  Position size: {position_size:,.0f} units ({position_size/1_000_000:.1f}M)")
        print(f"  Actual risk: ${actual_risk:.0f} ({actual_risk_pct:.2f}%)")
        print(f"  Target risk: {expected_risk_pct:.1f}%")
        
        # Check if risk is within acceptable range (±0.2% due to rounding)
        if abs(actual_risk_pct - expected_risk_pct) <= 0.2:
            print(f"  ✅ PASS - Risk within acceptable range")
        else:
            print(f"  ❌ FAIL - Risk outside acceptable range")
            all_correct = False
    
    print(f"\\nOVERALL RESULT: {'✅ ALL TESTS PASSED' if all_correct else '❌ SOME TESTS FAILED'}")
    return all_correct

if __name__ == "__main__":
    test_position_sizing_scenarios()