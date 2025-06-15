#!/usr/bin/env python3
"""
Debug script to analyze TP vs TSL exit dynamics
"""

import pandas as pd
import numpy as np
from strategy_code.Prod_strategy import OptimizedProdStrategy, OptimizedStrategyConfig
from technical_indicators_custom import TIC
import warnings

warnings.filterwarnings('ignore')

def analyze_tp_vs_tsl_config():
    """Analyze why TP exits aren't happening"""
    
    print("🔍 DEBUGGING TP vs TSL EXIT DYNAMICS")
    print("="*60)
    
    # Current configurations
    configs = {
        "Ultra-Tight Risk": {
            "tp_atr_multipliers": (0.2, 0.3, 0.5),
            "tsl_activation_pips": 3,
            "trailing_atr_multiplier": 0.8,
            "sl_atr_multiplier": 1.0,
            "max_tp_percent": 0.003
        },
        "Scalping": {
            "tp_atr_multipliers": (0.1, 0.2, 0.3),
            "tsl_activation_pips": 2,
            "trailing_atr_multiplier": 0.5,
            "sl_atr_multiplier": 0.5,
            "max_tp_percent": 0.002
        }
    }
    
    # Typical ATR values for AUDUSD (from experience)
    typical_atr = 0.0056  # ~56 pips
    typical_price = 0.6500  # Example AUDUSD price
    
    for config_name, config in configs.items():
        print(f"\n📊 {config_name} Configuration Analysis:")
        print("-" * 40)
        
        # Calculate actual distances
        atr_pips = typical_atr / 0.0001  # Convert to pips
        
        # TP distances
        tp1_atr_dist = config["tp_atr_multipliers"][0] * atr_pips
        tp2_atr_dist = config["tp_atr_multipliers"][1] * atr_pips
        tp3_atr_dist = config["tp_atr_multipliers"][2] * atr_pips
        
        # Max TP constraint
        max_tp_price_dist = config["max_tp_percent"] * typical_price
        max_tp_pips = max_tp_price_dist / 0.0001
        
        # Actual TP distances (limited by max_tp_percent)
        tp1_actual = min(tp1_atr_dist, max_tp_pips)
        tp2_actual = min(tp2_atr_dist, max_tp_pips)
        tp3_actual = min(tp3_atr_dist, max_tp_pips)
        
        # TSL parameters
        tsl_activation = config["tsl_activation_pips"]
        tsl_trail_dist = config["trailing_atr_multiplier"] * atr_pips
        
        # SL distance
        sl_dist = config["sl_atr_multiplier"] * atr_pips
        
        print(f"ATR: {atr_pips:.1f} pips")
        print(f"SL Distance: {sl_dist:.1f} pips")
        print(f"")
        print(f"TP Distances (before max constraint):")
        print(f"  TP1: {tp1_atr_dist:.1f} pips")
        print(f"  TP2: {tp2_atr_dist:.1f} pips") 
        print(f"  TP3: {tp3_atr_dist:.1f} pips")
        print(f"")
        print(f"Max TP Constraint: {max_tp_pips:.1f} pips")
        print(f"")
        print(f"Actual TP Distances (after constraint):")
        print(f"  TP1: {tp1_actual:.1f} pips")
        print(f"  TP2: {tp2_actual:.1f} pips")
        print(f"  TP3: {tp3_actual:.1f} pips")
        print(f"")
        print(f"TSL Settings:")
        print(f"  Activation: {tsl_activation} pips")
        print(f"  Trail Distance: {tsl_trail_dist:.1f} pips")
        print(f"")
        
        # Analysis
        print("🎯 ANALYSIS:")
        if tsl_activation < tp1_actual:
            print(f"  ⚠️  TSL activates at {tsl_activation} pips, but TP1 is at {tp1_actual:.1f} pips")
            print(f"  ⚠️  TSL will activate BEFORE any TP can be reached!")
            
            gap = tp1_actual - tsl_activation
            print(f"  📏 Gap between TSL activation and TP1: {gap:.1f} pips")
            
            if tsl_trail_dist < gap:
                print(f"  ❌ TSL trail distance ({tsl_trail_dist:.1f} pips) < gap ({gap:.1f} pips)")
                print(f"  ❌ ANY retracement >{tsl_trail_dist:.1f} pips will trigger TSL before TP1!")
            else:
                print(f"  ✅ TSL trail distance ({tsl_trail_dist:.1f} pips) > gap ({gap:.1f} pips)")
                print(f"  ✅ TP1 might be reachable if price moves smoothly")
        else:
            print(f"  ✅ TSL activates at {tsl_activation} pips, TP1 at {tp1_actual:.1f} pips")
            print(f"  ✅ TP1 can be reached before TSL activation")
        
        print("")

def propose_fixes():
    """Propose configuration fixes"""
    
    print("\n🔧 PROPOSED FIXES")
    print("="*60)
    
    print("Option 1: DELAY TSL ACTIVATION (Recommended)")
    print("-" * 50)
    print("Ultra-Tight Risk Management:")
    print("  tsl_activation_pips=15,     # From 3 → 15")
    print("  trailing_atr_multiplier=1.2 # From 0.8 → 1.2")
    print("")
    print("Scalping Strategy:")
    print("  tsl_activation_pips=8,      # From 2 → 8") 
    print("  trailing_atr_multiplier=0.8 # From 0.5 → 0.8")
    
    print("\nOption 2: TIGHTER TP LEVELS")
    print("-" * 50)
    print("Ultra-Tight Risk Management:")
    print("  tp_atr_multipliers=(0.05, 0.1, 0.15) # From (0.2, 0.3, 0.5)")
    print("")
    print("Scalping Strategy:")
    print("  tp_atr_multipliers=(0.03, 0.05, 0.08) # From (0.1, 0.2, 0.3)")
    
    print("\nOption 3: INCREASE MAX TP CONSTRAINT")
    print("-" * 50)
    print("Ultra-Tight Risk Management:")
    print("  max_tp_percent=0.005,  # From 0.003 → 0.005")
    print("")
    print("Scalping Strategy:")
    print("  max_tp_percent=0.004,  # From 0.002 → 0.004")

if __name__ == "__main__":
    analyze_tp_vs_tsl_config()
    propose_fixes()