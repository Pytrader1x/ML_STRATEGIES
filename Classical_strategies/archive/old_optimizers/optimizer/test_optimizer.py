#!/usr/bin/env python3
"""
Test script for the intelligent optimizer
"""

import sys
import os
from intelligent_optimizer import run_optimization

def main():
    """Run optimization test"""
    print("\n🚀 TESTING INTELLIGENT OPTIMIZER")
    print("="*60)
    
    # Test Strategy 1 (Ultra-Tight Risk) with Bayesian optimization
    print("\n1️⃣ Testing Strategy 1 - Ultra-Tight Risk Management")
    print("   Using Bayesian Optimization")
    print("-"*60)
    
    try:
        optimizer1, best_result1 = run_optimization(
            strategy_type=1,
            optimization_method='bayesian',
            currency='AUDUSD',
            n_iterations=15,  # Reduced for testing
            sample_size=3000  # Reduced for speed
        )
        
        print("\n✅ Strategy 1 optimization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error optimizing Strategy 1: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test Strategy 2 (Scalping) with Grid Search
    print("\n\n2️⃣ Testing Strategy 2 - Scalping Strategy")
    print("   Using Grid Search Optimization")
    print("-"*60)
    
    try:
        optimizer2, best_result2 = run_optimization(
            strategy_type=2,
            optimization_method='grid',
            currency='AUDUSD',
            n_iterations=20,  # Grid search needs more iterations
            sample_size=3000  # Reduced for speed
        )
        
        print("\n✅ Strategy 2 optimization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error optimizing Strategy 2: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()