#!/usr/bin/env python3
"""
Direct optimizer runner that finds optimal parameters
Focuses on achieving Sharpe > 1.0 with minimal iterations
"""

import sys
sys.path.insert(0, 'optimizer')
from intelligent_optimizer import run_optimization

def main():
    """Run optimization with focused parameters"""
    
    print("\n🚀 RUNNING FOCUSED PARAMETER OPTIMIZATION")
    print("="*60)
    
    # Run optimization for Strategy 1
    print("\n📊 Optimizing Strategy 1 - Ultra-Tight Risk Management")
    print("   Goal: Achieve Sharpe > 1.0")
    print("-"*60)
    
    try:
        # Run with moderate iterations for faster results
        optimizer1, best_result1 = run_optimization(
            strategy_type=1,
            optimization_method='bayesian',
            currency='AUDUSD',
            n_iterations=20,  # Focused search
            sample_size=3500,  # Balanced for speed vs accuracy
            use_previous_results=True
        )
        
        print(f"\n✅ Strategy 1 optimization completed!")
        if best_result1:
            print(f"   Best Sharpe: {best_result1.sharpe_ratio:.3f}")
            print(f"   Best Fitness: {best_result1.fitness:.3f}")
            
            if best_result1.sharpe_ratio >= 1.0:
                print(f"\n🎉 TARGET ACHIEVED! Sharpe ratio > 1.0")
            else:
                print(f"\n⚠️  Target not reached. Consider running more iterations.")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # If Strategy 1 didn't achieve target, try Strategy 2 with refined parameters
    if best_result1 and best_result1.sharpe_ratio < 1.0:
        print("\n\n📊 Optimizing Strategy 2 - Scalping (with improved bounds)")
        print("-"*60)
        
        try:
            optimizer2, best_result2 = run_optimization(
                strategy_type=2,
                optimization_method='bayesian',
                currency='AUDUSD',
                n_iterations=20,
                sample_size=3500,
                use_previous_results=True
            )
            
            print(f"\n✅ Strategy 2 optimization completed!")
            if best_result2:
                print(f"   Best Sharpe: {best_result2.sharpe_ratio:.3f}")
                print(f"   Best Fitness: {best_result2.fitness:.3f}")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Optimization completed! Check optimizer_results/ for detailed results.")

if __name__ == "__main__":
    main()