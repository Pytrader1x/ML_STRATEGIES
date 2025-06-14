#!/usr/bin/env python3
"""
Script to run trading strategy specifically for Feb-March 2025
"""

import subprocess
import sys
import os

def main():
    """Run the strategy for Feb-March 2025 period"""
    
    print("="*80)
    print("Running Trading Strategy for Feb-March 2025")
    print("="*80)
    
    # Define the command to run
    cmd = [
        sys.executable,  # Use current Python interpreter
        "run_Strategy.py",
        "--date-range", "2025-02-01", "2025-03-31",
        "--iterations", "1",  # Single iteration since we're using the full date range
        "--sample-size", "50000",  # Large sample size to use all available data
        "--save-plots",  # Save the plots
        "--realistic-costs",  # Use realistic trading costs
    ]
    
    # Run the command
    print(f"\nExecuting command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode
    except Exception as e:
        print(f"Error running strategy: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())