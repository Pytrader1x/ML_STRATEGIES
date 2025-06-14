#!/usr/bin/env python3
"""
Test script for exit metrics functionality
"""

import subprocess
import sys

def main():
    """Run the strategy with minimal iterations to test exit metrics"""
    print("🧪 Testing Exit Metrics Functionality")
    print("="*60)
    
    # Run with minimal settings for quick test
    cmd = [
        sys.executable,
        "run_strategy_oop.py",
        "--iterations", "3",
        "--sample-size", "2000",
        "--no-plots",
        "--currency", "AUDUSD"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("Looking for exit statistics output...")
    print("="*60)
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if exit statistics are in the output
        output = result.stdout
        
        if "EXIT STATISTICS ANALYSIS" in output:
            print("✅ Exit statistics section found!")
            
            # Extract and display the exit statistics section
            lines = output.split('\n')
            in_exit_section = False
            exit_lines = []
            
            for line in lines:
                if "EXIT STATISTICS ANALYSIS" in line:
                    in_exit_section = True
                elif in_exit_section and ("SUMMARY STATISTICS" in line or "YEARLY PERFORMANCE" in line):
                    break
                
                if in_exit_section:
                    exit_lines.append(line)
            
            print("\n📊 Exit Statistics Output:")
            print("\n".join(exit_lines))
            
            # Check for specific metrics
            checks = [
                ("TP1", "TP1 reached"),
                ("TP2", "TP2 reached"),
                ("TP3", "TP3 reached"),
                ("TSL exits", "TSL exits:"),
                ("SL exits", "SL exits:")
            ]
            
            print("\n✓ Checking for specific metrics:")
            for metric_name, search_str in checks:
                if any(search_str in line for line in exit_lines):
                    print(f"  ✅ {metric_name} metric found")
                else:
                    print(f"  ❌ {metric_name} metric NOT found")
            
        else:
            print("❌ Exit statistics section NOT found in output!")
            print("\nFull output:")
            print(output)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()