"""
Debug exit reason logic
"""

# Test the condition logic
exit_reasons = [
    'take_profit_3',
    'tp1_pullback', 
    'stop_loss',
    'trailing_stop',
    'ExitReason.TP1_PULLBACK'
]

for exit_reason in exit_reasons:
    print(f"\nTesting exit_reason: '{exit_reason}'")
    exit_reason_str = str(exit_reason)
    
    # Current logic
    if exit_reason == 'trailing_stop':
        result = 'TSL'
    elif exit_reason == 'stop_loss':
        result = 'SL'
    elif 'take_profit' in exit_reason_str:
        tp_num = exit_reason.split('_')[-1] if '_' in exit_reason else '3'
        result = f'TP{tp_num}'
    elif exit_reason == 'tp1_pullback' or 'tp1_pullback' in exit_reason_str:
        result = 'TP1 PB'
    else:
        result = 'Other'
    
    print(f"  Result: {result}")
    
    # Check the faulty condition
    faulty_condition = exit_reason == 'tp1_pullback' or 'tp1_pullback' in exit_reason_str
    print(f"  Faulty condition result: {faulty_condition}")
    
    # Show what's happening
    part1 = exit_reason == 'tp1_pullback'
    part2 = 'tp1_pullback' in exit_reason_str
    print(f"  Part 1 (exit_reason == 'tp1_pullback'): {part1}")
    print(f"  Part 2 ('tp1_pullback' in exit_reason_str): {part2}")

print("\n" + "="*60)
print("The issue: 'tp1_pullback' in exit_reason_str is always evaluated")
print("when exit_reason_str is non-empty due to operator precedence!")

# Show correct logic
print("\n" + "="*60)
print("CORRECT LOGIC:")
for exit_reason in exit_reasons:
    exit_reason_str = str(exit_reason)
    
    if exit_reason == 'trailing_stop':
        result = 'TSL'
    elif exit_reason == 'stop_loss':
        result = 'SL'
    elif 'take_profit' in exit_reason_str:
        tp_num = exit_reason.split('_')[-1] if '_' in exit_reason else '3'
        result = f'TP{tp_num}'
    elif 'tp1_pullback' in exit_reason_str.lower():  # Fixed!
        result = 'TP1 PB'
    else:
        result = 'Other'
    
    print(f"{exit_reason} -> {result}")