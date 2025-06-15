"""
Test the exit reason fix
"""

# Test cases
test_cases = [
    ('ExitReason.TP1_PULLBACK', 'TP1 PB'),
    ('ExitReason.TAKE_PROFIT_1', 'TP1'),
    ('ExitReason.TAKE_PROFIT_2', 'TP2'),
    ('ExitReason.TAKE_PROFIT_3', 'TP3'),
    ('ExitReason.STOP_LOSS', 'SL'),
    ('ExitReason.TRAILING_STOP', 'TSL'),
    ('take_profit_1', 'TP1'),
    ('tp1_pullback', 'TP1 PB'),
]

print("Testing exit reason logic fix:")
print("="*60)

for exit_reason, expected in test_cases:
    exit_reason_str = str(exit_reason)
    
    # Apply the new logic
    if 'TP1_PULLBACK' in exit_reason_str or 'tp1_pullback' in exit_reason_str.lower():
        result = 'TP1 PB'
    elif 'trailing_stop' in exit_reason_str.lower():
        result = 'TSL'
    elif 'stop_loss' in exit_reason_str.lower():
        result = 'SL'
    elif 'take_profit' in exit_reason_str.lower():
        if 'TAKE_PROFIT_' in exit_reason_str:
            tp_num = exit_reason_str.split('TAKE_PROFIT_')[-1]
        elif 'take_profit_' in exit_reason_str:
            tp_num = exit_reason_str.split('take_profit_')[-1]
        else:
            tp_num = '3'
        result = f'TP{tp_num}'
    else:
        result = 'Other'
    
    status = "✅" if result == expected else "❌"
    print(f"{status} {exit_reason:30} -> {result:8} (expected: {expected})")