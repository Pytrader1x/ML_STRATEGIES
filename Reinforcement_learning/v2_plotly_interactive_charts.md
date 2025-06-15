# Interactive Plotly Charts Implementation

## Overview

Replaced static matplotlib plots with fully interactive Plotly HTML charts for episode visualization. Each episode now generates a self-contained HTML file with zoom, pan, and hover tooltips.

## Key Features

### 1. Three-Row Subplot Layout
- **Top Panel (50%)**: AUDUSD price with entry/exit markers
- **Middle Panel (25%)**: Position size and direction (Long/Short/Flat)
- **Bottom Panel (25%)**: Cumulative P&L in USD

### 2. Interactive Elements
- **Zoom & Pan**: Click and drag to zoom into specific time periods
- **Hover Tooltips**: See exact values for price, P&L, and trade details
- **Unified Hover**: X-axis synchronized between panels
- **Legend Toggle**: Click legend items to show/hide traces

### 3. Visual Enhancements
- **Entry Markers**: Green triangles (▲) with hover labels
- **Exit Markers**: Red triangles (▼) with hover labels
- **Position Visualization**: Step plot showing exact position size (1M units)
- **Position Fill**: Blue shaded area for position exposure
- **P&L Coloring**: Green for profit, red for loss
- **Fill Area**: Shaded area under P&L curve

## Implementation Details

### Import Changes
```python
# Removed:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Added:
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

### Chart Generation (Every 5th Episode)
```python
# Create 3-row subplot
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.05,
    subplot_titles=("Episode X - Price & Trades", 
                    "Position Size & Direction",
                    "Cumulative P&L (USD)")
)

# Add traces: price line, entry/exit markers, position, cumulative P&L
# Save as HTML
fig.write_html(f'plots/episode_{episode+1:03d}.html')
```

### Performance Optimization
- **Async Saving**: Uses ThreadPoolExecutor for non-blocking file I/O
- **Selective Saving**: Only saves every 5th episode to avoid too many files
- **Efficient Data**: Minimal memory footprint with direct numpy arrays

## Output Files

```
plots/
├── episode_005.html  # Interactive chart for episode 5
├── episode_010.html  # Interactive chart for episode 10
├── episode_015.html  # Interactive chart for episode 15
└── ...
```

## Usage

1. **View Charts**: Open any HTML file in a web browser
2. **Zoom**: Click and drag to zoom into specific bars
3. **Pan**: Hold shift and drag to pan
4. **Reset**: Double-click to reset view
5. **Hover**: Move mouse over data points for details

## Benefits vs Matplotlib

| Feature | Matplotlib | Plotly |
|---------|------------|---------|
| Interactivity | Static PNG | Full zoom/pan/hover |
| File Size | ~50KB PNG | ~500KB HTML |
| Speed | Fast render | Fast render + async save |
| Analysis | Visual only | Inspect exact values |
| Sharing | Image file | Self-contained HTML |

## Example Hover Information

- **Price Line**: Bar number, AUDUSD price
- **Entry Marker**: "Entry 1", Bar: 1234, Price: 0.68234
- **Exit Marker**: "Exit 1", Bar: 1456, Price: 0.68567
- **Position Line**: Bar: 1500, Position: 1,000,000, Direction: Long
- **P&L Line**: Bar: 1500, P&L: $12,345

## Future Enhancements

1. Add trade annotations (P&L per trade)
2. Include technical indicators on price panel
3. Add drawdown visualization
4. Export to static images for reports
5. Real-time streaming updates during training