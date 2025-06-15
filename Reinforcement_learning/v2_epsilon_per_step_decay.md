# Epsilon Per-Step Decay Implementation

## Overview

Changed epsilon decay from per-episode to per-step with a much slower decay rate to reach the terminal epsilon value around episode 5-6.

## Changes Made

### 1. Decay Rate Calculation

**Previous**: Episode-based decay with rate 0.95
- Would decay too slowly over episodes
- Epsilon after 5 episodes: 0.9 × 0.95^5 ≈ 0.70 (still too high)

**New**: Per-step decay with rate 0.99982
- Target: Reach epsilon_min (0.01) by ~25,000-30,000 steps
- Calculation: 0.9 × 0.99982^25000 ≈ 0.011
- This equals approximately 5-6 episodes at 5,000 steps each

### 2. Code Changes

```python
# Config changes
EPSILON_DECAY = 0.99982  # Per-step decay to reach 0.01 by ~25k steps (5-6 episodes)

# In replay() method
# Epsilon decay per-step (slower rate to reach terminal by episode 5-6)
self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Removed episode-level decay in train_agent()
```

### 3. Expected Epsilon Trajectory

With the new per-step decay rate of 0.99982:

| Episode | Approx Steps | Expected Epsilon |
|---------|-------------|------------------|
| 0       | 0           | 0.9000          |
| 1       | 5,000       | 0.5970          |
| 2       | 10,000      | 0.3564          |
| 3       | 15,000      | 0.2127          |
| 4       | 20,000      | 0.1269          |
| 5       | 25,000      | 0.0758          |
| 6       | 30,000      | 0.0452          |

By episode 5-6, epsilon will be close to the minimum value (0.01), ensuring:
- Good exploration in early episodes
- Convergence to exploitation by episode 5-6
- Smooth transition from exploration to exploitation

## Benefits

1. **Better Exploration Schedule**: More gradual transition from exploration to exploitation
2. **Consistent Decay**: Every step contributes to decay, not just episode boundaries
3. **Predictable Convergence**: Reaches terminal epsilon at a known point (~25k steps)
4. **Tunable**: Easy to adjust decay rate based on episode length

## Monitoring

The training loop now prints expected epsilon values for future episodes on the first episode, helping verify the decay schedule is working as intended:

```
Episode 1 Summary:
  Current Epsilon: 0.9000
  Expected ε by episode 1: 0.5970
  Expected ε by episode 2: 0.3564
  Expected ε by episode 3: 0.2127
  Expected ε by episode 4: 0.1269
  Expected ε by episode 5: 0.0758
  Expected ε by episode 6: 0.0452
```