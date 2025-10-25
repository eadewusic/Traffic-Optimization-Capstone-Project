# Training Summary - run_4b

**Variant:** Deeper Network (Testing Hypothesis)
**Timestamp:** 2025-10-13 12:43:44

## Hypothesis Test
Does deeper network architecture perform better?
- Network: [128, 128, 64] (vs 4a's [64, 64])
- All other parameters kept the same
- Batch size: 64 (same as 4a)
- Entropy: 0.01 (same as 4a)
- Domain rand: NO (same as 4a)

## Configuration
- Total timesteps: 200,000
- Domain randomization: False
- Network parameters: ~28,000 (vs 4a's ~10,000)

## Training Performance
- Best mean reward: 1293.03

## Quick Test Results (10 episodes)
- Average reward: 339.8
- Std deviation: 30.4
- Average cleared: 236 vehicles

