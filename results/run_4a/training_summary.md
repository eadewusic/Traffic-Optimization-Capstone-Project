# Training Summary - run_4a

**Variant:** Simple Network (Friend's Hypothesis)
**Timestamp:** 2025-10-13 10:53:19

## Hypothesis
Simpler network architecture generalizes better:
- Network: [64, 64] (was [128, 64, 32])
- Batch size: 64 (was 128)
- Entropy: 0.01 (was 0.02)
- Domain rand: NO (was YES)

## Configuration
- Total timesteps: 200,000
- Domain randomization: False
- Network parameters: ~10,000 (vs Run 3's ~20,000)

## Training Performance
- Best mean reward: 1306.07

## Quick Test Results (10 episodes)
- Average reward: 326.2
- Std deviation: 51.6
- Average cleared: 236 vehicles

