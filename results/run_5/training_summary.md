# Training Summary - run_5

**Variant:** Production-Ready (Fixed Reward + Hardware DR)
**Timestamp:** 2025-10-13 18:34:52

## Key Improvements in Run 5
**From Run 4a (simple network):**
- Network: [64, 64] (proven to work)
- Domain rand: Enabled (needed for hardware)
- Reward: FIXED to align with effective control

**New reward function:**
- Primary: Minimize longest queue (-2.0)
- Secondary: Maximize throughput (+0.5)
- Tertiary: Minimize total waiting (-0.1)

**Hardware-aware domain randomization:**
- GPIO latency: 1-10ms
- Button debounce: 50-200ms
- Processing jitter: 0-5ms
- Arrival rate: 0.15-0.45

## Configuration
- Total timesteps: 250,000
- Domain randomization: True
- Network parameters: ~10,000

## Training Performance
- Best mean reward: -2377.21

## Quick Test Results (10 episodes)
- Average reward: -652.2
- Std deviation: 133.6
- Average cleared: 239 vehicles

## Expected vs Previous Runs
| Run | Network | DR | Reward | Test Perf | Status |
|-----|---------|----|---------|-----------|---------|
| 4a | [64,64] | No | Old | 290.8 | Baseline |
| 4b | [128,128,64] | No | Old | 181.7 | Worse |
| 5 | [64,64] | Yes | Fixed | TBD | Expected: +5-15% |
