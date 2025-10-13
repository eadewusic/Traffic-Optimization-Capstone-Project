# Training Summary - run_6

**Variant:** Properly Balanced Rewards (Positive for Good Performance)
**Timestamp:** 2025-10-13 19:47:52

## What's Different in Run 6
**Problem with Run 5:**
- Rewards always negative (even for good performance)
- Supervisor concern: 'Expected positive rewards'
- Academic issue: Confusing reward scale

**Solution in Run 6:**
- Throughput rewards DOMINATE: 3.0 (was 0.5 in Run 5)
- Congestion penalties SMALLER: -0.4 (was -2.0 in Run 5)
- Strategic bonuses added: +5.0 for attacking longest queue
- Result: POSITIVE rewards for good performance

**What's Kept from Run 5 (that worked):**
- Strategic alignment: Prioritize longest queue
- Simple network: [64, 64]
- Hardware DR: GPIO delays, button debounce
- Extended training: 250k steps

## Configuration
- Total timesteps: 250,000
- Domain randomization: True
- Network: [64, 64]
- Parameters: ~10,000

## Training Performance
- Best mean reward: 2040.26
- Expected: +500 to +1500 (POSITIVE!)

## Quick Test Results (10 episodes)
- Average reward: 548.9
- Std deviation: 48.4
- Average cleared: 140 vehicles
- Expected: +300 to +600 (POSITIVE!)

## Comparison to Previous Runs
| Run | Network | DR | Reward Scale | Test Result | Status |
|-----|---------|----|--------------|--------------|---------|
| 5 | [64,64] | Yes | Negative | -652 (beat baseline +6.9%) | Worked but confusing |
| 6 | [64,64] | Yes | Positive | TBD | Expected: +300 to +600 |
