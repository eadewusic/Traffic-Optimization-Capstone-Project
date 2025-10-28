# Run 8 Training Summary - Seed 789

## Training Configuration

**Run:** run_8  
**Seed:** 789  
**Date:** 2025-10-28 10:02:02

## Performance Statistics

| Metric | Value |
|--------|-------|
| Initial Reward | 1964.25 |
| Best Reward | 2066.27 |
| Final Reward | 2066.27 |
| Best Step | 1,000,000 |
| Total Improvement | 102.02 |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0003 |
| Steps per Update | 2048 |
| Batch Size | 64 |
| Epochs | 10 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |

## Environment Configuration

- **Environment:** Run7TrafficEnv
- **Max Queue Length:** 20 vehicles
- **Cars Cleared per Cycle:** 5 vehicles
- **Observation Space:** Box(4,) - Queue lengths for N, S, E, W
- **Action Space:** Discrete(2) - N/S or E/W green phase

## Training Details

- **Total Training Steps:** 1,000,000
- **Evaluation Frequency:** Every 10,000 steps
- **Checkpoint Frequency:** Every 100,000 steps
- **Visualization Frequency:** Every 50,000 steps

## Notes

This is part of Run 8 multi-seed validation experiment. Results from all seeds 
(42, 123, 456, 789, 1000) will be aggregated to compute mean ± standard deviation 
for statistical robustness.
