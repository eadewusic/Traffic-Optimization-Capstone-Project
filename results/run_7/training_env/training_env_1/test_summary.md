# Run 7 - Training Environment Test

**Test Date:** 2025-10-19 14:25:32

**Environment:** Run7TrafficEnv (comparative reward function)

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| Run 7 PPO | 557.1 | 88.6 | 1.4 |
| Longest Queue | 528.5 | 85.2 | 2.2 |
| Round Robin | 364.7 | 88.8 | 2.4 |
| Fixed Time | -655.6 | 79.6 | 7.8 |

## Run 7 vs Baseline

- Reward difference: +5.4%
- Throughput difference: +4.0%
- Statistical test: t=1.799, p=0.1098 (not statistically significant)
- Win rate: 4/5 scenarios (80%)

## Scenario Wins

- **Run 7 PPO:** 4/5 scenarios
- **Longest Queue:** 1/5 scenarios
- **Round Robin:** 0/5 scenarios
- **Fixed Time:** 0/5 scenarios

## Verdict

**LEARNING_SUCCESS**

