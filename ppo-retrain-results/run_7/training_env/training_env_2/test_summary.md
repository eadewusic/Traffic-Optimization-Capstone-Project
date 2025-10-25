# Run 7 - Training Environment Test

**Test Date:** 2025-10-19 14:49:07

**Environment:** Run7TrafficEnv (comparative reward function)

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| Run 7 PPO | 555.5 | 89.6 | 1.8 |
| Longest Queue | 533.3 | 85.4 | 2.0 |
| Round Robin | 376.6 | 84.2 | 2.8 |
| Fixed Time | -623.3 | 81.4 | 6.6 |

## Run 7 vs Baseline

- Reward difference: +4.2%
- Throughput difference: +4.9%
- Statistical test: t=1.363, p=0.2101 (not statistically significant)
- Win rate: 4/5 scenarios (80%)

## Scenario Wins

- **Run 7 PPO:** 4/5 scenarios
- **Longest Queue:** 1/5 scenarios
- **Round Robin:** 0/5 scenarios
- **Fixed Time:** 0/5 scenarios

## Verdict

**MARGINAL_SUCCESS**

