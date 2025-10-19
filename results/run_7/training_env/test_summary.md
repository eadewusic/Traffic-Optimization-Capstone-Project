# Run 7 - Training Environment Test

**Test Date:** 2025-10-19 09:57:36

**Environment:** Run7TrafficEnv (comparative reward function)

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| Run 7 PPO | 551.1 | 86.6 | 2.2 |
| Longest Queue | 529.2 | 85.4 | 2.6 |
| Round Robin | 381.9 | 84.8 | 2.2 |
| Fixed Time | -709.9 | 81.2 | 8.0 |

## Run 7 vs Baseline

- Reward difference: +4.1%
- Throughput difference: +1.4%
- Statistical test: t=2.266, p=0.0532 (not statistically significant)
- Win rate: 5/5 scenarios (100%)

## Scenario Wins

- **Run 7 PPO:** 5/5 scenarios
- **Longest Queue:** 0/5 scenarios
- **Round Robin:** 0/5 scenarios
- **Fixed Time:** 0/5 scenarios

## Verdict

**MARGINAL_SUCCESS**

