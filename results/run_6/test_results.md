# Test Results - run_6

**Timestamp:** 2025-10-13 23:54:24

## Model Information
- Run: run_6
- Model: `../models/hardware_ppo/run_6/best_model`
- VecNormalize: `../models/hardware_ppo/run_6/vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | 575.9 | 147.8 | 3.2 |
| Longest Queue | -204.2 | 92.8 | 40.0 |
| Round Robin | 320.7 | 145.6 | 5.2 |
| Fixed Time | 287.5 | 133.4 | 5.8 |

## PPO vs Longest Queue Baseline
- Reward improvement: +382.0%
- Throughput improvement: +59.3%
- Queue reduction: +92.0%

## Wins by Controller
- PPO (Retrained): 5/5 scenarios
- Longest Queue: 0/5 scenarios
- Round Robin: 0/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: PPO (Retrained)

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=604.2, Cleared=148, Queue=0
- Longest Queue: Reward=-465.1, Cleared=40, Queue=61
- Round Robin: Reward=362.3, Cleared=136, Queue=2
- Fixed Time: Reward=342.0, Cleared=148, Queue=7

### North Heavy Congestion
- PPO (Retrained): Reward=592.8, Cleared=150, Queue=1
- Longest Queue: Reward=76.3, Cleared=136, Queue=5
- Round Robin: Reward=297.5, Cleared=173, Queue=13
- Fixed Time: Reward=245.2, Cleared=123, Queue=5

### East-West Rush Hour
- PPO (Retrained): Reward=587.5, Cleared=154, Queue=4
- Longest Queue: Reward=-112.8, Cleared=108, Queue=35
- Round Robin: Reward=333.9, Cleared=144, Queue=4
- Fixed Time: Reward=285.0, Cleared=128, Queue=6

### Random Traffic Pattern
- PPO (Retrained): Reward=596.0, Cleared=158, Queue=3
- Longest Queue: Reward=-303.2, Cleared=85, Queue=58
- Round Robin: Reward=320.8, Cleared=145, Queue=3
- Fixed Time: Reward=248.0, Cleared=123, Queue=2

### Single Lane Blocked
- PPO (Retrained): Reward=498.9, Cleared=129, Queue=8
- Longest Queue: Reward=-216.1, Cleared=95, Queue=41
- Round Robin: Reward=289.0, Cleared=130, Queue=4
- Fixed Time: Reward=317.1, Cleared=145, Queue=9

## Visualizations
- Comparison plot: `../visualizations\run_6\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_6\scenario_heatmap.png`
