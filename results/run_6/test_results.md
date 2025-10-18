# Test Results - run_6

**Timestamp:** 2025-10-18 22:21:00

## Model Information
- Run: run_6
- Model: `../models/hardware_ppo/run_6/best_model`
- VecNormalize: `../models/hardware_ppo/run_6/vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | 512.1 | 134.2 | 2.0 |
| Longest Queue | 546.9 | 139.6 | 2.2 |
| Round Robin | 426.6 | 125.0 | 3.8 |
| Fixed Time | 206.5 | 125.0 | 11.0 |

## PPO vs Longest Queue Baseline
- Reward improvement: -6.4%
- Throughput improvement: -3.9%
- Queue reduction: +9.1%

## Wins by Controller
- PPO (Retrained): 1/5 scenarios
- Longest Queue: 4/5 scenarios
- Round Robin: 0/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=505.8, Cleared=124, Queue=4
- Longest Queue: Reward=534.3, Cleared=127, Queue=3
- Round Robin: Reward=399.0, Cleared=115, Queue=3
- Fixed Time: Reward=143.5, Cleared=111, Queue=17

### North Heavy Congestion
- PPO (Retrained): Reward=495.4, Cleared=133, Queue=2
- Longest Queue: Reward=616.5, Cleared=158, Queue=0
- Round Robin: Reward=416.0, Cleared=125, Queue=5
- Fixed Time: Reward=266.2, Cleared=135, Queue=21

### East-West Rush Hour
- PPO (Retrained): Reward=520.1, Cleared=138, Queue=0
- Longest Queue: Reward=561.5, Cleared=143, Queue=2
- Round Robin: Reward=443.2, Cleared=133, Queue=2
- Fixed Time: Reward=208.8, Cleared=140, Queue=4

### Random Traffic Pattern
- PPO (Retrained): Reward=479.8, Cleared=129, Queue=2
- Longest Queue: Reward=502.9, Cleared=128, Queue=3
- Round Robin: Reward=423.1, Cleared=116, Queue=6
- Fixed Time: Reward=163.2, Cleared=118, Queue=8

### Single Lane Blocked
- PPO (Retrained): Reward=559.1, Cleared=147, Queue=2
- Longest Queue: Reward=519.0, Cleared=142, Queue=3
- Round Robin: Reward=451.6, Cleared=136, Queue=3
- Fixed Time: Reward=250.6, Cleared=121, Queue=5

## Visualizations
- Comparison plot: `../visualizations\run_6\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_6\scenario_heatmap.png`
