# Test Results - run_3

**Timestamp:** 2025-10-18 23:00:33

## Model Information
- Run: run_3
- Model: `../models/hardware_ppo/run_3/best_model`
- VecNormalize: `../models/hardware_ppo/run_3/vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | -323.6 | 69.6 | 43.2 |
| Longest Queue | 569.0 | 144.0 | 3.0 |
| Round Robin | 462.5 | 135.4 | 5.2 |
| Fixed Time | 228.4 | 125.2 | 12.8 |

## PPO vs Longest Queue Baseline
- Reward improvement: -156.9%
- Throughput improvement: -51.7%
- Queue reduction: -1340.0%

## Wins by Controller
- PPO (Retrained): 0/5 scenarios
- Longest Queue: 4/5 scenarios
- Round Robin: 1/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=-320.3, Cleared=68, Queue=43
- Longest Queue: Reward=552.6, Cleared=137, Queue=0
- Round Robin: Reward=430.9, Cleared=128, Queue=3
- Fixed Time: Reward=162.0, Cleared=108, Queue=18

### North Heavy Congestion
- PPO (Retrained): Reward=-324.6, Cleared=61, Queue=44
- Longest Queue: Reward=620.0, Cleared=161, Queue=4
- Round Robin: Reward=505.2, Cleared=145, Queue=8
- Fixed Time: Reward=313.2, Cleared=152, Queue=18

### East-West Rush Hour
- PPO (Retrained): Reward=-312.0, Cleared=86, Queue=40
- Longest Queue: Reward=571.7, Cleared=144, Queue=5
- Round Robin: Reward=421.0, Cleared=126, Queue=5
- Fixed Time: Reward=199.2, Cleared=128, Queue=4

### Random Traffic Pattern
- PPO (Retrained): Reward=-427.0, Cleared=55, Queue=43
- Longest Queue: Reward=509.6, Cleared=127, Queue=3
- Round Robin: Reward=547.2, Cleared=158, Queue=4
- Fixed Time: Reward=161.9, Cleared=107, Queue=13

### Single Lane Blocked
- PPO (Retrained): Reward=-233.9, Cleared=78, Queue=46
- Longest Queue: Reward=591.2, Cleared=151, Queue=3
- Round Robin: Reward=407.8, Cleared=120, Queue=6
- Fixed Time: Reward=305.5, Cleared=131, Queue=11

## Visualizations
- Comparison plot: `../visualizations\run_3\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_3\scenario_heatmap.png`
