# Test Results - run_4a

**Timestamp:** 2025-10-18 22:59:43

## Model Information
- Run: run_4a
- Model: `../models/hardware_ppo/run_4a/best_model`
- VecNormalize: `../models/hardware_ppo/run_4a/vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | -651.6 | 6.8 | 80.0 |
| Longest Queue | 580.8 | 144.8 | 4.0 |
| Round Robin | 472.9 | 141.8 | 3.2 |
| Fixed Time | 245.4 | 132.0 | 13.2 |

## PPO vs Longest Queue Baseline
- Reward improvement: -212.2%
- Throughput improvement: -95.3%
- Queue reduction: -1900.0%

## Wins by Controller
- PPO (Retrained): 0/5 scenarios
- Longest Queue: 5/5 scenarios
- Round Robin: 0/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=-530.3, Cleared=20, Queue=80
- Longest Queue: Reward=569.7, Cleared=140, Queue=1
- Round Robin: Reward=473.3, Cleared=136, Queue=4
- Fixed Time: Reward=192.3, Cleared=111, Queue=12

### North Heavy Congestion
- PPO (Retrained): Reward=-633.0, Cleared=8, Queue=80
- Longest Queue: Reward=586.2, Cleared=144, Queue=8
- Round Robin: Reward=447.4, Cleared=131, Queue=5
- Fixed Time: Reward=276.0, Cleared=138, Queue=10

### East-West Rush Hour
- PPO (Retrained): Reward=-718.2, Cleared=0, Queue=80
- Longest Queue: Reward=637.9, Cleared=166, Queue=2
- Round Robin: Reward=474.8, Cleared=149, Queue=0
- Fixed Time: Reward=288.1, Cleared=153, Queue=10

### Random Traffic Pattern
- PPO (Retrained): Reward=-699.2, Cleared=0, Queue=80
- Longest Queue: Reward=566.5, Cleared=140, Queue=3
- Round Robin: Reward=487.6, Cleared=139, Queue=5
- Fixed Time: Reward=245.8, Cleared=136, Queue=15

### Single Lane Blocked
- PPO (Retrained): Reward=-677.0, Cleared=6, Queue=80
- Longest Queue: Reward=543.8, Cleared=134, Queue=6
- Round Robin: Reward=481.2, Cleared=154, Queue=2
- Fixed Time: Reward=224.8, Cleared=122, Queue=19

## Visualizations
- Comparison plot: `../visualizations\run_4a\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_4a\scenario_heatmap.png`
