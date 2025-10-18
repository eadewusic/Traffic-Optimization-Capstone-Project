# Test Results - run_1

**Timestamp:** 2025-10-18 23:03:46

## Model Information
- Run: run_1
- Model: `../models/hardware_ppo/run_1/best_model`
- VecNormalize: `../models/hardware_ppo/run_1/hardware_ppo_final_20251012_164818_vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | -361.2 | 61.6 | 43.6 |
| Longest Queue | 533.0 | 135.6 | 3.0 |
| Round Robin | 479.8 | 143.4 | 3.4 |
| Fixed Time | 217.2 | 123.0 | 10.0 |

## PPO vs Longest Queue Baseline
- Reward improvement: -167.8%
- Throughput improvement: -54.6%
- Queue reduction: -1353.3%

## Wins by Controller
- PPO (Retrained): 0/5 scenarios
- Longest Queue: 4/5 scenarios
- Round Robin: 1/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=-308.7, Cleared=68, Queue=42
- Longest Queue: Reward=556.6, Cleared=143, Queue=5
- Round Robin: Reward=442.5, Cleared=129, Queue=4
- Fixed Time: Reward=230.9, Cleared=121, Queue=2

### North Heavy Congestion
- PPO (Retrained): Reward=-319.4, Cleared=67, Queue=40
- Longest Queue: Reward=570.2, Cleared=144, Queue=0
- Round Robin: Reward=519.8, Cleared=154, Queue=4
- Fixed Time: Reward=233.1, Cleared=139, Queue=16

### East-West Rush Hour
- PPO (Retrained): Reward=-476.5, Cleared=47, Queue=52
- Longest Queue: Reward=474.2, Cleared=125, Queue=0
- Round Robin: Reward=476.6, Cleared=140, Queue=7
- Fixed Time: Reward=195.6, Cleared=120, Queue=9

### Random Traffic Pattern
- PPO (Retrained): Reward=-429.3, Cleared=52, Queue=44
- Longest Queue: Reward=480.5, Cleared=123, Queue=2
- Round Robin: Reward=462.0, Cleared=144, Queue=1
- Fixed Time: Reward=187.6, Cleared=118, Queue=9

### Single Lane Blocked
- PPO (Retrained): Reward=-272.1, Cleared=74, Queue=40
- Longest Queue: Reward=583.5, Cleared=143, Queue=8
- Round Robin: Reward=498.0, Cleared=150, Queue=1
- Fixed Time: Reward=238.9, Cleared=117, Queue=14

## Visualizations
- Comparison plot: `../visualizations\run_1\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_1\scenario_heatmap.png`
