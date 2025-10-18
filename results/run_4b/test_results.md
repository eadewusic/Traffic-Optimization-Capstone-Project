# Test Results - run_4b

**Timestamp:** 2025-10-18 22:59:15

## Model Information
- Run: run_4b
- Model: `../models/hardware_ppo/run_4b/best_model`
- VecNormalize: `../models/hardware_ppo/run_4b/vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | 196.9 | 133.0 | 9.0 |
| Longest Queue | 547.5 | 141.2 | 2.4 |
| Round Robin | 471.3 | 144.4 | 4.8 |
| Fixed Time | 224.2 | 126.6 | 14.2 |

## PPO vs Longest Queue Baseline
- Reward improvement: -64.0%
- Throughput improvement: -5.8%
- Queue reduction: -275.0%

## Wins by Controller
- PPO (Retrained): 0/5 scenarios
- Longest Queue: 3/5 scenarios
- Round Robin: 2/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=276.8, Cleared=122, Queue=9
- Longest Queue: Reward=439.2, Cleared=114, Queue=2
- Round Robin: Reward=502.5, Cleared=155, Queue=5
- Fixed Time: Reward=174.1, Cleared=120, Queue=24

### North Heavy Congestion
- PPO (Retrained): Reward=230.0, Cleared=143, Queue=4
- Longest Queue: Reward=676.0, Cleared=175, Queue=2
- Round Robin: Reward=446.0, Cleared=132, Queue=8
- Fixed Time: Reward=222.7, Cleared=126, Queue=9

### East-West Rush Hour
- PPO (Retrained): Reward=162.1, Cleared=125, Queue=13
- Longest Queue: Reward=546.6, Cleared=137, Queue=3
- Round Robin: Reward=582.6, Cleared=174, Queue=5
- Fixed Time: Reward=212.9, Cleared=133, Queue=17

### Random Traffic Pattern
- PPO (Retrained): Reward=65.8, Cleared=136, Queue=14
- Longest Queue: Reward=502.9, Cleared=130, Queue=2
- Round Robin: Reward=439.1, Cleared=133, Queue=4
- Fixed Time: Reward=219.3, Cleared=128, Queue=15

### Single Lane Blocked
- PPO (Retrained): Reward=250.0, Cleared=139, Queue=5
- Longest Queue: Reward=572.6, Cleared=150, Queue=3
- Round Robin: Reward=386.1, Cleared=128, Queue=2
- Fixed Time: Reward=291.7, Cleared=126, Queue=6

## Visualizations
- Comparison plot: `../visualizations\run_4b\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_4b\scenario_heatmap.png`
