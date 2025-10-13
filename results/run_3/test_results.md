# Test Results - run_3

**Timestamp:** 2025-10-13 10:00:55

## Model Information
- Model: `../models/hardware_ppo\run_3\final_model.zip`
- VecNormalize: `../models/hardware_ppo\run_3\vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | 122.0 | 242.2 | 18.4 |
| Longest Queue | 255.6 | 246.8 | 22.6 |
| Round Robin | -21.9 | 234.4 | 30.6 |
| Fixed Time | -143.7 | 221.4 | 41.2 |

## PPO vs Longest Queue Baseline
- Reward improvement: -52.3%
- Throughput improvement: -1.9%
- Queue reduction: +18.6%

## Wins by Controller
- PPO (Retrained): 1/5 scenarios
- Longest Queue: 4/5 scenarios
- Round Robin: 0/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=175.2, Cleared=243, Queue=7
- Longest Queue: Reward=310.5, Cleared=239, Queue=29
- Round Robin: Reward=-29.2, Cleared=241, Queue=52
- Fixed Time: Reward=-81.5, Cleared=214, Queue=36

### North Heavy Congestion
- PPO (Retrained): Reward=286.0, Cleared=220, Queue=8
- Longest Queue: Reward=191.8, Cleared=250, Queue=37
- Round Robin: Reward=78.8, Cleared=222, Queue=17
- Fixed Time: Reward=-110.8, Cleared=223, Queue=48

### East-West Rush Hour
- PPO (Retrained): Reward=37.0, Cleared=249, Queue=23
- Longest Queue: Reward=242.2, Cleared=247, Queue=8
- Round Robin: Reward=54.2, Cleared=230, Queue=26
- Fixed Time: Reward=-227.2, Cleared=225, Queue=46

### Random Traffic Pattern
- PPO (Retrained): Reward=156.5, Cleared=250, Queue=25
- Longest Queue: Reward=279.5, Cleared=248, Queue=10
- Round Robin: Reward=-52.8, Cleared=239, Queue=29
- Fixed Time: Reward=-214.8, Cleared=230, Queue=43

### Single Lane Blocked
- PPO (Retrained): Reward=-45.0, Cleared=249, Queue=29
- Longest Queue: Reward=254.0, Cleared=250, Queue=29
- Round Robin: Reward=-160.2, Cleared=240, Queue=29
- Fixed Time: Reward=-84.2, Cleared=215, Queue=33

## Visualizations
- Comparison plot: `../visualizations\run_3\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_3\scenario_heatmap.png`
