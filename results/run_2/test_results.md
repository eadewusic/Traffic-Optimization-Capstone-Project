# Test Results - run_2

**Timestamp:** 2025-10-18 23:02:38

## Model Information
- Run: run_2
- Model: `../models/hardware_ppo/run_2/best_model`
- VecNormalize: `../models/hardware_ppo/run_2/hardware_ppo_final_20251012_213423_vecnormalize.pkl`

## Overall Performance

| Controller | Avg Reward | Avg Cleared | Final Queue |
|------------|------------|-------------|-------------|
| PPO (Retrained) | -131.0 | 97.6 | 39.6 |
| Longest Queue | 548.3 | 137.4 | 1.6 |
| Round Robin | 458.7 | 134.6 | 3.8 |
| Fixed Time | 224.1 | 126.2 | 12.4 |

## PPO vs Longest Queue Baseline
- Reward improvement: -123.9%
- Throughput improvement: -29.0%
- Queue reduction: -2375.0%

## Wins by Controller
- PPO (Retrained): 0/5 scenarios
- Longest Queue: 5/5 scenarios
- Round Robin: 0/5 scenarios
- Fixed Time: 0/5 scenarios

## Champion: Longest Queue

## Scenario Results

### Balanced Traffic
- PPO (Retrained): Reward=-117.3, Cleared=92, Queue=38
- Longest Queue: Reward=518.3, Cleared=135, Queue=5
- Round Robin: Reward=386.0, Cleared=112, Queue=9
- Fixed Time: Reward=194.9, Cleared=116, Queue=10

### North Heavy Congestion
- PPO (Retrained): Reward=-63.6, Cleared=113, Queue=40
- Longest Queue: Reward=504.1, Cleared=128, Queue=1
- Round Robin: Reward=456.1, Cleared=128, Queue=6
- Fixed Time: Reward=279.9, Cleared=148, Queue=20

### East-West Rush Hour
- PPO (Retrained): Reward=-179.4, Cleared=89, Queue=40
- Longest Queue: Reward=551.3, Cleared=139, Queue=2
- Round Robin: Reward=544.7, Cleared=156, Queue=1
- Fixed Time: Reward=179.2, Cleared=123, Queue=16

### Random Traffic Pattern
- PPO (Retrained): Reward=-105.1, Cleared=108, Queue=40
- Longest Queue: Reward=553.1, Cleared=133, Queue=0
- Round Robin: Reward=495.1, Cleared=142, Queue=1
- Fixed Time: Reward=148.0, Cleared=104, Queue=9

### Single Lane Blocked
- PPO (Retrained): Reward=-189.6, Cleared=86, Queue=40
- Longest Queue: Reward=614.8, Cleared=152, Queue=0
- Round Robin: Reward=411.8, Cleared=135, Queue=2
- Fixed Time: Reward=318.7, Cleared=140, Queue=7

## Visualizations
- Comparison plot: `../visualizations\run_2\controller_comparison.png`
- Scenario heatmap: `../visualizations\run_2\scenario_heatmap.png`
