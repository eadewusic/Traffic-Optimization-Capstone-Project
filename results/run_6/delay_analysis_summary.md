# Vehicle Delay Analysis - Run 6

**Analysis Date:** 2025-10-14 03:20:47
**Model:** ../models/hardware_ppo/run_6/final_model

## Overall Results

- **PPO Average Delay:** 1.40 steps per vehicle
- **Baseline Average Delay:** 42.39 steps per vehicle
- **Overall Delay Reduction:** 96.7%
- **Target Achievement:**  EXCEEDS 50% target

## Scenario-by-Scenario Results

| Scenario | PPO Delay | Baseline Delay | Reduction | Target Met |
|----------|-----------|----------------|-----------|------------|
| Balanced Traffic | 1.39 | 55.40 | 97.5% | ✓ |
| North Heavy Congestion | 1.31 | 50.02 | 97.4% | ✓ |
| E-W Rush Hour | 1.41 | 19.00 | 92.6% | ✓ |
| Random Pattern | 1.32 | 45.17 | 97.1% | ✓ |
| Single Lane Blocked | 1.56 | 42.35 | 96.3% | ✓ |

## Interpretation

The PPO agent achieves a **96.7% reduction** in average vehicle delay compared to the longest-queue baseline controller. This significantly exceeds the 50% target set for the project.

**Key Findings:**
- Average wait time per vehicle was reduced from **42.39 steps** (baseline) to **1.40 steps** (PPO)
- All 5 test scenarios exceeded the 50% delay reduction target
- The PPO agent consistently outperforms the baseline across diverse traffic patterns

## For Thesis

> Validation testing demonstrates that the PPO agent achieves a **96.7% reduction** in average vehicle delay compared to the longest-queue baseline, significantly exceeding the 50% target. Average wait time per vehicle was reduced from 42.39 steps (baseline) to 1.40 steps (PPO) across five diverse traffic scenarios.
