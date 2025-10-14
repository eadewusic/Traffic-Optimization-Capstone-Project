# Vehicle Delay Analysis - Run 6

**Analysis Date:** 2025-10-14 13:32:20
**Model:** ../models/hardware_ppo/run_6/final_model

## Overall Results

- **PPO Average Delay:** 3.59 steps per vehicle
- **Baseline Average Delay:** 18.92 steps per vehicle
- **Overall Delay Reduction:** 81.0%
- **Target Achievement:**  EXCEEDS 50% target

## Scenario-by-Scenario Results

| Scenario | PPO Delay | Baseline Delay | Reduction | Target Met |
|----------|-----------|----------------|-----------|------------|
| Balanced Traffic | 5.18 | 14.86 | 65.1% | ✓ |
| North Heavy Congestion | 2.44 | 15.43 | 84.2% | ✓ |
| E-W Rush Hour | 4.39 | 15.84 | 72.3% | ✓ |
| Random Pattern | 3.61 | 26.70 | 86.5% | ✓ |
| Single Lane Blocked | 2.33 | 21.76 | 89.3% | ✓ |

## Interpretation

The PPO agent achieves a **81.0% reduction** in average vehicle delay compared to the longest-queue baseline controller. This significantly exceeds the 50% target set for the project.

**Key Findings:**
- Average wait time per vehicle was reduced from **18.92 steps** (baseline) to **3.59 steps** (PPO)
- The PPO agent consistently outperforms the baseline across diverse traffic patterns
- Individual vehicle tracking provides mathematically accurate delay measurements

## For Thesis

> Validation testing demonstrates that the PPO agent achieves a **81.0% reduction** in average vehicle delay compared to the longest-queue baseline, significantly exceeding the 50% target. Using individual vehicle lifecycle tracking, average wait time per vehicle was reduced from 18.92 steps (baseline) to 3.59 steps (PPO) across five diverse traffic scenarios.
