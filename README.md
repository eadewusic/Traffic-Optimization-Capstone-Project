# Traffic Light Optimization using Deep Reinforcement Learning and IoT

This project deploys a trained PPO reinforcement learning agent on Raspberry Pi hardware with push-button inputs and LED traffic lights, demonstrating adaptive traffic control for African intersections to reduce congestion.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%204%20Model%20B%202GB%20RAM-red.svg)](https://www.raspberrypi.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Firebase](https://img.shields.io/badge/Firebase-yellow.svg?logo=firebase&labelColor=grey)](https://console.firebase.google.com/)

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features & Achievements](#key-features--achievements)
- [Demo Video](#demo-video)
- [Project Links](#project-links)
- [Complete Training Evolution (Runs 1-8)](#complete-training-evolution-runs-1-8)
- [Technical Architecture](#technical-architecture)
- [Multi-Seed Validation](#multi-seed-validation-run-8)
- [Statistical Testing](#statistical-testing)
- [Hardware Deployment](#hardware-deployment)
- [Firebase Cloud Integration](#firebase-cloud-integration)
- [Installation & Setup](#installation--setup)
- [Key Contributions](#key-contributions)
- [Performance Metrics](#performance-metrics)
- [Tech Stack](#tech-stack)
- [Production Deployment Economics](#production-deployment-economics)
- [License & Attribution](#license--attribution)
- [Contact](#contact)

## Project Overview

This capstone project implements an **intelligent traffic light control system** powered by **Deep Reinforcement Learning (DRL)** using the **Proximal Policy Optimization (PPO)** algorithm. The system learns to minimize vehicle waiting times at a four-way intersection by dynamically adjusting traffic light phases based on real-time traffic conditions.

### **What Makes This Project Special:**

- **AI-Powered:** Uses PPO (Deep RL) to learn optimal traffic control strategies
- **Statistically Validated:** Wilcoxon test shows significant improvement (p=0.0002)
- **Hardware Deployed:** Real-time operation on Raspberry Pi 4 with LED visualization
- **Multi-Seed Validation:** 5-seed validation ensuring reproducibility (CV = 1.3%)
- **Cost-Effective:** $108 prototype, $8K-$10K production cost vs $250K-$500K traditional systems
- **Research-Grade:** Publication-ready documentation and scientific rigor

## Problem Statement

Rapid urbanization is one of the most significant global transformations of the 21st century, with Africa being the fastest-urbanizing continent. The United Nations projects that by 2050, over half of Africa's population will reside in urban areas, and the continent's total population will reach 2.5 billion. This explosive growth places immense pressure on urban infrastructure, particularly transportation networks, which are often inadequate to meet the surging demand.

### **Urban Traffic Congestion Crisis:**

- **Economic Losses:** $4 billion annually in Lagos, Nigeria alone due to congestion
- **Environmental Impact:** Excessive idling increases CO₂ emissions and fuel consumption
- **Lower Quality of Life:** Average commuter can spend up to 30 hours in traffic every week (70-minute average one-way commutes in Lagos)
- **Global Impact:** Lagos ranked world's most congested city in 2025 (traffic index 365.9)

### **Traditional Solutions Fall Short:**

| Approach | Limitation |
|----------|------------|
| **Fixed-Timing Signals** | Cannot adapt to changing traffic patterns |
| **Actuated Signals** | Rule-based, not optimal for complex scenarios & Expensive infrastructure ($250K-$500K per intersection) |
| **Coordinated Systems (SCATS/SCOOT)** | Limited flexibility |

### **Our Solution:**

Implement a **Deep Reinforcement Learning agent** that:
1. Observes real-time traffic conditions (queue lengths per direction)
2. Learns optimal control policies through systematic training (8 runs, multi-seed validation)
3. Adapts dynamically to varying traffic patterns
4. Deploys on low-cost hardware (Raspberry Pi 4, $108 prototype)
5. Achieves statistically validated improvements (p<0.05) over traditional controllers

## Key Features & Achievements

### Research Methodology: 3-Phase Systematic Approach

**Phase 1: Foundation Research - Algorithm Selection**
- Comprehensive evaluation of 4 RL algorithm families across 17 configurations
- Tested: PPO, DQN, REINFORCE, Actor-Critic
- Environment: TrafficJunctionEnv (113D state space, 9 actions, Rwanda-specific patterns)
- **Result:** PPO achieved 100% success rate vs DQN's 0% failure
- **Champion:** PPO-AGGRESSIVE with 74.7% improvement over random baseline
- See detailed analysis: [RL Summative Report](https://github.com/eadewusic/Eunice_Adewusi_RL_Summative)

**Phase 2: Iterative Training Refinement (Runs 1-6)**
- Systematic hyperparameter optimization through 6 complete training runs
- Environment simplification: 113D→4D state space, 9→4 actions
- Custom Python/Gymnasium environments: SimpleButtonTrafficEnv → Run7TrafficEnv
- Reward function evolution: Direct absolute → Comparative rewards
- Identified optimal 6:1 throughput:queue ratio (Run 3)
- **Breakthrough:** Run 6 achieved 5/5 scenario wins, 75.8% delay reduction

**Phase 3: Multi-Seed Validation & Hardware Deployment (Runs 7-8)**
- Run 7: Fine-tuned architecture (1.5M steps, single seed)
- Run 8: Multi-seed statistical validation (1M steps × 5 seeds)
- Seeds tested: 42, 123, 456, 789, 1000
- **Champion Model:** Run 8 Seed 789 (highest final reward: 2,066.3)
- Hardware deployment on Raspberry Pi 4 with direct GPIO control
- Firebase Cloud Storage integration for remote monitoring

### Key Achievements

| Metric | Value | Benchmark/Significance |
|--------|-------|------------------------|
| **Simulation Performance** | | |
| Statistical Significance (Reward) | p = 0.0002 | Highly significant (***) |
| Delay Reduction | 8.9% (7.89s → 7.19s) | p = 0.018 (significant *) |
| Queue Reduction | 8.8% (3.42 → 3.12 vehicles) | p = 0.025 (significant *) |
| Baseline Win Rate | 72% (18/25 scenarios) | vs Fixed-Timing controller |
| Control Efficiency | 233% better (2.0 vs 0.6 cars/switch) | Adaptive vs fixed timing |
| **Hardware Validation** | | |
| Mean Inference Time | 6.84ms | Raspberry Pi 4 edge device |
| Real-Time Safety Margin | 14.6× faster | Below 100ms threshold |
| Max Inference Time | 11.16ms | Worst-case performance |
| Throughput (Hardware) | +2.8% improvement | 100% win rate (2/2 tests) |
| Event-Driven Throughput | 100% (53/53, 32/32) | Zero wasted cycles |
| **Reproducibility** | | |
| Coefficient of Variation | 1.3% | Across 5 independent seeds |
| Multi-Seed Range | [1,987.7 - 2,066.3] | All within ±2% of mean |
| Training Efficiency | ~35 min/seed | 1M steps to convergence |
| **Cost Economics** | | |
| Prototype Hardware Cost | $108 (156,700 RWF) | Research validation |
| Production Deployment Cost | $8,000 - $10,000 | Per intersection (estimated) |
| Traditional System Cost | $250,000 - $500,000 | 96-98% cost savings |
| Payback Period (Lagos) | 7 months | 1,000 intersection deployment |

### **Innovation Highlights**

- **Systematic Algorithm Selection:** Empirical comparison of 4 RL families establishing PPO superiority
- **Honest Sim-to-Real Reporting:** Transparent gap documentation (8.8% simulation → 2.8% hardware)
- **Statistical Rigor:** Wilcoxon signed-rank test validation across 25 diverse scenarios
- **Engineering Excellence:** Hybrid rule-based + learned control (Safety Wrapper: 100% efficiency)
- **Hardware-First Design:** State reduction (113D→4D) prioritized over model compression for deployment
- **Real-World Focus:** Hardware validation with LED traffic lights and push-button simulation
- **Reproducibility:** Complete documentation enabling full replication (CV=1.3%)
- **Economic Viability:** Cost analysis demonstrating feasibility for African cities

### Demo Video

> **Watch the full system demonstration:**  
> **[Project Demo Video](https://drive.google.com/drive/folders/1qrGOvRicvj90Pvv2kZNsHaj0y3hPftWn?usp=sharing)**  

**What's in the demo:**
- Training process visualization across multiple runs
- Real-time agent decision-making with inference timing
- Hardware deployment with LED lights and push-button inputs
- Performance comparison: Fixed-Timing vs PPO (Record & Replay methodology)
- Terminal workflows and deployment commands
- Firebase cloud integration demonstration

### Project Links

| Resource | Link | Description |
|----------|------|-------------|
| **GitHub Repository** | [GitHub](https://github.com/eadewusic/Traffic-Optimization-Capstone-Project) | Complete source code |
| **Trained Models (Runs 7 & 8)** | [Google Drive - Models](https://drive.google.com/drive/folders/1Ik6iulDhcPMBermv-7wRNP02IbwNJRua?usp=drive_link) | All 5 seed models (~0.13MB each) |
| **Training Data & Logs** | [Google Drive - Data](https://drive.google.com/drive/folders/1Q8K8wo0kLMLhonOluAwU3bSakkX6rm7T?usp=drive_link) | Raw training data and performance logs |
| **Phase 1 Algorithm Comparison** | [RL Summative Report](https://docs.google.com/document/d/e/2PACX-1vRgDXbk0JeR4inCIdFiJgBXPihQkZIn-WNyjCnIByFqJ3u_PKt7N-ieJ0esTVzH3s2Jbs-eQu97EkQ1/pub) | 17-configuration systematic evaluation |

## Complete Training Evolution (Runs 1-8)

### Run 1: Original Baseline - "The Starting Point"

**Configuration:**
```python
Architecture:     [64, 64] (10K parameters)
Training Steps:   100,000
Learning Rate:    5e-4 (fixed)
Batch Size:       64
Reward Ratio:     13:1 (throughput-heavy)
  - Throughput:   +2.0 per vehicle cleared
  - Queue:        -0.15 per vehicle waiting
```

**Results:**
```
Average Reward:    302.6
Throughput:        239.2 vehicles cleared
Average Queue:     23.0 vehicles
Scenarios Won:     2/5
```

**Key Findings:**
- High rewards in balanced scenarios
- Good throughput performance
- ❌ Critical failure in directional congestion (North Heavy: 87 vs baseline 491)
- ❌ Over-emphasized throughput at expense of queue management
- Training instability: Peak 1677 → Final 1273 (-24% drop)

**Lesson Learned:** Reward function imbalance (13:1 ratio) prioritizes clearing vehicles but ignores dangerous queue buildup.

### Run 2: Deep Network Attempt - "The Kitchen Sink Failure"

**Configuration:**
```python
Architecture:     [128, 64, 32] (20K parameters - DOUBLED)
Training Steps:   150,000 (+50% longer)
Learning Rate:    5e-4 → 5e-5 (linear decay added)
Batch Size:       128 (DOUBLED)
Entropy Coef:     0.02 (DOUBLED for exploration)
Reward Ratio:     2.5:1 (TOO conservative)
```

**Results:**
```
Average Reward:    -170.3 (ALL NEGATIVE)
Throughput:        244.2 vehicles
Average Queue:     24.6 vehicles
Scenarios Won:     0/5 (LOST BOTH of Run 1's wins)
```

**Lesson Learned:** NEVER change everything at once! 2.5:1 reward ratio made agent too risk-averse. "More complex" ≠ "better" - overfitting with 20K parameters for simple 4D state space.

### Run 3: Balanced Reward - "The Goldilocks Breakthrough"

**Configuration:**
```python
Architecture:     [64, 64] (REVERTED to simple)
Training Steps:   150,000
Learning Rate:    5e-4 → 5e-5 (decay kept)
Reward Ratio:     6:1 (THE GOLDILOCKS RATIO)
  - Throughput:   +1.5 per vehicle
  - Queue:        -0.25 per vehicle
```

**Results:**
```
Average Reward:    122.0
Throughput:        242.2 vehicles
Average Queue:     18.4 (BEST)
Scenarios Won:     1/5
```

**Key Achievement:** 
- SOLVED the North Heavy problem: 286 vs Run 1's 87 (+229% improvement)
- Best queue management: 18.4 avg (20% better than Run 1)
- Most stable training: Peak 1081 → Final 852 (-21% drop)

**Lesson Learned:** The 6:1 reward ratio is optimal - balances throughput incentive with queue penalty. Not too aggressive (13:1), not too conservative (2.5:1).

### Run 4a: Extended Training - "The Simple Champion"

**Configuration:**
```python
Architecture:     [64, 64] (simple network)
Training Steps:   200,000 (LONGEST YET)
Learning Rate:    5e-4 → 5e-5 (decay)
Reward Ratio:     6:1 (kept from Run 3)
```

**Results:**
```
Average Reward:    290.8 (HIGHEST POSITIVE)
Throughput:        243.6 vehicles
Average Queue:     20.2 vehicles
Scenarios Won:     4/5 (MOST WINS)
```

**Lesson Learned:** Simple architecture + extended training + balanced rewards = winner. Proves hypothesis that architectural simplicity with sufficient training beats complex networks.

### Run 4b: Friend's Deep Network - "The Overfitting Lesson"

**Configuration:**
```python
Architecture:     [128, 64, 32] (deep network)
Training Steps:   200,000 (same as Run 4a)
Reward Ratio:     6:1 (same as Run 4a)
Purpose:          Direct comparison - architecture impact only
```

**Results:**
```
Average Reward:    181.7
Throughput:        241.2 vehicles
Average Queue:     16.6 (lowest, but misleading)
Scenarios Won:     0/5 ❌ (LOST ALL)
Training Issues:   15 KL constraint violations
```

**Lesson Learned:** Architecture complexity must match problem complexity. For 4-dimensional state space, [64, 64] is optimal. More parameters ≠ better performance.

### Run 5: Broken Rewards - "The Cautionary Tale"

**Configuration:**
```python
Architecture:     Unknown
Training Steps:   250,000 (LONGEST)
Reward Function:  BROKEN ❌ (sign error or massive penalty)
```

**Results:**
```
Average Reward:    -781.3 (WORST EVER)
All Training Rewards: NEGATIVE throughout 250K steps
Scenarios Won:     3/5* (MISLEADING - wins by being "less terrible")
```

**Lesson Learned:** Training duration cannot fix a broken reward function. 250K steps with wrong rewards < 100K steps with correct rewards. Always validate reward function on small episodes first.

### Run 6: Comparative Reward - "The Capstone Foundation"

**Configuration:**
```python
Architecture:     [128, 64, 32] (3-layer network)
Training Steps:   ~150,000
Learning Rate:    3e-4 (lowered from 5e-4)
Reward Function:  COMPARATIVE (reward relative to baseline)
Environment:      Run7TrafficEnv (enhanced version)
Key Innovation:   Reward = Agent Performance - Baseline Performance
```

**Results:**
```
Average Reward:    +575.9 (vs baseline -204.2)
Win Rate:          5/5 (100% of test scenarios)
Delay Reduction:   75.8% (massive improvement)
Throughput:        86.6 vs 85.4 cars cleared
Status:            BREAKTHROUGH - Selected for Capstone
```

**Lesson Learned:** Comparative rewards explicitly optimize for superiority over baseline. This run proved PPO could consistently beat traditional controllers, establishing foundation for capstone refinement.

### Run 7: Fine-Tuning Run 6

**Configuration:**
```python
Architecture:     [128, 64, 32] (from Run 6)
Training Steps:   1,502,000 (10× longer than Run 6!)
Learning Rate:    3e-4
Seed:             NOT explicitly set (random initialization)
Environment:      Run7TrafficEnv (comparative rewards)
```

**Results:**
```
Initial Reward:   1,703.3
Best Reward:      2,066.9 (at step 778,000)
Final Reward:     2,041.5 ± 17.9
Improvement:      +363.5 points (+21.3%)
Training Time:    ~2-3 hours
```

**Key Finding:** Strong performance but ⚠️ not reproducible (no explicit seed). Led directly to Run 8's multi-seed validation strategy.

### Run 8: Multi-Seed Validation - "The Statistical Champion"

**Configuration:**
```python
Architecture:     [128, 64, 32] (same as Run 7)
Training Steps:   1,000,000 per seed (33% less than Run 7!)
Learning Rate:    3e-4
Batch Size:       64
Seeds:            5 independent runs (42, 123, 456, 789, 1000)
Purpose:          Prove reproducibility and statistical significance
```

**Individual Seed Results:**

| Seed | Final Reward | Best Reward | Duration | Status |
|------|-------------|-------------|----------|---------|
| 42   | 1,987.7     | ~2,000      | 1:36:42  |         |
| 123  | 2,042.2     | ~2,100      | 0:36:41  |         |
| 456  | 2,029.9     | 2,074.7     | 0:32:42  |         |
| **789** | **2,066.3** | **2,066.3** | **0:34:00** | **Champion** |
| 1000 | ~2,010      | ~2,050      | 0:35:00  |         |

**Aggregate Statistics:**
```
Mean Reward:      2,025.3 ± 4.7
Median Reward:    2,029.9
Range:            [1,987.7, 2,066.3]
Coefficient of Variation (CV): 1.3% (EXCELLENT - industry standard <5%)
Champion Model:   Seed 789 (highest final reward)
```

**Statistical Testing (Champion Model - 25 Scenarios):**

**Simulation Results:**
- Win Rate: 72% (18/25 scenarios)
- Reward improvement: p = 0.0002 (highly significant ***)
- Delay reduction: 8.9% (p = 0.018, significant *)
- Queue reduction: 8.8% (p = 0.025, significant *)

**Hardware Validation (Raspberry Pi 4):**
- Mean inference time: 6.84ms (real-time capable)
- Max inference time: 11.16ms (14.6× safety margin under 100ms)
- Throughput improvement: +2.8% (hardware tests)
- Phase efficiency: 2.0 cars/switch vs 0.6 baseline (233% better)

**Lesson Learned:** Multi-seed validation proves the approach is robust, reproducible, and statistically significant. Seed 789 champion model ready for production deployment with high confidence.

### Summary: Training Evolution Tables

**Table 1: Complete Run Comparison**

| Run | Architecture | Steps | Seeds | Final Reward | Key Achievement |
|-----|--------------|-------|-------|--------------|-----------------|
| 1 | [64,64] | 100K | 1 | 302.6 | Identified North Heavy problem |
| 2 | [128,64,32] | 150K | 1 | -170.3 | Multi-variable failure lesson |
| 3 | [64,64] | 150K | 1 | 122.0 | Found 6:1 Goldilocks ratio |
| 4a | [64,64] | 200K | 1 | 290.8 | Simple + long training wins |
| 4b | [128,64,32] | 200K | 1 | 181.7 | Proved simple > complex |
| 5 | Unknown | 250K | 1 | -781.3 | Reward validation importance |
| 6 | [128,64,32] | ~150K | 1 | +575.9 | Comparative rewards, 5/5 wins |
| 7 | [128,64,32] | 1,502K | 1 | 2,041.5 | Fine-tuning start, no seed |
| 8 | [128,64,32] | 1,000K | **5** | **2,066.3** | **Multi-seed validated** |

**Table 2: Key Lessons by Run**

| Run | Primary Lesson | Evidence | Impact on Next Run |
|-----|----------------|----------|-------------------|
| 1 | Reward imbalance causes failure | North Heavy: 87 score, 44 car queue | Run 3: Balanced reward to 6:1 |
| 2 | Don't change everything at once | 6 simultaneous changes → impossible debug | Run 3: Changed only rewards |
| 3 | 6:1 is Goldilocks ratio | Solved North Heavy (286), best queue (18.4) | Run 4a: Kept 6:1, extended training |
| 4a | Simple + long > complex + short | [64,64] + 200K > [128,64,32] + 200K | Proved simplicity thesis |
| 4b | Match complexity to problem | 20K params overfits 4D state space | Validated [64,64] choice |
| 5 | Validate rewards before training | -781.3 avg across 250K steps | Check rewards on pilot episodes |
| 6 | Comparative rewards beat baseline | 5/5 wins, 75.8% delay reduction | Foundation for capstone |
| 7 | Need reproducibility validation | Single seed, no statistical proof | Run 8: Multi-seed protocol |
| 8 | Multi-seed proves robustness | CV=1.3%, p=0.0002 significance | Ready for deployment |

## Technical Architecture

### Environment Specification

**Custom Simulation**: `SimpleButtonTrafficEnv` → `Run7TrafficEnv`

**State Space Evolution:**

| Phase | Environment | State Dimensions | Action Space | Rationale |
|-------|------------|------------------|--------------|-----------|
| **Phase 1** | TrafficJunctionEnv | 113D (grids, queues, flags, timer) | Discrete(9) | Algorithm selection, comprehensive |
| **Phase 2-3** | Run7TrafficEnv | 4D (queue lengths only) | Discrete(2) | Hardware deployment, sim-to-real |

**Final Deployment Specification:**
- **Observation Space**: 4-dimensional continuous
  ```python
  state = [q_north, q_south, q_east, q_west]
  Range: [0, 20] vehicles per direction
  Type: Box(low=0, high=20, shape=(4,), dtype=np.float32)
  ```

- **Action Space**: Discrete(2)
  ```python
  Action 0: North-South lanes get green light
  Action 1: East-West lanes get green light
  ```

- **Step Duration**: 5 seconds per decision cycle (hardware deployment)
- **Episode Length**: Variable (event-driven mode) or 60s (timed mode)
- **Domain Randomization**: Hardware-aware (GPIO latency, button debounce, processing jitter)

**Traffic Dynamics**:
```python
Vehicle arrival:       Push-button simulation (manual input for HIL testing)
Queue capacity:        20 vehicles per lane
Yellow transition:     2.0 seconds (mandatory safety buffer)
Clearing rate:         2 vehicles per 5-second green phase
```

### Reward Function Evolution

**Runs 1-5: Direct Absolute Rewards**
```python
reward = throughput_reward + queue_penalty
optimal_ratio = 6:1 (discovered in Run 3)
```

**Runs 6-8: Comparative Rewards (Final Approach)**
```python
reward = agent_performance - baseline_performance
baseline = longest_queue_heuristic()
philosophy = explicitly_beat_baseline()
```

### PPO Agent Architecture (Run 8 - Deployed Model)

**Neural Network Structure:**

![PPO Agent Architecture](./images/PPO-Agent-Architecture.png)

*Figure: Three-layer fully-connected neural network (128-64-32) with dual heads for policy (action probabilities) and value (state estimation). VecNormalize wrapper standardizes 4D queue state inputs before processing.*
```
Input Layer (4D)
    ↓
[Observation Normalization: VecNormalize]
    ↓
Hidden Layer 1: 128 neurons (ReLU activation)
    ↓
Hidden Layer 2: 64 neurons (ReLU activation)
    ↓
Hidden Layer 3: 32 neurons (ReLU activation)
    ↓
    ├──→ Policy Head: 2 neurons (Softmax) → Action probabilities
    └──→ Value Head: 1 neuron (Linear) → State value V(s)
```

**Training Configuration (Run 8):**
```python
Learning rate:      3e-4 (fixed)
Batch size:         64
N-steps:            2048 (rollout buffer)
Epochs per update:  10
Gamma (γ):          0.99 (discount factor)
GAE lambda (λ):     0.95 (advantage estimation)
Clip range:         0.2 (PPO clipping)
Entropy coef:       0.01 (exploration)
Value coef:         0.5 (critic loss weight)
Max grad norm:      0.5 (gradient clipping)
Total timesteps:    1,000,000 per seed
```

**Model Size:**
- Parameters: ~20K (3-layer architecture)
- File size: 0.13MB (ppo_final_seed789.zip)
- VecNormalize: 1.8KB (vec_normalize_seed789.pkl)

**Why GAE (Generalized Advantage Estimation)?**

GAE calculates "how much better was this action than expected" using:

```python
Advantage = Σ (γλ)^t δ_t   (sum from t=0 to n)

where:
  δ_t = r_t + γV(s_{t+1}) - V(s_t)  # temporal difference error
  γ = 0.99   # discount factor for future rewards
  λ = 0.95   # bias-variance tradeoff parameter
```

**Why λ = 0.95?**
- λ = 0: Only 1-step (high bias, low variance)
- λ = 1: All future steps (low bias, high variance)
- λ = 0.95: Balanced (standard for PPO)

### System Architecture

**End-to-End System Overview:**

![System Architecture](./images/Capstone-System-Architecture.jpg)

*Figure: Complete system architecture showing three integrated phases: (1) Training Phase - PPO algorithm trained in simulation environment across multiple seeds, (2) Transfer Phase - Trained model deployed to Raspberry Pi 4 edge device, (3) Operation Phase - Real-time GPIO control with local-first logging and asynchronous Firebase cloud backup.*

The system architecture demonstrates:
- **Training**: Simulation-based learning with multi-seed validation (Run 8: seeds 42, 123, 456, 789, 1000)
- **Deployment**: Model compression (0.13MB) enables edge inference on resource-constrained hardware
- **Operation**: Hybrid control (PPO + Safety Wrapper) ensures 100% reliability with zero wasted cycles
- **Monitoring**: Local-first data storage with optional cloud synchronization for remote analysis

### IoT-to-Cloud Data Flow

![IoT-to-Cloud Pipeline](./images/IoT-to-Cloud-Pipeline.png)

*Figure: IoT-to-Cloud architecture showing process isolation between real-time GPIO control (Process 1) and asynchronous cloud upload (Process 2). Raspberry Pi 4 maintains 100% operational capability during internet outages, with local CSV logs, JSON stats, and PNG plots stored for later synchronization to Firebase Cloud Storage.*

**Key Design Decisions:**
- **Process Isolation**: Separate processes prevent Firebase SDK threading conflicts with GPIO interrupts
- **Local-First**: System continues operating without internet (critical for African infrastructure)
- **Selective Upload**: Duplicate detection prevents redundant cloud storage costs
- **Resilient**: Graceful failure handling preserves local data if upload fails

## Multi-Seed Validation (Run 8)

### Reproducibility Protocol

**Seeds Selected**: 42, 123, 456, 789, 1000

**Justification**:
- **Seed 42**: ML community standard (Hitchhiker's Guide reference)
- **123, 456, 789**: Sequential for traceability
- **1000**: Different magnitude to test scale independence

**Training Configuration** (Identical across all seeds):
```python
Architecture:  [128, 64, 32] (fixed)
Steps:         1,000,000 (fixed)
Learning rate: 3e-4 (fixed)
Batch size:    64 (fixed)
Environment:   Run7TrafficEnv (fixed)
Only variable: Random seed initialization
```

### Multi-Seed Results Analysis

**Individual Performance:**

```
Seed 42:   1,987.7 (lowest, but still strong)
Seed 123:  2,042.2 (good)
Seed 456:  2,029.9 (median)
Seed 789:  2,066.3 (champion - highest final reward)
Seed 1000: 2,010.0 (good)

Mean ± Std: 2,025.3 ± 4.7
```

**Statistical Metrics:**
- **Coefficient of Variation**: 1.3% (excellent - industry standard <5%)
- **Range**: 78.6 points (1,987.7 to 2,066.3)
- **Consistency**: All seeds within 4% of mean

**Champion Selection:**
- **Seed 789** selected based on highest final reward (2,066.3)
- Matched Run 7's best performance (2,066.9 vs 2,066.3)
- Most consistent across test scenarios
- Deployed to Raspberry Pi 4 hardware for validation

### Training Progression Visualization

![Multi-Seed Training Curves](./images/run8_multiseed_training.png)

*Figure: Training reward progression across 5 independent seeds (42, 123, 456, 789, 1000) showing consistent convergence around 600K steps with final rewards clustered within ±2% (CV=1.3%), validating reproducibility.*

**Key Observations:**
- All seeds converge within 600K-800K steps
- Final performance cluster: [1,987.7 - 2,066.3]
- Seed 789 (champion) maintains highest final reward
- Low variance demonstrates training stability

## Statistical Testing

### Baseline Comparison (Champion Model - Seed 789)

**Test Setup:**
- **Controller 1**: Run 8 Seed 789 (PPO agent)
- **Controller 2**: Fixed-timing baseline (longest-queue heuristic)
- **Scenarios**: 25 diverse traffic patterns
- **Method**: Paired testing (same traffic seed for both controllers)
- **Statistical Test**: Wilcoxon signed-rank (paired, non-parametric)

**Hypotheses:**
```
H₀: No difference between PPO and baseline
H₁: PPO ≠ Baseline
α = 0.05 (significance level)
```

### Simulation Results (25 Scenarios)

| Metric | Fixed-Timing | PPO (Run 8) | Improvement | p-value | Significance |
|--------|--------------|-------------|-------------|---------|--------------|
| **Mean Reward** | 2,073.8 ± 11.9 | 2,078.5 ± 12.3 | +0.2% | **0.0002** | *** Highly Sig |
| **Mean Delay (s)** | 7.89 ± 0.91 | 7.19 ± 0.84 | **-8.9%** | **0.018** | * Significant |
| **Mean Queue** | 3.42 ± 0.67 | 3.12 ± 0.61 | **-8.8%** | **0.025** | * Significant |
| **Throughput (%)** | 96.8 ± 1.3 | 97.1 ± 1.2 | +0.3pp | 0.234 | ns Not Sig |
| **Win Rate** | 28% (7/25) | **72% (18/25)** | +44pp | - | Dominant |
| **Control Efficiency** | 0.6 cars/switch | 2.0 cars/switch | **+233%** | - | Adaptive |

*Significance: *** p<0.001 (highly), ** p<0.01 (very), * p<0.05 (significant), ns = not significant, pp = percentage points*

**Key Findings:**
- **Highly significant reward improvement** (p=0.0002) indicates PPO consistently outperforms
- **Significant delay reduction** (8.9%, p=0.018) translates to 7 seconds saved per vehicle
- **Significant queue reduction** (8.8%, p=0.025) demonstrates better congestion management
- **Similar throughput** (96.8% vs 97.1%, p=0.234) shows PPO achieves efficiency without sacrificing capacity
- **233% better control efficiency** (2.0 vs 0.6 cars/switch) proves adaptive superiority

**Interpretation:**
The p-value of 0.0002 for reward metric provides extremely strong evidence that PPO consistently outperforms the baseline, even though the magnitude (0.2%) is small in that metric. This occurs because:
1. High consistency across 25 scenarios (low variance)
2. PPO wins 72% of scenarios (strong directionality)
3. Paired testing eliminates scenario difficulty confounds

## Hardware Deployment

### Raspberry Pi 4 Specifications

**Hardware Platform:**
- **Board**: Raspberry Pi 4 Model B (2GB RAM version)
- **CPU**: Quad-core ARM Cortex-A72 @ 1.5 GHz
- **RAM**: 2GB LPDDR4-3200 SDRAM
- **OS**: Raspberry Pi OS (64-bit, Debian 12-based)
- **Python**: 3.9.2
- **Power**: 5V DC @ 3A via USB-C (15W)
- **Storage**: 32GB microSD Class 10
- **Cost**: 156,700 RWF (≈$108 USD)

### Complete Bill of Materials

| Category | Item | Qty | Cost (RWF) | Purpose |
|----------|------|-----|------------|---------|
| **COMPUTING** | Raspberry Pi 4 Model B (2GB) | 1 | 76,900 | Main computing unit for PPO inference |
| | Power Supply (5V 3A USB-C) | 1 | 17,000 | Stable 15W power delivery |
| | MicroSD Card (32GB, Class 10) | 1 | 16,000 | OS, models, deployment logs |
| **INPUT** | Tactile Push Buttons (11mm) | 4 | 4,000 | Simulate vehicle arrivals (N/E/S/W) |
| **OUTPUT** | Traffic Light LED Modules | 4 | 2,800 | 12 LEDs total (4 directions × 3 colors) |
| **ASSEMBLY** | Solderless Breadboard (830 points) | **4** | 14,000 | **Modular cross-pattern design (1 per direction)** |
| | Jumper Wires (M-M, M-F, F-F) | 3 packs | 6,000 | GPIO connections |
| **COOLING** | Heatsink + Cooling Fan Kit | 1 | 10,000 | Thermal management (<70°C) |
| | Raspberry Pi 4 Case (Red/White) | 1 | 9,000 | Protection & fan mounting |
| | Heatsink (17×15×7mm U-shaped) | 5 | 1,000 | Cover sensitive Pi components |
| **TOTAL** | | | **156,700** | **≈$108 USD** |

### Hardware Architecture: 4-Breadboard Modular Design

The system uses a **modular 4-breadboard (BB) cross-pattern layout** that physically mirrors a real 4-way intersection:
```
          [BB1 - NORTH]
                 ↑
                 |
[BB4 - WEST] ← [Pi] → [BB2 - EAST]
                 |
                 ↓
          [BB3 - SOUTH]
```

![4-Breadboard Prototype Architecture](./images/Figure-4.9-Prototype-4Breadboard-Setup.JPG)

*Figure: Prototype on 4-breadboard architecture setup showing modular cross-pattern layout. Raspberry Pi 4 (center) connects to four independent breadboards arranged in cardinal directions (North-top, East-right, South-bottom, West-left). Each breadboard contains one LED traffic light module (3 LEDs: Red/Yellow/Green) and one tactile push button for vehicle arrival simulation. Common ground rails interconnect all modules for synchronized GPIO control.*

**Design Philosophy:**
- **One breadboard per direction** (North, East, South, West)
- **One LED module + One button** per breadboard
- **Physical layout mimics intersection geometry** for intuitive operation
- **Common ground rails** connect all 4 breadboards to Raspberry Pi GND

**Module Distribution:**

| Breadboard | Position | Components | GPIO Pins Used |
|------------|----------|------------|----------------|
| **BB1 (North)** | Top | 1 LED module (R/Y/G) + 1 button | GPIO 16, 20, 21, 9 |
| **BB2 (East)** | Right | 1 LED module (R/Y/G) + 1 button | GPIO 5, 6, 13, 10 |
| **BB3 (South)** | Bottom | 1 LED module (R/Y/G) + 1 button | GPIO 23, 24, 25, 22 |
| **BB4 (West)** | Left | 1 LED module (R/Y/G) + 1 button | GPIO 14, 4, 18, 17 |

**Advantages:**
- **Modularity**: Each direction is independent, simplifying debugging
- **Scalability**: Easy to add/remove directions during prototyping
- **Visual Clarity**: Physical layout matches mental model of intersection
- **Clean Wiring**: Organized GPIO connections reduce errors

### GPIO Pin Configuration

**LED Outputs (12 LEDs across 4 breadboards):**

| Breadboard | Lane | Signal | GPIO | Pin | Connection |
|------------|------|--------|------|-----|------------|
| **BB1 (North)** | North | Red | GPIO 16 | Pin 36 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Yellow | GPIO 20 | Pin 38 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Green | GPIO 21 | Pin 40 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| **BB2 (East)** | East | Red | GPIO 5 | Pin 29 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Yellow | GPIO 6 | Pin 31 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Green | GPIO 13 | Pin 33 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| **BB3 (South)** | South | Red | GPIO 23 | Pin 16 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Yellow | GPIO 24 | Pin 18 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Green | GPIO 25 | Pin 22 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| **BB4 (West)** | West | Red | GPIO 14 | Pin 8 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Yellow | GPIO 4 | Pin 7 | GPIO → 331Ω → LED(+) → LED(–) → GND |
| | | Green | GPIO 18 | Pin 12 | GPIO → 331Ω → LED(+) → LED(–) → GND |

**Button Inputs (4 buttons across 4 breadboards):**

| Breadboard | Direction | GPIO | Pin | Pull | Connection |
|------------|-----------|------|-----|------|------------|
| **BB1 (North)** | North | GPIO 9 | Pin 21 | DOWN | Button → GND (Pi Pin 39) |
| **BB2 (East)** | East | GPIO 10 | Pin 19 | DOWN | Button → GND (Pi Pin 39) |
| **BB3 (South)** | South | GPIO 22 | Pin 15 | DOWN | Button → GND (Pi Pin 39) |
| **BB4 (West)** | West | GPIO 17 | Pin 11 | DOWN | Button → GND (Pi Pin 39) |

**Cooling Fan:**

| Component | Signal | Pin | Connection |
|-----------|--------|-----|------------|
| Fan +5V | 5V | Pin 4 | Power |
| Fan GND | GND | Pin 6 | Ground |
| Fan PWM | GPIO 3 | Pin 5 | Speed control (PWM) |

**Common Grounds:**
- All LEDs → GND (Pin 25) via Breadboard 2
- All Buttons → GND (Pin 39) via Breadboard 1
- Breadboards interconnected for ground synchronization

### Circuit Schematic

![Circuit Schematic Diagram](./hardware-diagrams/SCH_Schematic.png)

*Figure: Complete electrical schematic showing Raspberry Pi 4 GPIO connections to 4-breadboard modular traffic light system. Each direction (N/E/S/W) includes 3 LEDs (Red/Yellow/Green) with 331Ω current-limiting resistors and 1 tactile push button with internal pull-down. All modules share common ground rails connected to Pi GND pins 25 and 39.*

**Electrical Specifications:**
- **LED Forward Voltage**: 2.0-2.2V (Red/Yellow), 3.0-3.2V (Green)
- **LED Forward Current**: 20mA per LED (max 240mA total for 12 LEDs)
- **Resistor Values**: 331Ω for 5V GPIO (calculated: (5V - 2V) / 20mA ≈ 150Ω, using 331Ω for safety margin)
- **Button Debounce**: 300ms software debounce to prevent false triggers
- **Total GPIO Pins**: 16 (12 LEDs + 4 buttons)

### Hardware Assembly & Wiring

**Raspberry Pi 4 Pinout Reference:**

![Raspberry Pi 4 GPIO Pins Description](./images/Raspberry-Pi-4-GPIO-Pins-Description.png)

**Assembly Steps:**

1. **Prepare Raspberry Pi:**
   - Install heatsinks on CPU, RAM, and power management chips
   - Mount cooling fan to official red/white case
   - Insert 32GB microSD card with Raspberry Pi OS pre-loaded
   - **Do not power on yet**

2. **Arrange 4 Breadboards in Cross Pattern:**
```
   Place breadboards physically matching intersection geometry:
   - BB1 (North) at top position
   - BB2 (East) at right position  
   - BB3 (South) at bottom position
   - BB4 (West) at left position
   - Center: Raspberry Pi in case
```

3. **Wire Each Breadboard Module (Repeat 4×):**
   
   **For Breadboard 1 (North) - Example:**
   - Insert 1 LED module (contains 3 LEDs: Red/Yellow/Green with built-in resistors)
   - Insert 1 push button
   - Connect LED Red → GPIO 16 (Pin 36)
   - Connect LED Yellow → GPIO 20 (Pin 38)
   - Connect LED Green → GPIO 21 (Pin 40)
   - Connect Button → GPIO 9 (Pin 21)
   - Create GND rail on breadboard
   - Connect all LED cathodes to GND rail
   - Connect button terminal to GND rail
   
   **Repeat for BB2 (East), BB3 (South), BB4 (West) with respective GPIO pins from table above**

4. **Interconnect Ground Rails:**
   - Use M-M jumper wires to link GND rails of all 4 breadboards
   - Connect unified GND to Pi Pin 39 (buttons) and Pin 25 (LEDs)
   - Verify continuity across all GND points with multimeter

5. **Power Connections:**
   - Fan +5V → Pi Pin 4
   - Fan GND → Pi Pin 6
   - Fan PWM → GPIO 3 (Pin 5)
   - **Do not connect Pi power yet**

6. **Pre-Power Checklist:**
   - ✓ All GPIO connections match pin table above
   - ✓ No shorts between power/ground rails
   - ✓ LED polarities correct (long leg = anode = +)
   - ✓ All 4 breadboard GND rails interconnected
   - ✓ Cooling fan oriented for exhaust airflow

7. **Power On:**
   - Connect 5V 3A USB-C power adapter to Pi
   - Verify cooling fan spins immediately
   - Boot Raspberry Pi OS (first boot ~60 seconds)

**Safety & Best Practices:**
- ⚠️ **Never hot-plug GPIO** - always power off before wiring changes
- Use **color-coded jumpers**: Red (5V), Black (GND), others (GPIO signals)
- Keep wiring **taut and organized** to prevent accidental disconnections
- **Test continuity** with multimeter before powering on
- Monitor Pi temperature: Should stay **<60°C** with fan during operation

**Thermal Management:**
- Heatsink + fan combo maintains <60°C during sustained PPO inference
- Case ventilation holes align with fan for optimal airflow
- **Monitor temperature**: `vcgencmd measure_temp` (should report <65°C under load)

### Physical Setup Circuit

**4-Breadboard Cross-Pattern Layout:**

- **LEDs OFF** (Standby State):

![LEDs OFF - 4-Breadboard Layout](./images/LEDs-OFF.JPG)

*Note: Physical cross arrangement (North-top, East-right, South-bottom, West-left) mirrors actual 4-way intersection geometry for intuitive operation. Each breadboard represents one intersection approach.*

- **LEDs ON** (Active State - North/South Green):

![LEDs ON - Active Traffic Control](./images/LEDs-ON.JPG)

*Demonstration: North and South lanes showing green signals (N/S Green phase), while East and West lanes display red signals. Button presses on each breadboard simulate vehicle arrivals from respective directions.*

**Architecture Highlights:**
- **Modular Design**: Each breadboard represents one intersection approach
- **Independent Modules**: Easy to debug individual directions
- **Scalable**: Can add/remove breadboards for different intersection types
- **Visual Clarity**: Physical layout matches traffic flow mental model

### Hardware Validation Results

**Mode: Event-Driven Validation (Continuous Operation)**

Two consecutive sessions validated continuous operation until all queues clear:

| Metric | Session 1 (Heavy) | Session 2 (Light) | Status |
|--------|------------------|------------------|--------|
| Duration | 114.7s | 74.0s | Adaptive |
| Vehicles Arrived | 53 | 32 | Variable load |
| Vehicles Cleared | 53 (100%) | 32 (100%) | Perfect throughput |
| Peak Congestion | 17 cars | 8 cars | Handled extremes |
| Phase Changes | 11 (61% of steps) | 6 (50% of steps) | Scaled with load |
| Mean Inference | 6.80ms | 7.79ms | Real-time |
| Max Inference | 9.73ms | 10.23ms | <11ms worst-case |

**Key Finding:** 100% throughput maintained across 65% traffic variance with zero wasted cycles.

**Mode: Comparative Performance (Record & Replay)**

Two tests using identical traffic patterns for fair comparison:

| Metric | Test 1 | Test 2 | Average |
|--------|--------|--------|---------|
| **Fixed-Timing Throughput** | 32/40 (80.0%) | 28/32 (87.5%) | 83.75% |
| **PPO Throughput** | 33/40 (82.5%) | 29/32 (90.6%) | 86.55% |
| **Improvement** | +2.5% (+1 car) | +3.1% (+1 car) | **+2.8%** |
| **Phase Changes (Fixed)** | 5 | 5 | 5 |
| **Phase Changes (PPO)** | 4 | 4 | 4 |
| **Phase Reduction** | -20% | -20% | **-20%** |
| **PPO Win Rate** | ✓ Win | ✓ Win | **100% (2/2)** |

**Key Finding:** PPO cleared +1 vehicle in both tests using 20% fewer phase changes, demonstrating consistent adaptive advantage.

**Real-Time Performance (All Modes Combined):**

| Metric | Value | Target | Safety Margin | Status |
|--------|-------|--------|---------------|--------|
| Mean Inference | 6.84ms | <100ms | 14.6× | ✓ Pass |
| Max Inference | 11.16ms | <100ms | 9.0× | ✓ Pass |
| Std Deviation | ±1.15ms | <5ms | Low variance | ✓ Pass |
| 95th Percentile | ~10.5ms | <50ms | 4.8× | ✓ Pass |

**System Resources:**

| Resource | Usage | Available | Headroom | Status |
|----------|-------|-----------|----------|--------|
| RAM | 363 MB peak | 2 GB | 82% free | Excellent |
| CPU | 0-10% | 100% | 90% available | Excellent |
| Temperature | 32-36°C | 70°C throttle | 48% margin | Safe |

**Stability Metrics:**
- Zero GPIO interference issues (process isolation successful)
- Zero thermal throttling events
- Zero crashes during extended testing
- 100% button detection reliability (300ms debounce)

### Sim-to-Real Transfer Analysis

**Transfer Success Validation:**

| Aspect | Simulation | Hardware | Transfer Status |
|--------|-----------|----------|-----------------|
| Real-time capability | <6ms | 6.84ms mean | ✓ Validated |
| Throughput improvement | +0.3% (ns) | +2.8% | ✓ Better in hardware |
| Win rate | 72% (18/25) | 100% (2/2) | ✓ Consistent |
| Phase optimization | Adaptive | 20% fewer changes | ✓ Confirmed |
| Sample size | n=25 | n=2 comparison, n=2 event | Limited hardware tests |

**Performance Gap Explanation:**

**Why Hardware +2.8% vs Simulation +8.8%?**

1. **Measurement Differences:**
   - Simulation: Continuous queue tracking every timestep
   - Hardware: Discrete button press counting

2. **Traffic Pattern Characteristics:**
   - Hardware tests: Relatively balanced traffic
   - Simulation: Included extreme imbalanced scenarios
   - Impact: PPO's adaptive advantage less pronounced with balanced traffic

3. **Sample Diversity:**
   - Simulation: 25 diverse scenarios (balanced, directional, random)
   - Hardware: 2 tests with similar manual button patterns
   - Impact: Limited coverage of PPO's full adaptive range

4. **Short Test Duration:**
   - Hardware: 60-second windows (10-12 decision cycles)
   - Simulation: Full episode convergence
   - Impact: Limited opportunity to demonstrate long-term optimization

**Transfer Validation Conclusion:**

Despite the numerical gap, hardware deployment successfully validated:
- ✓ Real-time operation: 14.6× safety margin confirmed
- ✓ Adaptive superiority: Consistent wins (2/2) with appropriate adaptations
- ✓ Production feasibility: Stable, reliable, resource-efficient
- ✓ Scalable architecture: Multi-session, edge-based, IoT-integrated

The +2.8% hardware improvement validates **realistic expectations** compared to simulation-only studies that report inflated gains.

## Firebase Cloud Integration

The system implements **Firebase Cloud Storage** for remote data backup, monitoring, and distributed traffic management. This IoT-to-Cloud architecture enables city-wide scalability while maintaining local-first resilience.

### Architecture Philosophy: Local-First Design

- **Primary Operation:** System runs 100% offline on Raspberry Pi
- **Cloud Role:** Backup and remote monitoring only
- **Critical Feature:** No cloud dependency for real-time control
- **Resilience:** Continues operating during internet outages

### Firebase Services Used

**Firebase Storage (Cloud Object Storage)**
- **Purpose:** Store deployment logs, performance metrics, visualization plots
- **Structure:** Organized by deployment run and timestamp hierarchy
- **Cost:** Free tier sufficient (<1GB/month for research MVP)
- **Access:** HTTPS URLs for web-based retrieval via Firebase Console

**Why NOT Firebase Realtime Database?**
- Our use case: File storage and backup (Firebase Storage is correct choice)
- Firestore/RTDB: For structured real-time queries (not needed for batch logging)

### Critical Technical Challenge: Threading Conflict

**Problem:** Firebase SDK background threads interfered with GPIO edge detection

**Symptoms:**
- Button presses stopped being detected during deployment
- GPIO callbacks not firing when Firebase was active
- System appeared frozen to user input

**Root Cause:**
```python
# BROKEN: Threading conflict
import firebase_admin
from RPi import GPIO

firebase_admin.initialize_app()  # Spawns background threads
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=on_button_press)

# CONFLICT: Firebase threads block GPIO interrupts
# Result: Callbacks never fire, buttons don't work
```

**5 Attempted Solutions:**
1. ❌ Different threading modes (`daemon=True` flags) → No effect
2. ❌ Callback priorities (higher priority for GPIO) → No effect
3. ❌ Thread-safe queues (`queue.Queue()`) → Still blocked
4. ❌ Event loop integration (`asyncio` with Firebase) → Complex, unreliable
5. **Process isolation** → **SOLVED!**

**Final Solution: Process Isolation**

```python
# WORKING: Separate processes (no shared threading context)

# Process 1: Hardware deployment (GPIO only)
python -u -m hardware.deploy_ppo_run8
# - Handles GPIO edge detection
# - Saves logs locally
# - Exits cleanly

# Process 2: Cloud upload (Firebase only)
python3 upload_to_firebase.py
# - Reads local files
# - Uploads to Firebase Storage
# - No GPIO interaction

# KEY: Processes don't share memory/threads
# Result: Both work independently without conflict
```

**Lesson Learned:** In embedded systems with real-time constraints, **process isolation** is safer than threading hacks.

### Firebase Upload Script Usage

```bash
# 1. Run deployment (local-first operation)
sudo python -u -m hardware.deploy_ppo_run8

# 2. Upload to cloud (separate process after deployment completes)
python3 upload_to_firebase.py
```

**Upload Script Features:**
- **Duplicate Detection:** Checks existing uploads to prevent redundancy
- **Selective Upload:** Only uploads new deployment runs
- **Progress Reporting:** Terminal output shows upload status
- **Error Handling:** Graceful failure with local data preservation

**What Gets Uploaded:**
- Deployment logs (CSV: timestamped performance metrics)
- Performance plots (PNG: training curves, comparison charts)
- Statistical summaries (JSON: aggregate metrics)
- Terminal output (TXT: complete console logs)

### Firebase Setup Instructions

**1. Create Firebase Project:**
```
1. Go to https://console.firebase.google.com
2. Create new project: "traffic-ppo-pi"
3. Enable Firebase Storage
4. Set up security rules (authenticated read/write)
```

**2. Generate Service Account Key:**
```
1. Project Settings → Service Accounts
2. Generate new private key (JSON)
3. Download as serviceAccountKey.json
4. Place in project root (add to .gitignore!)
```

**3. Install Firebase Admin SDK:**
```bash
pip install firebase-admin --break-system-packages
```

**4. Test Connection:**
```bash
python3 upload_to_firebase.py --test
# Should print: "Firebase connection successful!"
```

## Installation & Setup

### Prerequisites

**Required:**
- Operating System: Ubuntu 20.04 LTS+ or Raspberry Pi OS (64-bit, Debian 12)
- Python: Version 3.8 or higher (3.9.2 recommended for Raspberry Pi)
- RAM: Minimum 2GB (Raspberry Pi 4 Model B)
- Disk Space: At least 10GB free space
- Internet: Required for installation and Firebase (optional for operation)

**Optional (Training):**
- GPU: NVIDIA GPU with CUDA 11.0+ for faster training
- RAM: 8GB+ recommended for training simulations
- CPU: Multi-core processor for parallel environments

### System Dependencies

```bash
# Update package lists
sudo apt-get update
sudo apt-get upgrade

# Install Python development tools
sudo apt-get install -y python3-dev python3-pip python3-venv
```

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/eadewusic/Traffic-Optimization-Capstone-Project
cd Traffic-Optimization-Capstone-Project

# Verify directory structure
ls -la
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate     # On Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify key installations
python -c "import stable_baselines3; print(f'SB3 version: {stable_baselines3.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# For Raspberry Pi hardware deployment
pip install RPi.GPIO --break-system-packages
```

### Step 4: Download Pretrained Models

**Option A: Download Champion Model (Run 8 Seed 789)**

```bash
# Create models directory
mkdir -p models/hardware_ppo/run_8

# Download from Google Drive (manual)
# Visit: https://drive.google.com/drive/folders/1Ik6iulDhcPMBermv-7wRNP02IbwNJRua

# Place files in models/hardware_ppo/run_8/:
# - ppo_final_seed789.zip (0.13MB)
# - vec_normalize_seed789.pkl (1.8KB)
```

**Option B: Train From Scratch (Reproduce Run 8)**

```bash
cd training

# Train each seed (5 independent runs)
python train_run8.py --seed 42 --total-steps 1000000
python train_run8.py --seed 123 --total-steps 1000000
python train_run8.py --seed 456 --total-steps 1000000
python train_run8.py --seed 789 --total-steps 1000000
python train_run8.py --seed 1000 --total-steps 1000000

# Aggregate results
python aggregate_run8_seeds.py
```

### Step 5: Hardware Setup (Raspberry Pi Only)

**Physical Wiring:**

Follow GPIO pin configuration from [Hardware Deployment](#hardware-deployment) section:
- 12 LEDs (4 directions × 3 colors) connected via 331Ω resistors
- 4 push buttons (1 per direction) with internal pull-down resistors
- Cooling fan connected to GPIO 3 for thermal management
- All components share common ground rails

**GPIO Access Setup:**

```bash
# Add user to gpio group (no sudo needed for GPIO)
sudo usermod -a -G gpio $USER

# Reboot to apply group changes
sudo reboot

# Test GPIO access after reboot
python -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); print('GPIO initialized successfully')"
```

### Step 6: Test Hardware Components

```bash
# Navigate to hardware directory
cd hardware

# Test all components before deployment
python test_simple_hardware.py

# Expected output:
# [PASS] LED North Red
# [PASS] LED North Yellow
# ...
# [PASS] Button North detection
# ALL TESTS PASSED - Hardware ready for deployment!
```

### Step 7: Deploy on Raspberry Pi

```bash
# Ensure you're in the hardware directory
cd ~/Traffic-Optimization-Capstone-Project/hardware

# Activate virtual environment
source ../.venv/bin/activate

# Run deployment script (requires root for GPIO)
sudo python -u -m hardware.deploy_ppo_run8
```

**Interactive Menu:**
```
═══════════════════════════════════════════
  PPO Traffic Light - Hardware Deployment
═══════════════════════════════════════════

Select mode:
1. PPO Event-Driven (Run until queues clear)
2. PPO Timed (60 seconds)
3. Comparison (Fixed vs PPO - Record & Replay)
4. Exit

Enter choice (1-4): 1

[LOADING] PPO model... ✓
[LOADING] VecNormalize... ✓
[READY] PPO controller initialized

[GPIO] Initializing pins...
[GPIO] Setup complete

Starting Event-Driven Demo...
Press Ctrl+C to stop

[Step 1] Phase: N/S Green | Queue: N=3 E=1 S=2 W=0 | Infer: 6.42ms
[Step 2] Phase: N/S Green | Queue: N=1 E=2 S=1 W=1 | Infer: 6.89ms
[SWITCH] Yellow transition (2.0s)
[Step 3] Phase: E/W Green | Queue: N=0 E=2 S=0 W=1 | Infer: 7.12ms
...

[COMPLETE] Session finished
Duration: 74.0s
Vehicles cleared: 32/32 (100%)
Phase changes: 6
Mean inference: 7.79ms

Logs saved to: ~/deployments/run_20250602_104523/
```

### Step 8: Upload Results to Firebase (Optional)

```bash
# After deployment completes, upload to cloud
cd ~/Traffic-Optimization-Capstone-Project

# Run upload script (separate process)
python3 upload_to_firebase.py

# Expected output:
# [CHECK] Connecting to Firebase...
# [CHECK] Storage bucket: traffic-ppo-pi.appspot.com
# [SCAN] Found 1 new deployment run
# [UPLOAD] deployments/run_20250602_104523/
#   ✓ performance_log.csv (2.4 KB)
#   ✓ summary.json (1.2 KB)
#   ✓ inference_times.png (45.3 KB)
# [SUCCESS] Upload complete
# [VIEW] https://console.firebase.google.com/project/traffic-ppo-pi/storage
```

### Step 9: Validation Testing

**Compare Champion Model vs Baseline:**

```bash
cd tests
python test_run8seed789_vs_baseline.py

# Runs Wilcoxon signed-rank test across 25 scenarios
# Outputs statistical significance results
```

**Evaluate Multi-Seed Reproducibility:**

```bash
cd evaluation
python run8seed789_ppo_evaluation.py --seed 789

# Displays CV, mean reward, and consistency metrics
```

### Project Workflow: From IoT to Firebase

#### 1. Hardware Testing

![Hardware Test Output](./images/test_simple_hardware-terminal-output.png)

*Figure: Hardware validation testing output showing successful GPIO initialization and component verification before deployment.*

#### 2. Deployment Menu & Execution

![Deployment Menu](./images/deployment-run-menu.png)

*Figure: Interactive deployment menu offering 3 operational modes: Event-Driven (continuous until queues clear), Timed (60-second demo), and Comparison (Record & Replay for statistical validation).*

![PPO Steps Decision Process](./images/ppo-steps-decision.png)

*Figure: Real-time terminal output showing PPO agent decision-making with queue states, phase selections, and inference timing (6-8ms per decision).*

![Deployment Results](./images/deployment-results.png)

*Figure: Session completion summary displaying 100% throughput (32/32 vehicles cleared), mean inference time (7.79ms), and zero wasted cycles achieved through adaptive control.*

#### 3. Firebase Cloud Integration

![Firebase Upload Terminal](./images/firebase-run-terminal-output.png)

*Figure: Asynchronous cloud upload process running in separate process after GPIO deployment completes, preventing threading conflicts.*

![Firebase Storage - Before Upload](./images/before-uploading-to-firebase.png)

*Figure: Firebase Cloud Storage state before upload showing existing deployment runs organized by timestamp hierarchy.*

![Firebase Storage - After Upload](./images/after-uploading-to-firebase.jpg)

*Figure: Firebase Cloud Storage after successful upload showing new deployment run with performance logs (CSV), statistical summaries (JSON), and visualization plots (PNG) backed up for remote analysis.*

## Key Contributions

### 1. First Hardware-Validated Low-Cost RL Traffic Control System

**Bridges simulation-to-hardware gap** with proven deployment:
- **Phase 1:** Algorithm selection (113D state, 9 actions, 74.7% improvement)
- **Phase 3:** Hardware deployment (4D state, 2 actions, sub-7ms inference)
- **Methodology:** Systematic environment simplification (96.5% state reduction)
- **Result:** Replicable framework for RL deployment to embedded platforms

**Impact:** Validates RL traffic control has matured from research to deployable technology.

### 2. Systematic Algorithm Selection Framework

**Empirical comparison establishing PPO superiority:**
- Tested 4 RL families (PPO, DQN, REINFORCE, Actor-Critic)
- Evaluated 17 distinct configurations
- **Evidence:** PPO 100% success vs DQN 0% failure, Actor-Critic catastrophic sensitivity
- **Contribution:** Evidence-based algorithm selection for deployment contexts

### 3. Record & Replay Fair Comparison Methodology

**Solves stochastic evaluation challenge:**
- Records exact traffic timestamps during baseline run
- Replays identical pattern for PPO evaluation
- **Result:** Fair comparison attributing performance to controller strategy, not traffic luck

**Generalizability:** Applicable to any RL traffic control research requiring rigorous validation.

### 4. Honest Sim-to-Real Gap Reporting

**Transparent performance documentation:**
- Simulation: 8.8% queue reduction (25 scenarios)
- Hardware: +2.8% throughput improvement (2 tests)
- **Analysis:** Gap explained by measurement differences, sample diversity, test duration
- **Contribution:** Realistic expectations vs simulation-only inflated claims

### 5. Hybrid Rule-Based + Learned Control Architecture

**Safety Wrapper mechanism:**
- Validates PPO decisions before execution
- Overrides selection of empty phases
- **Result:** 100% vehicle clearance, zero crashes, zero wasted cycles

**Impact:** Production-ready safety guarantees while preserving adaptive intelligence.

### 6. Open-Source Blueprint for Resource-Constrained Deployment

**Complete system documentation:**
- Prototype hardware: $108 (156,700 RWF)
- Production deployment: $8,000-$10,000 per intersection
- 96-98% cost savings vs traditional systems ($250K-$500K)
- **Accessibility:** Enables intelligent traffic control for African cities

### 7. Statistical Rigor & Reproducibility

**Multi-seed validation:**
- 5 independent training runs (CV=1.3%)
- Wilcoxon signed-rank testing (p=0.0002)
- Comprehensive documentation enabling full replication

**Contribution:** Research-grade validation suitable for peer-reviewed publication.

### 8. IoT-to-Cloud Architecture with Process Isolation

**Production-ready cloud integration:**
- Local-first design (operates offline)
- Firebase Storage for remote monitoring
- Process isolation solving threading conflicts
- **Result:** Scalable architecture for multi-intersection deployments

## Performance Metrics

### Statistical Validation Summary

**Wilcoxon Signed-Rank Test Results (25 Scenarios)**

| Metric | Fixed-Timing | PPO (Run 8) | Improvement | p-value | Significance |
|--------|--------------|-------------|-------------|---------|--------------|
| **Mean Reward** | 2,073.8 ± 11.9 | 2,078.5 ± 12.3 | +0.2% | **0.0002** | *** Highly Sig |
| **Mean Delay (s)** | 7.89 ± 0.91 | 7.19 ± 0.84 | **-8.9%** (-7s/vehicle) | **0.018** | * Significant |
| **Mean Queue** | 3.42 ± 0.67 | 3.12 ± 0.61 | **-8.8%** (-0.30 cars) | **0.025** | * Significant |
| **Throughput (%)** | 96.8 ± 1.3 | 97.1 ± 1.2 | +0.3pp | 0.234 | ns Not Sig |
| **Win Rate** | 28% (7/25) | **72% (18/25)** | +44pp | - | Dominant |
| **Control Efficiency** | 0.6 cars/switch | 2.0 cars/switch | **+233%** | - | Adaptive |

*Significance: *** p<0.001 (highly), ** p<0.01 (very), * p<0.05 (significant), ns = not significant, pp = percentage points*

### Hardware Validation Results

**Real-Time Performance (Raspberry Pi 4):**

| Metric | Value | Target | Safety Margin | Status |
|--------|-------|--------|---------------|--------|
| Mean Inference | 6.84ms | <100ms | 14.6× | ✓ Pass |
| Max Inference | 11.16ms | <100ms | 9.0× | ✓ Pass |
| Std Deviation | ±1.15ms | <5ms | Low variance | ✓ Pass |

**Mode 3 Comparison (Record & Replay):**

| Metric | Test 1 | Test 2 | Average |
|--------|--------|--------|---------|
| Fixed-Timing Throughput | 80.0% | 87.5% | 83.75% |
| PPO Throughput | 82.5% | 90.6% | **86.55%** |
| Improvement | +2.5% | +3.1% | **+2.8%** |
| Phase Reduction | -20% | -20% | **-20%** |
| Win Rate | ✓ | ✓ | **100% (2/2)** |

**Mode 1 Event-Driven (Continuous):**

| Metric | Session 1 | Session 2 | Consistency |
|--------|-----------|-----------|-------------|
| Throughput | 100% (53/53) | 100% (32/32) | Perfect |
| Peak Congestion | 17 cars | 8 cars | Handled extremes |
| Phase Changes | 11 (61%) | 6 (50%) | Scaled with load |
| Mean Inference | 6.80ms | 7.79ms | Real-time |

### Reproducibility Metrics

**Multi-Seed Validation (Run 8):**

```
Seeds Tested: 5 (42, 123, 456, 789, 1000)
Mean Performance: 2,025.3 ± 4.7
Coefficient of Variation: 1.3% (industry standard <5%)
Range: [1,987.7 - 2,066.3] (all within ±2% of mean)
Training Efficiency: ~35 minutes per seed
```

## Tech Stack

### Machine Learning & AI

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **RL Framework** | Stable-Baselines3 | 2.0.0 | PPO implementation |
| **Deep Learning** | PyTorch | 2.0.1 | Neural network backend |
| **Training Interface** | Gymnasium | 0.28.0 | RL environment API |
| **Monitoring** | TensorBoard | 2.8.0 | Training visualization |

### Hardware Deployment

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Platform** | Raspberry Pi 4B | 2GB RAM | Edge computing device |
| **GPIO Control** | RPi.GPIO | 0.7.1 | LED/button control |
| **OS** | Raspberry Pi OS | Debian 12 | Operating system |

### Data Analysis & Visualization

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Numerical Computing** | NumPy | 1.21.0 | Array operations |
| **Data Manipulation** | Pandas | 1.4.0 | Statistical analysis |
| **Visualization** | Matplotlib | 3.5.1 | Performance plotting |

### Cloud Integration

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Cloud Storage** | Firebase Storage | Latest | Remote data backup |
| **Admin SDK** | firebase-admin | Latest | Asynchronous upload |

### Development Tools

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Version Control** | Git | 2.34.1 | Source control |
| **Python Environment** | venv | 3.9.2 | Virtual environments |
| **IDE** | VS Code | 1.75.0 | Code editor |

## Production Deployment Economics

### Cost Comparison

| Configuration | Cost per Intersection | Key Components | Intelligence Level |
|---------------|----------------------|----------------|-------------------|
| **Research Prototype** | **$108** (156,700 RWF) | Raspberry Pi 4, breadboard, LEDs, buttons | Adaptive (RL) |
| **Production-Ready** | **$8,000 - $10,000** | PCB, weatherproof enclosure, professional sensors, installation | Adaptive (RL) |
| **Traditional Fixed-Timing** | **$250,000 - $500,000** | Industrial controllers, signal heads, poles, installation | None (preset) |

### Production Cost Breakdown ($8K-$10K)

| Category | Cost | Components |
|----------|------|------------|
| Hardware & PCB | $2,000 - $2,500 | Raspberry Pi, PCB fabrication, enclosure |
| Professional Sensors | $1,500 - $2,000 | Camera/ultrasonic (replaces buttons) |
| Installation Labor | $3,000 - $4,000 | Electrical work, mounting, commissioning |
| Weatherproofing | $1,000 - $1,500 | Industrial enclosure, cable management |
| Permits & Regulatory | $1,500 - $2,000 | Traffic authority approval, compliance |
| Training & Commissioning | $1,000 - $1,500 | Operator training, system handover |

### Economic Impact Analysis

**Lagos Case Study (1,000 Intersections):**

| Metric | Calculation | Result |
|--------|-------------|--------|
| **Deployment Cost** | 1,000 × $10,000 | $10 million |
| **Traditional Cost** | 1,000 × $250,000 | $250 million |
| **Cost Savings** | $250M - $10M | **$240 million (96%)** |
| **Annual Congestion Losses** | Current losses | $4 billion/year |
| **Conservative Benefit (0.5%)** | 0.5% × $4B | $20 million/year |
| **Payback Period** | $10M ÷ $20M | **6 months** |
| **10-Year ROI** | ($20M × 10) - $10M | **$190 million** |

**Note:** Economic analysis uses conservative 0.5% benefit assumption (far below 2.8-8.9% demonstrated improvements).

### Key Economic Insights

1. **Affordability:** 96-98% cheaper than traditional adaptive systems
2. **Scalability:** Low per-unit cost enables citywide deployment
3. **Fast Payback:** <12 months in high-congestion cities
4. **Accessibility:** Feasible for resource-constrained African cities
5. **Production Path:** Clear cost structure from prototype ($108) to deployment ($8K-$10K)

## Limitations & Future Work

### Current Limitations

1. **Single Intersection Scope:** Validated for individual intersections only (no network-level coordination)
2. **Straight-Through Traffic Only:** No protected left-turn phases or turning movement detection
3. **Simulation Input:** Push-button HIL testing (requires camera/sensor integration for production)
4. **Breadboard Prototype:** PCB designs created but not fabricated (production manufacturing needed)
5. **Manual Cloud Upload:** Firebase integration requires manual script execution (automation needed for 24/7)
6. **Limited Hardware Testing:** 2 comparison tests + 2 event-driven sessions (more extensive validation needed)
7. **Retrofit Dependency:** Requires existing electrical infrastructure (15W power, 5V/3A)

### Future Work Recommendations

**Immediate (Pilot Deployment):**
- Field pilot at low-traffic Kigali intersection
- Replace buttons with ultrasonic sensors (HC-SR04) or camera-based detection
- Validate Emergency Vehicle Prioritization (EMV) with BLE beacons
- PCB fabrication for weatherproof enclosure
- Automate Firebase cloud sync (cron job or MQTT protocol)

**Medium-Term (Feature Expansion):**
- Multi-intersection network coordination (MARL)
- Turning movement detection (12D state space, protected left-turns)
- Computer vision integration (YOLOv5-nano for vehicle detection)
- Multi-modal optimization (pedestrian/cyclist equity in reward function)
- Production storage management (three-tier: local SQLite, cloud archive, city server)

**Long-Term (Research Extensions):**
- Offline RL for safety-critical deployment (CQL/BCQ algorithms)
- Transfer learning for different African cities (Lagos, Nairobi adaptation)
- Standardized sim-to-real benchmark (community comparison framework)
- Autonomous vehicle integration (SPaT broadcasts via DSRC/C-V2X)

See Chapter 6 of Project Report for detailed recommendations.

## Citation

If you use this work in your research, please cite:

```bibtex
@bscthesis{adewusi2025traffic,
  title={Sim-to-Real Transfer for Intelligent Traffic Control: Hardware-Aware Policy Optimization with Domain Randomization, Multi-Seed Validation, and Low-Cost Edge Deployment in Sub-Saharan Africa},
  author={Adewusi, Eunice Adebusayo},
  year={2025},
  school={African Leadership University},
  address={Kigali, Rwanda},
  note={BSc Software Engineering Capstone Project},
  url={https://github.com/eadewusic/Traffic-Optimization-Capstone-Project}
}
```

## License & Attribution

This project utilizes hybrid licensing but with larger part of it licensed under the **Apache 2.0 License** - see the [LICENSE](LICENSE) file for details.

**Rationale:** Provides patent protection and appropriate attribution requirements while allowing commercial use with proper credit.

**Before using, deploying, or modifying this system, please review:**

| Document | Purpose | Link |
|----------|---------|------|
| **End-User License Agreement (EULA)** | Terms of use, deployment restrictions, contribution guidelines | [Read EULA](https://docs.google.com/document/d/1wqHr0carY_SPtRfDgFwOrLlrvRmnz1RYWvpWwtktNes/edit?usp=sharing) |
| **Privacy Policy** | Data handling, security measures, compliance framework | [Read Privacy Policy](https://docs.google.com/document/d/1BzAlUTpPYfr22CTsre0Lzpy6w9OLcFFqweO8INaqPP4/edit?usp=sharing) |
| **Contributing Guidelines** | How to propose changes and collaborate | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Contact

**Author:** Eunice Adebusayo Adewusi  
**Email:** [e.adewusi@alustudent.com](mailto:e.adewusi@alustudent.com)  
**LinkedIn:** [linkedin.com/in/euniceadewusic](https://www.linkedin.com/in/euniceadewusic/)  
**Other Links:** [https://linktr.ee/climiradi](https://linktr.ee/climiradi)  
**Institution:** African Leadership University (ALU), Kigali, Rwanda  
**Program:** BSc Software Engineering (Machine Learning)

---

**Project Journey:** 4 RL algorithms → 17 configurations → PPO champion → 8 training runs → 5-seed validation (CV=1.3%) → Wilcoxon testing (p=0.0002) → Raspberry Pi deployment (6.84ms) → Firebase integration → **Intelligent Traffic Control for Africa** 
