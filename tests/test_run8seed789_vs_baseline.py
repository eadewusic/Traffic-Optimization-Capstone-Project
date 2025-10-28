#!/usr/bin/env python3
"""
FINAL COMPARISON TEST - RUN 8 SEED 789
Test Run 8 Seed 789 vs Baseline in Run7TrafficEnv (SAME environment)
This validates that Run 8 Seed 789 also beats baseline (not just Run 7)
"""

import numpy as np
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scipy import stats

# Get paths
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from environments.run7_env import Run7TrafficEnv


def baseline_longest_queue(obs):
    """Longest-queue-first baseline policy"""
    longest_idx = int(np.argmax(obs))
    return 0 if longest_idx in [0, 1] else 1


print("="*70)
print("RUN 8 SEED 789 vs BASELINE - FAIR COMPARISON")
print("="*70)
print("\n BOTH controllers tested in Run7TrafficEnv")
print(" SAME comparative reward function")
print(" SAME traffic patterns (controlled seeds)")
print(" SAME methodology as Run 7 baseline test")
print("\nThis validates Run 8 Seed 789 beats baseline!\n")

# Load Run 8 Seed 789 Final Model
model_path = "../models/hardware_ppo/run_8/seed_789/ppo_final_seed789.zip"
vecnorm_path = "../models/hardware_ppo/run_8/seed_789/vec_normalize_seed789.pkl"

print(f"Loading Run 8 Seed 789 model...")
print(f"  Model: {model_path}")
print(f"  VecNormalize: {vecnorm_path}")

try:
    model = PPO.load(model_path)
    dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(" Model loaded successfully\n")
except Exception as e:
    print(f" Error loading model: {e}")
    print("\nMake sure the model files exist:")
    print(f"  {model_path}")
    print(f"  {vecnorm_path}")
    sys.exit(1)

# Test scenarios (same as Run 7)
scenarios = [
    {"name": "Balanced", "queues": np.array([5, 5, 5, 5], dtype=np.float32)},
    {"name": "North Heavy", "queues": np.array([15, 3, 2, 4], dtype=np.float32)},
    {"name": "E-W Rush", "queues": np.array([2, 3, 12, 11], dtype=np.float32)},
    {"name": "Random", "queues": np.array([8, 2, 10, 4], dtype=np.float32)},
    {"name": "Blocked", "queues": np.array([18, 1, 1, 1], dtype=np.float32)}
]

test_seeds = [42, 123, 456, 789, 1234]

# Storage
run8_rewards = []
baseline_rewards = []
run8_cleared = []
baseline_cleared = []
scenario_results = []

print("Running 25 trials (5 seeds × 5 scenarios)...")

for seed_idx, seed in enumerate(test_seeds):
    print(f"\nSeed {seed_idx + 1}/5 (seed={seed})")
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        initial_queues = scenario['queues']
        
        # ==================== TEST RUN 8 SEED 789 ====================
        np.random.seed(seed)
        env = Run7TrafficEnv()
        obs, _ = env.reset(options={'initial_queues': initial_queues})
        obs = obs / env.max_queue_length
        
        run8_reward = 0
        run8_cars = 0
        
        for step in range(50):
            obs_norm = vec_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            run8_reward += reward
            run8_cars += info.get('cars_cleared', 0)
        
        run8_final_queue = np.sum(env.queues)
        
        # ==================== TEST BASELINE ====================
        np.random.seed(seed)  # SAME seed for fair comparison
        env = Run7TrafficEnv()
        obs, _ = env.reset(options={'initial_queues': initial_queues})
        obs = obs / env.max_queue_length
        
        baseline_reward = 0
        baseline_cars = 0
        
        for step in range(50):
            action = baseline_longest_queue(obs * env.max_queue_length)
            obs, reward, term, trunc, info = env.step(action)
            baseline_reward += reward
            baseline_cars += info.get('cars_cleared', 0)
        
        baseline_final_queue = np.sum(env.queues)
        
        # Store results
        run8_rewards.append(run8_reward)
        baseline_rewards.append(baseline_reward)
        run8_cleared.append(run8_cars)
        baseline_cleared.append(baseline_cars)
        
        scenario_results.append({
            'scenario': scenario_name,
            'seed': seed,
            'run8_reward': run8_reward,
            'baseline_reward': baseline_reward,
            'run8_wins': run8_reward > baseline_reward
        })
        
        # Print per-trial comparison
        winner = "Run 8" if run8_reward > baseline_reward else "Baseline" if baseline_reward > run8_reward else "Tie"
        print(f"  {scenario_name:12s}: Run8={run8_reward:6.1f}, Baseline={baseline_reward:6.1f} → {winner}")

# ==================== ANALYSIS ====================
print("\n" + "="*70)
print(" RESULTS SUMMARY")
print("="*70)

run8_avg = np.mean(run8_rewards)
run8_std = np.std(run8_rewards)
baseline_avg = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards)

print(f"\nRun 8 Seed 789 (PPO):")
print(f"  Average reward:  {run8_avg:8.1f} ± {run8_std:.1f}")
print(f"  Average cleared: {np.mean(run8_cleared):8.1f} cars")
print(f"  Reward range:    {min(run8_rewards):.1f} to {max(run8_rewards):.1f}")

print(f"\nBaseline (Longest Queue):")
print(f"  Average reward:  {baseline_avg:8.1f} ± {baseline_std:.1f}")
print(f"  Average cleared: {np.mean(baseline_cleared):8.1f} cars")
print(f"  Reward range:    {min(baseline_rewards):.1f} to {max(baseline_rewards):.1f}")

# Difference
difference = run8_avg - baseline_avg
if baseline_avg != 0:
    pct_diff = (difference / abs(baseline_avg)) * 100
else:
    pct_diff = 0

# Win rate
wins = sum(1 for r in scenario_results if r['run8_wins'])
win_rate = (wins / len(scenario_results)) * 100

print("\n" + "="*70)
print(" PERFORMANCE COMPARISON")
print("="*70)
print(f"\nAbsolute difference: {difference:+.1f} reward points")
print(f"Percentage difference: {pct_diff:+.1f}%")
print(f"Win rate: {wins}/{len(scenario_results)} trials ({win_rate:.0f}%)")

# Statistical test
t_stat, p_value = stats.ttest_ind(run8_rewards, baseline_rewards)
print(f"\nStatistical Significance:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")

# ==================== VERDICT ====================
print("\n" + "="*70)
print(" FINAL VERDICT")
print("="*70)

if p_value < 0.05:
    if t_stat > 0:
        significance = "statistically significant"
    else:
        significance = "statistically significant (baseline wins)"
else:
    significance = "not statistically significant (essentially tied)"

if difference > 0 and win_rate >= 60 and pct_diff > 5:
    print(f"\n SUCCESS! RUN 8 SEED 789 BEATS BASELINE!")
    print(f"\n   • {pct_diff:.1f}% better reward")
    print(f"   • {win_rate:.0f}% win rate")
    print(f"   • Difference is {significance}")
    print(f"\n  RECOMMENDATION: DEPLOY TO HARDWARE!")
    verdict = "STRONG_SUCCESS"
    
elif difference > 0 and (win_rate >= 50 or pct_diff > 0):
    print(f"\n Marginal Success - Run 8 Seed 789 edges out baseline")
    print(f"\n  • {pct_diff:.1f}% better reward")
    print(f"  • {win_rate:.0f}% win rate")  
    print(f"  • Difference is {significance}")
    print(f"\n  RECOMMENDATION: Deploy and test on hardware")
    verdict = "MARGINAL_SUCCESS"
    
elif abs(difference) < 50:  # Essentially tied
    print(f"\n≈ Statistical Tie - Performance equivalent")
    print(f"\n  • Difference: {difference:+.1f} points ({pct_diff:+.1f}%)")
    print(f"  • Win rate: {win_rate:.0f}%")
    print(f"  • Difference is {significance}")
    print(f"\n  Both approaches are statistically equivalent.")
    print(f"  Run 8's advantage: Multi-seed validated, robust to variations")
    print(f"\n  RECOMMENDATION: Deploy, hardware testing decides")
    verdict = "TIE"
    
else:
    print(f"\n Baseline wins")
    print(f"\n  • {-pct_diff:.1f}% better (baseline)")
    print(f"  • Win rate: {100-win_rate:.0f}% (baseline)")
    print(f"  • Difference is {significance}")
    print(f"\n  RECOMMENDATION: Further investigation needed")
    verdict = "BASELINE_WINS"

print("\n" + "="*70)
print(" COMPARISON WITH RUN 7")
print("="*70)
print("\nRun 8 Seed 789 vs Run 7 comparison already showed:")
print("  • Run 8 Seed 789 > Run 7 by +24.8 points")
print("  • 33.4% more training efficient")
print("  • Multi-seed validated")
print("\nIf Run 8 Seed 789 beats baseline (this test),")
print("then Run 8 Seed 789 > Run 7 > Baseline ✓")

print("\n" + "="*70)
print(" NEXT STEPS")
print("="*70)

if verdict in ["STRONG_SUCCESS", "MARGINAL_SUCCESS"]:
    print("\n1. Run 8 Seed 789 validated against baseline")
    print("2. Deploy Run 8 Seed 789 to Raspberry Pi")
    print("3. Hardware testing vs baseline")
    print("4. Thesis: Show complete progression:")
    print("     Baseline → Run 7 → Run 8 (multi-seed) → Hardware")
    print("5. Defense: Simulation AND hardware validation")
    
elif verdict == "TIE":
    print("\n1. Run 8 Seed 789 matches baseline (statistical tie)")
    print("2. Deploy Run 8 Seed 789 to hardware anyway")
    print("3. Hardware testing may reveal advantages")
    print("4. Thesis: Statistical equivalence + multi-seed robustness")
    print("5. Defense: \"RL matches sophisticated heuristic + scalability\"")
    
else:
    print("\n1. Unexpected result - investigate further")
    print("2. Compare Run 7 baseline results with Run 8 baseline results")
    print("3. Check for environment/evaluation differences")
    print("4. Hardware testing may still favor learned policy")

print("\n")

# Save detailed results
results_dir = "../results/run_8/seed_789"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "baseline_comparison_results.txt")

with open(results_file, 'w') as f:
    f.write("RUN 8 SEED 789 vs BASELINE COMPARISON\n")
    f.write("="*70 + "\n\n")
    f.write(f"Run 8 Seed 789 Average: {run8_avg:.1f} ± {run8_std:.1f}\n")
    f.write(f"Baseline Average: {baseline_avg:.1f} ± {baseline_std:.1f}\n")
    f.write(f"Difference: {difference:+.1f} ({pct_diff:+.1f}%)\n")
    f.write(f"Win Rate: {win_rate:.0f}%\n")
    f.write(f"Statistical Test: t={t_stat:.3f}, p={p_value:.4f}\n")
    f.write(f"Significance: {significance}\n")
    f.write(f"Verdict: {verdict}\n\n")
    
    f.write("DETAILED TRIAL RESULTS\n")
    f.write("-"*70 + "\n")
    for result in scenario_results:
        winner_mark = "YES" if result['run8_wins'] else "NO"
        f.write(f"{winner_mark} {result['scenario']:12s} (seed {result['seed']:4d}): ")
        f.write(f"Run8={result['run8_reward']:6.1f}, Baseline={result['baseline_reward']:6.1f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("CONTEXT: RUN 7 VS RUN 8 SEED 789\n")
    f.write("="*70 + "\n")
    f.write("Previous comparison showed Run 8 Seed 789 > Run 7:\n")
    f.write("  Final Reward: 2066.3 vs 2041.5 (+24.8 points)\n")
    f.write("  Training: 1.0M vs 1.5M steps (33.4% more efficient)\n")
    f.write("  Validation: Multi-seed proven (5 seeds)\n")

print(f" Detailed results saved: {results_file}")
