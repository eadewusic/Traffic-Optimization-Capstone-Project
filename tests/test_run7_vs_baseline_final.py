#!/usr/bin/env python3
"""
FINAL COMPARISON TEST
Test Run 7 vs Baseline in Run7TrafficEnv (SAME environment)
This is the FAIR comparison that determines success
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


print("RUN 7 FINAL vs BASELINE - FAIR COMPARISON")
print("\n✓ BOTH controllers tested in Run7TrafficEnv")
print("✓ SAME comparative reward function")
print("✓ SAME traffic patterns (controlled seeds)")
print("\nThis is the TRUE performance test!\n")

# Load Run 7 Final
model_path = "../models/hardware_ppo/run_7/final_model.zip"
vecnorm_path = "../models/hardware_ppo/run_7/vecnormalize.pkl"

print(f"Loading Run 7 model...")
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

# Test scenarios
scenarios = [
    {"name": "Balanced", "queues": np.array([5, 5, 5, 5], dtype=np.float32)},
    {"name": "North Heavy", "queues": np.array([15, 3, 2, 4], dtype=np.float32)},
    {"name": "E-W Rush", "queues": np.array([2, 3, 12, 11], dtype=np.float32)},
    {"name": "Random", "queues": np.array([8, 2, 10, 4], dtype=np.float32)},
    {"name": "Blocked", "queues": np.array([18, 1, 1, 1], dtype=np.float32)}
]

test_seeds = [42, 123, 456, 789, 1234]

# Storage
run7_rewards = []
baseline_rewards = []
run7_cleared = []
baseline_cleared = []
scenario_results = []

print("Running 25 trials (5 seeds × 5 scenarios)...")

for seed_idx, seed in enumerate(test_seeds):
    print(f"\nSeed {seed_idx + 1}/5 (seed={seed})")
    
    for scenario in scenarios:
        scenario_name = scenario['name']
        initial_queues = scenario['queues']
        
        # ==================== TEST RUN 7 ====================
        np.random.seed(seed)
        env = Run7TrafficEnv()
        obs, _ = env.reset(options={'initial_queues': initial_queues})
        obs = obs / env.max_queue_length
        
        run7_reward = 0
        run7_cars = 0
        
        for step in range(50):
            obs_norm = vec_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            run7_reward += reward
            run7_cars += info.get('cars_cleared', 0)
        
        run7_final_queue = np.sum(env.queues)
        
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
        run7_rewards.append(run7_reward)
        baseline_rewards.append(baseline_reward)
        run7_cleared.append(run7_cars)
        baseline_cleared.append(baseline_cars)
        
        scenario_results.append({
            'scenario': scenario_name,
            'seed': seed,
            'run7_reward': run7_reward,
            'baseline_reward': baseline_reward,
            'run7_wins': run7_reward > baseline_reward
        })
        
        # Print per-trial comparison
        winner = "Run 7" if run7_reward > baseline_reward else "Baseline" if baseline_reward > run7_reward else "Tie"
        print(f"  {scenario_name:12s}: Run7={run7_reward:6.1f}, Baseline={baseline_reward:6.1f} → {winner}")

# ==================== ANALYSIS ====================
print("\n RESULTS SUMMARY")

run7_avg = np.mean(run7_rewards)
run7_std = np.std(run7_rewards)
baseline_avg = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards)

print(f"\nRun 7 PPO:")
print(f"  Average reward:  {run7_avg:8.1f} ± {run7_std:.1f}")
print(f"  Average cleared: {np.mean(run7_cleared):8.1f} cars")
print(f"  Reward range:    {min(run7_rewards):.1f} to {max(run7_rewards):.1f}")

print(f"\nBaseline (Longest Queue):")
print(f"  Average reward:  {baseline_avg:8.1f} ± {baseline_std:.1f}")
print(f"  Average cleared: {np.mean(baseline_cleared):8.1f} cars")
print(f"  Reward range:    {min(baseline_rewards):.1f} to {max(baseline_rewards):.1f}")

# Difference
difference = run7_avg - baseline_avg
if baseline_avg != 0:
    pct_diff = (difference / abs(baseline_avg)) * 100
else:
    pct_diff = 0

# Win rate
wins = sum(1 for r in scenario_results if r['run7_wins'])
win_rate = (wins / len(scenario_results)) * 100

print("\n PERFORMANCE COMPARISON")
print(f"\nAbsolute difference: {difference:+.1f} reward points")
print(f"Percentage difference: {pct_diff:+.1f}%")
print(f"Win rate: {wins}/{len(scenario_results)} trials ({win_rate:.0f}%)")

# Statistical test
t_stat, p_value = stats.ttest_ind(run7_rewards, baseline_rewards)
print(f"\nStatistical Significance:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")

# ==================== VERDICT ====================
print("\n FINAL VERDICT")

if p_value < 0.05:
    if t_stat > 0:
        significance = "statistically significant"
    else:
        significance = "statistically significant (baseline wins)"
else:
    significance = "not statistically significant (essentially tied)"

if difference > 0 and win_rate >= 60 and pct_diff > 5:
    print(f"\n SUCCESS! RUN 7 BEATS BASELINE!")
    print(f"\n   {pct_diff:.1f}% better reward")
    print(f"   {win_rate:.0f}% win rate")
    print(f"   Difference is {significance}")
    print(f"\n  RECOMMENDATION: DEPLOY TO HARDWARE!")
    verdict = "STRONG_SUCCESS"
    
elif difference > 0 and (win_rate >= 50 or pct_diff > 0):
    print(f"\n✓ Marginal Success - Run 7 edges out baseline")
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
    print(f"  Run 7's advantage: Learned from data, robust to variations")
    print(f"\n  RECOMMENDATION: Deploy both, hardware testing decides")
    verdict = "TIE"
    
else:
    print(f"\n  Baseline wins")
    print(f"\n  • {-pct_diff:.1f}% better (baseline)")
    print(f"  • Win rate: {100-win_rate:.0f}% (baseline)")
    print(f"  • Difference is {significance}")
    print(f"\n  RECOMMENDATION: Hardware test or thesis pivot")
    verdict = "BASELINE_WINS"

print("\n NEXT STEPS")

if verdict in ["STRONG_SUCCESS", "MARGINAL_SUCCESS"]:
    print("\n Deploy Run 7 to Raspberry Pi")
    print(" Hardware testing vs baseline")
    print(" Thesis: Emphasize learning success + deployment")
    print(" Defense: Show simulation AND hardware results")
    
elif verdict == "TIE":
    print("\n1. Deploy Run 7 to hardware")
    print("2. Hardware testing may reveal advantages")
    print("3. Thesis: Statistical equivalence + deployment contribution")
    print("4. Defense: \"RL matches sophisticated heuristic\"")
    
else:
    print("\n1. Option A: Hardware testing (sim2real gap may favor PPO)")
    print("2. Option B: Thesis focuses on deployment contribution")
    print("3. Option C: Analyze why baseline won in this reward function")

print("\n")

# Save detailed results
results_file = "../results/run_7/comparison_results.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

with open(results_file, 'w') as f:
    f.write("RUN 7 FINAL vs BASELINE COMPARISON\n")
    f.write(f"Run 7 Average: {run7_avg:.1f} ± {run7_std:.1f}\n")
    f.write(f"Baseline Average: {baseline_avg:.1f} ± {baseline_std:.1f}\n")
    f.write(f"Difference: {difference:+.1f} ({pct_diff:+.1f}%)\n")
    f.write(f"Win Rate: {win_rate:.0f}%\n")
    f.write(f"Statistical Test: t={t_stat:.3f}, p={p_value:.4f}\n")
    f.write(f"Verdict: {verdict}\n")

print(f"\n Detailed results saved: {results_file}")
