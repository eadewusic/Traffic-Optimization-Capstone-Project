#!/usr/bin/env python3
"""
diagnose_run6_failure.py
Understand WHY Run 6 didn't beat baseline
This guides Run 7 design decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environments.simple_button_env import SimpleButtonTrafficEnv


def baseline_longest_queue(obs):
    longest_idx = int(np.argmax(obs))
    return 0 if longest_idx in [0, 1] else 1


print("RUN 6 FAILURE ANALYSIS")
print("\nDiagnosing why PPO didn't beat baseline...\n")

# Load best checkpoint from Run 6
model_path = "../models/hardware_ppo/run_6/checkpoint_250000_steps.zip"
vecnorm_path = "../models/hardware_ppo/run_6/checkpoint_vecnormalize_250000_steps.pkl"

model = PPO.load(model_path)
dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
vec_env = VecNormalize.load(vecnorm_path, dummy_env)
vec_env.training = False
vec_env.norm_reward = False

# Test scenario
test_scenario = np.array([15, 3, 2, 4], dtype=np.float32)  # North heavy

# Run detailed trace
np.random.seed(42)

env = SimpleButtonTrafficEnv(domain_randomization=False)
obs, _ = env.reset()
env.queues = test_scenario.copy()
obs = env.queues / env.max_queue_length

print("DIAGNOSTIC 1: ACTION SELECTION COMPARISON")
print("Scenario: North Heavy Congestion [15, 3, 2, 4]\n")

ppo_actions = []
baseline_actions = []
ppo_rewards = []
baseline_rewards_sim = []
queues_ppo = []
queues_baseline = []

# PPO trace
np.random.seed(42)
env_ppo = SimpleButtonTrafficEnv(domain_randomization=False)
obs_ppo, _ = env_ppo.reset()
env_ppo.queues = test_scenario.copy()
obs_ppo = env_ppo.queues / env_ppo.max_queue_length

for step in range(20):
    obs_norm = vec_env.normalize_obs(obs_ppo)
    action, _ = model.predict(obs_norm, deterministic=True)
    ppo_actions.append(action)
    queues_ppo.append(env_ppo.queues.copy())
    
    obs_ppo, reward, _, _, _ = env_ppo.step(action)
    ppo_rewards.append(reward)

# Baseline trace
np.random.seed(42)
env_baseline = SimpleButtonTrafficEnv(domain_randomization=False)
obs_baseline, _ = env_baseline.reset()
env_baseline.queues = test_scenario.copy()
obs_baseline = env_baseline.queues / env_baseline.max_queue_length

for step in range(20):
    action = baseline_longest_queue(obs_baseline * env_baseline.max_queue_length)
    baseline_actions.append(action)
    queues_baseline.append(env_baseline.queues.copy())
    
    obs_baseline, reward, _, _, _ = env_baseline.step(action)
    baseline_rewards_sim.append(reward)

# Compare first 10 steps
print(f"{'Step':<6} {'Queues (NESW)':<20} {'PPO Action':<12} {'Baseline':<12} {'Match?'}")
for i in range(10):
    q = queues_ppo[i]
    ppo_act = ['N/S', 'E/W', 'Red', 'Emerg'][ppo_actions[i]]
    base_act = ['N/S', 'E/W'][baseline_actions[i]]
    match = "YES" if ppo_actions[i] == baseline_actions[i] else "NO"
    print(f"{i:<6} [{q[0]:4.1f}, {q[2]:4.1f}, {q[1]:4.1f}, {q[3]:4.1f}]  {ppo_act:<12} {base_act:<12} {match}")

# Calculate agreement rate
agreement = sum(1 for i in range(20) if ppo_actions[i] == baseline_actions[i]) / 20
print(f"\nAction Agreement Rate: {agreement*100:.1f}%")

if agreement > 0.8:
    print(" FINDING 1: PPO is basically copying the baseline (>80% agreement)")
    print("    PPO learned the heuristic but didn't surpass it")
    print("    Need stronger reward signal to encourage better strategies")
elif agreement < 0.5:
    print(" FINDING 1: PPO uses different strategy (<50% agreement)")
    print("    PPO learned something different, but it's not better")
    print("    Strategy might be suboptimal or undertrained")

print("\n DIAGNOSTIC 2: REWARD FUNCTION ANALYSIS")

# Test if baseline gets high rewards with current reward function
env_test = SimpleButtonTrafficEnv(domain_randomization=False)

baseline_policy_rewards = []
for scenario_queues in [
    [5, 5, 5, 5],
    [15, 3, 2, 4],
    [2, 3, 12, 11]
]:
    np.random.seed(42)
    env_test = SimpleButtonTrafficEnv(domain_randomization=False)
    obs_test, _ = env_test.reset()
    env_test.queues = np.array(scenario_queues, dtype=np.float32)
    obs_test = env_test.queues / env_test.max_queue_length
    
    total_reward = 0
    for step in range(50):
        action = baseline_longest_queue(obs_test * env_test.max_queue_length)
        obs_test, reward, _, _, _ = env_test.step(action)
        total_reward += reward
    
    baseline_policy_rewards.append(total_reward)

avg_baseline_reward = np.mean(baseline_policy_rewards)
print(f"\nBaseline policy gets average reward: {avg_baseline_reward:.1f}")
print("\nThis means:")
if avg_baseline_reward > 500:
    print(" FINDING 2: Reward function ALREADY rewards baseline heavily")
    print("    Current reward function is 'baseline-aligned'")
    print("    PPO has no incentive to do better than baseline")
    print("    Need to restructure rewards to encourage superior strategies")
else:
    print(" FINDING 2: Reward function doesn't over-reward baseline")
    print("   Room for PPO to find better strategies")

print("\n DIAGNOSTIC 3: EXPLORATION ANALYSIS")

# Check if PPO uses all actions
action_distribution = {}
for action in ppo_actions:
    action_distribution[int(action)] = action_distribution.get(int(action), 0) + 1

print("\nPPO Action Usage (first 20 steps):")
for action, count in sorted(action_distribution.items()):
    pct = (count / len(ppo_actions)) * 100
    action_name = ['N/S Green', 'E/W Green', 'All Red', 'Emergency'][action]
    print(f"  Action {action} ({action_name}): {count}/20 ({pct:.1f}%)")

productive_actions = action_distribution.get(0, 0) + action_distribution.get(1, 0)
unproductive_actions = action_distribution.get(2, 0) + action_distribution.get(3, 0)

if unproductive_actions > 2:
    print("\n FINDING 3: PPO wastes cycles on unproductive actions")
    print(f"    {unproductive_actions} unproductive actions in 20 steps")
    print("    Empty action penalty (-3.0) too weak")
    print("    Need stronger penalty or remove these actions")
else:
    print("\n FINDING 3: PPO avoids unproductive actions")

print("\n DIAGNOSTIC 4: TRAINING STABILITY")

# Load training logs
eval_path = "../logs/hardware_ppo/run_6/evaluations.npz"
eval_data = np.load(eval_path)
eval_rewards = eval_data['results'].mean(axis=1)
eval_timesteps = eval_data['timesteps']

# Calculate learning metrics
initial_reward = eval_rewards[0]
best_reward = np.max(eval_rewards)
final_reward = eval_rewards[-1]
improvement = best_reward - initial_reward
stability = np.std(eval_rewards[len(eval_rewards)//2:])  # Second half std

print(f"\nTraining Progression:")
print(f"  Initial reward: {initial_reward:.1f}")
print(f"  Best reward: {best_reward:.1f}")
print(f"  Final reward: {final_reward:.1f}")
print(f"  Improvement: {improvement:.1f} ({(improvement/abs(initial_reward)*100):.1f}%)")
print(f"  Late-training stability (std): {stability:.1f}")

if improvement < 100:
    print("\n FINDING 4: Minimal learning occurred")
    print("    Only {:.1f} point improvement over training".format(improvement))
    print("    Possible causes:")
    print("     - Learning rate too low")
    print("     - Reward signal too weak")
    print("     - Training too short")
    print("     - Environment stochasticity drowning signal")
elif stability > 50:
    print("\n FINDING 4: High variance in late training")
    print("    Model not converging stably")
    print("    Need more training or different hyperparameters")
else:
    print("\n FINDING 4: Training progressed reasonably")

print("\n DIAGNOSTIC 5: ENVIRONMENT STOCHASTICITY")

# Test variance from environment alone
same_policy_rewards = []
for trial in range(10):
    np.random.seed(42 + trial)
    env_test = SimpleButtonTrafficEnv(domain_randomization=False)
    obs_test, _ = env_test.reset()
    env_test.queues = np.array([5, 5, 5, 5], dtype=np.float32)
    obs_test = env_test.queues / env_test.max_queue_length
    
    total_reward = 0
    for step in range(50):
        action = 0  # Fixed action
        obs_test, reward, _, _, _ = env_test.step(action)
        total_reward += reward
    
    same_policy_rewards.append(total_reward)

env_variance = np.std(same_policy_rewards)
print(f"\nSame policy, different seeds: std = {env_variance:.1f}")

if env_variance > 30:
    print("\n FINDING 5: High environment stochasticity")
    print("    Random traffic arrivals create huge variance")
    print("    Signal-to-noise ratio is poor for learning")
    print("    Need to reduce stochasticity or train much longer")
else:
    print("\n FINDING 5: Environment variance acceptable")

print("\n SUMMARY - ROOT CAUSES")
print("\nBased on diagnostics, Run 6 failed because:")

findings = []
if agreement > 0.8:
    findings.append("1. PPO learned to mimic baseline, not surpass it")
if avg_baseline_reward > 500:
    findings.append("2. Reward function already favors baseline strategy")
if unproductive_actions > 2:
    findings.append("3. Weak penalties allow wasted actions")
if improvement < 100:
    findings.append("4. Insufficient learning/improvement over training")
if env_variance > 30:
    findings.append("5. High environment noise drowns learning signal")

for finding in findings:
    print(f"  {finding}")

print("\n RECOMMENDATIONS FOR RUN 7")

if avg_baseline_reward > 500:
    print("\n1. REWARD REDESIGN (Critical)")
    print("   - Baseline gets high rewards with current function")
    print("   - Need to reward BETTER-than-baseline behavior")
    print("   - Suggestion: Add 'beat baseline' bonus in reward")

if env_variance > 30:
    print("\n2. REDUCE STOCHASTICITY (Critical)")
    print("   - Environment noise too high for effective learning")
    print("   - Make traffic arrivals more predictable")
    print("   - Or train 5-10x longer to overcome noise")

if improvement < 100:
    print("\n3. TRAIN LONGER")
    print("   - 250k steps showed minimal improvement")
    print("   - Recommend 1M steps minimum")

if unproductive_actions > 2:
    print("\n4. SIMPLIFY ACTION SPACE")
    print("   - Remove Actions 2 & 3 (All Red, Emergency)")
    print("   - Only keep Actions 0 & 1 (N/S, E/W)")
    print("   - Forces PPO to always take productive actions")
