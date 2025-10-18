#!/usr/bin/env python3
"""
Test multiple checkpoints with deterministic methodology
Find the TRUE best checkpoint under fair testing
"""

import numpy as np
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environments.simple_button_env import SimpleButtonTrafficEnv


def baseline_longest_queue(obs):
    longest_idx = int(np.argmax(obs))
    return 0 if longest_idx in [0, 1] else 1


def test_checkpoint_deterministic(model_path, vecnorm_path, scenarios, test_seeds):
    """Test a checkpoint with deterministic methodology"""
    
    model = PPO.load(model_path)
    dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    all_rewards = []
    all_cleared = []
    all_queues = []
    
    for seed in test_seeds:
        for scenario in scenarios:
            np.random.seed(seed)
            
            env = SimpleButtonTrafficEnv(domain_randomization=False)
            obs, _ = env.reset()
            env.queues = scenario['initial_queues'].copy()
            obs = env.queues / env.max_queue_length
            
            episode_reward = 0
            episode_cleared = 0
            
            for step in range(50):
                obs_norm = vec_env.normalize_obs(obs)
                action, _ = model.predict(obs_norm, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                episode_reward += reward
                episode_cleared += info.get('cars_cleared', 0)
            
            all_rewards.append(episode_reward)
            all_cleared.append(episode_cleared)
            all_queues.append(np.sum(env.queues))
    
    return {
        'avg_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'avg_cleared': np.mean(all_cleared),
        'avg_queue': np.mean(all_queues)
    }


# Test scenarios
scenarios = [
    {"name": "Balanced", "initial_queues": np.array([5, 5, 5, 5], dtype=np.float32)},
    {"name": "North Heavy", "initial_queues": np.array([15, 3, 2, 4], dtype=np.float32)},
    {"name": "E-W Rush", "initial_queues": np.array([2, 3, 12, 11], dtype=np.float32)},
    {"name": "Random", "initial_queues": np.array([8, 2, 10, 4], dtype=np.float32)},
    {"name": "Blocked", "initial_queues": np.array([18, 1, 1, 1], dtype=np.float32)}
]

test_seeds = [42, 123, 456, 789, 1234]

print("CHECKPOINT COMPARISON - DETERMINISTIC METHODOLOGY")
print("\nTesting 4 checkpoints with 5 seeds × 5 scenarios = 25 trials each")
print("This finds the TRUE best checkpoint under fair conditions\n")

checkpoints = [
    ("50k", "../models/hardware_ppo/run_6/checkpoint_50000_steps.zip",
     "../models/hardware_ppo/run_6/checkpoint_vecnormalize_50000_steps.pkl"),
    ("180k (best)", "../models/hardware_ppo/run_6/best_model.zip",
     "../models/hardware_ppo/run_6/checkpoint_vecnormalize_180000_steps.pkl"),
    ("200k", "../models/hardware_ppo/run_6/checkpoint_200000_steps.zip",
     "../models/hardware_ppo/run_6/checkpoint_vecnormalize_200000_steps.pkl"),
    ("250k (final)", "../models/hardware_ppo/run_6/final_model.zip",
     "../models/hardware_ppo/run_6/vecnormalize.pkl"),
]

results = {}

for name, model_path, vecnorm_path in checkpoints:
    print(f"Testing checkpoint {name}...", flush=True)
    result = test_checkpoint_deterministic(model_path, vecnorm_path, scenarios, test_seeds)
    results[name] = result
    print(f"  Avg Reward: {result['avg_reward']:.1f} (±{result['std_reward']:.1f})")

# Test baseline for comparison
print("\nTesting Baseline (Longest Queue)...", flush=True)
baseline_rewards = []
for seed in test_seeds:
    for scenario in scenarios:
        np.random.seed(seed)
        env = SimpleButtonTrafficEnv(domain_randomization=False)
        obs, _ = env.reset()
        env.queues = scenario['initial_queues'].copy()
        obs = env.queues / env.max_queue_length
        
        episode_reward = 0
        for step in range(50):
            action = baseline_longest_queue(obs * env.max_queue_length)
            obs, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
        
        baseline_rewards.append(episode_reward)

baseline_avg = np.mean(baseline_rewards)
baseline_std = np.std(baseline_rewards)
print(f"  Avg Reward: {baseline_avg:.1f} (±{baseline_std:.1f})")

# Results table
print("\n RESULTS SUMMARY")
print(f"\n{'Checkpoint':<20} {'Avg Reward':>12} {'Std Dev':>10} {'vs Baseline':>12}")

for name, _ , _ in checkpoints:
    r = results[name]
    diff = r['avg_reward'] - baseline_avg
    pct = (diff / baseline_avg) * 100
    print(f"{name:<20} {r['avg_reward']:>12.1f} {r['std_reward']:>10.1f} {pct:>11.1f}%")

print(f"{'Baseline':<20} {baseline_avg:>12.1f} {baseline_std:>10.1f} {'---':>12}")

# Find best checkpoint
best_checkpoint = max(results.items(), key=lambda x: x[1]['avg_reward'])

print("\n BEST CHECKPOINT")
print(f"\nWinner: {best_checkpoint[0]}")
print(f"  Avg Reward: {best_checkpoint[1]['avg_reward']:.1f}")
print(f"  Avg Cleared: {best_checkpoint[1]['avg_cleared']:.1f}")
print(f"  vs Baseline: {((best_checkpoint[1]['avg_reward'] - baseline_avg) / baseline_avg * 100):+.1f}%")

if best_checkpoint[1]['avg_reward'] > baseline_avg:
    print("\n SUCCESS: This checkpoint BEATS baseline!")
    print(f"   RECOMMENDATION: Use {best_checkpoint[0]} for deployment")
else:
    print("\n ISSUE: Best checkpoint still loses to baseline")
    print("   RECOMMENDATION: Consider retraining or focus on hardware")
