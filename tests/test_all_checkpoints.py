#!/usr/bin/env python3
"""
Test all checkpoints to find the best performing model
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
    """Corrected longest queue baseline"""
    longest_idx = int(np.argmax(obs))
    return 0 if longest_idx in [0, 1] else 1


def test_checkpoint(checkpoint_path, vecnorm_path, scenarios):
    """Test a single checkpoint on all scenarios"""
    
    # Load model
    model = PPO.load(checkpoint_path)
    
    # Load normalization
    dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    total_reward = 0
    total_cleared = 0
    total_final_queue = 0
    
    for scenario in scenarios:
        env = SimpleButtonTrafficEnv(domain_randomization=False)
        obs, info = env.reset()
        env.queues = scenario['initial_queues'].copy()
        obs = env.queues / env.max_queue_length
        
        episode_reward = 0
        episode_cleared = 0
        
        for step in range(50):
            obs_norm = vec_env.normalize_obs(obs)
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_cleared += info.get('cars_cleared', 0)
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
        total_cleared += episode_cleared
        total_final_queue += np.sum(env.queues)
    
    avg_reward = total_reward / len(scenarios)
    avg_cleared = total_cleared / len(scenarios)
    avg_final_queue = total_final_queue / len(scenarios)
    
    return {
        'avg_reward': avg_reward,
        'avg_cleared': avg_cleared,
        'avg_final_queue': avg_final_queue
    }


# Test scenarios
scenarios = [
    {
        "name": "Balanced Traffic",
        "initial_queues": np.array([5, 5, 5, 5], dtype=np.float32)
    },
    {
        "name": "North Heavy Congestion",
        "initial_queues": np.array([15, 3, 2, 4], dtype=np.float32)
    },
    {
        "name": "East-West Rush Hour",
        "initial_queues": np.array([2, 3, 12, 11], dtype=np.float32)
    },
    {
        "name": "Random Traffic Pattern",
        "initial_queues": np.array([8, 2, 10, 4], dtype=np.float32)
    },
    {
        "name": "Single Lane Blocked",
        "initial_queues": np.array([18, 1, 1, 1], dtype=np.float32)
    }
]

print("CHECKPOINT PERFORMANCE COMPARISON - RUN 6")
print("\nTesting checkpoints from 50k to 250k steps")
print("This will take a few minutes...\n")

# Checkpoints to test
checkpoints_to_test = [
    ('50k', '../models/hardware_ppo/run_6/checkpoint_50000_steps.zip',
     '../models/hardware_ppo/run_6/checkpoint_vecnormalize_50000_steps.pkl'),
    ('100k', '../models/hardware_ppo/run_6/checkpoint_100000_steps.zip',
     '../models/hardware_ppo/run_6/checkpoint_vecnormalize_100000_steps.pkl'),
    ('150k', '../models/hardware_ppo/run_6/checkpoint_150000_steps.zip',
     '../models/hardware_ppo/run_6/checkpoint_vecnormalize_150000_steps.pkl'),
    ('180k (best)', '../models/hardware_ppo/run_6/best_model.zip',
     '../models/hardware_ppo/run_6/checkpoint_vecnormalize_180000_steps.pkl'),
    ('200k', '../models/hardware_ppo/run_6/checkpoint_200000_steps.zip',
     '../models/hardware_ppo/run_6/checkpoint_vecnormalize_200000_steps.pkl'),
    ('250k (final)', '../models/hardware_ppo/run_6/final_model.zip',
     '../models/hardware_ppo/run_6/vecnormalize.pkl'),
]

results = []

for name, model_path, vecnorm_path in checkpoints_to_test:
    if not os.path.exists(model_path):
        print(f"Skipping {name}: file not found")
        continue
    
    print(f"Testing {name}...", end=' ', flush=True)
    
    try:
        result = test_checkpoint(model_path, vecnorm_path, scenarios)
        result['name'] = name
        results.append(result)
        print(f"Reward: {result['avg_reward']:.1f}, Cleared: {result['avg_cleared']:.1f}")
    except Exception as e:
        print(f"ERROR: {e}")

# Also test baseline for comparison
print("\nTesting Baseline (Longest Queue)...", end=' ', flush=True)
baseline_reward = 0
baseline_cleared = 0
baseline_queue = 0

for scenario in scenarios:
    env = SimpleButtonTrafficEnv(domain_randomization=False)
    obs, info = env.reset()
    env.queues = scenario['initial_queues'].copy()
    obs = env.queues / env.max_queue_length
    
    episode_reward = 0
    episode_cleared = 0
    
    for step in range(50):
        action = baseline_longest_queue(obs * env.max_queue_length)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_cleared += info.get('cars_cleared', 0)
    
    baseline_reward += episode_reward
    baseline_cleared += episode_cleared
    baseline_queue += np.sum(env.queues)

baseline_avg_reward = baseline_reward / len(scenarios)
baseline_avg_cleared = baseline_cleared / len(scenarios)
baseline_avg_queue = baseline_queue / len(scenarios)

print(f"Reward: {baseline_avg_reward:.1f}, Cleared: {baseline_avg_cleared:.1f}")

# Print results table
print("\n RESULTS SUMMARY")
print(f"\n{'Checkpoint':<20} {'Avg Reward':>12} {'Avg Cleared':>12} {'Avg Queue':>12}")

for result in results:
    print(f"{result['name']:<20} {result['avg_reward']:>12.1f} "
          f"{result['avg_cleared']:>12.1f} {result['avg_final_queue']:>12.1f}")

print(f"{'Baseline':<20} {baseline_avg_reward:>12.1f} "
      f"{baseline_avg_cleared:>12.1f} {baseline_avg_queue:>12.1f}")

# Find best checkpoint
if results:
    best_checkpoint = max(results, key=lambda x: x['avg_reward'])
    
    print("\n BEST CHECKPOINT")
    print(f"\nBest performing checkpoint: {best_checkpoint['name']}")
    print(f"  Avg Reward: {best_checkpoint['avg_reward']:.1f}")
    print(f"  Avg Cleared: {best_checkpoint['avg_cleared']:.1f}")
    print(f"  Avg Final Queue: {best_checkpoint['avg_final_queue']:.1f}")
    
    # Compare to baseline
    reward_diff = best_checkpoint['avg_reward'] - baseline_avg_reward
    cleared_diff = best_checkpoint['avg_cleared'] - baseline_avg_cleared
    
    print(f"\nVs Baseline:")
    print(f"  Reward: {reward_diff:+.1f} ({(reward_diff/baseline_avg_reward*100):+.1f}%)")
    print(f"  Cleared: {cleared_diff:+.1f} ({(cleared_diff/baseline_avg_cleared*100):+.1f}%)")
    
    if best_checkpoint['avg_reward'] > baseline_avg_reward:
        print("\n SUCCESS: PPO checkpoint outperforms baseline!")
        print(f"\nRECOMMENDATION: Deploy checkpoint {best_checkpoint['name']}")
    else:
        print("\n RESULT: Baseline still outperforms all checkpoints")

