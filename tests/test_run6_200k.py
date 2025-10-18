#!/usr/bin/env python3
"""
Full test of Run 6 checkpoint 200k
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


def baseline_round_robin(step):
    return step % 2


def baseline_fixed_time(step):
    return 0 if (step % 20) < 10 else 1


# Load checkpoint 200k
print("TESTING RUN 6 CHECKPOINT 200K")

model_path = "../models/hardware_ppo/run_6/checkpoint_200000_steps.zip"
vecnorm_path = "../models/hardware_ppo/run_6/checkpoint_vecnormalize_200000_steps.pkl"

print(f"\nLoading model: {model_path}")
model = PPO.load(model_path)

dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
vec_env = VecNormalize.load(vecnorm_path, dummy_env)
vec_env.training = False
vec_env.norm_reward = False

print("Model loaded successfully\n")

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

controllers = [
    ("PPO (200k checkpoint)", "ppo"),
    ("Longest Queue", "longest"),
    ("Round Robin", "round_robin"),
    ("Fixed Time", "fixed_time")
]

all_results = []

for scenario in scenarios:
    print(f"\n Scenario: {scenario['name']}")
    
    scenario_results = []
    
    for controller_name, controller_type in controllers:
        env = SimpleButtonTrafficEnv(domain_randomization=False)
        obs, info = env.reset()
        env.queues = scenario['initial_queues'].copy()
        obs = env.queues / env.max_queue_length
        
        episode_reward = 0
        episode_cleared = 0
        
        for step in range(50):
            if controller_type == "ppo":
                obs_norm = vec_env.normalize_obs(obs)
                action, _ = model.predict(obs_norm, deterministic=True)
            elif controller_type == "longest":
                action = baseline_longest_queue(obs * env.max_queue_length)
            elif controller_type == "round_robin":
                action = baseline_round_robin(step)
            elif controller_type == "fixed_time":
                action = baseline_fixed_time(step)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_cleared += info.get('cars_cleared', 0)
            
            if terminated or truncated:
                break
        
        final_queue = np.sum(env.queues)
        
        result = {
            "controller": controller_name,
            "scenario": scenario['name'],
            "reward": float(episode_reward),
            "cleared": int(episode_cleared),
            "final_queue": int(final_queue)
        }
        
        scenario_results.append(result)
        all_results.append(result)
        
        print(f"  {controller_name:30s}: Reward={episode_reward:7.1f}, "
              f"Cleared={int(episode_cleared):3d}, Final Queue={int(final_queue):3d}")
    
    best_result = max(scenario_results, key=lambda x: x['reward'])
    print(f"\n  Best: {best_result['controller']}")

# Overall summary
print("\n OVERALL PERFORMANCE SUMMARY")

controller_stats = {}
for controller_name, _ in controllers:
    controller_results = [r for r in all_results if r['controller'] == controller_name]
    
    avg_reward = np.mean([r['reward'] for r in controller_results])
    avg_cleared = np.mean([r['cleared'] for r in controller_results])
    avg_final_queue = np.mean([r['final_queue'] for r in controller_results])
    
    controller_stats[controller_name] = {
        'avg_reward': float(avg_reward),
        'avg_cleared': float(avg_cleared),
        'avg_final_queue': float(avg_final_queue)
    }

print(f"\n{'Controller':<35} {'Avg Reward':>12} {'Avg Cleared':>12} {'Final Queue':>12}")

for controller_name, _ in controllers:
    stats = controller_stats[controller_name]
    print(f"{controller_name:<35} {stats['avg_reward']:>12.1f} "
          f"{stats['avg_cleared']:>12.1f} {stats['avg_final_queue']:>12.1f}")

# Compare to baseline
ppo_stats = controller_stats["PPO (200k checkpoint)"]
baseline_stats = controller_stats["Longest Queue"]

reward_improvement = ((ppo_stats['avg_reward'] - baseline_stats['avg_reward']) / 
                     abs(baseline_stats['avg_reward']) * 100)
throughput_improvement = ((ppo_stats['avg_cleared'] - baseline_stats['avg_cleared']) / 
                         baseline_stats['avg_cleared'] * 100)

print("\n PPO (200k) vs Baseline")
print(f"  Reward improvement:     {reward_improvement:+.1f}%")
print(f"  Throughput improvement: {throughput_improvement:+.1f}%")

if reward_improvement > 0:
    print("\n SUCCESS: PPO CHECKPOINT 200K BEATS BASELINE!")
    print("\nREADY FOR DEPLOYMENT")