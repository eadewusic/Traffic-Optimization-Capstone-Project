"""
Comprehensive Testing: Hardware PPO vs Baselines
Demonstrates new model outperforms simple strategies
"""

import numpy as np
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

# Define the project root (one level up from 'training')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the system path so Python can find 'environments'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import using the absolute path from the project root
from environments.simple_button_env import SimpleButtonTrafficEnv

# LOAD TRAINED MODEL
print(" HARDWARE PPO MODEL PERFORMANCE TEST")

print("\nLoading trained model...")

# Load the final model and its VecNormalize stats
model_path = "../models/hardware_ppo/hardware_ppo_final_20251012_100838"
vecnorm_path = "../models/hardware_ppo/hardware_ppo_final_20251012_100838_vecnormalize.pkl"

if not os.path.exists(model_path + ".zip"):
    print(f" Error: Model not found at {model_path}.zip")
    sys.exit(1)

model = PPO.load(model_path)

# Load normalization stats
dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
vec_env = VecNormalize.load(vecnorm_path, dummy_env)
vec_env.training = False
vec_env.norm_reward = False

print(" Model loaded successfully")
print(f"  Model: {model_path}.zip")
print(f"  Normalization: {vecnorm_path}")
print()

# DEFINE BASELINE CONTROLLERS

def baseline_longest_queue(obs):
    """Simple baseline: Always pick longest queue"""
    return int(np.argmax(obs))

def baseline_round_robin(step):
    """Round-robin: Cycle through lanes"""
    return step % 4

def baseline_fixed_time(step):
    """Fixed-time: N-S for 10 steps, E-W for 10 steps"""
    cycle_position = step % 20
    if cycle_position < 10:
        return 0 if cycle_position % 2 == 0 else 1  # North or South
    else:
        return 2 if cycle_position % 2 == 0 else 3  # East or West

# TEST SCENARIOS

scenarios = [
    {
        "name": "Balanced Traffic",
        "description": "Equal arrivals from all directions",
        "initial_queues": np.array([5, 5, 5, 5], dtype=np.float32)
    },
    {
        "name": "North Heavy Congestion",
        "description": "One direction very busy",
        "initial_queues": np.array([15, 3, 2, 4], dtype=np.float32)
    },
    {
        "name": "East-West Rush Hour",
        "description": "Cross-traffic both busy",
        "initial_queues": np.array([2, 3, 12, 11], dtype=np.float32)
    },
    {
        "name": "Random Traffic Pattern",
        "description": "Unpredictable mixed loads",
        "initial_queues": np.array([8, 2, 10, 4], dtype=np.float32)
    },
    {
        "name": "Single Lane Blocked",
        "description": "Extreme congestion in one direction",
        "initial_queues": np.array([18, 1, 1, 1], dtype=np.float32)
    }
]

# RUN COMPARATIVE TESTS

print("\n COMPARATIVE PERFORMANCE TEST")
print("\nTesting 5 traffic scenarios × 4 controllers = 20 tests")
print("Each test: 50 steps per scenario\n")

controllers = [
    ("PPO (Trained)", "ppo"),
    ("Longest Queue", "longest"),
    ("Round Robin", "round_robin"),
    ("Fixed Time", "fixed_time")
]

all_results = []

for scenario in scenarios:
    print(f"\n Scenario: {scenario['name']}")
    print(f"   {scenario['description']}")
    print(f"   Initial queues: {scenario['initial_queues']}")
    print()
    
    scenario_results = []
    
    for controller_name, controller_type in controllers:
        # Create new environment
        env = SimpleButtonTrafficEnv(domain_randomization=False)
        obs, info = env.reset()
        
        # Set initial queues to scenario
        env.queues = scenario['initial_queues'].copy()
        obs = env.queues / env.max_queue_length
        
        # Run episode
        episode_reward = 0
        episode_cleared = 0
        final_queue = 0
        
        for step in range(50):
            # Select action based on controller type
            if controller_type == "ppo":
                obs_norm = vec_env.normalize_obs(obs)
                action, _ = model.predict(obs_norm, deterministic=True)
            elif controller_type == "longest":
                action = baseline_longest_queue(obs)
            elif controller_type == "round_robin":
                action = baseline_round_robin(step)
            elif controller_type == "fixed_time":
                action = baseline_fixed_time(step)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_cleared += info.get('cars_cleared', 0)
            
            if terminated or truncated:
                break
        
        final_queue = np.sum(env.queues)
        
        result = {
            "controller": controller_name,
            "scenario": scenario['name'],
            "reward": episode_reward,
            "cleared": int(episode_cleared),
            "final_queue": int(final_queue)
        }
        
        scenario_results.append(result)
        all_results.append(result)
        
        # Print result
        print(f"  {controller_name:15s}: Reward={episode_reward:7.1f}, "
              f"Cleared={int(episode_cleared):3d}, Final Queue={int(final_queue):3d}")
    
    # Find best in scenario
    best_result = max(scenario_results, key=lambda x: x['reward'])
    print(f"\n Best: {best_result['controller']} "
          f"(Reward: {best_result['reward']:.1f})")

# OVERALL SUMMARY

print("\n OVERALL PERFORMANCE SUMMARY")

# Aggregate by controller
controller_stats = {}
for controller_name, _ in controllers:
    controller_results = [r for r in all_results if r['controller'] == controller_name]
    
    avg_reward = np.mean([r['reward'] for r in controller_results])
    avg_cleared = np.mean([r['cleared'] for r in controller_results])
    avg_final_queue = np.mean([r['final_queue'] for r in controller_results])
    
    controller_stats[controller_name] = {
        'avg_reward': avg_reward,
        'avg_cleared': avg_cleared,
        'avg_final_queue': avg_final_queue
    }

print("\nAverage Performance Across All 5 Scenarios:")
print(f"{'Controller':<20} {'Avg Reward':>12} {'Avg Cleared':>12} {'Final Queue':>12}")

for controller_name, _ in controllers:
    stats = controller_stats[controller_name]
    print(f"{controller_name:<20} {stats['avg_reward']:>12.1f} "
          f"{stats['avg_cleared']:>12.1f} {stats['avg_final_queue']:>12.1f}")


# Calculate improvements
ppo_stats = controller_stats["PPO (Trained)"]
baseline_stats = controller_stats["Longest Queue"]

reward_improvement = ((ppo_stats['avg_reward'] - baseline_stats['avg_reward']) / 
                     abs(baseline_stats['avg_reward']) * 100)
throughput_improvement = ((ppo_stats['avg_cleared'] - baseline_stats['avg_cleared']) / 
                         baseline_stats['avg_cleared'] * 100)
queue_reduction = ((baseline_stats['avg_final_queue'] - ppo_stats['avg_final_queue']) / 
                   baseline_stats['avg_final_queue'] * 100)

print("\nPPO vs Best Baseline (Longest Queue):")
print(f"  Reward improvement:     {reward_improvement:+.1f}%")
print(f"  Throughput improvement: {throughput_improvement:+.1f}%")
print(f"  Queue reduction:        {queue_reduction:+.1f}%")

# DETAILED SCENARIO BREAKDOWN

print("\n SCENARIO-BY-SCENARIO WINNER")

for scenario in scenarios:
    scenario_results = [r for r in all_results if r['scenario'] == scenario['name']]
    best = max(scenario_results, key=lambda x: x['reward'])
    
    print(f"\n{scenario['name']}:")
    print(f"  Winner: {best['controller']}")
    print(f"  Performance: Reward={best['reward']:.1f}, "
          f"Cleared={best['cleared']}, Queue={best['final_queue']}")

# FINAL VERDICT

print("\n FINAL VERDICT")

wins = {}
for controller_name, _ in controllers:
    wins[controller_name] = 0

for scenario in scenarios:
    scenario_results = [r for r in all_results if r['scenario'] == scenario['name']]
    best = max(scenario_results, key=lambda x: x['reward'])
    wins[best['controller']] += 1

print(f"\nWins by Controller (out of {len(scenarios)} scenarios):")
for controller_name, _ in controllers:
    print(f"  {controller_name}: {wins[controller_name]}/{len(scenarios)} scenarios")

overall_best = max(wins.items(), key=lambda x: x[1])
print(f"\n Overall Champion: {overall_best[0]}")

if overall_best[0] == "PPO (Trained)":
    print("\n SUCCESS: PPO model outperforms all baseline strategies!")
    print("  Hardware-adapted model is ready for deployment.")
else:
    print(f"\n Note: {overall_best[0]} performed best")
    print("  Consider additional training or reward tuning.")

print("\n TEST COMPLETE")