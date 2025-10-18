#!/usr/bin/env python3
"""
Comprehensive Testing: Hardware PPO vs Baselines
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import json
from datetime import datetime
import re

# Define the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the system path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environments.simple_button_env import SimpleButtonTrafficEnv


def get_latest_run_folder(base_models_dir="../models/hardware_ppo"):
    """
    Automatically detect the latest run folder.
    Returns the run folder name (e.g., 'run_5', 'run_6')
    """
    if not os.path.exists(base_models_dir):
        return None
    
    run_folders = []
    for item in os.listdir(base_models_dir):
        if os.path.isdir(os.path.join(base_models_dir, item)) and item.startswith('run_'):
            # Extract run number
            match = re.search(r'run_(\d+)', item)
            if match:
                run_num = int(match.group(1))
                run_folders.append((run_num, item))
    
    if not run_folders:
        return None
    
    # Return the folder with highest run number
    latest_run = max(run_folders, key=lambda x: x[0])
    return latest_run[1]


def extract_run_name_from_path(model_path):
    """
    Extract run name from model path.
    E.g., "../models/hardware_ppo/run_5/final_model" -> "run_5"
    """
    match = re.search(r'(run_\d+)', model_path)
    if match:
        return match.group(1)
    return None


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# BASELINE CONTROLLERS

def baseline_longest_queue(obs):
    """
    Always serve the longest queue.
    
    Strategy: Find which individual lane has most cars, serve its direction.
    
    Args:
        obs: [N, S, E, W] queue lengths
    
    Returns:
        0 (N/S green) or 1 (E/W green)
    """
    longest_idx = int(np.argmax(obs))
    return 0 if longest_idx in [0, 1] else 1


def baseline_round_robin(step):
    """
    Alternate between N/S and E/W every step.
    
    Simple time-sharing: gives each direction equal opportunity.
    
    Args:
        step: Current timestep
    
    Returns:
        0 (N/S green) or 1 (E/W green)
    """
    return step % 2


def baseline_fixed_time(step):
    """
    Fixed-time controller: 10 steps N/S green, 10 steps E/W green, repeat.
    
    Classic traffic light timing strategy.
    
    Args:
        step: Current timestep
    
    Returns:
        0 (N/S green) or 1 (E/W green)
    """
    return 0 if (step % 20) < 10 else 1


# LOAD RETRAINED MODEL

print("HARDWARE PPO MODEL PERFORMANCE TEST")

print("\nLoading retrained model...")

# Auto-detect latest run
RUN_TO_TEST = "run_1"

if RUN_TO_TEST is None:
    print("Error: No run folders found in ../models/hardware_ppo/")
    print("Available folders should be named like 'run_5', 'run_6', etc.")
    sys.exit(1)

print(f"Testing run: {RUN_TO_TEST}")

# Model paths
model_path = f"../models/hardware_ppo/{RUN_TO_TEST}/hardware_ppo_final_20251012_164818.zip"
vecnorm_path = f"../models/hardware_ppo/{RUN_TO_TEST}/hardware_ppo_final_20251012_164818_vecnormalize.pkl"

# Check if best_model exists (use it if available, otherwise use final_model)
best_model_path = f"../models/hardware_ppo/{RUN_TO_TEST}/best_model"
if os.path.exists(best_model_path + ".zip"):
    model_path = best_model_path
    print("Using best_model.zip (better than final_model)")
elif not os.path.exists(model_path + ".zip"):
    print(f"Error: Model not found at {model_path}.zip")
    print(f"Also checked: {best_model_path}.zip")
    sys.exit(1)

# SETUP DIRECTORIES (RUN-SPECIFIC)
BASE_RESULTS_DIR = "../results"
BASE_VISUALIZATIONS_DIR = "../visualizations"

# Create run-specific directories
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, RUN_TO_TEST)
VISUALIZATIONS_DIR = os.path.join(BASE_VISUALIZATIONS_DIR, RUN_TO_TEST)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

print(f"Results will be saved to: {RESULTS_DIR}")
print(f"Visualizations will be saved to: {VISUALIZATIONS_DIR}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load model
model = PPO.load(model_path)

# Load normalization stats
dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
vec_env = VecNormalize.load(vecnorm_path, dummy_env)
vec_env.training = False
vec_env.norm_reward = False

print("Model loaded successfully")
print(f"  Model: {model_path}.zip")
print(f"  Normalization: {vecnorm_path}")
print()

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
print(f"\nTesting: {RUN_TO_TEST}")
print("Testing 5 traffic scenarios x 4 controllers = 20 tests")
print("Each test: 50 steps per scenario\n")

controllers = [
    ("PPO (Retrained)", "ppo"),
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
        env = SimpleButtonTrafficEnv(domain_randomization=False)
        obs, info = env.reset()
        
        env.queues = scenario['initial_queues'].copy()
        obs = env.queues / env.max_queue_length
        
        episode_reward = 0
        episode_cleared = 0
        final_queue = 0
        
        for step in range(50):
            if controller_type == "ppo":
                obs_norm = vec_env.normalize_obs(obs)
                action, _ = model.predict(obs_norm, deterministic=True)
            elif controller_type == "longest":
                # Pass actual queue values, not normalized
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
    print(f"\n  Best: {best_result['controller']} "
          f"(Reward: {best_result['reward']:.1f})")

# OVERALL SUMMARY

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

print("\nAverage Performance Across All 5 Scenarios:")
print(f"{'Controller':<35} {'Avg Reward':>12} {'Avg Cleared':>12} {'Final Queue':>12}")

for controller_name, _ in controllers:
    stats = controller_stats[controller_name]
    print(f"{controller_name:<35} {stats['avg_reward']:>12.1f} "
          f"{stats['avg_cleared']:>12.1f} {stats['avg_final_queue']:>12.1f}")

ppo_stats = controller_stats["PPO (Retrained)"]
baseline_stats = controller_stats["Longest Queue"]

reward_improvement = ((ppo_stats['avg_reward'] - baseline_stats['avg_reward']) / 
                     abs(baseline_stats['avg_reward']) * 100)
throughput_improvement = ((ppo_stats['avg_cleared'] - baseline_stats['avg_cleared']) / 
                         baseline_stats['avg_cleared'] * 100)
queue_reduction = ((baseline_stats['avg_final_queue'] - ppo_stats['avg_final_queue']) / 
                   baseline_stats['avg_final_queue'] * 100)

print("\n PPO vs Best Baseline (Longest Queue):")
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
print(f"\nOverall Champion: {overall_best[0]}")

if overall_best[0] == "PPO (Retrained)":
    print("\nSUCCESS: PPO model outperforms all baseline strategies!")
    print(f"  {RUN_TO_TEST} is ready for deployment.")
else:
    print(f"\nNote: {overall_best[0]} performed best")
    print("  Consider additional training or reward tuning.")

# GENERATE COMPARISON VISUALIZATIONS

print("\n GENERATING COMPARISON PLOTS")

# 1. Bar chart comparing average performance
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Controller Performance Comparison - {RUN_TO_TEST} - {timestamp}', fontsize=14)

controller_names = [name for name, _ in controllers]
avg_rewards = [controller_stats[name]['avg_reward'] for name in controller_names]
avg_cleared = [controller_stats[name]['avg_cleared'] for name in controller_names]
avg_queues = [controller_stats[name]['avg_final_queue'] for name in controller_names]

colors = ['green' if name == 'PPO (Retrained)' else 'steelblue' for name in controller_names]

axes[0].bar(controller_names, avg_rewards, color=colors, alpha=0.7)
axes[0].set_ylabel('Average Reward')
axes[0].set_title('Average Reward Across All Scenarios')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(controller_names, avg_cleared, color=colors, alpha=0.7)
axes[1].set_ylabel('Average Vehicles Cleared')
axes[1].set_title('Average Throughput')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

axes[2].bar(controller_names, avg_queues, color=colors, alpha=0.7)
axes[2].set_ylabel('Average Final Queue Length')
axes[2].set_title('Average Final Queue (Lower is Better)')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
comparison_plot_path = os.path.join(VISUALIZATIONS_DIR, f"controller_comparison.png")
plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Comparison plot saved to: {comparison_plot_path}")

# 2. Scenario-by-scenario heatmap
scenario_names = [s['name'] for s in scenarios]
performance_matrix = []

for controller_name, _ in controllers:
    row = []
    for scenario in scenarios:
        result = next(r for r in all_results 
                     if r['controller'] == controller_name and r['scenario'] == scenario['name'])
        row.append(result['reward'])
    performance_matrix.append(row)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto')

ax.set_xticks(np.arange(len(scenario_names)))
ax.set_yticks(np.arange(len(controller_names)))
ax.set_xticklabels(scenario_names)
ax.set_yticklabels(controller_names)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(controller_names)):
    for j in range(len(scenario_names)):
        text = ax.text(j, i, f"{performance_matrix[i][j]:.0f}",
                      ha="center", va="center", color="black", fontsize=10)

ax.set_title(f"Reward by Controller and Scenario - {RUN_TO_TEST}")
fig.tight_layout()

heatmap_plot_path = os.path.join(VISUALIZATIONS_DIR, f"scenario_heatmap.png")
plt.savefig(heatmap_plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Scenario heatmap saved to: {heatmap_plot_path}")

# SAVE TEST RESULTS

print("\n SAVING TEST RESULTS")

test_results = {
    "run": RUN_TO_TEST,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": {
        "path": model_path,
        "vecnorm_path": vecnorm_path
    },
    "overall_stats": controller_stats,
    "improvements": {
        "reward_improvement_pct": float(reward_improvement),
        "throughput_improvement_pct": float(throughput_improvement),
        "queue_reduction_pct": float(queue_reduction)
    },
    "wins": wins,
    "champion": overall_best[0],
    "detailed_results": all_results,
    "plots": {
        "comparison": comparison_plot_path,
        "heatmap": heatmap_plot_path
    }
}

# Save JSON
json_path = os.path.join(RESULTS_DIR, f"test_results.json")
with open(json_path, 'w') as f:
    test_results_serializable = convert_numpy_types(test_results)
    json.dump(test_results_serializable, f, indent=2)

print(f"Results saved to: {json_path}")

# Save Markdown
md_path = os.path.join(RESULTS_DIR, f"test_results.md")
with open(md_path, 'w') as f:
    f.write(f"# Test Results - {RUN_TO_TEST}\n\n")
    f.write(f"**Timestamp:** {test_results['timestamp']}\n\n")
    
    f.write("## Model Information\n")
    f.write(f"- Run: {RUN_TO_TEST}\n")
    f.write(f"- Model: `{model_path}`\n")
    f.write(f"- VecNormalize: `{vecnorm_path}`\n\n")
    
    f.write("## Overall Performance\n\n")
    f.write("| Controller | Avg Reward | Avg Cleared | Final Queue |\n")
    f.write("|------------|------------|-------------|-------------|\n")
    for controller_name, _ in controllers:
        stats = controller_stats[controller_name]
        f.write(f"| {controller_name} | {stats['avg_reward']:.1f} | "
               f"{stats['avg_cleared']:.1f} | {stats['avg_final_queue']:.1f} |\n")
    
    f.write("\n## PPO vs Longest Queue Baseline\n")
    f.write(f"- Reward improvement: {reward_improvement:+.1f}%\n")
    f.write(f"- Throughput improvement: {throughput_improvement:+.1f}%\n")
    f.write(f"- Queue reduction: {queue_reduction:+.1f}%\n\n")
    
    f.write("## Wins by Controller\n")
    for controller_name, _ in controllers:
        f.write(f"- {controller_name}: {wins[controller_name]}/{len(scenarios)} scenarios\n")
    
    f.write(f"\n## Champion: {overall_best[0]}\n\n")
    
    f.write("## Scenario Results\n\n")
    for scenario in scenarios:
        f.write(f"### {scenario['name']}\n")
        scenario_results = [r for r in all_results if r['scenario'] == scenario['name']]
        for result in scenario_results:
            f.write(f"- {result['controller']}: Reward={result['reward']:.1f}, "
                   f"Cleared={result['cleared']}, Queue={result['final_queue']}\n")
        f.write("\n")
    
    f.write("## Visualizations\n")
    f.write(f"- Comparison plot: `{comparison_plot_path}`\n")
    f.write(f"- Scenario heatmap: `{heatmap_plot_path}`\n")

print(f"Summary saved to: {md_path}")

print(f"\n TEST COMPLETE - {RUN_TO_TEST}")
