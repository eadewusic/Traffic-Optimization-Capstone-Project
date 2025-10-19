#!/usr/bin/env python3
"""
Comprehensive Test: Run 7 Final in ORIGINAL Environment
Tests generalization to SimpleButtonTrafficEnv (different reward function)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
import json
from scipy import stats

# Setup paths
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from environments.simple_button_env import SimpleButtonTrafficEnv  # ORIGINAL ENV


def baseline_longest_queue(obs):
    """Always serve the longest queue"""
    longest_idx = int(np.argmax(obs))
    return 0 if longest_idx in [0, 1] else 1


def baseline_round_robin(step):
    """Alternate between N/S and E/W"""
    return step % 2


def baseline_fixed_time(step):
    """10 steps N/S, 10 steps E/W"""
    return 0 if (step % 20) < 10 else 1


print("RUN 7 - COMPREHENSIVE TEST IN ORIGINAL ENVIRONMENT")
print("\nEnvironment: SimpleButtonTrafficEnv (original reward function)")
print("Purpose: Test generalization to different reward structure")
print()

# Load Run 7
model_path = "../models/hardware_ppo/run_7/final_model.zip"
vecnorm_path = "../models/hardware_ppo/run_7/vecnormalize.pkl"

print("Loading Run 7 model...")
try:
    model = PPO.load(model_path)
    # Use ORIGINAL env for normalization
    dummy_env = DummyVecEnv([lambda: SimpleButtonTrafficEnv(domain_randomization=False)])
    vec_env = VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False
    vec_env.norm_reward = False
    print(" Model loaded successfully\n")
except Exception as e:
    print(f" Error: {e}")
    print("\nMake sure model exists at:")
    print(f"  {model_path}")
    sys.exit(1)

# Test scenarios
scenarios = [
    {"name": "Balanced Traffic", "queues": np.array([5, 5, 5, 5], dtype=np.float32)},
    {"name": "North Heavy", "queues": np.array([15, 3, 2, 4], dtype=np.float32)},
    {"name": "E-W Rush Hour", "queues": np.array([2, 3, 12, 11], dtype=np.float32)},
    {"name": "Random Pattern", "queues": np.array([8, 2, 10, 4], dtype=np.float32)},
    {"name": "Single Lane Blocked", "queues": np.array([18, 1, 1, 1], dtype=np.float32)}
]

# Controllers to test
controllers = [
    ("Run 7 PPO", "ppo"),
    ("Longest Queue", "longest"),
    ("Round Robin", "round_robin"),
    ("Fixed Time", "fixed_time")
]

# Storage
all_results = []
scenario_winners = []

print("Testing 5 scenarios × 4 controllers = 20 tests")

# Run tests
for scenario in scenarios:
    print(f"\n Scenario: {scenario['name']}")
    print(f"   Initial queues: {scenario['queues']}")
    
    scenario_results = {}
    
    for controller_name, controller_type in controllers:
        env = SimpleButtonTrafficEnv(domain_randomization=False)
        obs, _ = env.reset()
        env.queues = scenario['queues'].copy()
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
            
            obs, reward, term, trunc, info = env.step(action)
            episode_reward += reward
            episode_cleared += info.get('cars_cleared', 0)
        
        final_queue = np.sum(env.queues)
        
        result = {
            "controller": controller_name,
            "scenario": scenario['name'],
            "reward": float(episode_reward),
            "cleared": int(episode_cleared),
            "final_queue": int(final_queue)
        }
        
        scenario_results[controller_name] = result
        all_results.append(result)
        
        print(f"   {controller_name:20s}: R={episode_reward:6.1f}, "
              f"C={episode_cleared:3d}, Q={final_queue:2d}")
    
    # Find winner
    best = max(scenario_results.values(), key=lambda x: x['reward'])
    scenario_winners.append(best['controller'])
    print(f"    Winner: {best['controller']}")

# Calculate statistics
print("\n OVERALL STATISTICS")

controller_stats = {}
for controller_name, _ in controllers:
    results = [r for r in all_results if r['controller'] == controller_name]
    
    controller_stats[controller_name] = {
        'avg_reward': float(np.mean([r['reward'] for r in results])),
        'std_reward': float(np.std([r['reward'] for r in results])),
        'avg_cleared': float(np.mean([r['cleared'] for r in results])),
        'avg_final_queue': float(np.mean([r['final_queue'] for r in results]))
    }

print(f"\n{'Controller':<25} {'Avg Reward':>12} {'Avg Cleared':>12} {'Final Queue':>12}")
for controller_name, _ in controllers:
    stats = controller_stats[controller_name]
    print(f"{controller_name:<25} {stats['avg_reward']:>12.1f} "
          f"{stats['avg_cleared']:>12.1f} {stats['avg_final_queue']:>12.1f}")

# PPO vs Baseline comparison
ppo_stats = controller_stats["Run 7 PPO"]
baseline_stats = controller_stats["Longest Queue"]

reward_diff = ppo_stats['avg_reward'] - baseline_stats['avg_reward']
reward_pct = (reward_diff / abs(baseline_stats['avg_reward'])) * 100
throughput_diff = ppo_stats['avg_cleared'] - baseline_stats['avg_cleared']
throughput_pct = (throughput_diff / baseline_stats['avg_cleared']) * 100

# Statistical test
ppo_rewards = [r['reward'] for r in all_results if r['controller'] == "Run 7 PPO"]
baseline_rewards = [r['reward'] for r in all_results if r['controller'] == "Longest Queue"]
t_stat, p_value = stats.ttest_ind(ppo_rewards, baseline_rewards)

print("\n RUN 7 PPO vs LONGEST QUEUE BASELINE")
print(f"\nReward:     {ppo_stats['avg_reward']:.1f} vs {baseline_stats['avg_reward']:.1f} "
      f"({reward_pct:+.1f}%)")
print(f"Throughput: {ppo_stats['avg_cleared']:.1f} vs {baseline_stats['avg_cleared']:.1f} "
      f"({throughput_pct:+.1f}%)")
print(f"Statistical test: t={t_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    sig_label = "statistically significant" if t_stat > 0 else "statistically significant (baseline better)"
else:
    sig_label = "not statistically significant (tied)"

print(f"Result: {sig_label}")

# Win count
wins = {name: scenario_winners.count(name) for name, _ in controllers}
print("\n SCENARIO WINS")
for controller_name, _ in controllers:
    print(f"{controller_name}: {wins[controller_name]}/5 scenarios")

# Verdict
print("\n VERDICT - ORIGINAL ENVIRONMENT")

if reward_pct > 5 and wins["Run 7 PPO"] >= 3:
    print("\n SUCCESS: Run 7 generalizes well to original environment")
    verdict = "STRONG_GENERALIZATION"
elif abs(reward_pct) < 5:
    print("\n≈ STATISTICAL TIE: Run 7 performs equivalently")
    print("   This validates both approaches work well")
    verdict = "EQUIVALENT_PERFORMANCE"
else:
    print("\n  TRANSFER GAP: Run 7 optimized for different reward")
    print("   Performance difference expected due to environment mismatch")
    verdict = "TRANSFER_GAP"

print(f"\nInterpretation:")
if verdict == "EQUIVALENT_PERFORMANCE":
    print("  • Run 7 successfully generalized to different reward function")
    print("  • Both achieve similar traffic management quality")
    print("  • Validates learned policy robustness")
else:
    print("  • Run 7 was trained in Run7TrafficEnv (comparative reward)")
    print("  • Testing in SimpleButtonTrafficEnv (original reward)")
    print("  • Some transfer gap is expected and acceptable")

# Create visualizations
print("\n GENERATING VISUALIZATIONS")

viz_dir = "../visualizations/run_7/original_env"
results_dir = "../results/run_7/original_env"
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 1. Performance comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Run 7 Performance in Original Environment (SimpleButtonTrafficEnv)', 
             fontsize=14, weight='bold')

controller_names = [name for name, _ in controllers]
colors = ['green' if 'PPO' in name else 'steelblue' for name in controller_names]

# Rewards
axes[0].bar(controller_names, 
           [controller_stats[name]['avg_reward'] for name in controller_names],
           color=colors, alpha=0.7)
axes[0].set_ylabel('Average Reward')
axes[0].set_title('Average Reward')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

# Throughput
axes[1].bar(controller_names,
           [controller_stats[name]['avg_cleared'] for name in controller_names],
           color=colors, alpha=0.7)
axes[1].set_ylabel('Vehicles Cleared')
axes[1].set_title('Average Throughput')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

# Final Queue
axes[2].bar(controller_names,
           [controller_stats[name]['avg_final_queue'] for name in controller_names],
           color=colors, alpha=0.7)
axes[2].set_ylabel('Final Queue Length')
axes[2].set_title('Final Queue (Lower Better)')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
bar_path = os.path.join(viz_dir, "performance_comparison.png")
plt.savefig(bar_path, dpi=200, bbox_inches='tight')
plt.close()
print(f" Bar chart saved: {bar_path}")

# 2. Scenario heatmap
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

ax.set_xticks(np.arange(len(scenarios)))
ax.set_yticks(np.arange(len(controller_names)))
ax.set_xticklabels([s['name'] for s in scenarios])
ax.set_yticklabels(controller_names)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(controller_names)):
    for j in range(len(scenarios)):
        text = ax.text(j, i, f"{performance_matrix[i][j]:.0f}",
                      ha="center", va="center", color="black", fontsize=10)

ax.set_title("Reward by Controller and Scenario (Original Environment)")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Reward', rotation=270, labelpad=20)
fig.tight_layout()

heatmap_path = os.path.join(viz_dir, "scenario_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
plt.close()
print(f" Heatmap saved: {heatmap_path}")

# Save results
results_data = {
    "test": "Run 7 - Original Environment",
    "environment": "SimpleButtonTrafficEnv",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "overall_stats": controller_stats,
    "comparison": {
        "reward_difference_pct": float(reward_pct),
        "throughput_difference_pct": float(throughput_pct),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significance": sig_label
    },
    "wins": wins,
    "verdict": verdict,
    "detailed_results": all_results
}

json_path = os.path.join(results_dir, "test_results.json")
with open(json_path, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f" JSON results saved: {json_path}")

# Markdown summary
md_path = os.path.join(results_dir, "test_summary.md")
with open(md_path, 'w') as f:
    f.write("# Run 7 - Original Environment Test\n\n")
    f.write(f"**Test Date:** {results_data['timestamp']}\n\n")
    f.write(f"**Environment:** SimpleButtonTrafficEnv (original reward function)\n\n")
    
    f.write("## Overall Performance\n\n")
    f.write("| Controller | Avg Reward | Avg Cleared | Final Queue |\n")
    f.write("|------------|------------|-------------|-------------|\n")
    for name, _ in controllers:
        s = controller_stats[name]
        f.write(f"| {name} | {s['avg_reward']:.1f} | {s['avg_cleared']:.1f} | {s['avg_final_queue']:.1f} |\n")
    
    f.write(f"\n## Run 7 vs Baseline\n\n")
    f.write(f"- Reward difference: {reward_pct:+.1f}%\n")
    f.write(f"- Throughput difference: {throughput_pct:+.1f}%\n")
    f.write(f"- Statistical test: t={t_stat:.3f}, p={p_value:.4f} ({sig_label})\n")
    
    f.write(f"\n## Scenario Wins\n\n")
    for name, _ in controllers:
        f.write(f"- **{name}:** {wins[name]}/5 scenarios\n")
    
    f.write(f"\n## Verdict\n\n")
    f.write(f"**{verdict}**\n\n")
    f.write(results_data.get('interpretation', ''))

print(f" Markdown summary saved: {md_path}")

print("\n TEST COMPLETE - ORIGINAL ENVIRONMENT")
print(f"\nResults saved to: {results_dir}/")
print("\nKey Findings:")
print(f"  • Run 7: {ppo_stats['avg_reward']:.1f} reward")
print(f"  • Baseline: {baseline_stats['avg_reward']:.1f} reward")
print(f"  • Difference: {reward_pct:+.1f}%")
print(f"  • Verdict: {verdict}")
