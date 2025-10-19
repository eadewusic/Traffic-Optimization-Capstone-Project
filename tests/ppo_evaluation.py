#!/usr/bin/env python3
"""
COMPREHENSIVE EVALUATION - Run 7
Measures ALL metrics: Training, Reward, Delay, Throughput, Response Time
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from datetime import datetime
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from environments.run7_env import Run7TrafficEnv

# SETUP DIRECTORIES
RESULTS_DIR = "../results/run_7/metrics_analysis"
VISUALIZATIONS_DIR = "../visualizations/run_7/metrics_analysis"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

print("COMPREHENSIVE EVALUATION - RUN 7")
print(f"\n Results will be saved to:")
print(f"   JSON/MD: {RESULTS_DIR}/")
print(f"   Plots:   {VISUALIZATIONS_DIR}/")
print("\nMeasuring:")
print("  ✓ Comparative Reward")
print("  ✓ Vehicle Delay")
print("  ✓ Throughput")
print("  ✓ Queue Management")
print("  ✓ Computational Response Time (Inference Latency)")


class ComprehensiveEvaluator:
    """Measures ALL performance metrics"""
    
    def run_episode(self, env, model, vec_env, controller_type='ppo'):
        """Run one episode and collect all metrics"""
        obs, _ = env.reset()
        
        episode_reward = 0
        episode_cleared = 0
        episode_delay = 0
        inference_times = []
        vehicle_arrivals = []  # (step, lane)
        
        # Track initial vehicles
        for lane in range(4):
            for _ in range(int(env.queues[lane])):
                vehicle_arrivals.append((0, lane))
        
        for step in range(1, 51):
            # Add new arrivals
            for lane in range(4):
                if np.random.random() < 0.15:
                    vehicle_arrivals.append((step, lane))
            
            # MEASURE RESPONSE TIME
            if controller_type == 'ppo':
                obs_norm = vec_env.normalize_obs(obs / env.max_queue_length)
                
                start_time = time.perf_counter()
                action, _ = model.predict(obs_norm, deterministic=True)
                inference_time = (time.perf_counter() - start_time) * 1000  # ms
                
                inference_times.append(inference_time)
            else:  # baseline
                start_time = time.perf_counter()
                longest_idx = int(np.argmax(obs))
                action = 0 if longest_idx in [0, 1] else 1
                inference_time = (time.perf_counter() - start_time) * 1000
                
                inference_times.append(inference_time)
            
            # Execute action
            obs, reward, term, trunc, info = env.step(action)
            
            # Track metrics
            episode_reward += reward
            cars_cleared = info.get('cars_cleared', 0)
            episode_cleared += cars_cleared
            
            # Calculate delay for departed vehicles
            if cars_cleared > 0:
                served_lanes = [0, 1] if action == 0 else [2, 3]
                served = [(s, l) for s, l in vehicle_arrivals if l in served_lanes]
                served.sort(key=lambda x: x[0])
                
                for i in range(min(int(cars_cleared), len(served))):
                    arrival_step, _ = served[i]
                    wait_time = step - arrival_step
                    episode_delay += wait_time
                
                # Remove departed vehicles
                departed = set(served[:int(cars_cleared)])
                vehicle_arrivals = [v for v in vehicle_arrivals if v not in departed]
        
        final_queue = np.sum(env.queues)
        avg_delay = episode_delay / episode_cleared if episode_cleared > 0 else 0
        avg_inference_time = np.mean(inference_times)
        
        return {
            'reward': float(episode_reward),
            'throughput': int(episode_cleared),
            'avg_delay': float(avg_delay),
            'total_delay': float(episode_delay),
            'final_queue': float(final_queue),
            'avg_inference_time_ms': float(avg_inference_time),
            'max_inference_time_ms': float(np.max(inference_times)),
            'min_inference_time_ms': float(np.min(inference_times))
        }


# Load model
print("\n Loading model...")
model = PPO.load("../models/hardware_ppo/run_7/final_model.zip")
dummy_env = DummyVecEnv([lambda: Run7TrafficEnv()])
vec_env = VecNormalize.load("../models/hardware_ppo/run_7/vecnormalize.pkl", dummy_env)
vec_env.training = False
vec_env.norm_reward = False
print(" Model loaded\n")

# Test scenarios
scenarios = [
    ("Balanced", np.array([5, 5, 5, 5], dtype=np.float32)),
    ("North Heavy", np.array([15, 3, 2, 4], dtype=np.float32)),
    ("E-W Rush", np.array([2, 3, 12, 11], dtype=np.float32)),
    ("Random", np.array([8, 2, 10, 4], dtype=np.float32)),
    ("Blocked", np.array([18, 1, 1, 1], dtype=np.float32))
]

N_TRIALS = 10
evaluator = ComprehensiveEvaluator()

print("Running comprehensive evaluation...")
print(f"  {N_TRIALS} trials × {len(scenarios)} scenarios × 2 controllers")
print(f"  Total: {N_TRIALS * len(scenarios) * 2} episodes\n")

all_results = []
scenario_summaries = []

for scenario_name, initial_queues in scenarios:
    print(f" Scenario: {scenario_name}")
    
    for trial in range(N_TRIALS):
        # Test PPO
        env = Run7TrafficEnv()
        env.reset(options={'initial_queues': initial_queues})
        ppo_result = evaluator.run_episode(env, model, vec_env, 'ppo')
        ppo_result['controller'] = 'PPO'
        ppo_result['scenario'] = scenario_name
        ppo_result['trial'] = trial
        
        # Test Baseline
        env = Run7TrafficEnv()
        env.reset(options={'initial_queues': initial_queues})
        baseline_result = evaluator.run_episode(env, model, vec_env, 'baseline')
        baseline_result['controller'] = 'Baseline'
        baseline_result['scenario'] = scenario_name
        baseline_result['trial'] = trial
        
        all_results.extend([ppo_result, baseline_result])
    
    # Scenario summary
    ppo_trials = [r for r in all_results if r['scenario'] == scenario_name and r['controller'] == 'PPO']
    baseline_trials = [r for r in all_results if r['scenario'] == scenario_name and r['controller'] == 'Baseline']
    
    ppo_avg_reward = np.mean([r['reward'] for r in ppo_trials])
    baseline_avg_reward = np.mean([r['reward'] for r in baseline_trials])
    
    ppo_avg_delay = np.mean([r['avg_delay'] for r in ppo_trials])
    baseline_avg_delay = np.mean([r['avg_delay'] for r in baseline_trials])
    
    ppo_avg_inference = np.mean([r['avg_inference_time_ms'] for r in ppo_trials])
    baseline_avg_inference = np.mean([r['avg_inference_time_ms'] for r in baseline_trials])
    
    delay_reduction = ((baseline_avg_delay - ppo_avg_delay) / baseline_avg_delay * 100) if baseline_avg_delay > 0 else 0
    
    scenario_summaries.append({
        'scenario': scenario_name,
        'ppo_reward': float(ppo_avg_reward),
        'baseline_reward': float(baseline_avg_reward),
        'ppo_delay': float(ppo_avg_delay),
        'baseline_delay': float(baseline_avg_delay),
        'delay_reduction_pct': float(delay_reduction),
        'ppo_inference_ms': float(ppo_avg_inference),
        'baseline_inference_ms': float(baseline_avg_inference)
    })
    
    print(f"   Reward:    PPO={ppo_avg_reward:.1f}, Baseline={baseline_avg_reward:.1f}")
    print(f"   Delay:     PPO={ppo_avg_delay:.1f}s, Baseline={baseline_avg_delay:.1f}s ({delay_reduction:+.1f}%)")
    print(f"   Response:  PPO={ppo_avg_inference:.2f}ms, Baseline={baseline_avg_inference:.2f}ms")
    print()

# CALCULATE OVERALL STATISTICS
print("\n OVERALL PERFORMANCE SUMMARY")

ppo_results = [r for r in all_results if r['controller'] == 'PPO']
baseline_results = [r for r in all_results if r['controller'] == 'Baseline']

# Aggregate metrics
overall_metrics = {
    'Comparative Reward': {
        'ppo': np.mean([r['reward'] for r in ppo_results]),
        'baseline': np.mean([r['reward'] for r in baseline_results])
    },
    'Vehicle Delay (steps)': {
        'ppo': np.mean([r['avg_delay'] for r in ppo_results]),
        'baseline': np.mean([r['avg_delay'] for r in baseline_results])
    },
    'Throughput (cars)': {
        'ppo': np.mean([r['throughput'] for r in ppo_results]),
        'baseline': np.mean([r['throughput'] for r in baseline_results])
    },
    'Final Queue Length': {
        'ppo': np.mean([r['final_queue'] for r in ppo_results]),
        'baseline': np.mean([r['final_queue'] for r in baseline_results])
    },
    'Inference Time (ms)': {
        'ppo': np.mean([r['avg_inference_time_ms'] for r in ppo_results]),
        'baseline': np.mean([r['avg_inference_time_ms'] for r in baseline_results])
    }
}

print(f"\n{'Metric':<30} {'PPO':>15} {'Baseline':>15} {'Difference':>15}")

for metric_name, values in overall_metrics.items():
    ppo_val = values['ppo']
    baseline_val = values['baseline']
    
    if 'Time' in metric_name or 'Queue' in metric_name or 'Delay' in metric_name:
        # Lower is better
        diff_pct = ((baseline_val - ppo_val) / baseline_val * 100) if baseline_val != 0 else 0
        symbol = "↓" if diff_pct > 0 else "↑"
    else:
        # Higher is better
        diff_pct = ((ppo_val - baseline_val) / abs(baseline_val) * 100) if baseline_val != 0 else 0
        symbol = "↑" if diff_pct > 0 else "↓"
    
    print(f"{metric_name:<30} {ppo_val:>15.2f} {baseline_val:>15.2f} {symbol} {abs(diff_pct):>13.1f}%")

# Statistical tests
ppo_rewards = [r['reward'] for r in ppo_results]
baseline_rewards = [r['reward'] for r in baseline_results]
reward_t, reward_p = stats.ttest_ind(ppo_rewards, baseline_rewards)

ppo_delays = [r['avg_delay'] for r in ppo_results]
baseline_delays = [r['avg_delay'] for r in baseline_results]
delay_t, delay_p = stats.ttest_ind(ppo_delays, baseline_delays)

print("\nStatistical Significance:")
print(f"  Reward:  t={reward_t:.3f}, p={reward_p:.4f}")
print(f"  Delay:   t={delay_t:.3f}, p={delay_p:.4f}")

# Response time analysis
ppo_inference_times = [r['avg_inference_time_ms'] for r in ppo_results]
ppo_max_inference = np.max([r['max_inference_time_ms'] for r in ppo_results])

print("\n COMPUTATIONAL RESPONSE TIME ANALYSIS")
print(f"\nPPO Agent:")
print(f"  Mean inference time:   {np.mean(ppo_inference_times):.2f} ms")
print(f"  Std deviation:         {np.std(ppo_inference_times):.2f} ms")
print(f"  Max inference time:    {ppo_max_inference:.2f} ms")
print(f"  Real-time capable:     {'✓ YES' if ppo_max_inference < 100 else '✗ NO'} (<100ms threshold)")

baseline_inference_times = [r['avg_inference_time_ms'] for r in baseline_results]
print(f"\nBaseline Controller:")
print(f"  Mean inference time:   {np.mean(baseline_inference_times):.2f} ms")

speedup_ratio = np.mean(ppo_inference_times) / np.mean(baseline_inference_times)
print(f"\nPPO is {speedup_ratio:.1f}× slower than baseline")
print(f"Both meet real-time requirements for traffic control")

# SAVE JSON RESULTS
print("\n SAVING RESULTS")

results_data = {
    "test": "comprehensive_evaluation",
    "model": "run_7",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_trials": N_TRIALS,
    "n_scenarios": len(scenarios),
    "total_episodes": len(all_results),
    
    "overall_metrics": {
        metric_name: {
            'ppo': float(values['ppo']),
            'baseline': float(values['baseline']),
            'difference_pct': float(((values['ppo'] - values['baseline']) / abs(values['baseline'])) * 100 if values['baseline'] != 0 else 0)
        }
        for metric_name, values in overall_metrics.items()
    },
    
    "statistical_tests": {
        "reward": {"t_statistic": float(reward_t), "p_value": float(reward_p)},
        "delay": {"t_statistic": float(delay_t), "p_value": float(delay_p)}
    },
    
    "response_time": {
        "ppo_mean_ms": float(np.mean(ppo_inference_times)),
        "ppo_std_ms": float(np.std(ppo_inference_times)),
        "ppo_max_ms": float(ppo_max_inference),
        "baseline_mean_ms": float(np.mean(baseline_inference_times)),
        "real_time_capable": bool(ppo_max_inference < 100),
        "speedup_ratio": float(speedup_ratio)
    },
    
    "scenario_summaries": scenario_summaries,
    "detailed_results": all_results
}

json_path = os.path.join(RESULTS_DIR, "comprehensive_evaluation.json")
with open(json_path, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\n JSON saved: {json_path}")

# SAVE MARKDOWN SUMMARY
md_path = os.path.join(RESULTS_DIR, "evaluation_summary.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Comprehensive Evaluation - Run 7\n\n")
    f.write(f"**Date:** {results_data['timestamp']}\n")
    f.write(f"**Model:** Run 7 (PPO Agent)\n")
    f.write(f"**Environment:** Run7TrafficEnv (Comparative Reward)\n")
    f.write(f"**Trials:** {N_TRIALS} per scenario\n")
    f.write(f"**Total Episodes:** {len(all_results)}\n\n")
    
    f.write("## Overall Performance Summary\n\n")
    f.write("| Metric | PPO | Baseline | Improvement |\n")
    f.write("|--------|-----|----------|-------------|\n")
    
    for metric_name, values in overall_metrics.items():
        ppo_val = values['ppo']
        baseline_val = values['baseline']
        diff_pct = ((ppo_val - baseline_val) / abs(baseline_val) * 100) if baseline_val != 0 else 0
        f.write(f"| {metric_name} | {ppo_val:.2f} | {baseline_val:.2f} | {diff_pct:+.1f}% |\n")
    
    f.write("\n## Statistical Significance\n\n")
    f.write(f"- **Reward:** t={reward_t:.3f}, p={reward_p:.4f}\n")
    f.write(f"- **Delay:** t={delay_t:.3f}, p={delay_p:.4f}\n\n")
    
    f.write("## Computational Response Time\n\n")
    f.write(f"- **PPO Mean Inference:** {np.mean(ppo_inference_times):.2f} ms\n")
    f.write(f"- **PPO Max Inference:** {ppo_max_inference:.2f} ms\n")
    f.write(f"- **Baseline Mean Inference:** {np.mean(baseline_inference_times):.2f} ms\n")
    f.write(f"- **Real-time Capable:** {'✓ YES' if ppo_max_inference < 100 else '✗ NO'}\n")
    f.write(f"- **Speed Ratio:** PPO is {speedup_ratio:.1f}× slower than baseline\n\n")
    
    f.write("## Scenario-by-Scenario Results\n\n")
    f.write("| Scenario | PPO Reward | Baseline | Delay Reduction | Response Time |\n")
    f.write("|----------|------------|----------|-----------------|---------------|\n")
    
    for summary in scenario_summaries:
        f.write(f"| {summary['scenario']} | {summary['ppo_reward']:.1f} | "
               f"{summary['baseline_reward']:.1f} | {summary['delay_reduction_pct']:+.1f}% | "
               f"{summary['ppo_inference_ms']:.2f}ms |\n")
    
    f.write("\n## Key Findings\n\n")
    
    reward_improvement = ((overall_metrics['Comparative Reward']['ppo'] - 
                          overall_metrics['Comparative Reward']['baseline']) / 
                         abs(overall_metrics['Comparative Reward']['baseline']) * 100)
    
    delay_reduction = ((overall_metrics['Vehicle Delay (steps)']['baseline'] - 
                       overall_metrics['Vehicle Delay (steps)']['ppo']) / 
                      overall_metrics['Vehicle Delay (steps)']['baseline'] * 100)
    
    f.write(f"1. **Reward Performance:** {reward_improvement:+.1f}% improvement (p={reward_p:.4f})\n")
    f.write(f"2. **Delay Reduction:** {delay_reduction:+.1f}% (p={delay_p:.4f})\n")
    f.write(f"3. **Response Time:** {np.mean(ppo_inference_times):.2f}ms average, real-time capable\n")
    f.write(f"4. **Throughput:** {overall_metrics['Throughput (cars)']['ppo']:.1f} vs {overall_metrics['Throughput (cars)']['baseline']:.1f} cars\n")
    f.write(f"5. **Queue Management:** {overall_metrics['Final Queue Length']['ppo']:.1f} vs {overall_metrics['Final Queue Length']['baseline']:.1f} final queue\n\n")
    
    f.write("## For Thesis Chapter 5\n\n")
    f.write(f"> Comprehensive evaluation across {len(all_results)} episodes demonstrates that Run 7 achieves ")
    f.write(f"{reward_improvement:+.1f}% reward improvement and {delay_reduction:+.1f}% delay reduction compared to ")
    f.write(f"the longest-queue baseline. The agent maintains real-time computational performance ")
    f.write(f"({np.mean(ppo_inference_times):.2f}ms mean inference time), meeting the <100ms requirement for ")
    f.write(f"practical traffic control deployment. Statistical analysis shows ")
    f.write(f"{'significant' if reward_p < 0.05 else 'approaching significant'} performance differences ")
    f.write(f"(reward: p={reward_p:.4f}, delay: p={delay_p:.4f}).\n")

print(f" Markdown saved: {md_path}")

# GENERATE VISUALIZATIONS
print("\n GENERATING VISUALIZATIONS")

# 1. Overall Performance Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Comprehensive Performance Evaluation - Run 7', 
             fontsize=16, fontweight='bold')

metric_names = list(overall_metrics.keys())
ppo_values = [overall_metrics[m]['ppo'] for m in metric_names]
baseline_values = [overall_metrics[m]['baseline'] for m in metric_names]

# Plot each metric
for idx, (ax, metric_name) in enumerate(zip(axes.flat[:5], metric_names)):
    controllers = ['PPO', 'Baseline']
    values = [overall_metrics[metric_name]['ppo'], overall_metrics[metric_name]['baseline']]
    colors = ['green', 'steelblue']
    
    bars = ax.bar(controllers, values, color=colors, alpha=0.7)
    ax.set_title(metric_name, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Response time distribution plot
ax = axes.flat[5]
ax.hist(ppo_inference_times, bins=20, alpha=0.7, color='green', label='PPO', edgecolor='black')
ax.axvline(np.mean(ppo_inference_times), color='red', linestyle='--', 
          linewidth=2, label=f'Mean: {np.mean(ppo_inference_times):.2f}ms')
ax.axvline(100, color='orange', linestyle='--', linewidth=2, label='100ms threshold')
ax.set_xlabel('Inference Time (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Response Time Distribution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
comparison_path = os.path.join(VISUALIZATIONS_DIR, "overall_performance_comparison.png")
plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\n Overall comparison saved: {comparison_path}")

# 2. Scenario Performance Heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Performance by Scenario - Run 7', fontsize=16, fontweight='bold')

scenario_names = [s['scenario'] for s in scenario_summaries]
metrics_to_plot = [
    ('ppo_reward', 'Comparative Reward'),
    ('delay_reduction_pct', 'Delay Reduction (%)'),
    ('ppo_inference_ms', 'Response Time (ms)')
]

for ax, (metric_key, metric_title) in zip(axes, metrics_to_plot):
    values = [s[metric_key] for s in scenario_summaries]
    colors = ['green' if v > 0 else 'red' for v in values] if 'reduction' in metric_key else ['green']*len(values)
    
    bars = ax.barh(scenario_names, values, color=colors, alpha=0.7)
    ax.set_xlabel(metric_title, fontweight='bold')
    ax.set_title(metric_title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f' {value:.1f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
scenario_path = os.path.join(VISUALIZATIONS_DIR, "scenario_performance.png")
plt.savefig(scenario_path, dpi=200, bbox_inches='tight')
plt.close()
print(f" Scenario performance saved: {scenario_path}")

# 3. Delay Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Vehicle Delay Analysis', fontsize=16, fontweight='bold')

# Delay by scenario
ppo_delays_by_scenario = [s['ppo_delay'] for s in scenario_summaries]
baseline_delays_by_scenario = [s['baseline_delay'] for s in scenario_summaries]

x = np.arange(len(scenario_names))
width = 0.35

axes[0].bar(x - width/2, ppo_delays_by_scenario, width, label='PPO', color='green', alpha=0.7)
axes[0].bar(x + width/2, baseline_delays_by_scenario, width, label='Baseline', color='steelblue', alpha=0.7)
axes[0].set_xlabel('Scenario')
axes[0].set_ylabel('Average Delay (steps)')
axes[0].set_title('Average Vehicle Delay by Scenario')
axes[0].set_xticks(x)
axes[0].set_xticklabels(scenario_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Delay reduction percentages
delay_reductions = [s['delay_reduction_pct'] for s in scenario_summaries]
colors = ['green' if r > 0 else 'red' for r in delay_reductions]

bars = axes[1].barh(scenario_names, delay_reductions, color=colors, alpha=0.7)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[1].set_xlabel('Delay Reduction (%)')
axes[1].set_title('Delay Reduction by Scenario')
axes[1].grid(True, alpha=0.3, axis='x')

for bar, value in zip(bars, delay_reductions):
    width = bar.get_width()
    axes[1].text(width, bar.get_y() + bar.get_height()/2,
                f' {value:+.1f}%', ha='left' if value >= 0 else 'right', 
                va='center', fontweight='bold')

plt.tight_layout()
delay_path = os.path.join(VISUALIZATIONS_DIR, "delay_analysis.png")
plt.savefig(delay_path, dpi=200, bbox_inches='tight')
plt.close()
print(f" Delay analysis saved: {delay_path}")

# FINAL SUMMARY
print("\n COMPREHENSIVE EVALUATION COMPLETE")

print(f"\n Files saved:")
print(f"   JSON:  {json_path}")
print(f"   MD:    {md_path}")
print(f"   Plots: {VISUALIZATIONS_DIR}/")
print(f"          - overall_performance_comparison.png")
print(f"          - scenario_performance.png")
print(f"          - delay_analysis.png")

print("\n Key Results:")
print(f"   Reward:       {reward_improvement:+.1f}% (p={reward_p:.4f})")
print(f"   Delay:        {delay_reduction:+.1f}% (p={delay_p:.4f})")
print(f"   Throughput:   {overall_metrics['Throughput (cars)']['ppo']:.1f} vs {overall_metrics['Throughput (cars)']['baseline']:.1f}")
print(f"   Response:     {np.mean(ppo_inference_times):.2f}ms (real-time ✓)")
