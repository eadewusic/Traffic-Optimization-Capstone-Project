#!/usr/bin/env python3
"""
Analyze Run 6 training logs to diagnose performance issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 70)
print("RUN 6 TRAINING LOG ANALYSIS")
print("=" * 70)

LOGS_DIR = "../logs/hardware_ppo/run_6"
VIZ_DIR = "../visualizations/run_6"
os.makedirs(VIZ_DIR, exist_ok=True)

# PART 1: EVALUATION ANALYSIS
print("\n" + "=" * 70)
print("EVALUATION METRICS ANALYSIS")
print("=" * 70)

eval_path = os.path.join(LOGS_DIR, "evaluations.npz")
data = np.load(eval_path)

print("\nAvailable data:", data.files)

timesteps = data['timesteps']
results = data['results']
ep_lengths = data['ep_lengths']

mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)

print(f"\nTotal evaluations: {len(timesteps)}")
print(f"Training range: {timesteps[0]:,} to {timesteps[-1]:,} steps")

# Find key checkpoints
best_idx = np.argmax(mean_rewards)
worst_idx = np.argmin(mean_rewards)
final_idx = len(mean_rewards) - 1

print(f"\nInitial Performance (step {timesteps[0]:,}):")
print(f"  Mean Reward: {mean_rewards[0]:.2f}")
print(f"  Std Reward: {std_rewards[0]:.2f}")

print(f"\nBest Performance (step {timesteps[best_idx]:,}):")
print(f"  Mean Reward: {mean_rewards[best_idx]:.2f}")
print(f"  Std Reward: {std_rewards[best_idx]:.2f}")

print(f"\nFinal Performance (step {timesteps[final_idx]:,}):")
print(f"  Mean Reward: {mean_rewards[final_idx]:.2f}")
print(f"  Std Reward: {std_rewards[final_idx]:.2f}")

print(f"\nWorst Performance (step {timesteps[worst_idx]:,}):")
print(f"  Mean Reward: {mean_rewards[worst_idx]:.2f}")
print(f"  Std Reward: {std_rewards[worst_idx]:.2f}")

# Calculate improvements
initial_to_best = ((mean_rewards[best_idx] - mean_rewards[0]) / abs(mean_rewards[0])) * 100
best_to_final = ((mean_rewards[final_idx] - mean_rewards[best_idx]) / abs(mean_rewards[best_idx])) * 100
initial_to_final = ((mean_rewards[final_idx] - mean_rewards[0]) / abs(mean_rewards[0])) * 100

print(f"\nTraining Progress:")
print(f"  Initial → Best: {initial_to_best:+.1f}%")
print(f"  Best → Final: {best_to_final:+.1f}%")
print(f"  Initial → Final: {initial_to_final:+.1f}%")

# Diagnose issues
print("\n" + "-" * 70)
print("DIAGNOSIS:")
print("-" * 70)

if best_to_final < -10:
    print("WARNING: Significant performance degradation after best checkpoint!")
    print("  → Model is likely overfitting")
    print("  → Should use best_model.zip, not final_model.zip")
elif best_to_final < -5:
    print("CAUTION: Some performance degradation after best checkpoint")
    print("  → Mild overfitting detected")
elif abs(best_to_final) < 5:
    print("OK: Performance stable after best checkpoint")
else:
    print("GOOD: Performance continued improving after best checkpoint")

if initial_to_best < 10:
    print("\nWARNING: Minimal learning detected!")
    print("  → Reward improved by only {:.1f}%".format(initial_to_best))
    print("  → Possible causes:")
    print("    - Reward function not well-aligned with learning")
    print("    - Hyperparameters suboptimal")
    print("    - Environment too difficult")
elif initial_to_best < 30:
    print("\nCAUTION: Modest learning")
    print("  → Reward improved by {:.1f}%".format(initial_to_best))
else:
    print("\nGOOD: Significant learning detected")
    print("  → Reward improved by {:.1f}%".format(initial_to_best))

# Check variance
if std_rewards[best_idx] > mean_rewards[best_idx] * 0.3:
    print("\nWARNING: High reward variance!")
    print(f"  → Std/Mean ratio: {(std_rewards[best_idx]/mean_rewards[best_idx]*100):.1f}%")
    print("  → Policy is inconsistent")

# PART 2: VISUALIZE EVALUATION CURVE
print("\n" + "=" * 70)
print("GENERATING EVALUATION CURVE")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Reward over time
ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
ax1.fill_between(timesteps, 
                  mean_rewards - std_rewards,
                  mean_rewards + std_rewards,
                  alpha=0.3, label='±1 Std Dev')
ax1.axvline(x=timesteps[best_idx], color='g', linestyle='--', 
            label=f'Best Model ({timesteps[best_idx]:,} steps)')
ax1.axhline(y=mean_rewards[best_idx], color='g', linestyle=':', alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='Zero Reward')

# Mark initial and final
ax1.scatter([timesteps[0]], [mean_rewards[0]], color='orange', s=100, 
            zorder=5, label='Initial')
ax1.scatter([timesteps[final_idx]], [mean_rewards[final_idx]], color='purple', 
            s=100, zorder=5, label='Final')

ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Evaluation Reward')
ax1.set_title('Run 6: Evaluation Reward Over Training', fontsize=14, weight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Episode lengths
mean_ep_lengths = ep_lengths.mean(axis=1)
ax2.plot(timesteps, mean_ep_lengths, 'r-', linewidth=2)
ax2.set_xlabel('Training Steps')
ax2.set_ylabel('Episode Length')
ax2.set_title('Episode Length Over Training')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
eval_plot_path = os.path.join(VIZ_DIR, "training_evaluation_curve.png")
plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
print(f"\nEvaluation curve saved to: {eval_plot_path}")
plt.close()

# PART 3: MONITOR LOG ANALYSIS
print("\n" + "=" * 70)
print("EPISODE TRAINING DATA ANALYSIS")
print("=" * 70)

monitor_path = os.path.join(LOGS_DIR, "monitor.monitor.csv")

try:
    df = pd.read_csv(monitor_path, skiprows=1)
    
    print(f"\nTotal training episodes: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    if 'r' in df.columns:
        print(f"\nEpisode Reward Statistics:")
        print(f"  Mean: {df['r'].mean():.2f}")
        print(f"  Median: {df['r'].median():.2f}")
        print(f"  Std: {df['r'].std():.2f}")
        print(f"  Min: {df['r'].min():.2f}")
        print(f"  Max: {df['r'].max():.2f}")
        
        # Analyze trend over time
        window = 100
        df['r_rolling'] = df['r'].rolling(window=window).mean()
        
        first_100_avg = df['r'].head(100).mean()
        last_100_avg = df['r'].tail(100).mean()
        trend = ((last_100_avg - first_100_avg) / abs(first_100_avg)) * 100
        
        print(f"\nLearning Trend (first vs last 100 episodes):")
        print(f"  First 100 episodes: {first_100_avg:.2f}")
        print(f"  Last 100 episodes: {last_100_avg:.2f}")
        print(f"  Improvement: {trend:+.1f}%")
        
        # Visualize training episodes
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['r'], alpha=0.2, color='blue', label='Episode Reward')
        ax.plot(df.index, df['r_rolling'], linewidth=2, color='red', 
                label=f'{window}-Episode Moving Average')
        ax.axhline(y=first_100_avg, color='green', linestyle='--', alpha=0.5,
                   label=f'First 100 Avg: {first_100_avg:.1f}')
        ax.axhline(y=last_100_avg, color='orange', linestyle='--', alpha=0.5,
                   label=f'Last 100 Avg: {last_100_avg:.1f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Run 6: Training Episode Rewards', fontsize=14, weight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        episode_plot_path = os.path.join(VIZ_DIR, "training_episode_rewards.png")
        plt.savefig(episode_plot_path, dpi=150, bbox_inches='tight')
        print(f"\nEpisode reward plot saved to: {episode_plot_path}")
        plt.close()
        
except Exception as e:
    print(f"\nError reading monitor file: {e}")

# PART 4: FINAL RECOMMENDATIONS
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("\n1. IMMEDIATE ACTIONS:")

if mean_rewards[best_idx] < 550:
    print("   - Your best model (reward {:.1f}) underperforms baseline (reward ~547)".format(
        mean_rewards[best_idx]))
    print("   - Consider testing earlier checkpoints (50k, 100k, 150k)")
    print("   - Run: python test_all_checkpoints.py")
else:
    print("   - Your best model may be competitive with baseline")
    print("   - Verify with full comparative testing")

print("\n2. TENSORBOARD INSPECTION:")
print("   - Run: tensorboard --logdir {}".format(LOGS_DIR))
print("   - Check for:")
print("     • Policy loss convergence")
print("     • Value loss stability")
print("     • Explained variance")
print("     • Policy entropy (should decrease gradually)")

print("\n3. IF RETRAINING NEEDED:")
print("   - Adjust reward weights to emphasize throughput:")
print("     'throughput': 5.0 (increase from 3.0)")
print("     'longest_queue': -1.0 (increase penalty from -0.4)")
print("   - Train longer: 500k steps instead of 250k")
print("   - Consider different PPO hyperparameters:")
print("     • Learning rate: 3e-4 → 1e-4 (more stable)")
print("     • Batch size: increase for stability")

print("\n4. THESIS NARRATIVE:")
if mean_rewards[best_idx] < 540:
    print("   - Focus on 'Single Lane Blocked' scenario where PPO excels")
    print("   - Emphasize hardware deployment and real-world robustness")
    print("   - Honest discussion of baseline competitiveness")
else:
    print("   - PPO shows competitive or superior performance")
    print("   - Emphasize learned adaptability vs fixed heuristic")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nNext steps:")
print("1. View generated plots in: {}".format(VIZ_DIR))
print("2. Run TensorBoard for detailed metrics")
print("3. Test all checkpoints to find best performer")
print("4. Make deployment decision based on findings")