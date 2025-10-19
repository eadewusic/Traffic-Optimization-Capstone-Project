#!/usr/bin/env python3
"""
Combines all features from both versions
"""

# CRITICAL: Set matplotlib backend BEFORE any other imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI)
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor 
import os
import sys
import numpy as np
import json
from datetime import datetime

# Get the absolute path of the directory containing the script
current_dir = os.path.abspath(os.path.dirname(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir) 

# Add the parent directory to the Python search path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from environments.run7_env import Run7TrafficEnv


class ProgressVisualizationCallback(BaseCallback):
    """
    Custom callback that creates visualizations during training
    Generates plots every 50k steps for progress monitoring
    """
    
    def __init__(self, viz_dir, eval_log_path, verbose=0):
        super().__init__(verbose)
        self.viz_dir = viz_dir
        self.eval_log_path = eval_log_path
        self.visualization_freq = 50000
        
    def _on_step(self):
        # Check if we should create visualization
        if self.num_timesteps % self.visualization_freq == 0:
            self._create_progress_plot()
        return True
    
    def _create_progress_plot(self):
        """Create training progress visualization"""
        try:
            # Load evaluation data
            eval_data = np.load(os.path.join(self.eval_log_path, 'evaluations.npz'))
            timesteps = eval_data['timesteps']
            results = eval_data['results']
            mean_rewards = results.mean(axis=1)
            std_rewards = results.std(axis=1)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Run 7 Mean Reward')
            ax.fill_between(timesteps, 
                           mean_rewards - std_rewards,
                           mean_rewards + std_rewards,
                           alpha=0.3, label='±1 Std Dev')
            
            # Reference line at zero (not Run 6 baseline - different reward function!)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, 
                      label='Zero Baseline')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Comparative Reward')
            ax.set_title(f'Run 7 Training Progress (Step {self.num_timesteps:,})')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Save
            plot_path = os.path.join(self.viz_dir, f'progress_step_{self.num_timesteps}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n Progress plot saved: {plot_path}")
            print(f"   Current reward: {mean_rewards[-1]:.1f}")
            print(f"   Target: Positive and increasing")
            
            # Decision support
            if mean_rewards[-1] > 1500:
                print("    STATUS: STRONG PERFORMANCE!")
            elif mean_rewards[-1] > 500:
                print("    STATUS: Good progress, continuing...")
            elif mean_rewards[-1] > 0:
                print("    STATUS: Positive and learning...")
            else:
                print("    STATUS: Still negative, monitor closely")
            
        except Exception as e:
            print(f"Warning: Could not create progress plot: {e}")


def create_analysis(logs_dir, viz_dir):
    """
    Create comprehensive analysis after training completes
    """
    print("\n CREATING TRAINING ANALYSIS")
    
    # Load Run 7 evaluation data
    eval_path = os.path.join(logs_dir, 'evaluations.npz')
    eval_data = np.load(eval_path)
    
    timesteps = eval_data['timesteps']
    results = eval_data['results']
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    
    # Statistics
    initial_reward = mean_rewards[0]
    best_reward = np.max(mean_rewards)
    final_reward = mean_rewards[-1]
    best_idx = np.argmax(mean_rewards)
    best_step = timesteps[best_idx]
    
    improvement = best_reward - initial_reward
    improvement_pct = (improvement / abs(initial_reward)) * 100 if initial_reward != 0 else 0
    
    print(f"\n Training Statistics:")
    print(f"   Initial reward: {initial_reward:.1f}")
    print(f"   Best reward: {best_reward:.1f} (at step {best_step:,})")
    print(f"   Final reward: {final_reward:.1f}")
    print(f"   Improvement: {improvement:.1f} ({improvement_pct:.1f}%)")
    
    # Success criteria for comparative reward
    # Success = positive final reward with clear improvement
    success = final_reward > 1000 and improvement > 5000
    
    if success:
        print(f"\n    SUCCESS! Strong learning demonstrated")
        print(f"   Next: Test vs baseline in SAME environment")
    else:
        print(f"\n    Training completed but performance moderate")
        print(f"   Next: Test vs baseline to determine effectiveness")
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Run 7 Training Analysis - Comparative Reward', fontsize=16, weight='bold')
    
    # Plot 1: Training curve
    ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Run 7 Mean')
    ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     alpha=0.3, label='±1 Std Dev')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero Line')
    ax1.axvline(x=best_step, color='g', linestyle=':', alpha=0.5, label=f'Best ({best_step:,})')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Comparative Reward')
    ax1.set_title('Evaluation Reward Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement trajectory
    improvement_curve = mean_rewards - initial_reward
    ax2.plot(timesteps, improvement_curve, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(timesteps, 0, improvement_curve, where=(improvement_curve >= 0),
                     alpha=0.3, color='green')
    ax2.fill_between(timesteps, 0, improvement_curve, where=(improvement_curve < 0),
                     alpha=0.3, color='red')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Improvement from Initial')
    ax2.set_title('Learning Progress')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance analysis
    ax3.plot(timesteps, std_rewards, 'purple', linewidth=2)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Reward Std Dev')
    ax3.set_title('Evaluation Stability')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution at different stages
    # Early, mid, late training
    early_idx = min(10, len(results)//4)
    mid_idx = len(results)//2
    late_idx = -1
    
    stages_data = {
        'Early\nTraining': results[early_idx],
        'Mid\nTraining': results[mid_idx],
        'Late\nTraining': results[late_idx]
    }
    
    positions = range(len(stages_data))
    bp = ax4.boxplot([stages_data[k] for k in stages_data.keys()], 
                      positions=positions,
                      labels=stages_data.keys(),
                      patch_artist=True)
    
    # Color the boxes
    colors = ['lightcoral', 'lightyellow', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Reward Distribution')
    ax4.set_title('Performance Evolution')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(viz_dir, 'run7_analysis.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n Analysis plot saved: {plot_path}")
    
    # Save summary JSON
    summary = {
        'run': 'run_7',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training': {
            'total_steps': int(timesteps[-1]),
            'initial_reward': float(initial_reward),
            'best_reward': float(best_reward),
            'best_step': int(best_step),
            'final_reward': float(final_reward),
            'improvement': float(improvement),
            'improvement_pct': float(improvement_pct),
            'final_std': float(std_rewards[-1])
        },
        'assessment': {
            'strong_learning': bool(success),
            'ready_for_testing': True
        },
        'next_steps': [
            'Test Run 7 vs Baseline in Run7TrafficEnv',
            'Both controllers in SAME environment',
            'Fair comparison with comparative reward'
        ]
    }
    
    summary_path = os.path.join(viz_dir, 'run7_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f" Summary saved: {summary_path}")
    
    return success

# MAIN TRAINING SCRIPT

print("RUN 7 - OPTIMIZED TRAINING")
print("\nKey Optimizations:")
print("  1. Comparative reward with balanced scaling (×8 win, ×5 loss)")
print("  2. Semi-deterministic arrivals (reduced variance)")
print("  3. 2 actions only (N/S, E/W)")
print("  4. 1.5M timesteps (50% more than before)")
print("  5. Automatic visualization every 50k steps")
print()

# Setup directories
MODELS_DIR = "../models/hardware_ppo/run_7"
LOGS_DIR = "../logs/hardware_ppo/run_7"
VIZ_DIR = "../visualizations/run_7"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Create training environment
env = Run7TrafficEnv()
env = DummyVecEnv([lambda: Run7TrafficEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Create PPO model with proven hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-5,       # Low for stability
    n_steps=4096,             # Long rollouts
    batch_size=256,           # Large batches
    n_epochs=20,              # Many gradient updates
    gamma=0.99,               # Standard discount
    gae_lambda=0.95,          # GAE parameter
    clip_range=0.2,           # PPO clipping
    ent_coef=0.02,            # Entropy for exploration
    vf_coef=0.5,              # Value function coefficient
    max_grad_norm=0.5,        # Gradient clipping
    verbose=1,
    tensorboard_log=LOGS_DIR
)

# Create evaluation environment with Monitor
eval_env = DummyVecEnv([lambda: Monitor(Run7TrafficEnv(), os.path.join(LOGS_DIR, "eval_monitor.csv"))])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# Setup callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR,
    eval_freq=2000,           # Evaluate every 2000 steps
    n_eval_episodes=10,       # 10 episodes per evaluation
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,          # Save checkpoint every 10k steps
    save_path=MODELS_DIR,
    name_prefix='checkpoint',
    save_vecnormalize=True
)

viz_callback = ProgressVisualizationCallback(
    viz_dir=VIZ_DIR,
    eval_log_path=LOGS_DIR
)

print("Configuration:")
print(f"  Total timesteps: 1,500,000 (1.5M)")
print(f"  Checkpoint frequency: Every 10,000 steps")
print(f"  Evaluation frequency: Every 2,000 steps")
print(f"  Visualization frequency: Every 50,000 steps")
print(f"\nDirectories:")
print(f"  Models: {MODELS_DIR}")
print(f"  Logs: {LOGS_DIR}")
print(f"  Visualizations: {VIZ_DIR}")
print("\nExpected training time: 18-36 hours")
print("\n STARTING TRAINING...")

# Train the model
try:
    model.learn(
        total_timesteps=1_500_000,  # 1.5M timesteps (50% more than original)
        callback=CallbackList([eval_callback, checkpoint_callback, viz_callback]),
        progress_bar=True
    )
    
    # Save final model
    model.save(os.path.join(MODELS_DIR, "final_model"))
    env.save(os.path.join(MODELS_DIR, "vecnormalize.pkl"))
    
    print("\n TRAINING COMPLETE!")
    
    # Create analysis
    success = create_analysis(LOGS_DIR, VIZ_DIR)

except KeyboardInterrupt:
    print("\n TRAINING INTERRUPTED BY USER")
    print("\nSaving current model...")
    model.save(os.path.join(MODELS_DIR, "interrupted_model"))
    env.save(os.path.join(MODELS_DIR, "interrupted_vecnormalize.pkl"))
    print(" Model saved as 'interrupted_model'")
    
    # Try to create analysis with available data
    print("\nAttempting to create analysis with available data...")
    try:
        create_analysis(LOGS_DIR, VIZ_DIR)
        print(" Analysis created successfully")
    except Exception as e:
        print(f"  Could not create analysis: {e}")
        print("   (Not enough training data yet)")
    
    print("\nYou can:")
    print("  1. Resume training from interrupted_model")
    print("  2. Test interrupted_model vs baseline")
    print("  3. Start fresh training")

except Exception as e:
    print(f"\n\n ERROR DURING TRAINING: {e}")
    print("Attempting to save model...")
    try:
        model.save(os.path.join(MODELS_DIR, "error_model"))
        env.save(os.path.join(MODELS_DIR, "error_vecnormalize.pkl"))
        print(" Model saved as 'error_model'")
    except:
        print(" Could not save model")
    raise

print("TRAINING SCRIPT COMPLETE")
