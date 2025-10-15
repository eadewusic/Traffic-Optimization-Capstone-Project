#!/usr/bin/env
"""
Train PPO for Hardware Deployment - RUN 6
PROPERLY BALANCED REWARDS - Positive for Good Performance

Key Changes from Run 5:
- Network: [64, 64] (KEPT - proven to work)
- Domain randomization: ENABLED (KEPT - needed for hardware)
- Reward function: FIXED SCALE (NEW - positive for good performance)
- Training: 250k steps (KEPT - adequate for DR)

The Problem with Run 5:
- Reward was always negative (even for good performance)
- Supervisor concern: "Expected positive rewards"
- Academic issue: Reward scale confusing

The Fix in Run 6:
- Throughput rewards DOMINATE (3.0 vs 0.5 in Run 5)
- Congestion penalties SMALLER (0.4 vs 2.0 in Run 5)
- Strategic bonuses added (+5 for attacking longest queue)
- Result: POSITIVE rewards for good performance

Expected Outcomes:
- Training rewards: +500 to +1500 (POSITIVE!)
- Test rewards: +300 to +600 (POSITIVE!)
- Still beats baseline (same strategy, better scale)
- Supervisor satisfied (proper reward scale)
"""

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environments.simple_button_env import SimpleButtonTrafficEnv

# CONFIGURATION
TOTAL_TIMESTEPS = 250000
EVAL_FREQ = 5000
SAVE_FREQ = 10000
N_EVAL_EPISODES = 15

# BASE DIRECTORIES
BASE_MODELS_DIR = "../models/hardware_ppo"
BASE_LOGS_DIR = "../logs/hardware_ppo"
BASE_RESULTS_DIR = "../results"
BASE_VISUALIZATIONS_DIR = "../visualizations"

for dir_path in [BASE_MODELS_DIR, BASE_LOGS_DIR, BASE_RESULTS_DIR, BASE_VISUALIZATIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# HARDWARE-AWARE DOMAIN RANDOMIZATION CONFIG (SAME AS RUN 5)
HARDWARE_DR_CONFIG = {
    'arrival_rate_range': (0.15, 0.45),
    'queue_capacity_range': (8, 12),
    'yellow_duration_range': (2, 4),
    'gpio_latency_range': (1, 10),
    'button_debounce_range': (50, 200),
    'processing_jitter_range': (0, 5),
}


def get_next_run_number():
    """Automatically detect next run number"""
    existing_runs = []
    
    for base_dir in [BASE_MODELS_DIR, BASE_RESULTS_DIR, BASE_VISUALIZATIONS_DIR]:
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('run_'):
                    try:
                        run_num = int(item.split('_')[1].replace('a', '').replace('b', '').replace('c', ''))
                        existing_runs.append(run_num)
                    except (IndexError, ValueError):
                        continue
    
    if existing_runs:
        return max(existing_runs) + 1
    else:
        return 6


def create_run_directories(run_number):
    """Create organized directory structure"""
    run_name = f"run_{run_number}"
    
    paths = {
        'models': os.path.join(BASE_MODELS_DIR, run_name),
        'logs': os.path.join(BASE_LOGS_DIR, run_name),
        'results': os.path.join(BASE_RESULTS_DIR, run_name),
        'visualizations': os.path.join(BASE_VISUALIZATIONS_DIR, run_name),
        'run_name': run_name,
        'run_number': run_number
    }
    
    for key, path in paths.items():
        if key not in ['run_name', 'run_number']:
            os.makedirs(path, exist_ok=True)
    
    return paths


def plot_training_results(log_path, save_path, run_info):
    """Generate training visualization plots"""
    try:
        results = load_results(log_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Results - {run_info["run_name"]} (Properly Balanced Rewards)', fontsize=16)
        
        x, y = ts2xy(results, 'timesteps')
        axes[0, 0].plot(x, y, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero Line')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards Over Time (Should be POSITIVE!)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        window = 50
        if len(y) >= window:
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(x[window-1:], moving_avg, color='orange', linewidth=2)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero Line')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Moving Average Reward')
            axes[0, 1].set_title(f'Smoothed Rewards (window={window})')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        if 'l' in results.columns:
            axes[1, 0].plot(results['l'], alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Episodes')
            axes[1, 0].set_ylabel('Episode Length')
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].grid(True, alpha=0.3)
        
        cumulative_reward = np.cumsum(y)
        axes[1, 1].plot(x, cumulative_reward, color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_title('Cumulative Reward Over Training')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, "training_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating training plot: {e}")
        return None


def save_training_summary(results_dict, results_dir):
    """Save training results to JSON and Markdown"""
    json_path = os.path.join(results_dir, "training_summary.json")
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    md_path = os.path.join(results_dir, "training_summary.md")
    with open(md_path, 'w') as f:
        f.write(f"# Training Summary - {results_dict['run_name']}\n\n")
        f.write(f"**Variant:** Properly Balanced Rewards (Positive for Good Performance)\n")
        f.write(f"**Timestamp:** {results_dict['timestamp']}\n\n")
        
        f.write("## What's Different in Run 6\n")
        f.write("**Problem with Run 5:**\n")
        f.write("- Rewards always negative (even for good performance)\n")
        f.write("- Supervisor concern: 'Expected positive rewards'\n")
        f.write("- Academic issue: Confusing reward scale\n\n")
        
        f.write("**Solution in Run 6:**\n")
        f.write("- Throughput rewards DOMINATE: 3.0 (was 0.5 in Run 5)\n")
        f.write("- Congestion penalties SMALLER: -0.4 (was -2.0 in Run 5)\n")
        f.write("- Strategic bonuses added: +5.0 for attacking longest queue\n")
        f.write("- Result: POSITIVE rewards for good performance\n\n")
        
        f.write("**What's Kept from Run 5 (that worked):**\n")
        f.write("- Strategic alignment: Prioritize longest queue\n")
        f.write("- Simple network: [64, 64]\n")
        f.write("- Hardware DR: GPIO delays, button debounce\n")
        f.write("- Extended training: 250k steps\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Total timesteps: {results_dict['config']['total_timesteps']:,}\n")
        f.write(f"- Domain randomization: {results_dict['config']['domain_randomization']}\n")
        f.write(f"- Network: {results_dict['config']['network_architecture']}\n")
        f.write(f"- Parameters: {results_dict['config']['parameters']}\n\n")
        
        f.write("## Training Performance\n")
        f.write(f"- Best mean reward: {results_dict['training']['best_mean_reward']:.2f}\n")
        f.write(f"- Expected: +500 to +1500 (POSITIVE!)\n\n")
        
        if 'test' in results_dict:
            f.write("## Quick Test Results (10 episodes)\n")
            f.write(f"- Average reward: {results_dict['test']['avg_reward']:.1f}\n")
            f.write(f"- Std deviation: {results_dict['test']['std_reward']:.1f}\n")
            f.write(f"- Average cleared: {results_dict['test']['avg_cleared']:.0f} vehicles\n")
            f.write(f"- Expected: +300 to +600 (POSITIVE!)\n\n")
        
        f.write("## Comparison to Previous Runs\n")
        f.write("| Run | Network | DR | Reward Scale | Test Result | Status |\n")
        f.write("|-----|---------|----|--------------|--------------|---------|\n")
        f.write("| 5 | [64,64] | Yes | Negative | -652 (beat baseline +6.9%) | Worked but confusing |\n")
        f.write("| 6 | [64,64] | Yes | Positive | TBD | Expected: +300 to +600 |\n")
    
    print(f"Summary saved to: {md_path}")


def make_env(domain_randomization=False, seed=None, log_dir=None):
    """Factory function to create environment"""
    def _init():
        env = SimpleButtonTrafficEnv(
            max_queue_length=20,
            cars_cleared_per_cycle=5,
            max_arrival_rate=3,
            domain_randomization=domain_randomization,
            randomization_config=HARDWARE_DR_CONFIG if domain_randomization else None
        )
        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def linear_schedule(initial_value, final_value):
    """Linear learning rate schedule"""
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


# ==================== MAIN TRAINING SCRIPT ====================

run_number = get_next_run_number()
run_paths = create_run_directories(run_number)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_info = {
    'run_name': run_paths['run_name'],
    'run_number': run_paths['run_number'],
    'timestamp': timestamp
}

# PRINT HEADER
print(f" RUN {run_number}: PROPERLY BALANCED REWARDS")
print(f"\nRun: {run_paths['run_name']}")
print(f"Timestamp: {timestamp}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"\n THE PROBLEM WITH RUN 5:")
print(f"  Rewards always negative (even for good performance)")
print(f"  Supervisor: 'Expected positive rewards'")
print(f"  Beat baseline (+6.9%) but reward scale confusing")
print(f"\n THE FIX IN RUN 6:")
print(f"  Throughput rewards: 3.0 (was 0.5 in Run 5) - DOMINATE")
print(f"  Longest queue penalty: -0.4 (was -2.0 in Run 5) - SMALLER")
print(f"  Strategic bonus: +5.0 for attacking longest queue - NEW")
print(f"  Result: POSITIVE rewards for good performance")
print(f"\n WHAT'S KEPT FROM RUN 5 (that worked):")
print(f"  Network: [64, 64] (proven)")
print(f"  Domain randomization: ENABLED (hardware-ready)")
print(f"  Strategic alignment: Longest queue priority")
print(f"\nGoal: Beat baseline + satisfy supervisor with positive rewards")
print()

# CREATE ENVIRONMENTS
print("Creating environments with hardware-aware DR...")

train_env = DummyVecEnv([make_env(domain_randomization=True, seed=42, log_dir=run_paths['logs'])])
train_env = VecNormalize(
    train_env, 
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)

eval_env = DummyVecEnv([make_env(domain_randomization=False, seed=123, log_dir=run_paths['logs'])])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    training=False,
    clip_obs=10.0
)

print("Environments created (Training: DR enabled, Eval: Standard)")
print()

# SETUP CALLBACKS
print("Setting up callbacks...")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=run_paths['models'],
    log_path=run_paths['logs'],
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=run_paths['models'],
    name_prefix="checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

callbacks = [eval_callback, checkpoint_callback]

print("Callbacks configured")
print()

# CREATE PPO MODEL
print("Initializing PPO model...")

model = PPO(
    policy="MlpPolicy",
    env=train_env,
    
    # Learning parameters (SAME AS RUN 5)
    learning_rate=linear_schedule(5e-4, 5e-5),
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    
    # PPO-specific
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    
    # Entropy
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    
    # Other
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=0.02,
    
    # Logging
    tensorboard_log=run_paths['logs'],
    
    # Network architecture (SAME AS RUN 5 - proven)
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        ),
        activation_fn=torch.nn.ReLU
    ),
    
    verbose=1,
    device='auto'
)

print("Model created successfully")
print(f"\nModel details:")
print(f"  Policy: MLP")
print(f"  Architecture: [4 inputs] → [64] → [64] → [4 outputs]")
print(f"  Parameters: ~10,000")
print(f"  Device: {model.device}")
print()

# TRAIN MODEL
print(" STARTING TRAINING - RUN 6")
print(f"\nProperly balanced rewards = Positive for success!")
print(f"Expected training rewards: +500 to +1500 (POSITIVE!)")
print(f"Expected training time: ~2.5 hours\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )
    
    print("\n TRAINING COMPLETE")
    
    # Save final model
    final_model_path = os.path.join(run_paths['models'], "final_model")
    vecnormalize_path = os.path.join(run_paths['models'], "vecnormalize.pkl")
    model.save(final_model_path)
    train_env.save(vecnormalize_path)
    
    print(f"\nFinal model saved:")
    print(f"  Model: {final_model_path}.zip")
    print(f"  VecNormalize: {vecnormalize_path}")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user (Ctrl+C)")
    interrupt_path = os.path.join(run_paths['models'], "interrupted_model")
    vecnormalize_path = os.path.join(run_paths['models'], "vecnormalize_interrupted.pkl")
    model.save(interrupt_path)
    train_env.save(vecnormalize_path)
    print(f"Model saved to: {interrupt_path}.zip")
    sys.exit(0)

except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# QUICK TEST
print("\n TESTING TRAINED MODEL")

test_env = SimpleButtonTrafficEnv(domain_randomization=False)
total_reward = 0
total_cleared = 0
episode_rewards = []

print("\nRunning 10 test episodes...")

for episode in range(10):
    obs, info = test_env.reset()
    episode_reward = 0
    episode_cleared = 0
    
    for step in range(50):
        obs_normalized = train_env.normalize_obs(obs)
        action, _states = model.predict(obs_normalized, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_reward += reward
        episode_cleared += info.get('cars_cleared', 0)
        
        if terminated or truncated:
            break
    
    total_reward += episode_reward
    total_cleared += episode_cleared
    episode_rewards.append(episode_reward)
    
    print(f"Episode {episode+1:2d}: Reward={episode_reward:7.1f}, Cleared={int(episode_cleared):3d} cars")

avg_reward = total_reward / 10
std_reward = np.std(episode_rewards)
avg_cleared = total_cleared / 10

print(f"\nTest Results:")
print(f"  Average reward:  {avg_reward:7.1f}")
print(f"  Reward std dev:  {std_reward:7.1f}")
print(f"  Average cleared: {int(avg_cleared):3d} vehicles")
print(f"\nExpected: +300 to +600 (POSITIVE!)")
print(f"Status: {' SUCCESS' if avg_reward > 200 else ' Review needed'}")

# GENERATE PLOTS
print("\n GENERATING TRAINING PLOTS")
training_plot_path = plot_training_results(run_paths['logs'], run_paths['visualizations'], run_info)
if training_plot_path:
    print(f"Training plot saved to: {training_plot_path}")

# SAVE SUMMARY
print("\n SAVING TRAINING SUMMARY")

best_mean_reward = eval_callback.best_mean_reward if hasattr(eval_callback, 'best_mean_reward') else 0.0

training_results = {
    "run_name": run_paths['run_name'],
    "run_number": run_paths['run_number'],
    "variant": "6_properly_balanced_rewards",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "total_timesteps": TOTAL_TIMESTEPS,
        "domain_randomization": True,
        "network_architecture": "[64, 64]",
        "parameters": "~10,000",
        "batch_size": 64,
        "entropy_coef": 0.01,
        "reward_function": "properly_balanced_positive_for_success",
        "hardware_aware_dr": True,
        "hypothesis": "Same strategy as Run 5, but reward scale shows success as positive"
    },
    "training": {
        "best_mean_reward": float(best_mean_reward)
    },
    "test": {
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "avg_cleared": float(avg_cleared),
        "total_cleared": int(total_cleared),
        "episodes": [float(r) for r in episode_rewards]
    },
    "files": {
        "best_model": os.path.join(run_paths['models'], "best_model.zip"),
        "final_model": final_model_path + ".zip",
        "vecnormalize": vecnormalize_path,
        "logs": run_paths['logs'],
        "training_plot": training_plot_path
    },
    "hardware_dr_config": HARDWARE_DR_CONFIG
}

save_training_summary(training_results, run_paths['results'])

# FINAL SUMMARY
print(f"\n RUN {run_number} COMPLETE - PROPERLY BALANCED REWARDS")

print(f"\nFiles saved in: {run_paths['models']}/")
print()