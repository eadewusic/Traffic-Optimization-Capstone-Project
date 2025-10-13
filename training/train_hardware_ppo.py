"""
Train PPO for Hardware Deployment - VERSION 3 with Run Management
Features:
- Automatic run folder creation (run_1, run_2, etc.)
- Organizes all outputs by run number
- Balanced reward function (fixes Run 2 failures)
- Proper experiment tracking
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
import glob
import re

# Define the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the system path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environments.simple_button_env import SimpleButtonTrafficEnv

# CONFIGURATION
TOTAL_TIMESTEPS = 150000
EVAL_FREQ = 5000
SAVE_FREQ = 10000
N_EVAL_EPISODES = 15

# BASE DIRECTORIES
BASE_MODELS_DIR = "../models/hardware_ppo"
BASE_LOGS_DIR = "../logs/hardware_ppo"
BASE_RESULTS_DIR = "../results"
BASE_VISUALIZATIONS_DIR = "../visualizations"

# Create base directories
for dir_path in [BASE_MODELS_DIR, BASE_LOGS_DIR, BASE_RESULTS_DIR, BASE_VISUALIZATIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def get_next_run_number():
    """
    Automatically detect the next run number by looking at existing run folders
    """
    # Check models directory for existing runs
    existing_runs = []
    
    for base_dir in [BASE_MODELS_DIR, BASE_RESULTS_DIR, BASE_VISUALIZATIONS_DIR]:
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)) and item.startswith('run_'):
                    try:
                        run_num = int(item.split('_')[1])
                        existing_runs.append(run_num)
                    except (IndexError, ValueError):
                        continue
    
    if existing_runs:
        return max(existing_runs) + 1
    else:
        return 1


def create_run_directories(run_number):
    """
    Create organized directory structure for this run
    
    Returns dict with all paths for this run
    """
    run_name = f"run_{run_number}"
    
    paths = {
        'models': os.path.join(BASE_MODELS_DIR, run_name),
        'logs': os.path.join(BASE_LOGS_DIR, run_name),
        'results': os.path.join(BASE_RESULTS_DIR, run_name),
        'visualizations': os.path.join(BASE_VISUALIZATIONS_DIR, run_name),
        'run_name': run_name,
        'run_number': run_number
    }
    
    # Create all directories
    for key, path in paths.items():
        if key not in ['run_name', 'run_number']:
            os.makedirs(path, exist_ok=True)
    
    return paths


def plot_training_results(log_path, save_path, run_info):
    """Generate and save training visualization plots"""
    try:
        results = load_results(log_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Results - {run_info["run_name"]} ({run_info["timestamp"]})', fontsize=16)
        
        # 1. Episode Rewards Over Time
        x, y = ts2xy(results, 'timesteps')
        axes[0, 0].plot(x, y, alpha=0.6)
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Moving Average of Rewards
        window = 50
        if len(y) >= window:
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(x[window-1:], moving_avg, color='orange', linewidth=2)
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Moving Average Reward')
            axes[0, 1].set_title(f'Smoothed Rewards (window={window})')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode Lengths
        if 'l' in results.columns:
            axes[1, 0].plot(results['l'], alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Episodes')
            axes[1, 0].set_ylabel('Episode Length')
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative Reward
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
    
    print(f"\nResults saved to: {json_path}")
    
    md_path = os.path.join(results_dir, "training_summary.md")
    with open(md_path, 'w') as f:
        f.write(f"# Training Summary - {results_dict['run_name']}\n\n")
        f.write(f"**Timestamp:** {results_dict['timestamp']}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Total timesteps: {results_dict['config']['total_timesteps']:,}\n")
        f.write(f"- Domain randomization: {results_dict['config']['domain_randomization']}\n")
        f.write(f"- Eval frequency: {results_dict['config']['eval_frequency']:,}\n")
        f.write(f"- Reward version: {results_dict['config']['reward_version']}\n\n")
        
        f.write("## Training Performance\n")
        f.write(f"- Best mean reward: {results_dict['training']['best_mean_reward']:.2f}\n")
        if results_dict['training'].get('convergence_timestep'):
            f.write(f"- Convergence timestep: {results_dict['training']['convergence_timestep']:,}\n")
        f.write("\n")
        
        if 'test' in results_dict:
            f.write("## Quick Test Results (10 episodes)\n")
            f.write(f"- Average reward: {results_dict['test']['avg_reward']:.1f}\n")
            f.write(f"- Std deviation: {results_dict['test']['std_reward']:.1f}\n")
            f.write(f"- Average cleared: {results_dict['test']['avg_cleared']:.0f} vehicles\n")
            f.write(f"- Total cleared: {results_dict['test']['total_cleared']} vehicles\n\n")
            
            f.write("## Episode Details\n")
            for i, reward in enumerate(results_dict['test']['episodes'], 1):
                f.write(f"- Episode {i}: {reward:.1f}\n")
            f.write("\n")
        
        f.write("## Model Files\n")
        f.write(f"- Best model: `{results_dict['files']['best_model']}`\n")
        f.write(f"- Final model: `{results_dict['files']['final_model']}`\n")
        f.write(f"- VecNormalize: `{results_dict['files']['vecnormalize']}`\n")
        f.write(f"- Logs: `{results_dict['files']['logs']}`\n")
        
        if results_dict['files'].get('training_plot'):
            f.write(f"- Training plot: `{results_dict['files']['training_plot']}`\n")
    
    print(f"Summary saved to: {md_path}")


def make_env(domain_randomization=True, seed=None, log_dir=None):
    """Factory function to create environment"""
    def _init():
        env = SimpleButtonTrafficEnv(
            max_queue_length=20,
            cars_cleared_per_cycle=5,
            max_arrival_rate=3,
            domain_randomization=domain_randomization
        )
        if log_dir:
            env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def linear_schedule(initial_value, final_value):
    """Linear learning rate schedule that decays over training"""
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


# ==================== MAIN TRAINING SCRIPT ====================

# Get run number and create directories
run_number = get_next_run_number()
run_paths = create_run_directories(run_number)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_info = {
    'run_name': run_paths['run_name'],
    'run_number': run_paths['run_number'],
    'timestamp': timestamp
}

# PRINT HEADER
print(f"\n RUN {run_number}: PPO TRAINING FOR HARDWARE DEPLOYMENT")
print(f"\nRun: {run_paths['run_name']}")
print(f"Timestamp: {timestamp}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Domain randomization: ENABLED")
print(f"\nReward Function: BALANCED (Version 3)")
print(f"  - Throughput: +1.5 per car (was +1.0 in Run 2)")
print(f"  - Queue penalty: -0.25 per car (was -0.4 in Run 2)")
print(f"  - Ratio: 6:1 (balanced, not too harsh)")
print(f"\nDirectory Structure:")
print(f"  Models: {run_paths['models']}")
print(f"  Logs: {run_paths['logs']}")
print(f"  Results: {run_paths['results']}")
print(f"  Visualizations: {run_paths['visualizations']}")
print()

# CREATE ENVIRONMENTS
print("Creating environments...")

train_env = DummyVecEnv([make_env(domain_randomization=True, seed=42, log_dir=run_paths['logs'])])
train_env = VecNormalize(
    train_env, 
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)

eval_env = DummyVecEnv([make_env(domain_randomization=False, seed=123)])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    training=False,
    clip_obs=10.0
)

print("Environments created")
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
    
    # Learning parameters with decay schedule
    learning_rate=linear_schedule(5e-4, 5e-5),
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    
    # PPO-specific parameters
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    
    # Entropy and value function coefficients
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    
    # Other parameters
    use_sde=False,
    sde_sample_freq=-1,
    target_kl=0.02,
    
    # Logging
    tensorboard_log=run_paths['logs'],
    
    # Network architecture
    policy_kwargs=dict(
        net_arch=dict(
            pi=[128, 64, 32],
            vf=[128, 64, 32]
        ),
        activation_fn=torch.nn.ReLU
    ),
    
    verbose=1,
    device='auto'
)

print("Model created successfully")
print()

# TRAIN MODEL
print(" STARTING TRAINING")
print(f"\nProgress will be displayed below.")
print(f"Expected training time: ~1.5-2 hours\n")

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

# QUICK TEST OF TRAINED MODEL
print("\n TESTING TRAINED MODEL")

test_env = SimpleButtonTrafficEnv(domain_randomization=False)
obs, info = test_env.reset()

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
print(f"  Total cleared:   {int(total_cleared):3d} vehicles")

# GENERATE TRAINING PLOTS
print("\n GENERATING TRAINING PLOTS")

training_plot_path = plot_training_results(run_paths['logs'], run_paths['visualizations'], run_info)
if training_plot_path:
    print(f"Training plot saved to: {training_plot_path}")

# SAVE TRAINING SUMMARY
print("\n SAVING TRAINING SUMMARY")

best_mean_reward = eval_callback.best_mean_reward if hasattr(eval_callback, 'best_mean_reward') else 0.0

training_results = {
    "run_name": run_paths['run_name'],
    "run_number": run_paths['run_number'],
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "total_timesteps": TOTAL_TIMESTEPS,
        "domain_randomization": True,
        "eval_frequency": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES,
        "reward_version": "v3_balanced",
        "improvements": {
            "network_architecture": "[128, 64, 32]",
            "entropy_coefficient": 0.02,
            "batch_size": 128,
            "learning_rate": "5e-4 → 5e-5 (decay)",
            "reward_ratio": "6:1 (throughput:queue)"
        }
    },
    "training": {
        "best_mean_reward": float(best_mean_reward),
        "convergence_timestep": None
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
    }
}

save_training_summary(training_results, run_paths['results'])

# FINAL SUMMARY
print(f"\n RUN {run_number} COMPLETE")

print(f"\nFiles saved in:")
print(f"  {run_paths['models']}/")
print(f"    ├── best_model.zip")
print(f"    ├── final_model.zip")
print(f"    └── vecnormalize.pkl")
print(f"  {run_paths['results']}/")
print(f"    ├── training_summary.json")
print(f"    └── training_summary.md")
print(f"  {run_paths['visualizations']}/")
print(f"    └── training_plot.png")
print()