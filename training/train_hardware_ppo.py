"""
Train PPO for Hardware Deployment
Simplified 4-button environment with domain randomization
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

# Define the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the system path so Python can find 'environments'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import using the absolute path from the project root
from environments.simple_button_env import SimpleButtonTrafficEnv

# CONFIGURATION
TOTAL_TIMESTEPS = 100000  # 100k steps (~30-60 min training time)
EVAL_FREQ = 5000          # Evaluate every 5k steps
SAVE_FREQ = 10000         # Save checkpoint every 10k steps
N_EVAL_EPISODES = 10      # Number of episodes per evaluation

MODELS_DIR = "../models/hardware_ppo"
LOGS_DIR = "../logs/hardware_ppo"
RESULTS_DIR = "../results"
VISUALIZATIONS_DIR = "../visualizations"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# PLOTTING FUNCTIONS
def plot_training_results(log_path, save_path, timestamp):
    """Generate and save training visualization plots"""
    try:
        # Load training results from monitor files
        results = load_results(log_path)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Results - {timestamp}', fontsize=16)
        
        # 1. Episode Rewards Over Time
        x, y = ts2xy(results, 'timesteps')
        axes[0, 0].plot(x, y, alpha=0.6)
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Episode Rewards Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Moving Average of Rewards (smoothed)
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
        
        # Save figure
        plot_path = os.path.join(save_path, f"training_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    except Exception as e:
        print(f"Error creating training plot: {e}")
        return None


def save_training_summary(results_dict, filename="training_summary.json"):
    """Save training results to JSON and Markdown"""
    
    # Save JSON
    json_path = os.path.join(RESULTS_DIR, filename)
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Also save as Markdown for easy reading
    md_path = json_path.replace('.json', '.md')
    with open(md_path, 'w') as f:
        f.write(f"# Training Summary - {results_dict['timestamp']}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Total timesteps: {results_dict['config']['total_timesteps']:,}\n")
        f.write(f"- Domain randomization: {results_dict['config']['domain_randomization']}\n")
        f.write(f"- Eval frequency: {results_dict['config']['eval_frequency']:,}\n\n")
        
        f.write("## Training Performance\n")
        f.write(f"- Best mean reward: {results_dict['training']['best_mean_reward']:.2f}\n")
        if results_dict['training'].get('convergence_timestep'):
            f.write(f"- Convergence timestep: {results_dict['training']['convergence_timestep']:,}\n")
        f.write("\n")
        
        # Write test results if they exist
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


# ENVIRONMENT FACTORY
def make_env(domain_randomization=True, seed=None):
    """Factory function to create environment"""
    def _init():
        env = SimpleButtonTrafficEnv(
            max_queue_length=20,
            cars_cleared_per_cycle=5,
            max_arrival_rate=3,
            domain_randomization=domain_randomization
        )
        # Wrap with Monitor for logging (required for plotting)
        env = Monitor(env, filename=os.path.join(LOGS_DIR, f"monitor_{timestamp}"))
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


# PRINT HEADER
print(" TRAINING PPO FOR HARDWARE DEPLOYMENT")
print(f"\nProject: Traffic Control with Button/LED Prototype")
print(f"Timestamp: {timestamp}")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Domain randomization: ENABLED")
print(f"\nEnvironment:")
print(f"  State: 4 queue counts [North, South, East, West]")
print(f"  Actions: 4 discrete (which lane gets green)")
print(f"  Reward: Minimize total queue length")
print()

# CREATE ENVIRONMENTS
print("Creating environments...")

# Training environment (with domain randomization)
train_env = DummyVecEnv([make_env(domain_randomization=True, seed=42)])
train_env = VecNormalize(
    train_env, 
    norm_obs=True,      # Normalize observations
    norm_reward=True,   # Normalize rewards
    clip_obs=10.0,      # Clip observations
    clip_reward=10.0,   # Clip rewards
    gamma=0.99          # Discount factor for reward normalization
)

# Evaluation environment (without domain randomization for consistent eval)
eval_env = DummyVecEnv([make_env(domain_randomization=False, seed=123)])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,  # Don't normalize rewards during eval
    training=False,     # Don't update running stats during eval
    clip_obs=10.0
)

print("Environments created")
print(f"  Training: With domain randomization")
print(f"  Evaluation: Without domain randomization (for consistent comparison)")
print()

# SETUP CALLBACKS
print("Setting up callbacks...")

# Evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=1
)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODELS_DIR,
    name_prefix=f"hardware_ppo_checkpoint_{timestamp}",
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
    
    # Learning parameters
    learning_rate=3e-4,     # Standard learning rate
    n_steps=2048,           # Steps per update
    batch_size=64,          # Minibatch size
    n_epochs=10,            # Training epochs per update
    
    # PPO-specific parameters
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE parameter
    clip_range=0.2,         # PPO clipping parameter
    clip_range_vf=None,     # Value function clipping (None = same as clip_range)
    normalize_advantage=True,
    
    # Entropy and value function coefficients
    ent_coef=0.01,          # Entropy bonus (encourage exploration)
    vf_coef=0.5,            # Value function loss coefficient
    max_grad_norm=0.5,      # Gradient clipping
    
    # Other parameters
    use_sde=False,          # State-dependent exploration
    sde_sample_freq=-1,
    target_kl=None,         # Target KL divergence (None = no limit)
    
    # Logging
    tensorboard_log=LOGS_DIR,
    
    # Network architecture
    policy_kwargs=dict(
        net_arch=dict(
            pi=[64, 64],    # Policy network: 2 hidden layers, 64 neurons each
            vf=[64, 64]     # Value network: 2 hidden layers, 64 neurons each
        ),
        activation_fn=torch.nn.Tanh  # Activation function
    ),
    
    verbose=1,
    device='auto'  # Use GPU if available, else CPU
)

print("Model created successfully")
print(f"\nModel details:")
print(f"  Policy: MLP (Multi-Layer Perceptron)")
print(f"  Architecture: [4 inputs] → [64] → [64] → [4 outputs]")
print(f"  Parameters: ~10,000 trainable parameters")
print(f"  Device: {model.device}")
print()

# TRAIN MODEL
print("\n STARTING TRAINING")
print(f"\nProgress will be displayed below.\n")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=10,        # Log every 10 updates
        progress_bar=True       # Show progress bar
    )
    
    print("\n TRAINING COMPLETE")
    
    # Save final model
    final_model_path = os.path.join(MODELS_DIR, f"hardware_ppo_final_{timestamp}")
    vecnormalize_path = final_model_path + "_vecnormalize.pkl"
    model.save(final_model_path)
    train_env.save(vecnormalize_path)
    
    print(f"\nFinal model saved:")
    print(f"  Model: {final_model_path}.zip")
    print(f"  VecNormalize: {vecnormalize_path}")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user (Ctrl+C)")
    interrupt_path = os.path.join(MODELS_DIR, f"hardware_ppo_interrupted_{timestamp}")
    vecnormalize_path = interrupt_path + "_vecnormalize.pkl"
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
    
    for step in range(50):  # 50 steps per episode
        # Normalize observation
        obs_normalized = train_env.normalize_obs(obs)
        
        # Get action from trained model
        action, _states = model.predict(obs_normalized, deterministic=True)
        
        # Take step
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

training_plot_path = plot_training_results(LOGS_DIR, VISUALIZATIONS_DIR, timestamp)
if training_plot_path:
    print(f"Training plot saved to: {training_plot_path}")

# SAVE TRAINING SUMMARY
print("\n SAVING TRAINING SUMMARY")

# Get best evaluation results from callback
best_mean_reward = eval_callback.best_mean_reward if hasattr(eval_callback, 'best_mean_reward') else 0.0

training_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "config": {
        "total_timesteps": TOTAL_TIMESTEPS,
        "domain_randomization": True,
        "eval_frequency": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES
    },
    "training": {
        "best_mean_reward": float(best_mean_reward),
        "convergence_timestep": None  # Could be tracked with custom callback
    },
    "test": {
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "avg_cleared": float(avg_cleared),
        "total_cleared": int(total_cleared),
        "episodes": [float(r) for r in episode_rewards]
    },
    "files": {
        "best_model": os.path.join(MODELS_DIR, "best_model.zip"),
        "final_model": final_model_path + ".zip",
        "vecnormalize": vecnormalize_path,
        "logs": LOGS_DIR,
        "training_plot": training_plot_path
    }
}

save_training_summary(training_results, f"training_summary_{timestamp}.json")

# TRAINING SUMMARY
print("\n TRAINING SUMMARY")

print(f"\nFiles saved:")
print(f"  Best model:  {MODELS_DIR}/best_model.zip")
print(f"  Final model: {final_model_path}.zip")
print(f"  Logs:        {LOGS_DIR}/")
if training_plot_path:
    print(f"  Training plot: {training_plot_path}")
print(f"\nTraining complete!")