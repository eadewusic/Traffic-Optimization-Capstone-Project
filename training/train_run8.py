#!/usr/bin/env python3
"""
Run 8: Multi-Seed Training for Statistical Validation
This script performs PPO training for traffic signal control using multiple random seeds.
It generates training visualizations and saves summaries for each seed.
The seeds used are: 42, 123, 456, 789, and 1000.
Each seed's results will be aggregated later for statistical analysis.
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
import random
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir) 

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from environments.run7_env import Run7TrafficEnv


def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    print(f"\n SETTING RANDOM SEED: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"   Python random: {seed}")
        print(f"   NumPy: {seed}")
        print(f"   PyTorch: {seed}")
    except ImportError:
        print(f"   Python random: {seed}")
        print(f"   NumPy: {seed}")


class ProgressVisualizationCallback(BaseCallback):
    """Custom callback for progress visualization during training"""
    
    def __init__(self, viz_dir, eval_log_path, seed, verbose=0):
        super().__init__(verbose)
        self.viz_dir = viz_dir
        self.eval_log_path = eval_log_path
        self.seed = seed
        self.visualization_freq = 50000
        
    def _on_step(self):
        if self.num_timesteps % self.visualization_freq == 0:
            self._create_progress_plot()
        return True
    
    def _create_progress_plot(self):
        """Create training progress visualization"""
        try:
            eval_data = np.load(os.path.join(self.eval_log_path, 'evaluations.npz'))
            timesteps = eval_data['timesteps']
            results = eval_data['results']
            mean_rewards = results.mean(axis=1)
            std_rewards = results.std(axis=1)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(timesteps, mean_rewards, 'b-', linewidth=2, label=f'Run 8 (Seed {self.seed})')
            ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                           alpha=0.3, label='±1 Std Dev')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Zero Baseline')
            ax.set_xlabel('Training Steps', fontsize=11)
            ax.set_ylabel('Comparative Reward', fontsize=11)
            ax.set_title(f'Run 8 Training Progress - Seed {self.seed} (Step {self.num_timesteps:,})', 
                        fontsize=12, weight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plot_path = os.path.join(self.viz_dir, f'progress_seed{self.seed}_step_{self.num_timesteps}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n Progress plot saved: {plot_path}")
            print(f"   Current reward: {mean_rewards[-1]:.1f}")
            
        except Exception as e:
            print(f"  Warning: Could not create progress plot: {e}")


def create_training_plots(logs_dir, viz_dir, seed):
    """Create comprehensive training analysis plots (matches run_6/run_7 style)"""
    print("\n Creating training visualizations...")
    
    try:
        eval_path = os.path.join(logs_dir, 'evaluations.npz')
        eval_data = np.load(eval_path)
        
        timesteps = eval_data['timesteps']
        results = eval_data['results']
        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)
        
        initial_reward = mean_rewards[0]
        best_reward = np.max(mean_rewards)
        final_reward = mean_rewards[-1]
        best_idx = np.argmax(mean_rewards)
        best_step = timesteps[best_idx]
        improvement = best_reward - initial_reward
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Run 8 Training Analysis - Seed {seed}', fontsize=16, weight='bold')
        
        # Plot 1: Training evaluation curve
        ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label=f'Seed {seed} Mean')
        ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                         alpha=0.3, label='±1 Std Dev')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero Line')
        ax1.axvline(x=best_step, color='g', linestyle=':', alpha=0.5, label=f'Best ({best_step:,})')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Evaluation Reward')
        ax1.set_title('Training Evaluation Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement trajectory
        improvement_curve = mean_rewards - initial_reward
        ax2.plot(timesteps, improvement_curve, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.fill_between(timesteps, 0, improvement_curve, where=(improvement_curve >= 0),
                         alpha=0.3, color='green', label='Improvement')
        ax2.fill_between(timesteps, 0, improvement_curve, where=(improvement_curve < 0),
                         alpha=0.3, color='red', label='Decline')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Reward Change from Initial')
        ax2.set_title('Learning Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Variance/stability
        ax3.plot(timesteps, std_rewards, 'purple', linewidth=2)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Reward Standard Deviation')
        ax3.set_title('Evaluation Stability')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance distribution
        early_idx = min(10, len(results)//4)
        mid_idx = len(results)//2
        late_idx = -1
        
        stages_data = {
            'Early': results[early_idx],
            'Mid': results[mid_idx],
            'Late': results[late_idx]
        }
        
        bp = ax4.boxplot([stages_data[k] for k in stages_data.keys()], 
                          labels=stages_data.keys(),
                          patch_artist=True)
        
        colors = ['lightcoral', 'lightyellow', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Reward Distribution')
        ax4.set_title('Performance Evolution')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        plot_path = os.path.join(viz_dir, 'training_plot.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f" Saved: {plot_path}")
        
        # Create separate evaluation curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(timesteps, mean_rewards, 'b-', linewidth=2, label=f'Seed {seed}')
        ax.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Zero Baseline')
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Evaluation Reward', fontsize=11)
        ax.set_title(f'Run 8 - Seed {seed}: Training Evaluation Curve', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        eval_curve_path = os.path.join(viz_dir, 'training_evaluation_curve.png')
        plt.savefig(eval_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Saved: {eval_curve_path}")
        
        return {
            'initial_reward': float(initial_reward),
            'best_reward': float(best_reward),
            'final_reward': float(final_reward),
            'best_step': int(best_step),
            'improvement': float(improvement)
        }
        
    except Exception as e:
        print(f"  Warning: Could not create visualizations: {e}")
        return None


def save_training_summary(stats, seed, results_dir):
    """Save training summary in JSON and Markdown formats"""
    print("\n Saving training summaries...")
    
    # JSON summary
    json_summary = {
        'run': 'run_8',
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'training_statistics': stats,
        'hyperparameters': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        },
        'environment': {
            'name': 'Run7TrafficEnv',
            'max_queue_length': 20,
            'cars_cleared_per_cycle': 5
        }
    }
    
    json_path = os.path.join(results_dir, 'training_summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f" Saved: {json_path}")
    
    # Markdown summary
    md_content = f"""# Run 8 Training Summary - Seed {seed}

## Training Configuration

**Run:** run_8  
**Seed:** {seed}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Statistics

| Metric | Value |
|--------|-------|
| Initial Reward | {stats['initial_reward']:.2f} |
| Best Reward | {stats['best_reward']:.2f} |
| Final Reward | {stats['final_reward']:.2f} |
| Best Step | {stats['best_step']:,} |
| Total Improvement | {stats['improvement']:.2f} |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.0003 |
| Steps per Update | 2048 |
| Batch Size | 64 |
| Epochs | 10 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| Entropy Coefficient | 0.01 |

## Environment Configuration

- **Environment:** Run7TrafficEnv
- **Max Queue Length:** 20 vehicles
- **Cars Cleared per Cycle:** 5 vehicles
- **Observation Space:** Box(4,) - Queue lengths for N, S, E, W
- **Action Space:** Discrete(2) - N/S or E/W green phase

## Training Details

- **Total Training Steps:** 1,000,000
- **Evaluation Frequency:** Every 10,000 steps
- **Checkpoint Frequency:** Every 100,000 steps
- **Visualization Frequency:** Every 50,000 steps

## Notes

This is part of Run 8 multi-seed validation experiment. Results from all seeds 
(42, 123, 456, 789, 1000) will be aggregated to compute mean ± standard deviation 
for statistical robustness.
"""
    
    md_path = os.path.join(results_dir, 'training_summary.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f" Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Run 8: Multi-Seed PPO Training')
    parser.add_argument('--seed', type=int, required=True, 
                       help='Random seed (42, 123, 456, 789, or 1000)')
    parser.add_argument('--total-steps', type=int, default=1000000,
                       help='Total training timesteps (default: 1000000)')
    args = parser.parse_args()
    
    seed = args.seed
    total_timesteps = args.total_steps
    
    print("="*70)
    print(" PPO TRAFFIC CONTROL - RUN 8 (MULTI-SEED VALIDATION)")
    print("="*70)
    print(f"\n Random Seed: {seed}")
    print(f" Total Steps: {total_timesteps:,}")
    
    set_random_seeds(seed)
    
    # Create directory structure
    project_root = Path(parent_dir)
    logs_run8 = project_root / 'logs' / 'hardware_ppo' / 'run_8' / f'seed_{seed}'
    models_run8 = project_root / 'models' / 'hardware_ppo' / 'run_8' / f'seed_{seed}'
    results_run8 = project_root / 'results' / 'run_8' / f'seed_{seed}'
    viz_run8 = project_root / 'visualizations' / 'run_8' / f'seed_{seed}'
    
    for directory in [logs_run8, models_run8, results_run8, viz_run8]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Directory Structure:")
    print(f"   Logs:           {logs_run8}")
    print(f"   Models:         {models_run8}")
    print(f"   Results:        {results_run8}")
    print(f"   Visualizations: {viz_run8}")
    
    # Create environments
    print("\n Creating environments...")
    env = Run7TrafficEnv(max_queue_length=20, cars_cleared_per_cycle=5)
    env = Monitor(env, str(logs_run8))
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    eval_env = Run7TrafficEnv(max_queue_length=20, cars_cleared_per_cycle=5)
    eval_env = Monitor(eval_env, str(logs_run8))
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    print("    Environments created")
    
    # Create PPO model
    print(f"\n Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(logs_run8 / f'PPO_8_seed{seed}')
    )
    print(f"    Model created with seed: {seed}")
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_run8),
        log_path=str(logs_run8),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=str(models_run8),
        name_prefix=f'ppo_checkpoint_seed{seed}'
    )
    
    viz_callback = ProgressVisualizationCallback(
        viz_dir=str(viz_run8),
        eval_log_path=str(logs_run8),
        seed=seed
    )
    
    callback_list = CallbackList([eval_callback, checkpoint_callback, viz_callback])
    
    # Train
    print(f"\n STARTING TRAINING...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Estimated time: ~{total_timesteps // 60000} minutes")
    print("\n" + "="*70)
    
    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "="*70)
    print(f" TRAINING COMPLETE (Seed {seed})")
    print(f"   Duration: {training_duration}")
    print("="*70)
    
    # Save final model
    final_model_path = models_run8 / f'ppo_final_seed{seed}.zip'
    model.save(str(final_model_path))
    print(f"\n Final model saved: {final_model_path}")
    
    vec_normalize_path = models_run8 / f'vec_normalize_seed{seed}.pkl'
    env.save(str(vec_normalize_path))
    print(f" VecNormalize saved: {vec_normalize_path}")
    
    # Create visualizations and summaries
    stats = create_training_plots(str(logs_run8), str(viz_run8), seed)
    
    if stats:
        save_training_summary(stats, seed, str(results_run8))
        
        print("\n" + "="*70)
        print(" SEED TRAINING COMPLETE")
        print("="*70)
        print(f"\n Seed: {seed}")
        print(f" Final Reward: {stats['final_reward']:.1f}")
        print(f" Best Reward: {stats['best_reward']:.1f}")
        print(f" Improvement: {stats['improvement']:.1f}")
        print(f"  Duration: {training_duration}")
        
        print(f"\n Files saved in:")
        print(f"   logs/hardware_ppo/run_8/seed_{seed}/")
        print(f"   models/hardware_ppo/run_8/seed_{seed}/")
        print(f"   results/run_8/seed_{seed}/")
        print(f"   visualizations/run_8/seed_{seed}/")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
